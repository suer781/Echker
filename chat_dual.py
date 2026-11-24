"""已归档的双半球 ByT5 CLI；新主线请运行 chat_qwen.py。

仅为兼容历史检查点而保留，不属于 Qwen 主线。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

import torch

from chat_cli import VramGovernor, build_byte_maps, decode_response, utf8_byte_mask
from byt5_dual import BASE, Byt5DualBrain, Seq2SeqSample

ROOT = Path(__file__).resolve().parent
FINETUNED = ROOT / "byt5_model"
DEFAULT_STATE = ROOT / "state_dual"


def aligned_response(ids: list[int], id_to_byte: dict[int, int]) -> tuple[bytearray, list[int]]:
    """从生成结果提取原始字节与对齐的 token 序列（跳过 pad/unk，停在 eos）。"""
    raw = bytearray()
    tokens: list[int] = []
    for token in ids:
        if token in (0, 1, 2):
            if token == 1:
                break
            continue
        byte = id_to_byte.get(token)
        if byte is not None:
            raw.append(byte)
            tokens.append(token)
    return raw, tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--learn", dest="learn", action="store_true", default=True)
    parser.add_argument("--no-learn", dest="learn", action="store_false")
    parser.add_argument("--no-save", action="store_true", help="冒烟模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（0=贪心；0.6-0.9 更活泼）")
    parser.add_argument("--history-turns", type=int, default=3,
                        help="编码器携带的最近对话轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")
    args = parser.parse_args()

    source = FINETUNED if (FINETUNED / "config.json").exists() else BASE
    print(f"双半球初始化（两侧均为 {source}）…", flush=True)
    brain = Byt5DualBrain(source, args.state_dir)
    restored = brain.load()
    print(f"检查点：{'已恢复' if restored else '全新（两侧继承微调权重）'}", flush=True)
    brain.ensure_active_gpu()
    active = brain.active_h()
    tok = active.tokenizer
    byte_to_id, id_to_byte = build_byte_maps(tok)

    governor = VramGovernor()
    if args.learn:
        print("双半球在线学习开启：每轮都更新活动侧权重，被动侧巩固，曲线自适应切换", flush=True)
    else:
        print("纯对话模式（--no-learn）：本轮起不学习、不巩固、不切换", flush=True)
    print("输入文字开始对话；/status 状态，/save 保存，/exit 退出。", flush=True)
    turns_since_save = 0
    history = []   # 多轮上下文（用户/助手）

    while True:
        try:
            user_text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        except UnicodeDecodeError as exc:
            print(f"[输入编码错误，已忽略该行：{exc}]", flush=True)
            continue
        if not user_text:
            continue
        # P2-9：以 / 开头但未定义的输入一律按命令处理，绝不喂给模型
        # （否则用户的手滑命令变成训练教材）
        cmd = user_text.lower()   # P1-1：门与分发统一用小写，/Exit 不再穿透
        if user_text.startswith("/") and cmd not in ("/exit", "/save", "/status"):
            print("[未知命令。可用命令：/status /save /exit]", flush=True)
            continue
        if cmd == "/exit":
            if not args.no_save:
                brain.save()
                print("已保存并退出。")
            else:
                print("冒烟模式：未保存，退出。")
            break
        if cmd == "/save":
            if args.no_save:
                print("冒烟模式（--no-save）：未保存。")
            else:
                brain.save()
                print("已保存。")
            continue
        if cmd == "/status":
            passive = brain.passive_h()
            print(
                f"active={brain.active} 切换次数={brain.switch_count} "
                f"紧迫度={brain.switch_urgency():.3f} 累计={brain.switch_urgency_acc:.3f} "
                f"被动侧容量={passive.debt.repayment_capacity:.3f} "
                f"被动侧待偿债务={len(passive.debt)}",
                flush=True,
            )
            continue

        enc_cap, gen_tokens, intensity, pressure, free, lr_scale = governor.plan(args.learn)
        intensity_text = f" · 学习强度 {intensity:.2f}" if args.learn else ""
        print(
            f"[自适应] 空闲 {free / 1024**3:.2f}GB · enc≤{enc_cap} · 回复≤{gen_tokens}"
            f"{intensity_text} · 压力 {pressure:.2f} · 半球 {brain.active}",
            flush=True,
        )
        if free < governor.SKIP_FREE:
            torch.cuda.empty_cache()
            print(f"[显存告急（{free / 1024**3:.2f}GB），本轮跳过。]", flush=True)
            continue

        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += "用户：" + h_user + chr(10) + "助手：" + h_resp + chr(10)
        full_text = context_text + "用户：" + user_text + chr(10) + "助手："
        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(full_text, add_special_tokens=True).input_ids[-enc_cap:]
            inputs = torch.tensor([enc], device="cuda")
            start = time.perf_counter()
            active_h.gpu_model.eval()
            active_h.gpu_model.config.use_cache = True   # 生成走 KV 缓存（快）
            with torch.no_grad():
                sampling = args.temperature > 0
                output = active_h.gpu_model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens, min_new_tokens=8,
                    do_sample=sampling,
                    temperature=max(args.temperature, 1e-5) if sampling else None,
                    top_k=40 if sampling else None,
                    num_beams=1,
                    repetition_penalty=1.5,
                )
            response_ids = output[0].tolist()
            active_h.gpu_model.config.use_cache = False
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            # P1-4：实时解析当前活动侧，防未绑定/陈旧引用崩溃
            gpu_model = brain.active_h().gpu_model
            if gpu_model is not None:
                gpu_model.config.use_cache = False
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（压力已上调）。输入继续。]", flush=True)
            continue

        raw, resp_tokens = aligned_response(response_ids, id_to_byte)
        response = raw.decode("utf-8", errors="replace")
        print(f"助手({brain.active}): {response}")
        print(f"[生成 {time.perf_counter() - start:.2f}s]", flush=True)

        if args.learn and response.strip():
            mask, ratio = utf8_byte_mask(bytes(raw))
            valid_bytes = sum(mask)
            if valid_bytes < 4 or ratio < 0.4:
                print(f"[本轮学习跳过：正确字节占比 {ratio:.0%} 过低，整轮拒学]", flush=True)
            else:
                # 错误字节（mask=False）被剔除：只让正确字词的字节进入学习
                clean_bytes = [raw[i] for i, ok in enumerate(mask) if ok]
                # P2-6：学习与生成使用同一段截断后的用户上下文
                user_bytes = list(user_text.encode("utf-8"))[-enc_cap:]
                # P0-2：单字节输入会让账本校验崩溃——保底到 ≥2 字节
                if len(user_bytes) < 2:
                    user_bytes = (user_bytes + [32, 32])[:2]
                exchange_ids = user_bytes[-256:]
                # TTT 语义：学习目标 = 用户的观测输入（真实人类语言），
                # 模型自己的回应永远不进训练——斩断自训练塌缩回路。
                # 观测样本无条件入账（拒学轮也入：退化半球借此积累切换压力）。
                brain.admit_exchange_sample(Seq2SeqSample(
                    ids=user_bytes[-256:], labels=user_bytes[-192:],
                    weight=1.0, source=f"stream:{brain.active}",
                ))
                try:
                    # TTT：编码器=多轮上下文+当前输入，解码器=当前用户输入
                    # （回忆当前轮）；回应永不进训练
                    result = active_h.learn(
                        list(full_text.encode("utf-8"))[-enc_cap:],
                        list(user_text.encode("utf-8"))[-192:], intensity)
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    active_h.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print("[显存不足，本轮学习跳过（梯度已回滚）]", flush=True)
                    continue
                except ValueError as exc:
                    print(f"[本轮学习跳过：样本构造异常 {exc}]", flush=True)
                    continue
                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} TTT 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)
                # 观测数据无条件入账（用户输入是真实语言；回应永不入账）
                brain.admit_exchange_sample(Seq2SeqSample(
                    ids=user_bytes[-256:], labels=user_bytes[-192:],
                    weight=1.0, source=f"stream:{brain.active}",
                ))
                # 上下文回忆巩固：随机抽一轮历史，给定前文回忆其用户输入
                # ——把上下文压进权重，编码器因此可以减负
                if history and intensity > 0:
                    try:
                        ri = random.randrange(len(history))
                        r_user, _r_resp = history[ri]
                        r_prefix = "".join(
                            "用户：" + u + chr(10) + "助手：" + r + chr(10)
                            for u, r in history[:ri])
                        r_text = r_prefix + "用户：" + r_user + chr(10) + "助手："
                        recall = active_h.learn(
                            list(r_text.encode("utf-8"))[-enc_cap:],
                            list(r_user.encode("utf-8"))[-192:],
                            intensity * 0.5)
                        if recall.get("learned"):
                            print(f"[回忆巩固 #{ri}] loss={recall['loss']:.4f}", flush=True)
                    except torch.cuda.OutOfMemoryError:
                        active_h.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                history.append((user_text, response))
                del history[:-args.history_turns]

        # 被动侧巩固（换入-训练-换回，债务驱动）——P1-1：--no-learn 时跳过
        if args.learn:
            try:
                result = brain.consolidate_passive(intensity)
                if result:
                    print(f"[{brain.passive_h().name} 巩固 steps={result['steps']} loss={result.get('loss')}]", flush=True)
            except torch.cuda.OutOfMemoryError:
                governor.bump_pressure()
                torch.cuda.empty_cache()
                brain.ensure_active_gpu()
                print("[显存不足，本轮巩固跳过]", flush=True)

            # 曲线自适应切换（紧迫度累加，满 1 触发）
            try:
                if brain.maybe_switch():
                    print(f"[半球切换 #{brain.switch_count}] → {brain.active}（紧迫度曲线已达标，债务交换+2%插值完成）", flush=True)
            except torch.cuda.OutOfMemoryError:
                governor.bump_pressure()
                torch.cuda.empty_cache()
                brain.ensure_active_gpu()
                print("[切换时显存不足，已回滚驻留，下轮重试]", flush=True)

        if turns_since_save >= args.save_every and not args.no_save:
            brain.save()
            turns_since_save = 0
            print("[自动保存]", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
