"""双半球 ByT5 交互 CLI：两侧真权重在线学习 + 曲线自适应切换。

每轮流程：
  活动侧生成 → 字节有效性闸门 → 活动侧真权重学习（强度曲线缩放学习率）
  → 新数据入活动侧债务（切换压力）→ 被动侧巩固（换入 GPU 训练后换回）
  → 曲线紧迫度累计 → 满足则切换（债务交换 + 2% 对称插值 + 换驻留）

Usage:
  python chat_dual.py            # 双半球对话（每轮都学习）
  python chat_dual.py --no-learn # 只对话
  /save  /status  /exit
"""
from __future__ import annotations

import argparse
from pathlib import Path
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
    print("双半球在线学习开启：每轮都更新活动侧权重，被动侧巩固，曲线自适应切换", flush=True)
    print("输入文字开始对话；/status 状态，/save 保存，/exit 退出。", flush=True)
    turns_since_save = 0

    while True:
        try:
            user_text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        if not user_text:
            continue
        if user_text == "/exit":
            if not args.no_save:
                brain.save()
                print("已保存并退出。")
            else:
                print("冒烟模式：未保存，退出。")
            break
        if user_text == "/save":
            if args.no_save:
                print("冒烟模式（--no-save）：未保存。")
            else:
                brain.save()
                print("已保存。")
            continue
        if user_text == "/status":
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
        print(
            f"[自适应] 空闲 {free / 1024**3:.2f}GB · enc≤{enc_cap} · 回复≤{gen_tokens}"
            f" · 学习强度 {intensity:.2f} · 压力 {pressure:.2f} · 半球 {brain.active}",
            flush=True,
        )
        if free < governor.SKIP_FREE:
            torch.cuda.empty_cache()
            print(f"[显存告急（{free / 1024**3:.2f}GB），本轮跳过。]", flush=True)
            continue

        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            inputs = torch.tensor([enc], device="cuda")
            start = time.perf_counter()
            active_h.gpu_model.eval()
            with torch.no_grad():
                output = active_h.gpu_model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                )
            response_ids = output[0].tolist()
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
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
                try:
                    result = active_h.learn(list(user_text.encode("utf-8")), clean_bytes, intensity)
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    print("[显存不足，本轮学习跳过]", flush=True)
                    continue
                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                    # 新数据进入活动侧债务（字节为单位）：构成被动侧的切换压力
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=list(user_text.encode("utf-8"))[-256:],
                        labels=clean_bytes[-192:],
                        weight=1.0,
                        source=f"stream:{brain.active}",
                    ))
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)

        # 被动侧巩固（换入-训练-换回，债务驱动）
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
