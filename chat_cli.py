"""Interactive chat with the fine-tuned byt5-small — and real online learning.

Every turn can update the model's own attention q/v weights on the exchange
(TTT-style, one Adafactor step), so the model keeps learning while you talk.
The language core is Google's pre-trained ByT5; what the project fine-tuned
and keeps learning online is the dialogue behaviour on top of it.

学习统计只认正确字词构成的字节：逐字节 UTF-8 有效性 + 字符类别合法性，
错误字节在损失里被屏蔽（-100），模型物理上学不到它们；有效率过低的整轮拒学。

Usage:
  python chat_cli.py                     # chat with per-turn online learning
  python chat_cli.py --no-learn          # chat only
  commands: /save  /exit
"""
from __future__ import annotations

import argparse
import codecs
import os
from pathlib import Path
import sys
import time
import unicodedata

# 可扩展分段分配器：显著减少长会话反复分配/释放造成的显存碎片。
# Windows 不支持（torch 会告警回退），仅在非 Windows 平台启用。
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor

ROOT = Path(__file__).resolve().parent
FINETUNED = ROOT / "byt5_model"
BASE = "google/byt5-small"
GEN_TOKENS = 128
ENC_CAP = 512
DEC_CAP = 192


def build_byte_maps(tok) -> tuple[dict[int, int], dict[int, int]]:
    vocab = tok.get_vocab()
    byte_to_id = {b: vocab[chr(b)] for b in range(256)}
    id_to_byte = {i: b for b, i in byte_to_id.items()}
    return byte_to_id, id_to_byte


class VramGovernor:
    """自适应显存治理：水位监测 + 分级降档 + OOM 拥塞退避。

    设计约束（用户要求）：**学习永不停止，只放慢节奏**——等级越低，
    上下文越小、学习周期越长（每 2/3 轮一次），但学习始终在进行。
    OOM 后强制保守若干轮（乘性降），平稳后逐级恢复（加性升），类似
    TCP 拥塞控制。水位用 cudaMemGetInfo 读取整机空闲显存（加回 torch
    可复用缓存），桌面应用波动会被自动感知。
    """

    # 等级 → (编码上限, 回复上限, 学习周期轮数) —— 学习周期 ≥ 1，永不为 0
    LEVELS = [
        (512, 128, 1),    # 等级 0：全速学习（每轮）
        (320, 96, 1),     # 等级 1：缩上下文，每轮学习
        (192, 64, 2),     # 等级 2：更小上下文，每 2 轮学习一次
        (128, 48, 3),     # 等级 3：最小上下文，每 3 轮学习一次
    ]
    # 空闲显存台阶（字节）：低于下一档就降一级。台阶按实测校准：
    # 学习一步的增量峰值 ~0.22GB，阈值留少量余量，激进档由 OOM 退避兜底。
    FREE_FLOORS = [int(0.40 * 1024**3), int(0.26 * 1024**3), int(0.15 * 1024**3)]
    SKIP_FREE = int(0.10 * 1024**3)           # 低于此值整轮跳过（生成也无法安全进行）
    CLEAN_TURNS_PER_RECOVERY = 10             # 每平稳 N 轮 OOM 退避恢复一级

    def __init__(self) -> None:
        self.backoff = 0        # OOM 强制等级下限（乘性降）
        self.clean_streak = 0
        self.last_level = None

    def free_bytes(self) -> int:
        try:
            free, _total = torch.cuda.mem_get_info()
            # 驱动层“空闲”不含 torch 缓存分配器扣住的空闲块——它们对
            # torch 是立即可复用的。真实余量 = 驱动空闲 + 可复用缓存，
            # 否则第一轮学习后缓存变大会被误判为 0。
            cached_freeable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            return int(free) + max(0, cached_freeable)
        except Exception:
            return 1 << 62      # 读不到就视为充裕，交给 OOM 兜底

    def note_ok(self) -> None:
        self.clean_streak += 1
        if self.clean_streak >= self.CLEAN_TURNS_PER_RECOVERY and self.backoff > 0:
            self.backoff -= 1
            self.clean_streak = 0

    def note_oom(self) -> None:
        self.backoff = min(len(self.LEVELS) - 1, max(self.backoff, 2))
        self.clean_streak = 0

    def plan(self, turn_index: int) -> tuple[int, int, int, bool, int, int]:
        """返回 (等级, enc_cap, gen_tokens, 本轮是否学习, 学习周期, 当前空闲字节)。

        学习永远被调度，只是低水位时周期变长（每 2/3 轮）+ 上下文变小。
        """
        free = self.free_bytes()
        level = len(self.LEVELS) - 1
        for index, floor in enumerate(self.FREE_FLOORS):
            if free >= floor:
                level = index
                break
        level = min(len(self.LEVELS) - 1, max(level, self.backoff))
        enc_cap, gen_tokens, learn_every = self.LEVELS[level]
        learn_this = turn_index % learn_every == 0
        return level, enc_cap, gen_tokens, learn_this, learn_every, free


def decode_response(ids: list[int], id_to_byte: dict[int, int]) -> str:
    raw = bytearray()
    for token in ids:
        if token in (0, 1, 2):  # pad / eos / unk
            if token == 1:
                break
            continue
        byte = id_to_byte.get(token)
        if byte is not None:
            raw.append(byte)
    return raw.decode("utf-8", errors="replace")


def response_bytes(ids: list[int], id_to_byte: dict[int, int]) -> bytearray:
    raw = bytearray()
    for token in ids:
        if token in (0, 1, 2):  # pad / eos / unk 不是字节
            if token == 1:
                break
            continue
        byte = id_to_byte.get(token)
        if byte is not None:
            raw.append(byte)
    return raw


def _char_ok(ch: str) -> bool:
    """正确字词的字符类别：字母/数字/标点/符号/空白；控制符与替换符是错误。"""
    if ch == "\ufffd":
        return False
    category = unicodedata.category(ch)
    return category[0] in {"L", "N", "P", "S"} or ch.isspace()


def utf8_byte_mask(raw: bytes) -> tuple[list[bool], float]:
    """逐字节有效性统计。

    mask[i]=True 表示 raw[i] 属于正确字词的字节：
    - 能构成合法 UTF-8 序列（坏序列字节、尾部截断字节标记 False）；
    - 解出的字符属于正常类别（控制符 / U+FFFD 对应字节标记 False）。
    返回 (mask, valid_ratio)。
    """
    mask = [True] * len(raw)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    i = 0
    while i < len(raw):
        pending = len(decoder.getstate()[0])
        try:
            emitted = decoder.decode(raw[i : i + 1])
        except UnicodeDecodeError:
            decoder.reset()
            mask[i] = False
            i += 1
            continue
        i += 1
        if emitted and not _char_ok(emitted):
            for j in range(max(0, i - pending - 1), i):
                mask[j] = False
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        buffered = decoder.getstate()[0]
        for j in range(max(0, len(raw) - len(buffered)), len(raw)):
            mask[j] = False
    valid = sum(mask)
    return mask, (valid / len(raw) if raw else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", dest="learn", action="store_true", default=True)
    parser.add_argument("--no-learn", dest="learn", action="store_false")
    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")
    args = parser.parse_args()

    source = FINETUNED if (FINETUNED / "config.json").exists() else BASE
    print(f"加载 {source}（fp32, GPU）…", flush=True)
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source, dtype=torch.float32)
    if not torch.cuda.is_available():
        print("[错误] 未检测到可用 CUDA。本系统按设计在 GPU 上推理与学习；请检查显卡驱动与 PyTorch CUDA 安装后重试。")
        return 1
    try:
        model.cuda()
    except torch.cuda.OutOfMemoryError:
        print("[错误] 显存不足（CUDA OOM）。byt5-small fp32 需要 ~1.7GB 空闲显存：")
        print("       - 关闭占用显存的应用（壁纸引擎、浏览器硬件加速、游戏等）后重试；")
        print("       - 或在显存更大的显卡上运行。")
        return 1
    model.gradient_checkpointing_enable()
    byte_to_id, id_to_byte = build_byte_maps(tok)

    if args.learn:
        model.train()
        trainable = [p for n, p in model.named_parameters() if "block" in n and (".q." in n or ".v." in n)]
        optimizer = Adafactor(trainable, lr=1e-4, scale_parameter=False, relative_step=False)
        print(f"在线学习开启：{sum(p.numel() for p in trainable):,} 个参数逐轮真权重更新", flush=True)
    else:
        model.eval()
        trainable = []
        optimizer = None
        print("在线学习关闭（--learn 可开启）", flush=True)

    print("输入文字开始对话；/save 保存，/exit 退出。", flush=True)
    governor = VramGovernor()
    turns_since_save = 0
    turn_index = 0
    while True:
        try:
            user_text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        if not user_text:
            continue
        if user_text == "/exit":
            if args.learn and not args.no_save:
                model.config.use_cache = True
                model.save_pretrained(FINETUNED, safe_serialization=True)
                tok.save_pretrained(FINETUNED)
                print("已保存并退出。")
            break
        if user_text == "/save":
            if args.no_save:
                print("冒烟模式（--no-save）：未保存。")
                continue
            model.config.use_cache = True
            model.save_pretrained(FINETUNED, safe_serialization=True)
            tok.save_pretrained(FINETUNED)
            model.config.use_cache = False
            print("已保存。")
            continue

        level, enc_cap, gen_tokens, learn_this, learn_every, free = governor.plan(turn_index)
        if level != governor.last_level:
            governor.last_level = level
            cadence = "每轮" if learn_every == 1 else f"每{learn_every}轮"
            print(
                f"[自适应] 空闲显存 {free / 1024**3:.2f}GB → 等级{level}"
                f"（编码≤{enc_cap}，回复≤{gen_tokens}，学习{cadence}）",
                flush=True,
            )
        if free < governor.SKIP_FREE:
            torch.cuda.empty_cache()
            print(f"[显存告急（{free / 1024**3:.2f}GB），本轮跳过。关闭占显存的应用后输入继续。]", flush=True)
            continue
        if level >= 2:
            torch.cuda.empty_cache()   # 低水位时把缓存还给驱动，降低整机压力
        response_ids = None
        try:
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            inputs = torch.tensor([enc], device="cuda")
            start = time.perf_counter()
            model.eval()
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                )
            response_ids = output[0].tolist()
        except torch.cuda.OutOfMemoryError:
            governor.note_oom()
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（已自动降档）。关闭占显存的应用后输入继续。]", flush=True)
            continue
        governor.note_ok()
        response = decode_response(response_ids, id_to_byte)
        print(f"助手: {response}")
        print(f"[生成 {time.perf_counter() - start:.2f}s]", flush=True)

        if learn_this and response.strip():
            raw = response_bytes(response_ids, id_to_byte)
            mask, ratio = utf8_byte_mask(bytes(raw))
            valid_bytes = sum(mask)
            # 错误字节直接从损失中屏蔽：labels 在非法/坏字符位置 = -100，
            # 模型对这些字节不产生梯度，物理上学不到错误字节。
            if valid_bytes < 4 or ratio < 0.4:
                print(f"[本轮学习跳过：正确字节占比 {ratio:.0%} 过低，整轮拒学]", flush=True)
            else:
                byte_labels = [
                    byte_to_id[byte] if ok else -100
                    for byte, ok in zip(raw, mask)
                ][-DEC_CAP:] + [tok.eos_token_id]
                label_tensor = torch.tensor([byte_labels], device="cuda")
                try:
                    model.train()
                    loss = model(input_ids=inputs, labels=label_tensor).loss
                    loss_value = float(loss.item())
                except torch.cuda.OutOfMemoryError:
                    governor.note_oom()
                    torch.cuda.empty_cache()
                    print("[显存不足，本轮学习跳过]", flush=True)
                    continue
                # 第二道闸门：屏蔽后的损失仍过高说明模型对正确部分也毫无把握
                # （多半是退化输出），把垃圾当教材会自我污染权重——跳过。
                if loss_value > 2.0:
                    print(f"[本轮学习跳过：loss={loss_value:.4f} 超过阈值 2.0]", flush=True)
                else:
                    try:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step()
                    except torch.cuda.OutOfMemoryError:
                        governor.note_oom()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        print("[显存不足，本轮学习跳过（部分梯度已回滚）]", flush=True)
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)
                    if turns_since_save >= args.save_every and not args.no_save:
                        model.config.use_cache = True
                        model.save_pretrained(FINETUNED, safe_serialization=True)
                        tok.save_pretrained(FINETUNED)
                        model.config.use_cache = False
                        turns_since_save = 0
                        print("[自动保存]", flush=True)
        turn_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
