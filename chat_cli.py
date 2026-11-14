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
    """曲线自适应显存治理（无档位）。

    三个连续控制量，全部是空闲显存的光滑函数（smoothstep 插值，无拐点）：
    - enc_cap：编码上下文上限，(0.08GB, 96) → (0.45GB, 512)；
    - gen_tokens：回复长度，(0.08GB, 48) → (0.45GB, 128)；
    - 学习占空比 p_learn：(0.08GB, 0.34) → (0.30GB, 1.0)，累加器（Bresenham）
      实现无级节奏——0.87 就是每 10 轮平滑地学 8.7 次，而不是死板周期。
    时间平滑：水位走 EMA；OOM 用"压力"标量平滑收紧并逐轮指数衰减。
    学习占空比永不为 0——学习可以无限放慢，但从不停止。
    """

    ENC_MAX, ENC_MIN = 512, 96
    GEN_MAX, GEN_MIN = 128, 48
    ANCHOR_HI = 0.45 * 1024**3      # 达到此空闲即全速
    ANCHOR_LO = 0.08 * 1024**3      # 低于此空闲贴地滑行
    DUTY_HI, DUTY_LO = 1.0, 0.34
    DUTY_ANCHOR_HI = 0.30 * 1024**3
    SKIP_FREE = int(0.10 * 1024**3)   # 连生成都不安全的物理底线
    EMA_ALPHA = 0.30
    OOM_JUMP, OOM_DECAY, OOM_DUTY_PENALTY = 0.5, 0.85, 0.5
    DUTY_FLOOR = 0.25

    def __init__(self) -> None:
        self.ema = None
        self.pressure = 0.0
        self.accumulator = 0.0

    def free_bytes(self) -> int:
        try:
            free, _total = torch.cuda.mem_get_info()
            # 驱动层"空闲"不含 torch 缓存分配器扣住的空闲块——它们对
            # torch 是立即可复用的。真实余量 = 驱动空闲 + 可复用缓存。
            cached_freeable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            return int(free) + max(0, cached_freeable)
        except Exception:
            return 1 << 62      # 读不到就视为充裕，交给 OOM 兜底

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _interp(self, free: float, lo_free: float, hi_free: float, lo_value: float, hi_value: float) -> float:
        t = self._smoothstep((free - lo_free) / max(1, hi_free - lo_free))
        return lo_value + (hi_value - lo_value) * t

    def bump_pressure(self) -> None:
        self.pressure = min(1.0, self.pressure + self.OOM_JUMP)
        torch.cuda.empty_cache()

    def plan(self, learn_requested: bool) -> tuple[int, int, bool, float, float, float]:
        """返回 (enc_cap, gen_tokens, 本轮是否学习, 学习占空比, 压力, 空闲字节)。"""
        free = self.observe()
        # OOM 压力平滑收紧有效余量：不必撞第二次墙才开始让路
        effective = free * (1.0 - 0.35 * self.pressure)
        enc_cap = int(self._interp(effective, self.ANCHOR_LO, self.ANCHOR_HI, self.ENC_MIN, self.ENC_MAX))
        gen_tokens = int(self._interp(effective, self.ANCHOR_LO, self.ANCHOR_HI, self.GEN_MIN, self.GEN_MAX))
        duty = self._interp(effective, self.ANCHOR_LO, self.DUTY_ANCHOR_HI, self.DUTY_LO, self.DUTY_HI)
        duty = max(self.DUTY_FLOOR, min(self.DUTY_HI, duty * (1.0 - self.OOM_DUTY_PENALTY * self.pressure)))
        self.accumulator += duty
        learn_this = learn_requested and self.accumulator >= 1.0
        if learn_this:
            self.accumulator -= 1.0
        return enc_cap, gen_tokens, learn_this, duty, self.pressure, free

    def observe(self) -> float:
        free = float(self.free_bytes())
        if self.ema is None:
            self.ema = free
        else:
            self.ema += self.EMA_ALPHA * (free - self.ema)
        self.pressure *= self.OOM_DECAY
        return self.ema


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

        enc_cap, gen_tokens, learn_this, duty, pressure, free = governor.plan(args.learn)
        print(
            f"[自适应] 空闲 {free / 1024**3:.2f}GB · enc≤{enc_cap} · 回复≤{gen_tokens}"
            f" · 学习占空 {duty:.2f} · 压力 {pressure:.2f}",
            flush=True,
        )
        if free < governor.SKIP_FREE:
            torch.cuda.empty_cache()
            print(f"[显存告急（{free / 1024**3:.2f}GB），本轮跳过。关闭占显存的应用后输入继续。]", flush=True)
            continue
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
            governor.bump_pressure()
            print("[显存不足，本轮生成失败（压力已上调，系统自动收紧）。输入继续。]", flush=True)
            continue
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
                    governor.bump_pressure()
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
                        governor.bump_pressure()
                        optimizer.zero_grad(set_to_none=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
