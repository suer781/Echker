"""Interactive chat with the fine-tuned byt5-small — and real online learning.

Every turn can update the model's own attention q/v weights on the exchange
(TTT-style, one Adafactor step), so the model keeps learning while you talk.
The language core is Google's pre-trained ByT5; what the project fine-tuned
and keeps learning online is the dialogue behaviour on top of it.

Usage:
  python chat_cli.py                     # chat with per-turn online learning
  python chat_cli.py --no-learn          # chat only
  commands: /save  /exit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

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
    model.cuda()
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

        enc = tok(user_text, add_special_tokens=True).input_ids[-ENC_CAP:]
        inputs = torch.tensor([enc], device="cuda")
        start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=GEN_TOKENS,
                do_sample=True,
                top_k=40,
                temperature=0.9,
                repetition_penalty=1.15,
            )
        response_ids = output[0].tolist()
        response = decode_response(response_ids, id_to_byte)
        print(f"助手: {response}")
        print(f"[生成 {time.perf_counter() - start:.2f}s]", flush=True)

        if args.learn and response.strip():
            labels = tok(response, add_special_tokens=False).input_ids[-DEC_CAP:] + [tok.eos_token_id]
            label_tensor = torch.tensor([labels], device="cuda")
            model.train()
            loss = model(input_ids=inputs, labels=label_tensor).loss
            loss_value = float(loss.item())
            # 学习质量闸门：交换损失过高说明模型对这段内容毫无把握（多半是
            # 自己的退化输出），把垃圾当教材会自我污染权重——跳过本轮学习。
            if loss_value > 2.0:
                print(f"[本轮学习跳过：loss={loss_value:.4f} 超过阈值 2.0]", flush=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
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
