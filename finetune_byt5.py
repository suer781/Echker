"""LoRA-free direct fine-tune of google/byt5-small on the dialogue pairs.

The model itself learns: the attention q/v projection weights (22.6M real
parameters) receive gradient updates via Adafactor, exactly as probed by
byt5_probe.py (peak 1.42 GB at seq512, ~0.44 s/step on the GTX 1060).
No adapters, no frozen-base indirection — the base's own weights move.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor

ROOT = Path(__file__).resolve().parent
NAME = "google/byt5-small"
DATA = ROOT / "finetune_data" / "dialogue_pairs.jsonl"
OUTPUT = ROOT / "byt5_model"
ENC_CAP = 512
DEC_CAP = 128


def encode_pair(tok, user: str, assistant: str):
    enc = tok(user, add_special_tokens=True).input_ids[-ENC_CAP:]
    dec = tok(assistant, add_special_tokens=False).input_ids[-DEC_CAP:] + [tok.eos_token_id]
    return enc, dec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", help="continue from byt5_model")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "finetune_data", help="数据目录（读取其中全部 .jsonl）")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT if args.resume else NAME, dtype=torch.float32)
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = ("block" in name) and (".q." in name or ".v." in name)
        if param.requires_grad:
            trainable.append(name)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"trainable_tensors": len(trainable), "trainable_params": count}, ensure_ascii=False))

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, scale_parameter=False, relative_step=False,
    )

    examples_raw = []
    for data_file in sorted(args.data_dir.glob("*.jsonl")):
        rows = 0
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                examples_raw.append((row["user"], row["assistant"]))
                rows += 1
        print(json.dumps({"data": data_file.name, "pairs": rows}, ensure_ascii=False))
    print(json.dumps({"examples": len(examples_raw)}, ensure_ascii=False))

    model.train()
    random.seed(11)
    ema = None
    step = 0
    skipped = 0
    started = time.time()
    enc_cap = 1024   # 自适应上限：OOM 降回 512（长指令不再截断）
    for epoch in range(args.epochs):
        random.shuffle(examples_raw)
        for user_text, assistant_text in examples_raw:
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            dec = tok(assistant_text, add_special_tokens=False).input_ids[-DEC_CAP:] + [tok.eos_token_id]
            inputs = torch.tensor([enc], device="cuda")
            labels = torch.tensor([dec], device="cuda")
            try:
                loss = model(input_ids=inputs, labels=labels).loss
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                if enc_cap > 512:
                    enc_cap = 512
                    print(json.dumps({"adaptive": "OOM → 上下文上限降至 512"}, ensure_ascii=False), flush=True)
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.item())
            ema = value if ema is None else 0.98 * ema + 0.02 * value
            if step % 200 == 0:
                rate = step / max(1e-9, time.time() - started)
                print(json.dumps({"step": step, "ema_loss": round(ema, 4), "steps_per_s": round(rate, 2),
                                  "enc_cap": enc_cap}, ensure_ascii=False), flush=True)
            if step % 2000 == 0:
                model.config.use_cache = True
                model.save_pretrained(OUTPUT, safe_serialization=True)
                tok.save_pretrained(OUTPUT)
                model.config.use_cache = False
                print(json.dumps({"checkpoint": step}), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(OUTPUT, safe_serialization=True)
    tok.save_pretrained(OUTPUT)
    print(json.dumps({"done": True, "steps": step, "ema_loss": round(ema, 4), "skipped_oom": skipped,
                      "enc_cap_final": enc_cap, "saved": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
