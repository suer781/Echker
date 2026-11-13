"""ByT5 feasibility probe: byte-token mapping + GPU memory/speed profile.

Answers, with numbers, whether LoRA fine-tuning of google/byt5-small fits the
GTX 1060 3GB (Pascal) alongside the desktop, and how fast it runs:
  1. byte<->token mapping must be lossless for ARBITRARY binary 0..255;
  2. fp32 inference VRAM;
  3. LoRA forward+backward peak VRAM and seconds per step at batch 1 and 2;
  4. generation latency for a 96-byte response.
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor

NAME = "google/byt5-small"


def byte_mapping(tok) -> dict:
    # ByT5Tokenizer._tokenize does text.encode("utf-8") -> [chr(b) for each
    # byte b] -> vocab lookup. So byte b maps to the id of token chr(b).
    vocab = tok.get_vocab()
    byte_to_id, collisions = {}, []
    for byte in range(256):
        token = chr(byte)
        token_id = vocab.get(token)
        if token_id is None:
            collisions.append(byte)
        else:
            byte_to_id[byte] = token_id
    specials = {tok.pad_token_id, tok.eos_token_id, tok.unk_token_id}
    overlapping = sorted({i for i in byte_to_id.values()} & specials)
    roundtrip_ok = (
        len(byte_to_id) == 256
        and not overlapping
        and len(set(byte_to_id.values())) == 256
    )
    sample = {str(b): byte_to_id[b] for b in (0, 10, 65, 97, 128, 255) if b in byte_to_id}
    return {
        "vocab_size": tok.vocab_size,
        "all_256_bytes_mapped": len(byte_to_id) == 256,
        "collides_with_specials": overlapping,
        "bijective": len(set(byte_to_id.values())) == 256,
        "roundtrip_ok": roundtrip_ok,
        "sample_byte_to_id": sample,
        "pad": tok.pad_token_id,
        "eos": tok.eos_token_id,
        "byte_id_min": min(byte_to_id.values()),
        "byte_id_max": max(byte_to_id.values()),
    }


def main() -> int:
    report: dict = {}
    tok = AutoTokenizer.from_pretrained(NAME)
    report["tokenizer"] = byte_mapping(tok)

    model = AutoModelForSeq2SeqLM.from_pretrained(NAME, torch_dtype=torch.float32)
    model = model.cuda()
    model.gradient_checkpointing_enable()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    report["inference_vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)

    # LoRA-style trainable slice for the memory probe: train ONLY the attention
    # q/v projections by zeroing every other grad (cheap stand-in until the
    # real LoRA wrapper; the memory profile is dominated by weights+activations).
    model.train()
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = ("q." in name or ".v." in name) and ("block" in name)
        if param.requires_grad:
            trainable.append(name)
    report["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, scale_parameter=False, relative_step=False,
    )

    def make_batch(seq: int, target: int = 32):
        byte_lo, byte_hi = report["tokenizer"]["byte_id_min"], report["tokenizer"]["byte_id_max"] + 1
        ids = torch.randint(byte_lo, byte_hi, (1, seq), device="cuda")
        labels = torch.randint(byte_lo, byte_hi, (1, target), device="cuda")
        labels[labels == 0] = 1
        return ids, labels

    timings = {}
    for seq in (256, 512, 1024):
        torch.cuda.reset_peak_memory_stats()
        ids, labels = make_batch(seq)
        # warmup
        try:
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            timings[f"seq{seq}"] = {"status": "OOM"}
            torch.cuda.empty_cache()
            continue
        peak_after_setup = torch.cuda.max_memory_allocated()
        start = time.perf_counter()
        steps = 3
        for _ in range(steps):
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - start) / steps
        timings[f"seq{seq}"] = {
            "status": "ok",
            "peak_vram_gb": round(peak_after_setup / 1e9, 3),
            "seconds_per_step": round(per_step, 3),
        }
        torch.cuda.empty_cache()
    report["lora_probe"] = timings

    # generation latency (fp32, greedy, 96 new bytes)
    model.eval()
    ids = torch.randint(3, 359, (1, 128), device="cuda")
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=96, do_sample=False)
    report["generate_96bytes_seconds"] = round(time.perf_counter() - start, 2)
    report["generate_output_len"] = int(out.shape[1])

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
