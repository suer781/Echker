"""ByT5-era layer-3 verdict: did dialogue fine-tuning generalize?

Scores teacher-forced cross-entropy on assistant responses drawn from the
1% path-hash holdout files (LogiQA-2.0 test.txt, CMMLU arts.csv) that the
fine-tune pipeline never consumes, for:
  (a) google/byt5-small (fresh pre-trained base), and
  (b) the fine-tuned byt5_model.

Positive reduction = the fine-tune transferred to unseen held-out data
(domain generalization), not just memorization of its training pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare_finetune_data import cmmlu_pairs, is_holdout, logiqa2_jsonl  # noqa: E402

DATA_ROOT = Path(r"O:\数据集")
BASE = "google/byt5-small"
FINETUNED = ROOT / "byt5_model"
MAX_PAIRS_PER_SOURCE = 120


def collect_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    test_txt = DATA_ROOT / "01_LogiQA/LogiQA-2.0/LogiQA2.0-main/logiqa/DATA/LOGIQA/test.txt"
    if test_txt.exists() and is_holdout(test_txt):
        for user, assistant in logiqa2_jsonl(test_txt)[:MAX_PAIRS_PER_SOURCE]:
            pairs.append(("logiqa2_test", user, assistant))
    arts = DATA_ROOT / "04_综合评测/CMMLU/test/arts.csv"
    if arts.exists() and is_holdout(arts):
        for user, assistant in cmmlu_pairs(arts)[:MAX_PAIRS_PER_SOURCE]:
            pairs.append(("cmmlu_arts", user, assistant))
    return pairs


@torch.no_grad()
def score(model, tok, pairs: list[tuple[str, str, str]], enc_cap: int = 512, dec_cap: int = 128) -> dict:
    totals = {"overall": 0.0, "logiqa2_test": 0.0, "cmmlu_arts": 0.0}
    counts = {"overall": 0, "logiqa2_test": 0, "cmmlu_arts": 0}
    for source, user, assistant in pairs:
        enc = tok(user, add_special_tokens=True).input_ids[-enc_cap:]
        dec = tok(assistant, add_special_tokens=False).input_ids[-dec_cap:] + [tok.eos_token_id]
        loss = model(
            input_ids=torch.tensor([enc], device="cuda"),
            labels=torch.tensor([dec], device="cuda"),
        ).loss
        totals["overall"] += float(loss.item())
        counts["overall"] += 1
        totals[source] += float(loss.item())
        counts[source] += 1
    return {key: (totals[key] / counts[key] if counts[key] else None) for key in totals}


def main() -> int:
    pairs = collect_pairs()
    if not pairs:
        print(json.dumps({"error": "no holdout pairs collected"}))
        return 1
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE, dtype=torch.float32).cuda().eval()
    tuned = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED, dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        base_scores = score(base, tok, pairs)
        tuned_scores = score(tuned, tok, pairs)
    reduction = {
        key: round(100.0 * (base_scores[key] - tuned_scores[key]) / base_scores[key], 2)
        for key in base_scores
        if base_scores[key]
    }
    verdict = "PASS" if reduction["overall"] > 0 else "FAIL"
    print(json.dumps({
        "holdout_pairs": len(pairs),
        "base_ce": {k: round(v, 4) for k, v in base_scores.items() if v},
        "finetuned_ce": {k: round(v, 4) for k, v in tuned_scores.items() if v},
        "reduction_percent": reduction,
        "verdict": verdict,
        "interpretation": "正的 reduction = 微调泛化到了从未训练的保留集；负 = 只会背训练对。",
    }, ensure_ascii=False, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
