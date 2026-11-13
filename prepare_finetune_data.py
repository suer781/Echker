"""Build the MB-scale dialogue fine-tune set from the local corpus.

With a pre-trained ByT5 backbone, the corpus is no longer a language teacher
(Google's pre-training did that); fine-tuning only has to inject the project's
dialogue format and the logic-reasoning domain flavor. This script extracts
user/assistant pairs — a few thousand pairs, a few MB — and writes JSONL.

Files in the 1% holdout (never used for training, layer-3 exam set) are
excluded through the same path-hash check the ingest pipeline uses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(r"O:\数据集")
OUTPUT = ROOT / "finetune_data" / "dialogue_pairs.jsonl"
MAX_PAIRS = 8000
MAX_CHARS = 1200
LETTERS = "ABCD"

from ingest_dataset import DatasetStreamer

_holdout_checker = DatasetStreamer(DATA_ROOT, 1, 0.01, [], [])


def is_holdout(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return True
    return _holdout_checker._holdout(relative)


def logiqa2_jsonl(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
            answer, text, question, options = row.get("answer"), row.get("text", ""), row.get("question", ""), row.get("options") or []
            if not isinstance(answer, int) or not 0 <= answer < len(options) or not question:
                continue
            choice_block = "；".join(f"{LETTERS[i]}.{option.strip()}" for i, option in enumerate(options))
            user = f"问题：{question}\n上下文：{text}\n选项：{choice_block}\n请回答。"
            pairs.append((user, f"{LETTERS[answer]}. {options[answer].strip()}"))
        except Exception:
            continue
    return pairs


def logiqa1_blocks(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    for start in range(0, len(lines) - 8 + 1, 8):
        chunk = lines[start : start + 8]
        answer, context, question = chunk[0], chunk[1], chunk[2]
        options = [line for line in chunk[3:7] if line]
        if not answer or not question or len(options) < 2:
            continue
        letter = answer.strip().rstrip(".").upper()[:1]
        index = LETTERS.find(letter)
        if index < 0 or index >= len(options):
            continue
        user = f"问题：{question}\n上下文：{context}\n选项：{'; '.join(options)}\n请回答。"
        pairs.append((user, f"{letter}. {options[index].split('.', 1)[-1].strip() or options[index]}"))
    return pairs


def cmmlu_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            question = row.get("Question", "")
            choices = [row.get(k, "") for k in ("A", "B", "C", "D")]
            answer = row.get("Answer", "")
            if not question or not answer or not all(choices):
                continue
            letter = answer.strip().upper()[:1]
            index = LETTERS.find(letter)
            if index < 0:
                continue
            user = f"问题：{question}\n选项：{'；'.join(f'{LETTERS[i]}.{c}' for i, c in enumerate(choices))}\n请回答。"
            pairs.append((user, f"{letter}. {choices[index]}"))
    return pairs


def main() -> int:
    pairs: list[tuple[str, str]] = []
    data_root = DATA_ROOT

    logiqa2_root = data_root / "01_LogiQA/LogiQA-2.0/LogiQA2.0-main/logiqa/DATA/LOGIQA"
    for name, cap in (("dev_zh.txt", 3500), ("train.txt", 2000)):
        path = logiqa2_root / name
        if path.exists() and not is_holdout(path):
            before = len(pairs)
            pairs += logiqa2_jsonl(path)[:cap]
            print(f"LogiQA2 {name}: +{len(pairs) - before}")
    for name in ("Eval.txt", "Test.txt"):
        path = data_root / "01_LogiQA/LogiQA-1.0/LogiQA-dataset-master" / name
        if path.exists() and not is_holdout(path):
            before = len(pairs)
            pairs += logiqa1_blocks(path)[:800]
            print(f"LogiQA1 {name}: +{len(pairs) - before}")
    cmmlu_dir = data_root / "04_综合评测/CMMLU/test"
    for path in sorted(cmmlu_dir.glob("*.csv"))[:14]:
        if is_holdout(path):
            continue
        before = len(pairs)
        pairs += cmmlu_pairs(path)[:150]
        print(f"CMMLU {path.name}: +{len(pairs) - before}")

    random.seed(7)
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for user, assistant in pairs:
            handle.write(
                json.dumps(
                    {"user": user[-MAX_CHARS:], "assistant": assistant[-MAX_CHARS // 3:]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    total_bytes = OUTPUT.stat().st_size
    print(f"wrote {len(pairs)} pairs, {total_bytes} bytes -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
