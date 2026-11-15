"""大型 SFT 数据构建器：从 13_对话指令 采样 + 既有 QA/闲聊混合。

水库采样控制内存；每源独立配额；输出 sft_mega_pairs.jsonl。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path



ROOT = Path(__file__).resolve().parent
DATA = Path(r"O:\数据集\13_对话指令")
OUT = ROOT / "finetune_data" / "sft_mega_pairs.jsonl"
MAX_CHARS_ENC = 1500
MAX_CHARS_DEC = 1200


def seen(digest: str, store: set) -> bool:
    if digest in store:
        return True
    store.add(digest)
    return False


def reservoir(path, n, store, parse):
    got = []
    total_estimate_line = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair = parse(line)
            if not pair:
                continue
            u, a = pair
            if not u or not a or len(a) > MAX_CHARS_DEC or len(u) > MAX_CHARS_ENC:
                continue
            digest = hashlib.md5((u[:200] + "|" + a[:200]).encode("utf-8")).hexdigest()
            if seen(digest, store):
                continue
            total_estimate_line += 1
            if len(got) < n:
                got.append((u, a))
            else:
                j = random.randrange(total_estimate_line)
                if j < n:
                    got[j] = (u, a)
    return got


def parse_firefly(line):
    try:
        row = json.loads(line)
        u, a = row.get("input", "").strip(), row.get("target", "").strip()
        if row.get("input"):
            u = row["input"].strip()
        if not u or not a:
            return None
        return (u, a)
    except Exception:
        return None


def parse_belle(line):
    try:
        row = json.loads(line)
        u = row.get("instruction", "").strip()
        extra = row.get("input", "").strip()
        a = row.get("output", "").strip()
        if extra:
            u = f"{u}\n{extra}"
        if not u or not a:
            return None
        return (u, a)
    except Exception:
        return None


def parse_lccc_gz(path, n, store):
    got = []
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                turns = json.loads(line)
            except Exception:
                continue
            if not isinstance(turns, list) or len(turns) < 2:
                continue
            # 相邻轮次配对：奇数位用户 / 偶数位助手
            for i in range(0, len(turns) - 1, 2):
                u, a = str(turns[i]).strip(), str(turns[i + 1]).strip()
                if not u or not a or len(a) > MAX_CHARS_DEC or len(u) > MAX_CHARS_ENC:
                    continue
                digest = hashlib.md5((u[:200] + "|" + a[:200]).encode("utf-8")).hexdigest()
                if seen(digest, store):
                    continue
                count += 1
                if len(got) < n:
                    got.append((u, a))
                else:
                    j = random.randrange(count)
                    if j < n:
                        got[j] = (u, a)
    return got


def main() -> int:
    random.seed(31)
    store: set = set()
    pairs = []

    firefly = DATA / "01_Firefly-1.1M" / "firefly-train-1.1M.jsonl"
    if firefly.exists():
        got = reservoir(firefly, 45000, store, parse_firefly)
        pairs += got
        print(f"Firefly: +{len(got)}")

    belle = DATA / "02_BELLE-train_2M_CN" / "train_2M_CN.json"
    if belle.exists():
        got = reservoir(belle, 45000, store, parse_belle)
        pairs += got
        print(f"BELLE: +{len(got)}")

    lccc = DATA / "13_LCCC" / "lccc_large.jsonl.gz"
    if lccc.exists():
        got = parse_lccc_gz(lccc, 20000, store)
        pairs += got
        print(f"LCCC: +{len(got)}")

    # 既有 QA/逻辑/CMMLU 全量
    existing = ROOT / "finetune_data" / "dialogue_pairs.jsonl"
    if existing.exists():
        cnt = 0
        for line in existing.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            digest = hashlib.md5((row["user"][:200] + "|" + row["assistant"][:200]).encode("utf-8")).hexdigest()
            if seen(digest, store):
                continue
            pairs.append((row["user"], row["assistant"]))
            cnt += 1
        print(f"既有 QA: +{cnt}")

    random.shuffle(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for u, a in pairs:
            fh.write(json.dumps({"user": u, "assistant": a}, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs -> {OUT} ({OUT.stat().st_size/1024/1024:.1f}MB)")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
