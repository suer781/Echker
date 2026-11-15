# 第二轮 SFT 数据：MOSS + ShareGPT-zh + NaturalConv + ruozhiba + LCCC 增量
from __future__ import annotations

import gzip
import hashlib
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(r"O:\数据集\13_对话指令")
OUT = ROOT / "finetune_data_r2" / "round2_pairs.jsonl"
DEDUP_SOURCES = [ROOT / "finetune_data" / "sft_mega_pairs.jsonl"]

store: set = set()


def digest(u: str, a: str) -> str:
    return hashlib.md5((u[:200] + "|" + a[:200]).encode("utf-8")).hexdigest()


def seen(u: str, a: str) -> bool:
    d = digest(u, a)
    if d in store:
        return True
    store.add(d)
    return False


# 既有 13.4 万对全部入去重库（防止与第一轮重复）
for src_path in DEDUP_SOURCES:
    if src_path.exists():
        for line in src_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                seen(row["user"], row["assistant"])
            except Exception:
                continue
    print(f"去重库累计: {len(store)}")


def reservoir_lines(path, n, parse):
    got = []
    count = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pair = parse(line)
            if not pair:
                continue
            u, a = pair
            if not u or not a or len(a) > 1200 or len(u) > 1500:
                continue
            if seen(u, a):
                continue
            count += 1
            if len(got) < n:
                got.append((u, a))
            else:
                j = random.randrange(count)
                if j < n:
                    got[j] = (u, a)
    return got, count


def parse_moss(line):
    try:
        row = json.loads(line)
        chat = row.get("chat")
        if not isinstance(chat, dict):
            return None
        for key in sorted(chat.keys()):
            turn = chat[key]
            if not isinstance(turn, dict):
                continue
            u_raw = str(turn.get("Human", ""))
            a_raw = str(turn.get("MOSS", ""))
            if "<|Human|>:" in u_raw and "<|MOSS|>:" in a_raw:
                u = u_raw.split("<|Human|>:", 1)[-1].split("<eoh>")[0].strip()
                a = a_raw.split("<|MOSS|>:", 1)[-1].split("<eoc>")[0].strip()
                if u and a:
                    return (u, a)
        return None
    except Exception:
        return None


def parse_sharegpt(line):
    try:
        row = json.loads(line)
        conv = row.get("conversation", [])
        if not isinstance(conv, list):
            return None
        for i in range(len(conv) - 1):
            eu, ea = conv[i], conv[i + 1]
            if not isinstance(eu, dict) or not isinstance(ea, dict):
                continue
            u = str(eu.get("human", eu.get("from", ""))).strip()
            a = str(ea.get("assistant", ea.get("value", ""))).strip()
            if u and a and len(a) <= 1200:
                return (u, a)
        return None
    except Exception:
        return None


def naturalconv_pairs(path, n):
    got = []
    count = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        data = json.load(fh)
    for dialog in data:
        content = dialog.get("content", []) if isinstance(dialog, dict) else []
        for i in range(0, len(content) - 1, 2):
            u = str(content[i]).strip()
            a = str(content[i + 1]).strip()
            if not u or not a or len(a) > 1200 or len(u) > 1500:
                continue
            if seen(u, a):
                continue
            count += 1
            if len(got) < n:
                got.append((u, a))
            else:
                j = random.randrange(count)
                if j < n:
                    got[j] = (u, a)
    return got


def ruozhiba_pairs(path):
    pairs = []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in (data if isinstance(data, list) else []):
        u = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        a = str(row.get("output", "")).strip()
        if extra:
            u = u + "\n" + extra
        if u and a and len(a) <= 1200:
            pairs.append((u, a))
    return pairs


def parse_lccc(line):
    try:
        turns = json.loads(line)
    except Exception:
        return None
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    for i in range(0, len(turns) - 1, 2):
        u, a = str(turns[i]).strip(), str(turns[i + 1]).strip()
        if u and a and len(a) <= 1200 and len(u) <= 1500:
            return (u, a)
    return None


def main() -> int:
    random.seed(77)
    pairs = []

    moss_zip = DATA / "03_MOSS-003-sft-data" / "moss-003-sft-no-tools.jsonl.zip"
    if moss_zip.exists():
        # 直接流式读 zip 成员（11GB 解压内容不落盘）
        with zipfile.ZipFile(moss_zip) as zf:
            inner = zf.namelist()[0]
            got = []
            count = 0
            with zf.open(inner) as fh:
                for raw_line in fh:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    pair = parse_moss(line)
                    if not pair:
                        continue
                    u, a = pair
                    if len(a) > 1200 or len(u) > 1500:
                        continue
                    if seen(u, a):
                        continue
                    count += 1
                    if len(got) < 30000:
                        got.append((u, a))
                    else:
                        j = random.randrange(count)
                        if j < 30000:
                            got[j] = (u, a)
            pairs += got
            print(f"MOSS: +{len(got)} / 池 {count}")

    zh = DATA / "06_ShareGPT-Chinese-English-90k" / "sharegpt_jsonl" / "common_zh_70k.jsonl"
    if zh.exists():
        got, _ = reservoir_lines(zh, 20000, parse_sharegpt)
        pairs += got
        print(f"ShareGPT-zh: +{len(got)}")

    nc = DATA / "14_NaturalConv" / "dialog_release.json"
    if nc.exists():
        got = naturalconv_pairs(nc, 10000)
        pairs += got
        print(f"NaturalConv: +{len(got)}")

    rz = DATA / "09_ruozhiba_gpt4" / "ruozhiba_qa2449_gpt4o.json"
    if rz.exists():
        got = [x for x in ruozhiba_pairs(rz) if not seen(*x)]
        pairs += got
        print(f"ruozhiba: +{len(got)}")

    # LCCC large 再采 15k（第一轮只用了 20k，池子很大，重叠有限）
    lccc = DATA / "13_LCCC" / "lccc_large.jsonl.gz"
    if lccc.exists():
        got, _ = reservoir_lines(lccc, 15000, parse_lccc)
        pairs += got
        print(f"LCCC 增量: +{len(got)}")

    random.shuffle(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for u, a in pairs:
            fh.write(json.dumps({"user": u, "assistant": a}, ensure_ascii=False) + "\n")
    print(f"第二轮共 {len(pairs)} 对 -> {OUT} ({OUT.stat().st_size/1024/1024:.1f}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
