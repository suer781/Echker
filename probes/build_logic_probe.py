"""构建逻辑域体检探测集（律 L8）。

来源（只许进探测集、永不许进训练粮）：
  1. LLMEval-Logic：Z3 求解器验证的中文逻辑题（base 196 + hard 154）
  2. LogiQA-1.0 的 test/eval split（zh_test.txt / zh_eval.txt）
  3. CMATH 的 cmath_test.jsonl

构建前把旧探测集备份为 probe_medical.txt.bak；构建后做重叠抽查：
随机取 5 个 100 字符片段，在全部训练粮来源中搜索，要求零命中。

用法：python probes/build_logic_probe.py [--target-kb 60]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.datasets import _iter_jsonl, serialize, iter_source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_PATH = os.path.join(ROOT, "probes", "probe.txt")
BAK_PATH = os.path.join(ROOT, "probes", "probe_medical.txt.bak")

DR = "O:/数据集"
LLMEVAL_BASE = DR + "/02_LLMEval-Logic/LLMEval-Logic-main/bench/base/llmeval_logic_base.json"
LLMEVAL_HARD = DR + "/02_LLMEval-Logic/LLMEval-Logic-main/bench/hard/llmeval_logic_hard.json"
LOGIQA1_DIR = DR + "/01_LogiQA/LogiQA-1.0/LogiQA-dataset-master"
LOGIQA2_TRAIN = DR + "/01_LogiQA/LogiQA-2.0/LogiQA2.0-main/logiqa/DATA/LOGIQA/train_zh.txt"
CMATH_TEST = DR + "/03_数学推理/CMATH/cmath_test.jsonl"
CMATH_DEV = DR + "/03_数学推理/CMATH/cmath_dev.jsonl"


def _read_logiqa1(path):
    """LogiQA 1.0 txt：空行分块，每块 7 行。"""
    out = []
    block = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            if len(block) == 7:
                out.append(block)
            block = []
            continue
        block.append(line)
    if len(block) == 7:
        out.append(block)
    return out


def collect_candidates():
    """返回 (items, dropped)——item 为 (来源名, 题面, 答案)；dropped 记录防重叠过滤。"""
    items, dropped = [], {"logiqa2_train": 0, "cmath_dev": 0}

    # ① LLMEval-Logic：base 是嵌套 original{...}，hard 是平铺字段——两种形状都接
    for p in (LLMEVAL_BASE, LLMEVAL_HARD):
        data = json.load(open(p, encoding="utf-8"))
        n0 = len(items)
        for rec in data:
            org = rec.get("original") or rec  # base 嵌套 / hard 平铺
            bg = str(org.get("background") or "").strip()
            q = str(org.get("question") or "").strip()
            a = str(org.get("answer") or "").strip()
            if bg and q and a:
                items.append(("llmeval_logic", bg + "\n" + q, a))
        print(f"[候选] {os.path.basename(p)}: {len(items) - n0} 题")

    # ② LogiQA-1.0 test/eval split（体检候选，永不进训练）
    # 防重叠预过滤：LogiQA 2.0 train 与 1.0 test 同源（公务员逻辑题再爬扩编），
    # 先在内存里比对，撞题的丢弃，保证抽查能真零命中
    lq2_text = open(LOGIQA2_TRAIN, encoding="utf-8").read()
    n0 = len(items)
    for name in ("zh_test.txt", "zh_eval.txt"):
        for b in _read_logiqa1(os.path.join(LOGIQA1_DIR, name)):
            letter = b[0].strip().lower()
            gold = "abcd".find(letter)
            opts = b[3:7]
            if gold < 0:
                continue
            q = b[1] + "\n" + b[2] + "\n" + "\n".join(opts)
            a = letter.upper() + ". " + opts[gold][2:]
            if b[1] in lq2_text or b[2] in lq2_text:
                dropped["logiqa2_train"] += 1
                continue
            items.append(("logiqa1_test", q, a))
    print(f"[候选] LogiQA-1.0 test/eval: {len(items) - n0} 题"
          f"（与 LogiQA2 train 撞题剔除 {dropped['logiqa2_train']} 条）")
    del lq2_text

    # ③ CMATH test split（禁进训练；与 dev 撞题剔除）
    dev_text = open(CMATH_DEV, encoding="utf-8").read()
    n0 = len(items)
    for rec in _iter_jsonl(CMATH_TEST):
        q = str(rec.get("question") or "").strip()
        a = str(rec.get("golden") or "").strip()
        if not q or not a:
            continue
        if q in dev_text:
            dropped["cmath_dev"] += 1
            continue
        items.append(("cmath_test", q, a))
    print(f"[候选] cmath_test: {len(items) - n0} 题"
          f"（与 cmath_dev 撞题剔除 {dropped['cmath_dev']} 条）")
    return items, dropped


def build_probe(items, target_bytes):
    """三域轮转抽取，拼成自然文本，目标 target_bytes 字节。

    返回 (文本, 实际入选的题目)——重叠抽查只认写入文件的这部分。
    """
    pools = {}
    for it in items:
        pools.setdefault(it[0], []).append(it)
    order = ["llmeval_logic", "logiqa1_test", "cmath_test"]
    idx = {k: 0 for k in order}
    parts, chosen, size = [], [], 0
    while size < target_bytes:
        progressed = False
        for src in order:
            pool = pools.get(src)
            if not pool or idx[src] >= len(pool):
                continue
            _, q, a = pool[idx[src]]
            idx[src] += 1
            parts.append("问：" + q + "\n答：" + a + "\n\n")
            chosen.append((src, q, a))
            size += len(parts[-1].encode("utf-8"))
            progressed = True
            if size >= target_bytes:
                break
        if not progressed:
            break
    return "".join(parts), chosen


def overlap_check(probe_items, n_frags=5, frag_len=100):
    """重叠抽查：随机片段在全部训练粮来源中必须零命中（律 L8 硬性要求）。

    片段取自单个探测题内部（而非拼接边界），与逐条喂入的训练记录同粒度比对。
    """
    rng = random.Random(20261003)
    long_items = [q + "\n答：" + a for _, q, a in probe_items if len(q) + len(a) >= frag_len + 20]
    frags = []
    for s in rng.sample(long_items, min(n_frags, len(long_items))):
        i = rng.randrange(0, len(s) - frag_len)
        frags.append(s[i:i + frag_len])
    print(f"\n[重叠抽查] 抽取 {len(frags)} 个 {frag_len} 字符片段：")
    for i, f in enumerate(frags):
        print(f"  片段{i + 1}: {f[:50]}...")

    sources = ["logiqa", "logiqa2", "cmath", "gsm8k", "distil",
               "coig", "synlogic", "metamath", "logiconbench"]
    hits = []
    for name in sources:
        n = 0
        try:
            for rec in iter_source(name):
                text = serialize(rec)
                if not text:
                    continue
                n += 1
                for fi, frag in enumerate(frags):
                    if frag in text:
                        hits.append((name, fi))
                        print(f"  !! 命中：来源 {name} 片段{fi + 1}")
        except Exception as e:  # 某来源缺失（如未装 pyarrow）不应掩盖其他来源的检查
            print(f"  [警告] 来源 {name} 检查中断：{e!r}")
            continue
        print(f"  来源 {name:<13} 扫描 {n} 条")
    if hits:
        print(f"\n[重叠抽查] 发现 {len(hits)} 处命中——律 L8 被违反，请处理后再上线！")
        return False
    print("\n[重叠抽查] 全部训练粮来源零命中 ✓（律 L8 达成）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-kb", type=int, default=60, help="目标大小 KB（30-100）")
    args = ap.parse_args()
    assert 30 <= args.target_kb <= 100, "探测集须在 30-100KB"

    # 旧医学探测集备份（只备份一次，重复运行不覆盖备份）
    if os.path.exists(PROBE_PATH) and not os.path.exists(BAK_PATH):
        os.replace(PROBE_PATH, BAK_PATH)
        print(f"[备份] 旧探测集 → {BAK_PATH}")
    elif os.path.exists(BAK_PATH):
        print(f"[备份] 已存在 {BAK_PATH}，跳过")

    items, _ = collect_candidates()
    print(f"[候选] 合计 {len(items)} 题")
    text, chosen = build_probe(items, args.target_kb * 1024)
    with open(PROBE_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    kb = os.path.getsize(PROBE_PATH) / 1024
    print(f"[构建] {PROBE_PATH}  {kb:.1f} KB  在 30-100KB 区间：{30 <= kb <= 100}"
          f"  实际入选 {len(chosen)} 题")

    ok = overlap_check(chosen)
    if not ok:
        sys.exit(1)
    print("PASS：逻辑域探测集构建完成。")


if __name__ == "__main__":
    main()
