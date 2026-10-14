"""只读验证脚本：验证新增数据源可读、内容为中文推理/数学文本。

用法：python312 tools/verify_new_sources.py [--count]
--count 时额外统计每个源可 yield 的样本数（大文件全量计数，可能较慢）。
只读：不训练、不写入任何训练数据文件。
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.datasets import SOURCES, _READERS, iter_source, serialize

NEW_SOURCES = ["school_math_r1", "cmid", "math_ds_r1_10k", "basic_math_1m",
               "math_hard_zh", "mathinstruct_zh", "belle_math_zh",
               "openqa_reasoning_50k", "math23k", "ape210k", "gaokao_math",
               "xnli_zh", "clue", "logic_cot_zh"]

OLD_SOURCES = ["logiqa", "logiqa2", "cmath", "gsm8k", "distil",
               "coig", "synlogic", "metamath", "logiconbench"]


def show_first(name, n=3):
    print(f"\n{'='*70}\n[源] {name}")
    try:
        it = iter_source(name)
        count = 0
        for rec in it:
            text = serialize(rec)
            if not text:
                continue
            print(f"  样本 {count + 1}: {text[:100]}...")
            count += 1
            if count >= n:
                break
        if count == 0:
            print("  !! 未能 yield 任何有效样本")
    except Exception as e:
        import traceback
        print(f"  !! 读取失败: {e}")
        traceback.print_exc()


def count_source(name):
    print(f"  计数 {name} ...", end="", flush=True)
    n = 0
    for rec in iter_source(name):
        if serialize(rec):
            n += 1
    print(f" {n}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="统计每个源可 yield 样本数")
    ap.add_argument("--source", default=None, help="只验证指定源（逗号分隔）")
    args = ap.parse_args()

    targets = NEW_SOURCES + OLD_SOURCES
    if args.source:
        targets = [s.strip() for s in args.source.split(",")]

    for name in targets:
        show_first(name, 3)

    if args.count:
        print("\n" + "=" * 70)
        print("样本数统计（可 yield 且 serialize 非空）：")
        total = 0
        counts = {}
        for name in targets:
            c = count_source(name)
            counts[name] = c
            total += c
        print("\n汇总：")
        for name, c in counts.items():
            print(f"  {name}: {c}")
        print(f"  总计（含旧源）: {total}")


if __name__ == "__main__":
    main()