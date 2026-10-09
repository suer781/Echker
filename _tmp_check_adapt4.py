# -*- coding: utf-8 -*-
"""临时验证脚本第四部分：真正替换 DATA_ROOT 模拟数据根不可达。"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dolphin.datasets as ds

# 记录原始模块属性
orig = {}
for n in ("DATA_ROOT", "_LOGIQA2_TRAIN", "_CMATH_TRAIN", "_CMATH_DEV"):
    orig[n] = getattr(ds, n)

# 彻底模拟：DATA_ROOT 指向不存在的盘（软链缺失 = O: 数据集不可达）
ds.DATA_ROOT = "Z:/nonexistent/数据集"
ds._LOGIQA2_TRAIN = "Z:/nonexistent/数据集/01_LogiQA/LogiQA-2.0/LogiQA2.0-main/logiqa/DATA/LOGIQA/train_zh.txt"
ds._CMATH_TRAIN = "Z:/nonexistent/数据集/03_数学推理/CMATH/cmath_train.jsonl"
ds._CMATH_DEV = "Z:/nonexistent/数据集/03_数学推理/CMATH/cmath_dev.jsonl"

try:
    print("=== 数据根不可达（软链缺失场景）各源行为 ===")
    for name in ds.SOURCES:
        try:
            it = ds.iter_source(name)
            first = next(it)
            print(f"[{name}] 意外有数据: {first['source']}")
        except FileNotFoundError as e:
            print(f"[{name}] FileNotFoundError（优雅）: {str(e)[:65]}")
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {str(e)[:80]}")
finally:
    for n, v in orig.items():
        setattr(ds, n, v)

print("\n=== 确认恢复 ===")
print("DATA_ROOT =", ds.DATA_ROOT)
print("_LOGIQA2_TRAIN =", ds._LOGIQA2_TRAIN)

print("\nALL_DONE")