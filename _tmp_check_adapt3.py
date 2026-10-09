# -*- coding: utf-8 -*-
"""临时验证脚本第三部分：彻底模拟数据根缺失时所有源的行为。"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dolphin.datasets as ds

# 保存原始常量
orig = {n: getattr(ds, n) for n in ("_LOGIQA2_TRAIN", "_CMATH_TRAIN", "_CMATH_DEV")}
# 替换为不存在路径（模拟软链缺失/DATA_ROOT 不可达）
ds._LOGIQA2_TRAIN = "Z:/nonexistent/LogiQA2/train_zh.txt"
ds._CMATH_TRAIN = "Z:/nonexistent/cmath_train.jsonl"
ds._CMATH_DEV = "Z:/nonexistent/cmath_dev.jsonl"

try:
    print("=== 模拟数据根不可达：各 reader 行为 ===")
    for name in ds.SOURCES:
        try:
            it = ds.iter_source(name)
            next(it)
            print(f"[{name}] 意外有数据")
        except FileNotFoundError as e:
            print(f"[{name}] FileNotFoundError（优雅，不崩溃）: {str(e)[:70]}")
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {str(e)[:80]}")
finally:
    for n, v in orig.items():
        setattr(ds, n, v)

print("\n=== 补充：synlogic 无 pyarrow 时优雅跳过 ===")
# 模拟 pyarrow 缺失
import builtins
real_import = builtins.__import__
def fake_import(name, *a, **k):
    if name == "pyarrow.parquet" or name.startswith("pyarrow"):
        raise ImportError("No module named 'pyarrow'")
    return real_import(name, *a, **k)
builtins.__import__ = fake_import
try:
    it = ds.iter_source("synlogic")
    # 应返回空生成器（打印提示后 return），不抛异常
    try:
        first = next(it)
        print("synlogic 意外有数据:", first)
    except StopIteration:
        print("synlogic 无 pyarrow → 优雅跳过（空生成器，不崩溃）")
finally:
    builtins.__import__ = real_import

print("\n=== 补充：serialize 短碎片 ===")
print("短碎片返回空串:", repr(ds.serialize({"prompt": "短", "answer": "短"})))
print("正常返回:", repr(ds.serialize({"prompt": "这是一个足够长的测试问题内容用于验证序列化逻辑", "answer": "这是答案"})))

print("\nALL_DONE")