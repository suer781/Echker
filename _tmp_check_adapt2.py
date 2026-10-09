# -*- coding: utf-8 -*-
"""临时验证脚本第二部分：缺失文件/编码回退/软链缺失行为。"""
import sys, io, os, json, tempfile, importlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dolphin.datasets import _utf8_ok, _open_text, _iter_jsonl, iter_source, serialize

print("=== A. _utf8_ok 边界 ===")
samples = [
    ("合法utf8", b'{"a": "\xe4\xb8\xad\xe6\x96\x87"}\n'),
    ("头截断(中间残缺)", b'{"a": "\xe4\xb8", "b": 1}\n'),
    ("尾截断", b'{"a": "x", "b": "\xe6\x96"}\n'),
    ("gbk可解", b'\x81\x40\x82\x40\n'),
    ("完全噪声", b'\xff\xfe\xfd\xfc\xfb\xfa\n'),
]
for name, s in samples:
    print(f"{name}: _utf8_ok={_utf8_ok(s)}")

print("\n=== B. 构造临时文件验证 _open_text 编码回退 ===")
tmpdir = tempfile.mkdtemp()
# 纯 GBK 文件
gbk_path = os.path.join(tmpdir, "gbk.txt")
gbk_text = "这是一段中文测试内容。" * 20
with open(gbk_path, "wb") as f:
    f.write(gbk_text.encode("gbk"))
with _open_text(gbk_path) as f:
    content = f.read()
print("GBK 文件解码成功:", content[:20], "| 编码一致:", content == gbk_text)

# UTF-8 文件
utf8_path = os.path.join(tmpdir, "utf8.txt")
with open(utf8_path, "wb") as f:
    f.write(("Hello 世界 " * 20).encode("utf-8"))
with _open_text(utf8_path) as f:
    content = f.read()
print("UTF-8 文件解码成功:", content[:20])

# jsonl 行迭代
jsonl_path = os.path.join(tmpdir, "data.jsonl")
with open(jsonl_path, "wb") as f:
    f.write(('{"x": 1}\n{"x": 2}\n').encode("utf-8"))
recs = list(_iter_jsonl(jsonl_path))
print("jsonl 迭代:", recs)

print("\n=== C. 缺失文件时的迭代行为（logiqa2）===")
try:
    it = iter_source("logiqa2")
    next(it)
    print("logiqa2: 竟然有数据？")
except FileNotFoundError as e:
    print("logiqa2 FileNotFoundError OK:", e)
except Exception as e:
    print(f"logiqa2 {type(e).__name__}: {e}")

print("\n=== D. CMATH train 缺失 → 防御性回退 dev ===")
from dolphin.datasets import _CMATH_TRAIN, _CMATH_DEV
print("train exists:", os.path.exists(_CMATH_TRAIN), "| dev exists:", os.path.exists(_CMATH_DEV))
it = iter_source("cmath")
recs = []
for i, r in zip(range(3), it):
    recs.append(r)
print("cmath 前3条 source:", [r["source"] for r in recs])

print("\n=== E. 软链不存在场景模拟（datasets/ 缺失不崩溃）===")
# 模拟：把软链重命名，运行读数据，再恢复 —— 但任务是只读，不能改软链。
# 改为验证：当 os.path.exists(DATA_ROOT) 为 False 时，各 reader 抛 FileNotFoundError 而非崩溃。
# 通过 monkeypatch DATA_ROOT 为不存在路径来验证错误形态。
import dolphin.datasets as ds
orig_root = ds.DATA_ROOT
ds.DATA_ROOT = "Z:/不存在的盘/数据集"
# 重新加载模块不影响已绑定的 _LOGIQA2_TRAIN 常量，故直接验证路径拼接后 FileNotFoundError
try:
    for name in ["logiqa", "logiqa2", "cmath", "gsm8k", "distil", "coig", "synlogic", "metamath", "logiconbench"]:
        try:
            it = iter_source(name)
            next(it)
            print(f"[{name}] 意外有数据")
        except FileNotFoundError as e:
            print(f"[{name}] FileNotFoundError OK: {str(e)[:60]}")
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {str(e)[:80]}")
finally:
    ds.DATA_ROOT = orig_root

print("\nALL_DONE")