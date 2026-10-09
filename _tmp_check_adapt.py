# -*- coding: utf-8 -*-
"""临时验证脚本：datasets 白名单/迭代器/缺失文件行为/编码回退；autotune 边界；adapt 边界。"""
import sys, io, os, json, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dolphin.datasets import (iter_source, _READERS, SOURCES, DEFAULT_ORDER,
                              _utf8_ok, _open_text, serialize)

print("=== 1. 白名单与默认顺序 ===")
print("SOURCES =", SOURCES)
print("DEFAULT_ORDER =", DEFAULT_ORDER)

print("\n=== 2. 白名单校验 iter_source ===")
try:
    iter_source("not_a_source")
except KeyError as e:
    print("KeyError OK:", e)

print("\n=== 3. 各白名单源迭代行为（每源取 1 条）===")
for name in SOURCES:
    try:
        it = iter_source(name)
        rec = next(it)
        print(f"[{name}] OK: keys={sorted(rec.keys())} prompt_len={len(rec['prompt'])} "
              f"answer_len={len(rec['answer'])} ser_len={len(serialize(rec))}")
    except FileNotFoundError as e:
        print(f"[{name}] FileNotFoundError: {e}")
    except StopIteration:
        print(f"[{name}] EMPTY (no records)")
    except Exception as e:
        print(f"[{name}] {type(e).__name__}: {e}")

print("\n=== 4. 编码回退逻辑 ===")
# 构造 utf-8 正常与含残缺字节的样本
samples = [
    b'{"a": "\xe4\xb8\xad\xe6\x96\x87"}\n',          # 合法 utf-8
    b'{"a": "\xe4\xb8", "b": 1}\n',                   # 头截断（残缺字节在行中间）
    b'{"a": "x", "b": "\xe6\x96"}\n',                  # 尾截断（行尾残缺）
    b'\x81\x40\x82\x40\n',                            # 非 utf-8（gbk 可解）
]
for i, s in enumerate(samples):
    print(f"sample{i}: _utf8_ok={_utf8_ok(s)}")

# 实际文件编码探测
for p in ["/o/数据集/01_LogiQA/LogiQA-1.0/LogiQA-dataset-master/zh_train.txt",
          "/o/数据集/03_数学推理/GSM8K_zh/GSM8K_zh.json"]:
    try:
        with open(p, "rb") as f:
            head = f.read(65536)
            f.seek(-min(65536, os.path.getsize(p)), 2)
            tail = f.read()
        print(f"{p}: utf8_ok(head)={_utf8_ok(head)} utf8_ok(tail)={_utf8_ok(tail)}")
    except Exception as e:
        print(f"{p}: {type(e).__name__}: {e}")

print("\n=== 5. serialize 短碎片过滤 ===")
print(repr(serialize({"prompt": "a", "answer": "b"})))  # 短 → 空串
print(repr(serialize({"prompt": "这是一个足够长的测试问题内容用于验证", "answer": "这是答案"})))

print("\n=== 6. autotune 边界（不会把参数调到非法范围）===")
from dolphin.autotune import Autotune, _DOMAINS, AUTOTUNE_EVERY
at = Autotune()
# 强制极端信号
at.signals.update({
    "note_hit_rate": 1.0, "resurrect_throughput": 100.0,
    "margin_ema": 1.0, "margin_volatility": 0.0,
    "kd_ratio": 1.0, "sleep_interval_ema": 1e6, "rollback_rate": 1.0,
})
for _ in range(500):
    at.adjust(None, {})
print("极端上调后 tunables =", at.tunables)
for k, (lo, hi) in _DOMAINS.items():
    v = at.tunables[k]
    assert lo <= v <= hi, f"{k}={v} 超出 [{lo},{hi}]"
print("全部在域内 OK")

# 极端下调
at2 = Autotune()
at2.signals.update({
    "note_hit_rate": 0.0, "resurrect_throughput": 0.0,
    "margin_ema": -1.0, "margin_volatility": 1.0,
    "kd_ratio": 0.0, "sleep_interval_ema": 0.0, "rollback_rate": 0.0,
})
for _ in range(500):
    at2.adjust(None, {})
print("极端下调后 tunables =", at2.tunables)
for k, (lo, hi) in _DOMAINS.items():
    v = at2.tunables[k]
    assert lo <= v <= hi, f"{k}={v} 超出 [{lo},{hi}]"
print("全部在域内 OK")

print("\n=== 7. adapt.next_lr 边界 ===")
from dolphin.adapt import Plasticity
pl = Plasticity()
# 体检否决：退火不应低于 lr_min
v = pl.next_lr(1e-5, 0, 0, 1, probe_passed=False)
print("否决退火 v =", v, ">= lr_min?", v >= pl.lr_min)
# 带内极端偏离
v = pl.next_lr(1e-4, 1e6, 0, 1, probe_passed=True)
print("带内极端偏离 v =", v, "<= lr_max?", v <= pl.lr_max)
# 带外极端偏离（异常输入挂起，应回落）
v = pl.next_lr(1e-4, 1e9, 0, 1, probe_passed=True)
print("带外极端偏离 v =", v, "<= lr_max?", v <= pl.lr_max, ">= lr_min?", v >= pl.lr_min)
# 世界如常 → 回落到 base
v = pl.next_lr(1e-4, 0, 0, 1, probe_passed=True)
print("常态 v =", v, "== lr_base?", v == pl.lr_base)

print("\n=== 8. datasets/ 软链不存在时的行为（模拟缺失路径）===")
# 不实际删除软链；直接验证缺失文件路径的 reader 行为（logiqa2 文件缺失）
try:
    it = iter_source("logiqa2")
    next(it)
except FileNotFoundError as e:
    print("logiqa2 FileNotFoundError OK:", e)
except Exception as e:
    print(f"logiqa2 {type(e).__name__}: {e}")

print("\nALL_DONE")