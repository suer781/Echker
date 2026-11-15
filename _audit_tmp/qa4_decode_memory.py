# -*- coding: utf-8 -*-
"""解码 memory.json 的 debt/replay 样本为可读文本(样本 ids 是原始字节, 非token id)。"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent

def dec(ids):
    return bytes(ids).decode("utf-8", errors="replace")

for side in ("left", "right"):
    p = ROOT / "_audit_tmp/qa4_dual_state/hemispheres" / side / "memory.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    print(f"===== {side} =====")
    print(" debt 条数:", len(m.get("debt", {}).get("items", [])),
          " total_clean_added:", m.get("debt", {}).get("total_clean_added"),
          " repayment_capacity:", m.get("debt", {}).get("repayment_capacity"))
    for it in m.get("debt", {}).get("items", []):
        print(f"  [debt src={it['source']} w={it['weight']:.3f}] 用户={dec(it['ids'])[:44]!r} → 回应={dec(it['labels'])[:36]!r}")
    for it in m.get("replay", {}).get("samples", []):
        print(f"  [replay src={it['source']} w={it['weight']:.3f}] 用户={dec(it['ids'])[:44]!r} → 回应={dec(it['labels'])[:36]!r}")
