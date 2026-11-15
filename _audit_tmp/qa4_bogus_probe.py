# -*- coding: utf-8 -*-
"""bogus brain.json 变体: 复用真实 Byt5DualBrain.load() 的元数据解析路径(跳过模型加载)。"""
import sys, traceback
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from byt5_dual import Byt5DualBrain

ROOT = Path(__file__).resolve().parent.parent
for tag in ("qa4_bogus2_state", "qa4_bogus3_state"):
    brain = Byt5DualBrain.__new__(Byt5DualBrain)   # 跳过 __init__ 的模型加载
    brain.state_dir = ROOT / "_audit_tmp" / tag
    brain.hemispheres = {}                          # 已有档案的恢复部分不在本测试范围
    try:
        ok = brain.load()
        print(f"[{tag}] load() 返回 {ok}, active={brain.active}, acc={brain.switch_urgency_acc}, count={brain.switch_count}")
    except Exception as exc:
        print(f"[{tag}] 崩溃: {type(exc).__name__}: {exc}")
