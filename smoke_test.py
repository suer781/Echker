"""M0 端到端冒烟测试。

出生同源 → 模拟交互（记录惊讶度/奖励）→ 睡眠周期（选拔→变异重放→
蒸馏训练→体检）→ 换班/回滚 → 记忆库生长 → 存档读档。

═══════════════════════════════════════════════════════════════════════════
生产资产隔离（2026-10-05，监督审计 P2-1）——与 tests/test_conformance.py 同款三防线
═══════════════════════════════════════════════════════════════════════════
事故档案：smoke_test 曾把存档直接写进 dolphin/state.pt，且其 Dolphin 以生产
冷层（dolphin/memory_cold.jsonl）为默认——审计期间亲跑一次 smoke 就覆写了
生产 state.pt 并重写生产冷层。本文件与 test_conformance 同款隔离：

  防线①  冷层改写：MemoryStore 默认解析指向生产冷层时，一律换成沙箱路径；
  防线②  存档改道：save/load 走沙箱内 state.pt（滚动备份随存档路径落沙箱
          archive/——dolphin.py._rotate_backup 按 dirname(path)/archive 取址）；
  防线③  指纹哨兵：开工记录 probe.txt / memory_cold.jsonl / state.pt /
          fed_state.pt 的 sha256，收尾逐一比对，任一变化即 FAIL——
          ①②是「让它碰不到」，③是「万一碰到了必须喊出来」，互不替代。
"""
import atexit
import hashlib
import os
import shutil
import sys
import tempfile

import torch

from dolphin.dolphin import Dolphin
from dolphin.memory import MemoryStore
from dolphin.model import Config

# ---- 隔离沙箱与防线①② ----

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_smoke_sandbox_")
atexit.register(shutil.rmtree, _SANDBOX, True)  # ignore_errors：Windows 句柄延迟
STATE_PATH = os.path.join(_SANDBOX, "state.pt")

_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    """MemoryStore.__init__ 的隔离版：解析结果若指向生产冷层，换成沙箱路径。"""
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = os.path.join(_SANDBOX, "memory_cold.jsonl")
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init

# ---- 防线③：指纹哨兵 ----

_GUARDED = ["probes/probe.txt", "dolphin/memory_cold.jsonl",
            "dolphin/state.pt", "dolphin/fed_state.pt"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_BEFORE = {p: (_sha256(os.path.join(_ROOT, p)) if os.path.exists(os.path.join(_ROOT, p))
               else None) for p in _GUARDED}


def verify_production_untouched():
    """收尾红线：生产资产指纹必须与开工时逐字节一致，否则判 FAIL。"""
    bad = []
    for p, before in _BEFORE.items():
        full = os.path.join(_ROOT, p)
        after = _sha256(full) if os.path.exists(full) else None
        if after != before:
            bad.append(f"{p}: {before} → {after}")
    if bad:
        print("FAIL：生产资产被改动！")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print(f"⑧ 生产资产零触碰（sha256 前后一致，{len(_BEFORE)} 项）✓")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)

    if os.path.exists("dolphin/birth.pt"):
        d = Dolphin.from_birth("dolphin/birth.pt", device=dev)
        print("① 从胎教基座出生 ✓")
    else:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=192), device=dev)
        print("① 无胎教基座，随机初始化出生")

    p1 = list(d.h[0].model.parameters())
    p2 = list(d.h[1].model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(p1, p2)), "出生时应同卵双生"
    print("   双半球出生同源 ✓")

    lines = []
    for fn in sorted(os.listdir("corpus")):
        if fn.endswith(".txt"):
            t = open(os.path.join("corpus", fn), encoding="utf-8").read()
            lines += [s.strip() for s in t.replace("\n", "。").split("。") if len(s.strip()) >= 8]
    assert len(lines) >= 10, "语料句子太少"
    print(f"② 语料句 {len(lines)} 条，开始模拟交互")

    for i in range(24):
        q = lines[(i * 7 + 3) % len(lines)]
        resp, eid = d.serve(q, max_new=16)
        if i % 3 == 0:
            d.feedback(eid, 1.0)
    print(f"③ 交互 24 次  缓冲 {d.buffer.bytes} 字节  压力 {d.buffer.pressure():.3f}")

    report = d.maybe_sleep(force=True, steps=150)
    print("④ 睡眠报告:", report)
    assert report is not None and "probe_new" in report, "睡眠周期未完成体检"
    assert 0.25 <= d.budget <= 0.60, "预算超出律定边界"
    assert len(d.memory.entries) > 0, "记忆库应有滞留/笔记条目"
    print(f"⑤ 记忆库 {len(d.memory.entries)} 条  传输预算 {d.budget:.3f}"
          f"  睡眠阈值 {d.buffer.sleep_threshold:.3f}")

    resp, _ = d.serve(lines[0], max_new=16)
    print("⑥ 换班后示例回复:", repr(resp[:40]))

    d.save(STATE_PATH)  # P2-1：存档落沙箱（原 dolphin/state.pt 是生产资产）
    d2 = Dolphin(cfg=Config(), device=dev)
    d2.load(STATE_PATH)
    assert d2.awake_idx == d.awake_idx and abs(d2.budget - d.budget) < 1e-9
    print("⑦ 存档/读档一致 ✓（沙箱内）")
    verify_production_untouched()
    print("PASS：M0 全循环跑通。")


if __name__ == "__main__":
    main()
