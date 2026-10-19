"""自动裁剪机制回归测试（2026-10-15）。

覆盖：
- 系统级自动裁剪触发：growth_blocked 达阈值 → param_adaptive 排 born-again。
- 深度睡眠专属精炼 deep_refine：裁剪低重要性非核心通道、残差流位红线。
- 循环链路复位：post_exam 验收通过/放弃 → growth_blocked 归零。
- _plan_grow 在显存预算拒绝时累计 growth_blocked。
隔离纪律：沿用 test_refactor_mechanisms 的沙箱方案（临时冷层/合成探测卷）。
"""
import atexit
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.life import (MEM_PRESSURE_WINDOW, STATE_NORMAL, STATE_WITHER_PLAN,
                          STATE_WITHERING)
from dolphin.memory import MemoryStore
from dolphin.model import Config
from dolphin.param_adaptive import ParamAdaptive
from dolphin.vitals import SiteLedger, Vitals

# ---------- 冷层隔离 ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_prune_sandbox_")
atexit.register(shutil.rmtree, _SANDBOX, True)
_sandbox_seq = 0


def _sandbox_cold_path():
    global _sandbox_seq
    _sandbox_seq += 1
    return os.path.join(_SANDBOX, f"cold_{_sandbox_seq:04d}.jsonl")


_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = _sandbox_cold_path()
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init

_SMALL_CFG = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
_PROBE_SEG = (
    "逻辑推理探测片段：若甲高于乙，乙高于丙，则甲高于丙；"
    "所有金属都导电，铁是金属，故铁导电；此段仅供体检评分，永不进训练粮。"
).encode("utf-8")


def make_dolphin():
    probe = os.path.join(_SANDBOX, "probe.txt")
    body = (_PROBE_SEG * 20)[:12 * _SMALL_CFG.block_size]
    with open(probe, "wb") as f:
        f.write(body)
    d = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=probe)
    d.param_adaptive_enabled = False
    d.memory = MemoryStore(cold_path=_sandbox_cold_path())
    return d


def _make_vitals_with_low_channels():
    """构造含低重要性通道的账本（前 N-2 高、后 2 极低）。"""
    v = Vitals()
    led = SiteLedger("b0.mlp_hidden", C=16)
    for c in range(14):
        led.stable_tay[c] = 1.0
        led.stable_act[c] = 1.0
    for c in (14, 15):
        led.stable_tay[c] = 1e-6
        led.stable_act[c] = 1e-6
    v.sites["b0.mlp_hidden"] = led
    return v


# ==================== 系统级自动裁剪触发 ====================

def test_auto_prune_trigger_growth_blocked():
    """growth_blocked 达阈值 → detect_pressure 返回 growth_blocked → 排 born-again。"""
    d = make_dolphin()
    pa = ParamAdaptive()
    d.life_ctl.growth_blocked = MEM_PRESSURE_WINDOW
    kind, why = pa.detect_pressure(d, gap_bar=0.0)
    assert kind == "growth_blocked"
    assert "无法扩张" in why
    report = {}
    pa.on_cycle(d, report, gap_bar=0.0)
    assert d.life_ctl.state == STATE_WITHER_PLAN
    assert d.life_ctl.plan is not None
    assert d.life_ctl.plan.get("mode") == "born_again"


def test_auto_prune_not_triggered_below_threshold():
    """growth_blocked 未达阈值 → 不触发 growth_blocked 压力。"""
    d = make_dolphin()
    pa = ParamAdaptive()
    d.life_ctl.growth_blocked = MEM_PRESSURE_WINDOW - 1
    kind, _ = pa.detect_pressure(d, gap_bar=0.0)
    assert kind != "growth_blocked"


def test_plan_grow_increments_growth_blocked_on_mem_oom(monkeypatch):
    """_plan_grow 在显存预算拒绝时累计 growth_blocked。"""
    d = make_dolphin()
    ctl = d.life_ctl
    # 构造一个能走到 _plan_grow 的环境
    stream = ("幽灵扫描测试数据" * 40).encode("utf-8")
    # 预填 thermo_hist：某轴位 headroom 过热已确认
    ctl.thermo_hist["b0.mlp_hidden"] = [[d.cycle - 2, 0.5, 1.5],
                                        [d.cycle - 1, 0.5, 1.5],
                                        [d.cycle, 0.5, 1.5]]
    # ghost_hist：基线低（1.0），最近两窗高增益（3.0）→ 确认窗越过 base*thr
    ctl.ghost_hist["0.mlp_hidden"] = [[d.cycle - 3, 1.0],
                                      [d.cycle - 2, 3.0],
                                      [d.cycle - 1, 3.0]]
    ctl.fresh_since_surgery = 1e6
    ctl.grown_since_surgery = 0
    # monkeypatch ghost_scan 返回固定增益（避免真实幽灵探测）
    import dolphin.life as life_mod
    monkeypatch.setattr(life_mod, "ghost_scan",
                        lambda model, data, seed=None: {(0, "mlp_hidden"): 3.0})
    # monkeypatch _mem_ok 恒失败（模拟显存满）
    monkeypatch.setattr(ctl, "_mem_ok", lambda dd, site, delta: (False, "mock OOM"))
    report = {}
    before = ctl.growth_blocked
    out = ctl._plan_grow(d, report, ctl.vitals_for(d, d.sleeping().name), stream)
    assert ctl.growth_blocked == before + 1
    assert out["kind"] == "hold"
    assert "生长受阻" in out["class"] or "显存预算" in out["class"]


# ==================== 深度睡眠专属精炼 ====================

def test_deep_refine_prunes_low_importance_channels():
    """deep_refine 裁剪 stable_taylor 最低的非核心通道并清零输出权重。"""
    d = make_dolphin()
    v = _make_vitals_with_low_channels()
    d.life_ctl.vitals["A"] = v
    d.life_ctl.growth_blocked = MEM_PRESSURE_WINDOW
    d.life_ctl.state = STATE_WITHERING
    d.life_ctl.campaign = {"mode": "born_again", "gen": 1, "max_gen": 3}
    report = {}
    d.life_ctl.deep_refine(d, report, v)
    assert "m4_deep_refine" in report
    assert report["m4_deep_refine"]["pruned_total"] >= 1
    assert d.life_ctl.prune_stats["prune_count"] == 1
    # 低重要性通道输出权重被清零（mlp proj 对应列）
    m = d.sleeping().model
    zero_cols = (m.blocks[0].mlp.out.weight[:, 14:16].abs().sum(dim=0) == 0).sum().item()
    assert zero_cols >= 1


def test_deep_refine_respects_residual_red_line():
    """deep_refine 不裁剪残差流位（ln1_out/ln2_out）。"""
    d = make_dolphin()
    v = Vitals()
    led = SiteLedger("b0.ln1_out", C=8)
    for c in range(8):
        led.stable_tay[c] = 1e-6  # 全是低重要性
    v.sites["b0.ln1_out"] = led
    d.life_ctl.vitals["A"] = v
    d.life_ctl.growth_blocked = MEM_PRESSURE_WINDOW
    d.life_ctl.state = STATE_WITHERING
    d.life_ctl.campaign = {"mode": "born_again", "gen": 1, "max_gen": 3}
    report = {}
    d.life_ctl.deep_refine(d, report, v)
    assert report["m4_deep_refine"]["pruned_total"] == 0


# ==================== 循环链路复位 ====================

def test_post_exam_resets_growth_blocked_on_withering_accept():
    """裁剪战役验收通过 → growth_blocked 归零（循环复位）。"""
    d = make_dolphin()
    d.life_ctl.growth_blocked = 5
    d.life_ctl.state = STATE_WITHERING
    d.life_ctl.campaign = {"mode": "born_again", "gen": 1, "max_gen": 3}
    d.life_ctl.consec_rollback = 0
    directive = d.life_ctl.post_exam(d, {"probe_new": 4.5, "gate_margin": 0.1}, passed=True)
    assert directive == "swap"
    assert d.life_ctl.growth_blocked == 0


def test_post_exam_resets_growth_blocked_on_abandon():
    """裁剪战役分代耗尽放弃 → growth_blocked 归零（避免无限重试）。"""
    d = make_dolphin()
    d.life_ctl.growth_blocked = 5
    d.life_ctl.state = STATE_WITHERING
    d.life_ctl.campaign = {"mode": "born_again", "gen": 99, "max_gen": 3}
    directive = d.life_ctl.post_exam(d, {"probe_new": 4.5, "gate_margin": -0.1}, passed=False)
    assert directive == "rollback"
    assert d.life_ctl.growth_blocked == 0