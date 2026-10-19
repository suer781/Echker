"""语义饱和机制与惊讶值衰减测试（2026-10-15）。

覆盖需求：
1. 语义饱和模块随 Dolphin 构造自动初始化（训练启动即生效，无需手动开启）。
2. 重复输入同一数据时，惊讶值按线性衰减（每次重复衰减 satiation_decay 比例）。
3. 单批重复数据占比超过阈值时自动触发饱和防护（saturation_events 累计、
   衰减率增强）。
4. 无异常语义识别失效：原件保持完整惊讶值；衰减有下限（min_ratio>0）；
   不同数据不受惩罚。
5. 与既有 G3 结构信号（负奖励压分）不回归。
"""
import atexit
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.memory import MemoryStore
from dolphin.model import Config

# ---------- 冷层隔离（与 test_adaptive_eval 相同的沙箱方案） ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_sat_sandbox_")
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
_PROBE = os.path.join(_SANDBOX, "probe_sat.txt")


@pytest.fixture()
def d():
    body = ("逻辑推理探测片段：若甲高于乙则甲高于丙；金属导电铁是金属故铁导电；"
            "此段仅供体检评分永不进训练粮。").encode("utf-8")
    with open(_PROBE, "wb") as f:
        f.write(body * 4)
    dol = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=_PROBE)
    dol.param_adaptive_enabled = False
    dol.memory = MemoryStore(cold_path=_sandbox_cold_path())
    return dol


def _raw_surprise(d, text):
    """当前醒脑对文本的原始惊讶度（未衰减）。"""
    return d.awake().model.mean_nll(text.encode("utf-8"), d.device)


def _buffered_items(d):
    return list(d.buffer.items)


# ==================== 1. 自动初始化 ====================

def test_saturation_state_initialized_automatically(d):
    """Dolphin 构造后语义饱和状态即就绪（2026-10-17 双尺度：短期+长期）。"""
    # 短期窗口（短时间大量重复 → 强抑制）
    assert d.satiation_window_short == 50
    assert 0.0 < d.satiation_decay_short < 1.0
    assert 0.0 < d.satiation_min_short < 1.0
    # 长期窗口（历史偶尔重复 → 弱抑制，避免某些字被排斥）
    assert d.satiation_window_long == 2000
    assert 0.0 < d.satiation_decay_long < 1.0
    assert 0.0 < d.satiation_min_long < 1.0
    assert 0.0 < d.saturation_threshold < 1.0
    assert isinstance(d.satiation_counts_short, dict) and len(d.satiation_counts_short) == 0
    assert len(d.satiation_order_short) == 0
    assert isinstance(d.satiation_counts_long, dict) and len(d.satiation_counts_long) == 0
    assert len(d.satiation_order_long) == 0
    assert d.saturation_events == 0
    assert d.saturation_last_ratio == 0.0


# ==================== 2. 惊讶值线性衰减 ====================

def test_surprise_linear_decay_on_repeat(d):
    """同一数据重复输入 5 次（窗口重复占比低于阈值）：
    学习权重衰减因子 sat_factor 双尺度线性衰减（长期弱抑制主导）：
    [1.0, 0.95, 0.90, 0.85, 0.80]；原始惊讶值 surprise 保持不变。"""
    text = "语义饱和线性衰减验证样本：同一句话重复五次。"
    raw = _raw_surprise(d, text)
    for i in range(5):
        eid = d.learn(text, source="test")
        assert eid is not None  # 原始惊讶值过门控，全部入缓冲
        # 喂入互不相同的干扰样本，保持窗口重复占比低于防护阈值（>0.5 才触发）
        for j in range(6):
            d.learn(f"干扰样本第{i}组第{j}条：内容互不相同以避免窗口重复占比抬升。{i}-{j}",
                    source="filler")

    dups = [e for e in d.buffer.items if e.data == text.encode("utf-8")]
    assert len(dups) == 5
    # 双尺度因子 = max(短期, 长期)。第 i 条（i=0..4）重复前计数 n=i：
    # factor_short = max(min_short, 1 - decay_short*i)
    # factor_long  = max(min_long,  1 - decay_long*i)
    # 取较大者 → 长期弱抑制主导：[1.0, 0.95, 0.90, 0.85, 0.80]
    expected = []
    for i in range(5):
        fs = max(d.satiation_min_short, 1.0 - d.satiation_decay_short * i)
        fl = max(d.satiation_min_long, 1.0 - d.satiation_decay_long * i)
        expected.append(max(fs, fl))
    for e, exp in zip(dups, expected):
        assert abs(e.sat_factor - exp) < 1e-4, f"衰减因子不符：{e.sat_factor} vs {exp}"
        # 原始惊讶值保留（learn 按窗口加权平均，与整体 mean_nll 允许多窗口微差）
        assert abs(e.surprise - raw) < 0.05, f"原始惊讶值被改动：{e.surprise} vs {raw}"
    # 原件衰减因子恒为 1.0
    assert abs(dups[0].sat_factor - 1.0) < 1e-4
    # 单调下降
    assert all(dups[i].sat_factor > dups[i + 1].sat_factor for i in range(4))
    # 有效惊讶值（原始×衰减因子）线性下降——观测口径的"惊讶值随重复衰减"
    eff_expected = [raw * exp for exp in expected]
    for e, exp in zip(dups, eff_expected):
        assert abs(e.effective_surprise - exp) < 0.05, (
            f"有效惊讶值衰减不符：{e.effective_surprise} vs {exp}")
    # 干扰样本（互不相同）不受衰减：sat_factor=1.0 且惊讶值=原始值
    fillers = [e for e in d.buffer.items if e.data != text.encode("utf-8")]
    assert len(fillers) == 30
    for e in fillers[:5]:
        raw_f = d.awake().model.mean_nll(e.data, d.device)
        assert abs(e.sat_factor - 1.0) < 1e-4
        assert abs(e.surprise - raw_f) < 0.05


# ==================== 3. 单批重复占比超阈值触发防护 ====================

def test_batch_duplicate_ratio_triggers_protection(d):
    """单批重复占比超过阈值 → 自动触发饱和防护（事件累计 + 重复占比观测）。"""
    text = "饱和防护触发验证：同一批次内反复出现的高重复样本。"
    n = 80  # 窗口内 80 条同一数据 → 重复占比 1 - 1/80 ≈ 0.9875 > 0.5
    for _ in range(n):
        d.learn(text, source="test")
    assert d.saturation_events > 0, "重复占比超阈值未触发饱和防护事件"
    assert d.saturation_last_ratio > d.saturation_threshold
    # 衰减下限仍生效：最后一条的 sat_factor 不低于长期下限（取较大者）
    last = _buffered_items(d)[-1]
    assert last.sat_factor >= d.satiation_min_long - 1e-4
    # 原始惊讶值仍被保留（learn 按窗口加权平均，允许多窗口微差）
    raw = _raw_surprise(d, text)
    assert abs(last.surprise - raw) < 0.05


# ==================== 4. 无异常语义识别失效 ====================

def test_first_copy_full_surprise_and_distinct_unaffected(d):
    """原件完整惊讶值与权重；衰减有下限；不同数据完全不受惩罚。"""
    dup = "同一数据反复出现，验证衰减下限与原件保护。"
    other = "完全不同的数据，不应受到任何语义饱和惩罚。"
    raw_dup = _raw_surprise(d, dup)
    raw_other = _raw_surprise(d, other)

    # 先喂 15 条重复 + 1 条不同
    for _ in range(15):
        d.learn(dup, source="test")
    d.learn(other, source="test")

    items = _buffered_items(d)
    # 原件（第一条 dup）衰减因子=1.0、惊讶值=原始值
    assert abs(items[0].sat_factor - 1.0) < 1e-4
    assert abs(items[0].surprise - raw_dup) < 0.05
    # 最后一条 dup 的衰减因子不低于长期下限（双尺度取较大者）
    assert items[14].sat_factor >= d.satiation_min_long - 1e-4
    assert abs(items[14].surprise - raw_dup) < 0.05
    # 不同数据（第 16 条）sat_factor=1.0、惊讶值=原始值（不受惩罚）
    assert abs(items[15].sat_factor - 1.0) < 1e-4
    assert abs(items[15].surprise - raw_other) < 0.05


def test_duplicate_flood_does_not_suppress_normal_data(d):
    """回归（可用性 D1）：重复洪泛不得扭曲带通中心、压伤正常数据。
    大量重复样本灌入后，正常数据仍应排在重复样本之前。"""
    dup = "洪泛重复样本：同一数据反复灌入以制造饱和。"
    for _ in range(150):
        d.learn(dup, source="dup")
    normal = "正常数据样本：内容完全不同且仅出现一次。"
    d.learn(normal, source="normal")
    sel, _ = d.buffer.select(1.0)
    rank = {e.data: r for r, (s, e) in enumerate(sel)}
    normal_rank = rank.get(normal.encode("utf-8"), -1)
    dup_ranks = [r for data, r in rank.items() if data != normal.encode("utf-8")]
    assert normal_rank >= 0, "正常数据未入选"
    assert dup_ranks, "重复样本未入选"
    assert normal_rank < min(dup_ranks), (
        f"正常数据被重复洪泛压后：normal_rank={normal_rank} 重复最低rank={min(dup_ranks)}")
    # 带通中心未被塌缩：仍接近正常数据的原始惊讶度
    assert abs(d.buffer.band()[0] - _raw_surprise(d, normal)) < 1.0, (
        f"带通中心被重复洪泛扭曲：{d.buffer.band()[0]:.3f}")


def test_saturation_events_persist_across_save_load(d, tmp_path):
    """saturation_events 随档持久化（save/load 往返一致）。"""
    text = "持久化验证：高重复样本触发防护后事件计数随档保存。"
    for _ in range(60):
        d.learn(text, source="test")
    assert d.saturation_events > 0
    path = os.path.join(str(tmp_path), "state.pt")
    d.save(path)
    d2 = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=_PROBE)
    d2.param_adaptive_enabled = False
    d2.memory = MemoryStore(cold_path=_sandbox_cold_path())
    d2.load(path)
    assert d2.saturation_events == d.saturation_events


# ==================== 5. 与 G3 结构信号不回归 ====================

def test_g3_structural_reward_no_regression(d):
    """G3 结构信号不回归：重复条目负奖励 [0,-1,-2]，重复分数低于原件。"""
    text = "重复条目压分回归验证题干，一字不差地出现三次以检验结构信号。"
    seen = {}
    eids = []
    for _ in range(3):
        eid = d.learn(text, source="fake")
        assert eid is not None
        eids.append(eid)
        # 复刻 feed.structural_reward 的自动第二信号
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        n = seen.get(h, 0)
        seen[h] = n + 1
        if n > 0:
            d.feedback(eid, float(-n))

    rewards = [e.reward for e in d.buffer.items]
    assert rewards == [0.0, -1.0, -2.0]
    sel, _ = d.buffer.select(1.0)
    head = max(s for s, e in sel if e.reward == 0.0)
    dup_scores = sorted((s for s, e in sel if e.reward < 0), reverse=True)
    assert len(dup_scores) >= 2 and dup_scores[-1] < head


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))