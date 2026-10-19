"""训完上线后的"有错就改"自适应评估测试（2026-10-15）。

设计定位：与前置拦截门（体检判决带）职责分离——本机制不拦截任何输入/更新，
只在异常事件已经发生（模型误删用户内容/服务错误/答非所问/空响应/乱码）时，
事后捕获负面反馈信号、对交互打负分，并驱动自评估阈值（值自成）自适应收紧。

覆盖：
- capture_anomaly：负面事件捕获 → 交互负分 + 事件日志 + negative_feedback_rate 信号。
- self_evaluate 使用 autotune 自适应阈值（乱码判定线 / 正奖励线 / 差评线），
  不再写死 0.30/0.50/0.55/0.30。
- autotune 负面反馈联动：只调事后评估阈值，不触碰体检判决带（k_margin）。
- anomaly_log 随档持久化（save/load 往返一致）。
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

# ---------- 冷层隔离（与 test_self_eval 相同的沙箱方案） ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_adaptive_eval_sandbox_")
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
_PROBE = os.path.join(_SANDBOX, "probe_adaptive_eval.txt")


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


def _add(d, text):
    return d.buffer.add(text.encode("utf-8"), 5.5)


# ==================== capture_anomaly：事后负面反馈捕获 ====================

def test_capture_anomaly_marks_interaction_negative(d):
    """异常事件 → 对该交互打 -1 分，并记录事件日志、抬升负面反馈率。"""
    eid = _add(d, "用户内容：重要文档")
    ok, ret_eid = d.capture_anomaly("model_deleted_user_content", eid=eid, detail="模型误删")
    assert ok is True
    assert ret_eid == eid
    exp = next(e for e in d.buffer.items if e.id == eid)
    assert exp.reward == -1.0
    assert d.anomaly_log[-1]["kind"] == "model_deleted_user_content"
    assert d.anomaly_log[-1]["eid"] == eid
    assert d.anomaly_log[-1]["detail"] == "模型误删"
    assert d.negative_feedback_rate() > 0.0


def test_capture_anomaly_without_eid_still_records(d):
    """无对应交互的纯事件（如服务错误）也能记录并抬升信号，不写缓冲。"""
    ok, _ = d.capture_anomaly("service_error", detail="500")
    assert ok is False
    assert d.anomaly_log[-1]["kind"] == "service_error"
    assert d.negative_feedback_rate() > 0.0


def test_anomaly_log_persisted(d, tmp_path):
    """anomaly_log 随档持久化（save/load 往返一致）。"""
    d.capture_anomaly("model_deleted_user_content", eid=_add(d, "x"), detail="d")
    p = str(tmp_path / "fed_anomaly.pt")
    d.save(p)
    d2 = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=_PROBE)
    d2.load(p)
    assert d2.anomaly_log == d.anomaly_log


# ==================== self_evaluate 使用自适应阈值 ====================

def test_self_evaluate_garbage_threshold_adaptive(d):
    """乱码判定线由 autotune 驱动：调高 repl 线后，轻度乱码不再被直接判差评。"""
    # 默认阈值：repl>0.30 判乱码。构造 repl_ratio≈0.333 的轻度乱码 → 默认判 -1。
    eid = _add(d, "用户诉求：你好")
    mild_garbage = b"abc\xff\xfe\xfddef"  # 9 字节，3 个非法字节 → repl≈0.333
    reward_default = d.self_evaluate("你好", mild_garbage, eid)
    assert reward_default == -1.0
    assert d.self_eval_log[-1]["dim"].get("reason") == "输出为乱码/不可读"
    # 调高乱码阈值（repl>0.90 才算乱码）→ 同一响应不再判乱码，reward != -1。
    d.autotune.tunables["eval_garbage_repl"] = 0.90
    d.autotune.tunables["eval_garbage_print"] = 0.10
    eid2 = _add(d, "用户诉求：你好")
    reward_adaptive = d.self_evaluate("你好", mild_garbage, eid2)
    assert reward_adaptive != -1.0


def test_self_evaluate_pass_threshold_adaptive(d):
    """正奖励线由 autotune 驱动：调高 pass 线后，同一响应不再得 +1。"""
    eid1 = _add(d, "用户诉求：今天天气怎么样？")
    r_lo = d.self_evaluate("今天天气怎么样？", "今日天气晴朗".encode("utf-8"), eid1)
    # 记录低 pass 线下的结果
    d.autotune.tunables["eval_pass_thresh"] = 0.80  # 域上限：很难给 +1
    d.autotune.tunables["eval_fail_thresh"] = 0.10
    eid2 = _add(d, "用户诉求：今天天气怎么样？")
    r_hi = d.self_evaluate("今天天气怎么样？", "今日天气晴朗".encode("utf-8"), eid2)
    # 同一响应的总分不变（同一模型），pass 线抬高后不应得到比低线下更高的奖励。
    assert r_hi <= r_lo
    # 且高线下不该出现 +1（除非总分本就 ≥0.80，本测试响应不可能）
    assert r_hi != 1.0


# ==================== autotune 负面反馈联动（职责分离） ====================

def test_autotune_negative_feedback_adjusts_eval_thresholds_only(d):
    """负面反馈率高 → 只收紧事后评估阈值（pass 线抬升/fail 线下移），
    不触碰前置体检判决带 k_margin（职责分离：新门槛不是提前预知拦截）。"""
    au = d.autotune
    au.signals["negative_feedback_rate"] = 0.5
    before_k = au.tunables["k_margin"]
    before_pass = au.tunables["eval_pass_thresh"]
    before_fail = au.tunables["eval_fail_thresh"]
    adj = au.adjust(d, {"gate_margin": 0.01})
    assert au.tunables["eval_pass_thresh"] > before_pass
    assert au.tunables["eval_fail_thresh"] < before_fail
    assert "eval_pass_thresh" in adj
    assert "eval_fail_thresh" in adj
    assert au.tunables["k_margin"] == before_k  # 职责分离：不动前置门


def test_autotune_negative_feedback_recovers_when_low(d):
    """负面反馈率低 → 评估阈值缓慢恢复（值自成，不写死）。"""
    au = d.autotune
    au.signals["negative_feedback_rate"] = 0.5
    au.adjust(d, {"gate_margin": 0.01})
    raised = au.tunables["eval_pass_thresh"]
    # 低负面反馈 → 恢复（pass 线下调）
    au.signals["negative_feedback_rate"] = 0.0
    au.adjust(d, {"gate_margin": 0.01})
    assert au.tunables["eval_pass_thresh"] < raised