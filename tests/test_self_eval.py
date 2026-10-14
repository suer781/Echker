"""模型自评估打分机制测试（2026-10-14）。

覆盖：
- 自评估 reward 与原手动打分数据结构完全兼容（-1/0/+1 浮点，经 feedback 写入经验缓冲）。
- 判断依据（dim 三维度）随存档持久化。
- 三维度分数范围与空响应边界。
- chat.py / feed.py 已无手动打分交互残留（无感知替换）。
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

# ---------- 冷层隔离（与 test_acceptance_scenarios 相同的沙箱方案） ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_selfeval_sandbox_")
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
_PROBE = os.path.join(_SANDBOX, "probe_selfeval.txt")


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


def test_self_evaluate_reward_format_compatible(d):
    """自评估 reward 必须落在 {-1, 0, +1}，且经 feedback 写入 Experience.reward（兼容原格式）。"""
    eid = _add(d, "用户诉求：今天天气怎么样？")
    reward = d.self_evaluate("今天天气怎么样？", "今日天气晴朗".encode("utf-8"), eid)
    assert reward in (-1.0, 0.0, 1.0)
    # 原手动打分链路：feedback(eid, reward) → Experience.reward
    assert d.buffer.feedback(eid, reward) is True
    exp = next(e for e in d.buffer.items if e.id == eid)
    assert exp.reward == reward


def test_self_evaluate_reason_logged_and_persisted(d, tmp_path):
    """判断依据写入 self_eval_log，且随存档持久化（save/load 往返一致）。"""
    eid = _add(d, "用户诉求：帮我规划周末")
    d.self_evaluate("帮我规划周末行程吧", "周末安排：周六爬山周日休息".encode("utf-8"), eid)
    assert len(d.self_eval_log) == 1
    rec = d.self_eval_log[0]
    assert "dim" in rec and "reward" in rec and "eid" in rec
    dim = rec["dim"]
    for k in ("relevance", "completeness", "accuracy", "total"):
        assert k in dim
    # 持久化往返
    p = str(tmp_path / "fed_selfeval.pt")
    d.save(p)
    d2 = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=_PROBE)
    d2.load(p)
    assert d2.self_eval_log == d.self_eval_log


def test_self_evaluate_dimensions_range_and_empty(d):
    """三维度分数均在 [0,1]；空响应给 -1 并记录依据。"""
    eid = _add(d, "用户诉求：你好")
    reward = d.self_evaluate("你好", b"", eid)  # 空响应
    assert reward == -1.0
    assert d.self_eval_log[-1]["dim"]["reason"] == "空响应"

    eid2 = _add(d, "用户诉求：算 1+1")
    r2 = d.self_evaluate("算 1+1", "1+1=2 的完整解答过程如下：".encode("utf-8"), eid2)
    assert r2 in (-1.0, 0.0, 1.0)
    dim = d.self_eval_log[-1]["dim"]
    for k in ("relevance", "completeness", "accuracy"):
        assert 0.0 <= dim[k] <= 1.0


def test_no_manual_score_prompt_in_code():
    """无感知替换：chat.py / feed.py 中不应再有用户手动打分交互残留。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fn in ("chat.py", "feed.py"):
        with open(os.path.join(root, fn), encoding="utf-8") as f:
            src = f.read()
        assert "[打分]" not in src, f"{fn} 仍含手动打分提示"
        assert "打分？（" not in src, f"{fn} 仍含打分引导"
        assert "reward_line = input" not in src and "reward_line = line" not in src, \
            f"{fn} 仍含手动打分读取"
        # 自评估已接线
        assert "self_evaluate" in src, f"{fn} 缺少 self_evaluate 调用"


# ==================== F-04 回归（2026-10-14） ====================

def test_self_evaluate_garbage_output_negative(d):
    """F-04：乱码/不可读输出（大量 U+FFFD 或控制字符）必须判差评，禁止正奖励。"""
    eid = _add(d, "用户诉求：今天天气")
    # 构造乱码响应：大量非法 UTF-8 字节 → decode 后出现高比例 U+FFFD
    garbage = b"\xff\xfe\xfd\x80\x81\x82" * 8
    reward = d.self_evaluate("今天天气", garbage, eid)
    assert reward == -1.0
    assert d.self_eval_log[-1]["dim"].get("reason") == "输出为乱码/不可读"


def test_self_evaluate_irrelevant_output_not_positive(d):
    """F-04：输入有实质内容但输出与输入零重叠时，reward 不得为 +1。"""
    eid = _add(d, "用户诉求：甲")
    reward = d.self_evaluate("甲", "乙".encode("utf-8") * 10, eid)
    assert reward in (-1.0, 0.0), f"零重叠响应不应得正奖励，实际 {reward}"
    # 若输出可读但完全无关，总分应被封顶到中性档（<0.55）
    if reward == 0.0:
        assert d.self_eval_log[-1]["dim"]["total"] < 0.55