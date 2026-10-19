# -*- coding: utf-8 -*-
"""在线 Qwen 蒸馏测试（2026-10-17）。

验证：
1. QwenTeacher.soft_targets_for 返回 [L, 256] 的 next-byte 软分布（行和≈1）；
2. QwenTeacher.kd_loss 返回标量且可反向；
3. run_cycle 集成：Dolphin 带 teacher 跑一个睡眠周期不报错，报告含 qwen_kd_bytes。

红线：不触碰生产资产；Dolphin 用沙箱冷层 + 临时 probe。
"""
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.memory import MemoryStore
from dolphin.model import Config

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "external_models", "Qwen2.5-0.5B-Instruct")

_SMALL_CFG = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
_PROBE_TMP = tempfile.mkdtemp(prefix="dolphin_qwenkd_")
_PROBE = os.path.join(_PROBE_TMP, "probe.txt")


def _make_dolphin(probe_path=None):
    d = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=probe_path)
    d.param_adaptive_enabled = False
    d.memory = MemoryStore(cold_path=os.path.join(_PROBE_TMP, "cold.jsonl"))
    return d


@pytest.fixture(scope="module")
def teacher():
    from dolphin.qwen_kd import QwenTeacher
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("Qwen 模型未下载，跳过在线蒸馏测试")
    return QwenTeacher(MODEL_DIR, device="cpu")


def test_soft_targets_shape(teacher):
    """BPM 软标签：长度=UTF-8 字节数，每行 256 维且行和≈1。"""
    text = "你好，世界！这是一个中文测试。"
    soft = teacher.soft_targets_for(text)
    expect_len = len(text.encode("utf-8"))
    assert soft is not None
    assert soft.dim() == 2 and soft.size(0) == expect_len
    assert soft.size(1) == 256
    row_sums = soft.sum(dim=1)
    assert (row_sums - 1.0).abs().max().item() < 0.05, \
        f"行和偏离：max={(row_sums - 1.0).abs().max().item():.4f}"


def test_kd_loss_scalar(teacher):
    """kd_loss：输入 [B,T,256] logits 与软标签，返回可反向标量。"""
    from dolphin.qwen_kd import QwenTeacher
    soft = teacher.soft_targets_for("你好，世界！")
    s_logits = torch.randn(1, 8, 256, requires_grad=True)
    st = soft[:8].unsqueeze(0)
    loss = teacher.kd_loss(s_logits, st)
    assert loss.dim() == 0
    loss.backward()
    assert s_logits.grad is not None


def test_run_cycle_with_teacher(teacher):
    """端到端：Dolphin 带 Qwen 教师跑一个睡眠周期不报错。"""
    body = ("逻辑推理探测片段：若甲高于乙则甲高于丙；金属导电铁是金属故铁导电；"
            "此段仅供体检评分永不进训练粮。").encode("utf-8")
    with open(_PROBE, "wb") as f:
        f.write(body * 4)
    d = _make_dolphin(probe_path=_PROBE)
    d.gate_exam_enabled = False  # 跳过考试，训练成果直接接受
    d.life_enabled = False        # 关闭恒温器手术，聚焦训练路径
    d.qwen_kd_enabled = True
    d.qwen_kd_weight = 0.3
    d.qwen_teacher = teacher
    for i in range(4):
        d.learn(f"在线蒸馏训练样本第{i}条：太阳从东方升起，河水向东流。{i}",
                 source="test")
    from dolphin.life import run_cycle
    report = run_cycle(d, feeding=True, steps=4)
    assert report is not None
    # 报告应包含 qwen_kd_bytes（Qwen 软标签已生成）
    assert report.get("qwen_kd_bytes", 0) > 0, f"报告缺少 qwen_kd_bytes：{report}"
    assert "error" not in str(report.get("qwen_kd_error", ""))
    shutil.rmtree(_PROBE_TMP, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))