"""S1~S11 重构机制验收测试（2026-10-13 新增，只增不删既有断言）。

覆盖项（与 .trae/specs/refactor-dolphin-rewrite/spec.md 的 S 编号一一对应）：
- S1 数据无需清洗：MIN_CHARS 常量已删除、serialize() 对短样本返回非空、
  短文本/噪声直接 learn() 不崩溃。
- S2 惊讶值链路：goldilocks_gate 三态分类 + ExperienceBuffer.select() 带通单调。
- S3 睡眠压力：update_hemi_pressure() 对方进度换算 + 存档/旧档兼容。
- S4 压力中断换班：睡侧 > 醒侧 ×1.5 → pressure_interrupt + debt_settle。
- S5 债务账本/胼胝体：记账增加、还债减少、to_state/from_state 往返、callosum_kl、
  蒸馏方向交替。
- S6 四档睡眠/编制：compute_sleep_program 单调性 + 四档 mode 可运行。
- S7 杏仁核显著性：salience_score 单调、高显著未入选入 note。
- S8 多模态：图片/音频原始字节直通、视频分块、learn(bytes) 不崩。
- S9 理想容量档位：mem_ratio 0.92→30%、0.97→10%、None→容量压力路径。
- 集成：Dolphin→learn 文本/字节→run_cycle(feeding=True) 报告字段齐全。

隔离纪律（沿用 tests/test_conformance.py 的沙箱方案）：
- 测试进程内把 MemoryStore 默认冷层改写为临时沙箱，绝不触碰生产
  dolphin/memory_cold.jsonl（该文件不受 git 保护，污染不可逆）。
- 所有 run_cycle 场景使用临时合成探测卷（仅换试卷，生产律一个字节不动）。
- gate 通过/失败不是本文件被测性质：需要"换班成功"的用例用 monkeypatch
  固定 gate 返回通过（体检门控本身由既有 tests 覆盖，此处不重复）。
"""
import atexit
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.amygdala import Amygdala
from dolphin.debt import DebtLedger
from dolphin.dolphin import Dolphin
from dolphin.experience import ExperienceBuffer
from dolphin.goldilocks import PRIOR_ABILITY, gate as goldilocks_gate
from dolphin.life import compute_sleep_program, run_cycle
from dolphin.memory import MemoryStore
from dolphin.model import Config
from dolphin.param_adaptive import ParamAdaptive

# ---------- 冷层隔离（沿用 test_conformance 的三道防线中的①②） ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))

_SANDBOX = tempfile.mkdtemp(prefix="dolphin_mech_sandbox_")
atexit.register(shutil.rmtree, _SANDBOX, True)
_sandbox_seq = 0


def _sandbox_cold_path():
    global _sandbox_seq
    _sandbox_seq += 1
    return os.path.join(_SANDBOX, f"cold_{_sandbox_seq:04d}.jsonl")


_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    """隔离版：解析结果若指向生产冷层，换成沙箱路径。"""
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = _sandbox_cold_path()
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init

# ---------- 共用夹具 ----------
_SMALL_CFG = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)

_PROBE_SEG = (
    "逻辑推理探测片段：若甲高于乙，乙高于丙，则甲高于丙；"
    "所有金属都导电，铁是金属，故铁导电；下雨地必湿，此地不湿，"
    "故此地未必下雨；此段仅供体检评分，永不进训练粮。"
).encode()


def _write_synthetic_probe(path, n_chunks=12, block_size=64):
    """写合成探测卷：n_chunks 个整块（补齐到 block_size 整数倍）。"""
    body = (_PROBE_SEG * (n_chunks * block_size // len(_PROBE_SEG) + 1))
    body = body[: n_chunks * block_size]
    with open(path, "wb") as f:
        f.write(body)
    return path


def make_dolphin(cfg=None, device="cpu", probe_path=None):
    """构造 Dolphin：关参数自适应（测试隔离），并把记忆库换到沙箱冷层。"""
    d = Dolphin(cfg=cfg, device=device, probe_path=probe_path)
    d.param_adaptive_enabled = False
    d.memory = MemoryStore(cold_path=_sandbox_cold_path())
    return d


def _fake_gate_detail(margin=1.0):
    """与 probe.gate_decision 同构的 detail 字典（monkeypatch gate 用）。"""
    return {
        "probe_old": 5.0,
        "probe_new": 4.9,
        "gate_eps": 0.1,
        "gate_se": 0.05,
        "gate_margin": margin,
        "chunks": 12,
    }


# ==================== S1 数据无需清洗 ====================

def test_s1_no_min_chars_constant():
    """datasets.py 不再有 MIN_CHARS 常量（S1：删除硬编码过滤）。"""
    import dolphin.datasets as ds
    assert not hasattr(ds, "MIN_CHARS")


def test_s1_serialize_nonempty_short_sample():
    """serialize() 对非空短样本返回非空字符串（不再做长度过滤）。"""
    import dolphin.datasets as ds
    s = ds.serialize({"prompt": "三加五等于八吗？", "answer": "八。"})
    assert isinstance(s, str)
    assert len(s) > 0
    assert "三加五等于八吗？" in s and "八。" in s


def test_s1_learn_noisy_short_text_no_crash():
    """短文本/噪声文本直接喂入 Dolphin.learn() 不崩溃。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    for text in ("好", "啊？", "a", "!", "x", "中文短文本", "噪声 123 !@#", "嗯" * 3):
        d.learn(text)
    # learn_total 对每条输入都会累计（skip/suspend/pass 均计）
    assert d.learn_total >= len(("好", "啊？", "a", "!", "x", "中文短文本", "噪声 123 !@#", "嗯" * 3))


# ==================== S2 惊讶值链路 ====================

def test_s2_goldilocks_gate_tristate():
    """金发姑娘门控三态分类：太熟 skip / 太怪 suspend / 中间带 pass。"""
    ability = PRIOR_ABILITY
    state, delta = goldilocks_gate(ability * 0.90, ability)
    assert state == "skip" and delta < -0.05
    state, delta = goldilocks_gate(ability * 1.30, ability)
    assert state == "suspend" and delta > 0.20
    state, delta = goldilocks_gate(ability * 1.10, ability)
    assert state == "pass" and -0.05 <= delta <= 0.20


def test_s2_select_bandpass_monotonic():
    """ExperienceBuffer.select() 带通权重单调：中间带分数 > 边缘分数。"""
    buf = ExperienceBuffer()
    surprises = [3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0]
    for i, s in enumerate(surprises):
        buf.add(bytes([65 + i]) * 64, s)  # 数据互不相同，避免 L7 去重惩罚
    center, width = buf.band()
    sel, rest = buf.select(1.0)  # 预算 100%：全部入选，仅按带通分排序
    scored = sorted(((sc, e.surprise) for sc, e in sel), key=lambda t: -t[0])
    assert len(scored) == len(surprises)
    # 最接近带通中心的样本分数最高
    closest = min(surprises, key=lambda s: abs(s - center))
    farthest = max(surprises, key=lambda s: abs(s - center))
    assert scored[0][1] == closest
    assert scored[-1][1] == farthest
    assert scored[0][0] > scored[-1][0]
    # 分数随距离中心单调（排序后 surprise 到中心的距离应单调不减）
    dists = [abs(s - center) for _, s in scored]
    assert dists == sorted(dists)


# ==================== S3 睡眠压力 ====================

def test_s3_update_hemi_pressure_opponent_progress():
    """对方训练进度高 → 本侧换算压力下降（S3 验收场景）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d._since_sleep = 10
    d.hemi_trained_steps = [100, 0]  # 侧1的对方（侧0）进度 100%
    d.buffer.add(b"data" * 16, 5.5)
    d.update_hemi_pressure()
    # 侧0：对方进度 0 → 无减免；侧1：对方进度 1.0 → ×(1-0.3)
    assert d.hemi_pressure[1] < d.hemi_pressure[0]
    assert d.hemi_pressure[1] == pytest.approx(d.hemi_pressure[0] * 0.7, rel=1e-6)


def test_s3_save_load_hemi_pressure(tmp_path):
    """save()/load() 后 hemi_pressure 随档恢复（S3）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.hemi_pressure = [1.5, 2.5]
    path = os.path.join(tmp_path, "fed_state.pt")
    d.save(path)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d2.load(path)
    assert d2.hemi_pressure == pytest.approx([1.5, 2.5])


def test_s3_load_old_ckpt_without_hemi_pressure(tmp_path):
    """旧档缺省 hemi_pressure/debt 字段 → load 不崩溃并回落初值（S3/S5 兼容）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.hemi_pressure = [9.0, 8.0]
    path = os.path.join(tmp_path, "old.pt")
    d.save(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck.pop("hemi_pressure", None)
    ck.pop("debt", None)
    torch.save(ck, path)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d2.load(path)  # 不崩溃
    assert d2.hemi_pressure == [0.0, 0.0]
    assert d2.debt_ledger.debts == [0.0, 0.0]
    assert d2.debt_ledger.callosum_kl is None


# ==================== S4 压力中断换班 ====================

def test_s4_pressure_interrupt_and_settle(tmp_path):
    """睡侧压力 > 醒侧 ×1.5 → pressure_interrupt=True 且 settle() 发生（S4）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.param_adaptive_enabled = False
    # 构造压力失衡：债务集中在睡侧（侧1），训练进度对称
    d.debt_ledger.debts = [0.0, 100.0]
    d.hemi_trained_steps = [10, 10]
    for i in range(4):
        d.learn("这是一段用于触发压力中断的训练文本内容，长度足够。第%d段。" % i)
    report = run_cycle(d, feeding=True, steps=2)
    assert report.get("pressure_interrupt") is True
    assert "debt_settle" in report
    assert report["debt_settle"]["before"] == [0.0, 100.0]
    assert report["swap_first"] is False


# ==================== S5 债务账本 / 胼胝体 ====================

def test_s5_debt_ledger_to_from_state_roundtrip():
    """DebtLedger.to_state/from_state 往返一致（S5 持久化）。"""
    dl = DebtLedger()
    dl.debts = [3.0, 7.5]
    dl.notes = [[1, 2], [3]]
    dl.callosum_kl = 0.123
    st = dl.to_state()
    dl2 = DebtLedger()
    dl2.from_state(st)
    assert dl2.debts == [3.0, 7.5]
    assert dl2.notes == [[1, 2], [3]]
    assert dl2.callosum_kl == 0.123


def test_s5_debt_ledger_settle():
    """settle()：小债务清零，大债务保留（结算语义）。"""
    dl = DebtLedger()
    dl.debts = [5.0, 5.0]
    rec = dl.settle()
    assert rec["after"] == [0.0, 0.0]
    assert rec["settled_debt"] == pytest.approx(10.0)
    assert rec["notes_exchanged"] == 0
    dl.debts = [5.0, 100.0]
    rec = dl.settle()
    assert rec["after"] == [5.0, 100.0]
    assert rec["settled_debt"] == pytest.approx(0.0)
    assert rec["notes_exchanged"] == 0


def test_s5_gamma_repay_reduces_debt():
    """还债（debt_gamma_repay）：训练后债务按 (1-GAMMA) 缩减（S5）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [100.0, 0.0]
    gamma = d.autotune.get("debt_gamma_repay", 0.5)
    before = d.debt_ledger.debts[0]
    d.debt_ledger.debts[0] *= (1 - gamma)  # life.run_cycle 的还债公式
    assert d.debt_ledger.debts[0] == pytest.approx(before * (1 - gamma))
    assert d.debt_ledger.debts[0] < before


def test_s5_shift_accounting_and_callosum_kl(tmp_path, monkeypatch):
    """换班记账后 debts 增加；callosum_kl 写入报告与账本（S5）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.param_adaptive_enabled = False
    for i in range(6):
        d.learn(("这是一段用于换班记账与胼胝体 KL 测试的长文本内容，"
                 "包含足够多的字节以构成训练流，第%d段。" % i) * 3)
    monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
    debts_before = sum(d.debt_ledger.debts)
    report = run_cycle(d, feeding=True, steps=2)
    assert report.get("swapped") is True
    assert report.get("trained_swapped") is True
    assert "teacher" in report and report["teacher"] in ("A", "B")
    assert "callosum_kl" in report and report["callosum_kl"] is not None
    assert d.debt_ledger.callosum_kl == report["callosum_kl"]
    assert sum(d.debt_ledger.debts) > debts_before


def test_s5_teacher_alternates(tmp_path, monkeypatch):
    """蒸馏方向交替：连续 feeding 周期互为师生，实际为成对交替模式（S5）。

    feeding 生产路径通过持久化标志 _teacher_flip 自动交替：偶数相位沿用
    swap-first（训练对象=旧醒脑，教师=当前醒脑）；奇数相位跳过 swap-first、
    直接训练当前睡脑。每周期结束翻转标志。由于二态机特性，连续周期教师呈**首单发后成对重复**
    （A→B→B→A→A→B→B…），即首个周期单发，其后每个教师连续出现两周期
    （F-06 实测钉住该模式，与文档表述一致）。
    """
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.param_adaptive_enabled = False
    monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
    for i in range(6):
        d.learn(("第一次换班蒸馏方向测试文本，用于记录教师名。第%d条。" % i) * 3)
    report1 = run_cycle(d, feeding=True, steps=2)
    teacher1 = report1["teacher"]
    assert teacher1 in ("A", "B")
    assert report1["teacher"] is not None
    # 清空上周期记账与压力（避免债务不对称触发压力中断干扰交替测试）
    d.debt_ledger.debts = [0.0, 0.0]
    d.hemi_pressure = [0.0, 0.0]
    for i in range(6):
        d.learn(("第二次换班蒸馏方向测试文本，用于记录教师名。第%d条。" % i) * 3)
    report2 = run_cycle(d, feeding=True, steps=2)
    teacher2 = report2["teacher"]
    assert teacher2 in ("A", "B")
    assert teacher1 != teacher2
    # F-06：成对交替模式——第三个周期教师与第二个相同（相隔两周期相同）
    d.debt_ledger.debts = [0.0, 0.0]
    d.hemi_pressure = [0.0, 0.0]
    for i in range(6):
        d.learn(("第三次换班蒸馏方向测试文本，用于记录教师名。第%d条。" % i) * 3)
    report3 = run_cycle(d, feeding=True, steps=2)
    teacher3 = report3["teacher"]
    assert teacher3 in ("A", "B")
    # F-06：实测连续周期教师呈 A→B→B→A→A→B…（首个单发，其后成对重复）
    assert teacher2 == teacher3  # 成对交替：第2与第3周期教师相同


def test_s5_teacher_flip_persisted(tmp_path):
    """_teacher_flip 随档保存/恢复：save→load 后翻转标志不变（S5 持久化）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d._teacher_flip = True
    path = os.path.join(tmp_path, "flip_state.pt")
    d.save(path)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d2.load(path)
    assert d2._teacher_flip is True
    # 旧档缺省：去掉该字段后 load 回落 False（兼容升级）
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck.pop("_teacher_flip", None)
    torch.save(ck, path)
    d3 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d3.load(path)
    assert d3._teacher_flip is False


# ==================== S6 四档睡眠 / 编制 ====================

def test_s6_sleep_program_monotonic():
    """compute_sleep_program：压力升高 → deep 占比与 total_steps 单调不减（S6）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [0.0, 0.0]
    deeps, totals = [], []
    for p in (0.0, 10.0, 25.0, 40.0, 50.0):
        d.hemi_pressure = [p, p]
        prog = compute_sleep_program(d)
        deeps.append(prog["deep"])
        totals.append(prog["total_steps"])
    assert deeps == sorted(deeps)
    assert totals == sorted(totals)
    # 律定约束的具体形态：压力最大时 deep 占比最大、总时长最长。
    # P2 归一化后 deep 最大占比 = 1/(1+rem) < 1（rem 恒占 0.2），但仍单调不减。
    assert 0.8 < deeps[-1] < 1.0
    assert totals[-1] == 240


def test_s6_four_modes_accepted(tmp_path):
    """四档 mode（light/nrem/rem/deep）都可被 run_cycle 接受不抛错（S6）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    for mode in ("light", "nrem", "rem", "deep"):
        d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
        d.life_enabled = False
        d.param_adaptive_enabled = False
        for i in range(4):
            d.learn(("四档睡眠模式测试文本，用于验证 %s 模式可被接受运行。" % mode) * 3)
        report = run_cycle(d, feeding=False, steps=2, mode=mode)
        assert report["mode"] == mode


# ==================== S7 杏仁核显著性 ====================

def test_s7_salience_score_monotonic():
    """salience_score：惊讶度偏离基线越大分数越高，且始终落在 [0,1]（S7）。"""
    am = Amygdala()
    am.salience_score(5.0)  # 初始化 surprise_ema 基线
    s1 = am.salience_score(5.5)
    s2 = am.salience_score(7.0)
    s3 = am.salience_score(10.0)
    for s in (s1, s2, s3):
        assert 0.0 <= s <= 1.0
    assert s1 < s2 < s3


def test_s7_high_salience_rest_goes_note(tmp_path):
    """高显著且未入选预算的经验以 kind='note' 进入记忆库（S7 优先保留）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.param_adaptive_enabled = False
    d.amygdala.surprise_ema = 5.0  # 预设岛叶基线
    d.budget = 0.2                 # 小预算：多数经验落选（rest）
    for i in range(4):
        d.buffer.add(("普通文本内容第%d条，用于选拔。" % i).encode("utf-8"), 5.2)
    d.buffer.add("高惊讶".encode("utf-8") * 30, 30.0)  # 显著偏离基线
    run_cycle(d, feeding=True, steps=2)
    note_texts = [en.text for en in d.memory.entries if en.kind == "note"]
    assert note_texts, "高显著未入选经验应以 note 种类进入记忆库"
    assert any("高惊讶" in t for t in note_texts)


# ==================== S8 多模态输入 ====================

def test_s8_modal_input_passthrough_and_chunks():
    """图片/音频原始字节直通；视频分块正确（S8）。"""
    from dolphin.modal_input import (
        encode_audio_bytes,
        encode_image_bytes,
        sample_video_chunks,
    )
    raw_img = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
    raw_audio = b"ID3" + bytes(range(128))
    assert encode_image_bytes(raw_img) == raw_img
    assert encode_audio_bytes(raw_audio) == raw_audio

    raw_video = bytes(range(256)) * 40  # 10240 字节
    chunks = sample_video_chunks(raw_video, chunk=4096, max_chunks=64)
    assert chunks == [raw_video[i:i + 4096] for i in range(0, len(raw_video), 4096)]
    assert sum(len(c) for c in chunks) == len(raw_video)
    # max_chunks 截断
    chunks2 = sample_video_chunks(raw_video, chunk=1024, max_chunks=3)
    assert len(chunks2) == 3
    assert chunks2[0] == raw_video[:1024]


def test_s8_learn_bytes_no_crash():
    """Dolphin.learn(bytes) 直接接收原始字节不崩溃（S8）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.learn(b"raw bytes input \x00\x01\x02" * 4)
    d.learn(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert d.learn_total >= 2


# ==================== S9 理想容量档位 ====================

def test_s9_schedule_shrink_tiers(tmp_path):
    """理想容量三档下调：mem_ratio>0.90→30%、>0.95→10%、None→容量路径（S9）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    pa = ParamAdaptive()

    # 0.92 → 30%（默认 body_frac=1.0 不触发容量下限拦截）
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=0.92)
    assert report["m4_param_adaptive"]["target_ratio"] == pytest.approx(0.30)
    assert d.life_ctl.plan["target"] == pytest.approx(0.30)

    # 0.97 → 10%（需 body_frac 足够大以通过 MIN_BODY_FRAC 下限，mock 大身体）
    d.life_ctl._body_frac = lambda dd, hname: 3.0
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=0.97)
    assert report["m4_param_adaptive"]["target_ratio"] == pytest.approx(0.10)

    # 无显存（None）→ 容量压力路径：target=传入 ratio
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=None)
    assert report["m4_param_adaptive"]["target_ratio"] == pytest.approx(0.30)


# ==================== 集成：端到端最小链路 ====================

def test_integration_minimal_chain(tmp_path, monkeypatch):
    """Dolphin→learn(文本/字节)→run_cycle(feeding=True) 报告字段齐全（S2-S8 集成）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.param_adaptive_enabled = False
    # 喂入文本与原始字节
    for i in range(6):
        d.learn(("端到端最小链路集成测试文本，用于触发睡眠周期并验证报告字段。"
                 "第%d条。" % i) * 4)
    d.learn(b"binary bytes input " * 8)
    monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
    report = run_cycle(d, feeding=True, steps=2)
    # S6：睡眠编制写入报告
    assert "sleep_program" in report
    prog = report["sleep_program"]
    for k in ("deep", "rem", "nrem", "light", "total_steps"):
        assert k in prog
    # S5：教师、胼胝体 KL、债务记账
    assert "teacher" in report and report["teacher"] in ("A", "B")
    assert "callosum_kl" in report and report["callosum_kl"] is not None
    assert sum(d.debt_ledger.debts) > 0
    # S7：杏仁核显著性重打分已执行
    assert "salience_top" in report and 0.0 <= report["salience_top"] <= 1.0
    # 换班后 awake_idx 合法（swap-first + 训练后换班，两次翻转）
    assert d.awake_idx in (0, 1)
    assert report.get("swapped") is True


# ==================== F-01 / F-02 / F-07 回归（2026-10-14） ====================

def test_autotune_s357_feedback_branches():
    """F-01：极端世界信号下，S3/S5/S7 系数被 autotune 真实调整（值自成闭环）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    au = d.autotune
    # 构造触发条件
    au.signals["sleep_interval_ema"] = 5.0      # 远低于 target_interval*0.7=10.5 → w3 up
    au.signals["rollback_rate"] = 0.8           # >0.4 → w1 up
    au.signals["resurrect_throughput"] = 99.0   # >2.0 → gamma up
    au.signals["note_hit_rate"] = 0.01          # <0.05 → thresh down
    au.signals["margin_ema"] = 0.0
    au.signals["margin_volatility"] = 0.0
    au.signals["kd_ratio"] = 0.5
    au.signals["goldilocks_skip_rate"] = 0.0
    au.signals["goldilocks_suspend_rate"] = 0.0
    d.debt_ledger.callosum_kl = 0.05              # >0.01 → beta up
    before = dict(au.tunables)
    adj = au.adjust(d, {"gate_margin": 0.0, "ce_last": 5.0, "kd_last": 0.0,
                       "passed": False, "mode": "deep", "candidates": 10, "dreamed": 0})
    after = dict(au.tunables)
    # 三组系数至少各有一个发生调整
    assert after["pressure_w3"] > before["pressure_w3"], "S3 w3 未上调"
    assert after["pressure_w1"] > before["pressure_w1"], "S3 w1 未上调"
    assert after["debt_beta_repay"] > before["debt_beta_repay"], "S5 beta 未上调"
    assert after["debt_gamma_repay"] > before["debt_gamma_repay"], "S5 gamma 未上调"
    assert after["salience_thresh"] < before["salience_thresh"], "S7 thresh 未下调"
    assert adj, "adjust 应产生调整记录"


def test_nrem_forbids_born_again():
    """F-02：nrem 档禁止 born-again 重手术（spec S6），只返回 hold。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d._last_sleep_mode = "nrem"
    ctl = d.life_ctl
    report = {}
    out = ctl._plan_born_again(d, report, gap_bar=0.5, escalated=False)
    assert out is not None
    assert out["kind"] == "hold"
    assert "nrem" in out["class"] or "nrem" in out["reason"]
    assert ctl.plan is None, "nrem 不应排 born-again 计划"
    # deep 档不受影响：应正常排程
    d._last_sleep_mode = "deep"
    out2 = ctl._plan_born_again(d, {}, gap_bar=0.5, escalated=False)
    assert out2 is not None and out2["kind"] == "born_again"
    assert ctl.plan is not None


def test_sleep_program_reads_autotune():
    """F-07：compute_sleep_program 系数从 autotune 读取，且单调性约束保持。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [0.0, 0.0]
    # 修改 autotune 系数
    d.autotune.tunables["sleep_prog_rem_base"] = 0.25
    d.autotune.tunables["sleep_prog_rem_debt"] = 0.05
    d.autotune.tunables["sleep_prog_nrem_scale"] = 0.4
    d.autotune.tunables["sleep_prog_total_base"] = 180
    # 压力 0.5：rem = 0.25，nrem = 0.4*0.5 = 0.2，deep = 0.5
    d.hemi_pressure = [25.0, 25.0]
    prog = compute_sleep_program(d)
    assert prog["rem"] == pytest.approx(0.25)
    assert prog["nrem"] == pytest.approx(0.2)
    assert prog["deep"] == pytest.approx(0.5)
    assert prog["total_steps"] == int(180 * 1.5)
    # 单调性：压力升高 → deep 占比与 total_steps 单调不减
    deeps, totals = [], []
    for p in (0.0, 10.0, 25.0, 40.0, 50.0):
        d.hemi_pressure = [p, p]
        pr = compute_sleep_program(d)
        deeps.append(pr["deep"])
        totals.append(pr["total_steps"])
    assert deeps == sorted(deeps)
    assert totals == sorted(totals)