"""生产级验收 B：全面可用性与易用性验证——多场景覆盖测试（2026-10-14 新增）。

对应《生产级验收 B 任务书》场景矩阵（常规 / 边界 / 异常）：

- 常规场景：
  * 正常喂食→睡眠→体检→换班（feeding 双半球流水线全链路）。
  * chat 对话 serve 返回（响应文本 + 经验 eid，输入即学习）。
  * 存档 / 读档往返（状态随档恢复）。
- 边界场景：
  * 空输入 / 空缓冲触发睡眠（早退不崩溃）。
  * 单条极短经验（1~2 字节）。
  * 超长文本（> block_size）。
  * 压力极端值（0 / 极大值）与债务全零下 compute_sleep_program 占比合法性。
  * _teacher_flip 旧档缺省回落 False（S5 存档兼容）。
  * mem_ratio 恰好 0.90 / 0.95 边界（严格大于语义，S9）。
  * sample_video_chunks 空文件 / chunk>len / max_chunks=0（S8）。
- 异常场景：
  * load() 损坏存档（随机字节）抛异常而非静默。
  * probe 卷面被篡改 → load() 抛 RuntimeError（G7）。
  * run_cycle 探测集缺失 → 抛 GateDisabled（G8）。
  * learn() 传 bytes 与 str 均不崩溃（S8）。
  * _schedule_shrink 在 CPU 设备（无 cuda）不崩溃（S9）。

隔离纪律（沿用 tests/test_conformance.py 与 tests/test_refactor_mechanisms.py 的
沙箱方案）：
- 测试进程内把 MemoryStore 默认冷层改写为临时沙箱，绝不触碰生产
  dolphin/memory_cold.jsonl（该文件不受 git 保护，污染不可逆）。
- 所有 run_cycle 场景使用临时合成探测卷（仅换试卷，生产 probes/probe.txt
  一个字节不动）。
- 需要"体检通过/换班成功"的用例用 monkeypatch 固定 gate 返回通过。
"""
import atexit
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin, GateDisabled
from dolphin.life import compute_sleep_program, run_cycle
from dolphin.memory import MemoryStore
from dolphin.model import Config
from dolphin.param_adaptive import ParamAdaptive

# ---------- 冷层隔离（沿用 test_conformance 的三道防线中的①②） ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))

_SANDBOX = tempfile.mkdtemp(prefix="dolphin_accept_sandbox_")
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
    """构造 Dolphin：关参数自适应（测试隔离），记忆库换沙箱冷层。"""
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


# ==================== 常规场景 ====================

def test_normal_feed_sleep_exam_shift(tmp_path, monkeypatch):
    """正常喂食→睡眠→体检→换班：feeding 流水线报告字段齐全、换班成功。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    for i in range(6):
        d.learn(("正常喂食睡眠体检换班测试文本，内容足够构成训练流。第%d条。" % i) * 3)
    monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
    report = run_cycle(d, feeding=True, steps=2)
    assert report.get("passed") is True
    assert report.get("swapped") is True
    assert "probe_old" in report and "probe_new" in report
    assert "sleep_program" in report
    for k in ("deep", "rem", "nrem", "light", "total_steps"):
        assert k in report["sleep_program"]
    assert d.awake_idx in (0, 1)
    assert d.learn_total >= 6


def test_chat_serve_returns(tmp_path):
    """chat 对话 serve：返回回复文本与经验 eid，输入自动成为学习信号。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    resp, eid = d.serve("你好，蓝蓟", max_new=16)
    assert isinstance(resp, str)
    assert len(resp) > 0
    assert isinstance(eid, int)
    assert d.learn_total >= 1        # serve 计入部署侧经验流入
    assert d.buffer.bytes > 0     # 用户输入已入缓冲等待睡眠选拔


def test_save_load_roundtrip(tmp_path):
    """存档/读档往返：周期、压力、债务、缓冲全部随档恢复。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    for i in range(4):
        d.learn(("存档读档往返测试文本，内容足够长。第%d条。" % i) * 2)
    d.hemi_pressure = [1.2, 2.3]
    d.debt_ledger.debts = [3.0, 4.0]
    d.cycle = 7
    d.sleep_steps_nrem = 95   # S6 四档：nrem 步数域随档透传
    path = os.path.join(tmp_path, "fed_state.pt")
    d.save(path)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d2.load(path)
    assert d2.cycle == 7
    assert d2.hemi_pressure == pytest.approx([1.2, 2.3])
    assert d2.debt_ledger.debts == pytest.approx([3.0, 4.0])
    assert d2.sleep_steps_nrem == 95
    assert d2.awake_idx == d.awake_idx
    assert d2.buffer.bytes == d.buffer.bytes
    assert len(d2.buffer.items) == len(d.buffer.items)


# ==================== 边界场景 ====================

def test_empty_input_and_empty_buffer_sleep(tmp_path):
    """空输入 + 空缓冲触发睡眠：learn("") 不崩溃，空缓冲周期早退不崩溃。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.learn("")   # 空字符串：mean_nll=0 → 金发姑娘 skip，仍计 learn_total
    assert d.learn_total >= 1
    report = run_cycle(d, feeding=True, steps=2)   # 空缓冲触发睡眠
    assert report.get("selected") == 0
    assert "note" in report                          # 早退路径有说明


def test_tiny_experience_1_2_bytes():
    """单条极短经验（1~2 字节）learn 不崩溃。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.learn(b"a")       # 1 字节
    d.learn(b"ab")      # 2 字节
    d.learn("好")        # 3 字节 UTF-8
    assert d.learn_total >= 3
    assert d.buffer.bytes >= 0


def test_long_text_gt_block_size():
    """超长文本（> block_size）learn 不崩溃且进入缓冲。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    long_text = "超长文本" * (_SMALL_CFG.block_size * 2)   # 每字 3 字节，远超 64
    d.learn(long_text)
    assert d.learn_total >= 1
    assert d.buffer.bytes > 0


def test_update_hemi_pressure_extreme_zero():
    """压力极端值（0 / 极大值）：update_hemi_pressure 数值合法不崩溃。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.hemi_trained_steps = [0, 0]      # 对方进度 0：max(1,0)=1 不除零
    d.debt_ledger.debts = [0.0, 0.0]
    d._since_sleep = 0
    d.update_hemi_pressure()
    assert d.hemi_pressure == [0.0, 0.0]
    # 极大债务：压力为正且无 NaN
    d.debt_ledger.debts = [1e9, 1e9]
    d.update_hemi_pressure()
    assert all(p > 0 for p in d.hemi_pressure)
    assert all(p == p for p in d.hemi_pressure)   # 无 NaN


def test_compute_sleep_program_extreme_pressure():
    """compute_sleep_program 压力极端值：压力 0 与压力 >50 时各占比非负。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [0.0, 0.0]
    # 压力 0：占比非负且和为 1
    d.hemi_pressure = [0.0, 0.0]
    prog0 = compute_sleep_program(d)
    for k in ("deep", "rem", "nrem", "light"):
        assert prog0[k] >= 0.0
    assert sum(prog0[k] for k in ("deep", "rem", "nrem", "light")) \
        == pytest.approx(1.0, abs=1e-9)
    # 压力极大（>50 归一化到 p=1）：各占比非负且和为 1（deep 被 rem 归一化到 <1）
    d.hemi_pressure = [100.0, 100.0]
    prog1 = compute_sleep_program(d)
    for k in ("deep", "rem", "nrem", "light"):
        assert prog1[k] >= 0.0
    assert 0.0 < prog1["deep"] < 1.0
    assert prog1["light"] == 0.0
    assert sum(prog1[k] for k in ("deep", "rem", "nrem", "light")) \
        == pytest.approx(1.0, abs=1e-9)


def test_compute_sleep_program_high_pressure_sum_is_one():
    """[验收语义] 压力 >50 时 compute_sleep_program 各占比和应为 1（P2 已修复）。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [0.0, 0.0]
    d.hemi_pressure = [100.0, 100.0]
    prog = compute_sleep_program(d)
    s = sum(prog[k] for k in ("deep", "rem", "nrem", "light"))
    assert s == pytest.approx(1.0, abs=1e-9)


def test_compute_sleep_program_zero_debt():
    """债务全零：compute_sleep_program 各档位数值正确且和为 1。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.debt_ledger.debts = [0.0, 0.0]
    d.hemi_pressure = [25.0, 25.0]     # p=0.5
    prog = compute_sleep_program(d)
    assert prog["deep"] == pytest.approx(0.5)
    assert prog["rem"] == pytest.approx(0.2)
    assert prog["nrem"] == pytest.approx(0.25)
    assert prog["light"] == pytest.approx(0.05)
    assert sum(prog[k] for k in ("deep", "rem", "nrem", "light")) \
        == pytest.approx(1.0, abs=1e-9)


def test_teacher_flip_old_ckpt_default(tmp_path):
    """_teacher_flip 旧档缺省：存档去掉该字段后 load 回落 False（S5 兼容）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d._teacher_flip = True
    path = os.path.join(tmp_path, "flip.pt")
    d.save(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck.pop("_teacher_flip", None)
    torch.save(ck, path)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d2.load(path)
    assert d2._teacher_flip is False


def test_mem_ratio_boundary_075_085(tmp_path):
    """S9 显存分档边界（2026-10-17 用户定案）：mem_ratio 恰好 0.75 / 0.85 时
    按严格大于语义分档；>85% 走动态 15%~3% 档。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    pa = ParamAdaptive()
    # 0.75：不 >0.75 → 正常理想容量档（1.571 ≥1.0），不需要收缩，不排程计划
    report = {}
    res = pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=0.75)
    assert res is None
    assert "m4_param_adaptive" not in report
    assert d.life_ctl.plan is None
    # 0.80：>0.75 但不 >0.85 → 30% 档（而非 15%/动态档）
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=0.80)
    assert report["m4_param_adaptive"]["target_ratio"] == pytest.approx(0.30)
    # 0.90：>0.85 → 动态档（15%~3% 之间，且 < 0.30）
    # （mock 大身体避免小模型 body_frac 触发容量下限拦截）
    d.life_ctl._body_frac = lambda dd, hname: 10.0
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=0.90)
    tr = report["m4_param_adaptive"]["target_ratio"]
    assert 0.03 <= tr < 0.15, f"target={tr}"


def test_modal_sample_video_chunks_edges():
    """S8 视频分块边界：空文件 / chunk>len / max_chunks=0。"""
    from dolphin.modal_input import sample_video_chunks
    # 空文件 → 空列表
    assert sample_video_chunks(b"") == []
    # chunk > len → 整块返回
    raw = b"abc"
    assert sample_video_chunks(raw, chunk=10, max_chunks=64) == [raw]
    # max_chunks=0 → 空列表
    assert sample_video_chunks(b"abcdef", chunk=2, max_chunks=0) == []


# ==================== 异常场景 ====================

def test_load_corrupted_save_raises(tmp_path):
    """损坏存档（随机字节文件）load 必须抛异常而非静默。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    bad = os.path.join(tmp_path, "corrupt.pt")
    with open(bad, "wb") as f:
        f.write(os.urandom(512))
    with pytest.raises(Exception):
        d.load(bad)


def test_load_tampered_probe_raises_runtimeerror(tmp_path):
    """G7：probe 卷面被篡改 → load() 抛 RuntimeError 拒绝加载。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    path = os.path.join(tmp_path, "state.pt")
    d.save(path)
    # 篡改卷面
    with open(probe, "wb") as f:
        f.write(b"tampered probe content, not the original exam." * 20)
    d2 = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    with pytest.raises(RuntimeError):
        d2.load(path)


def test_run_cycle_missing_probe_raises_gate_disabled(tmp_path):
    """G8：run_cycle 在探测集缺失时抛 GateDisabled（fail-fast，非静默回滚）。"""
    missing = os.path.join(tmp_path, "no_probe", "probe.txt")
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=missing)
    with pytest.raises(GateDisabled):
        run_cycle(d, feeding=True, steps=2)


def test_learn_bytes_and_str_no_crash():
    """S8：learn() 传入 bytes 与 str 均不崩溃。"""
    d = make_dolphin(cfg=_SMALL_CFG)
    d.learn("字符串输入")
    d.learn(b"raw bytes input")
    d.learn(b"\x00\x01\x02\x03")
    d.learn("x")
    assert d.learn_total >= 4


def test_schedule_shrink_cpu_no_crash():
    """S9：_schedule_shrink 在 CPU 设备（无 cuda）不崩溃并走容量压力路径。"""
    d = make_dolphin(cfg=_SMALL_CFG, device="cpu")
    pa = ParamAdaptive()
    report = {}
    pa._schedule_shrink(d, report, ratio=0.30, mem_ratio=None)
    assert report["m4_param_adaptive"]["target_ratio"] == pytest.approx(0.30)


def test_autotune_update_signals_candidates_zero_no_crash(tmp_path):
    """[长时间运行] report['candidates']==0 时 update_signals 不崩溃（P1 已修复）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    report = {"candidates": 0, "dreamed": 0, "passed": True, "mode": "deep",
              "gate_margin": 0.1, "gate_eps": 0.05, "ce_last": 5.0, "kd_last": 0.1}
    d.autotune.update_signals(d, report)   # 不应抛异常
    # candidates 缺省/None 同样不崩溃
    d.autotune.update_signals(d, {"dreamed": None, "passed": True, "mode": "deep"})


def test_retrieve_large_cold_no_index_no_crash(monkeypatch):
    """[抗压回归] 大冷层且倒排索引未就绪时 retrieve 不抛 NameError（memory.py import sys）。

    生产抗压测试暴露：memory.py 的 retrieve 大冷层跳过分支使用 file=sys.stderr，
    但文件顶部缺 import sys → NameError。本用例钉住该路径不崩溃。
    """
    from dolphin.memory import MemoryEntry, _SYNC_BUILD_THRESHOLD

    ms = MemoryStore(cap_entries=10, promote_hits=3,
                      cold_path=os.path.join(_SANDBOX, "cold_big.jsonl"))
    # 构造超过同步阈值的大量内存内冷层条目（不落盘、不触发真实构建）
    n = _SYNC_BUILD_THRESHOLD + 10
    ms.cold_entries = [
        MemoryEntry(f"cold item {i} " + "内容" * 10, 0.5, 0, "residue")
        for i in range(n)
    ]
    ms._cold_loaded = True
    ms._cold_inverted = None
    ms._cold_skip_warned = False
    # 阻止真实后台构建线程（测试隔离）
    monkeypatch.setattr(MemoryStore, "_maybe_start_index_build", lambda self: None)

    # 不应抛任何异常（曾因 memory.py 缺 import sys 而 NameError）
    out = ms.retrieve(b"cold item 1")
    assert isinstance(out, list)
    assert len(out) == 0  # 大冷层索引未就绪：本次跳过冷层，仅热层（空）结果


# ==================== S12 记忆库自适应睡眠机制 ====================

def _fill_cold_entries(n, prefix="S12冷层夹具"):
    """构造含 n 条冷层条目的沙箱冷层文件，返回 (文件路径, 源 MemoryStore)。"""
    path = _sandbox_cold_path()
    src = MemoryStore(cap_entries=4, promote_hits=3, cold_path=path)
    for i in range(4 + n):
        src.add(f"{prefix}第{i:04d}号记忆，内容足够长以避免与其它条目重复。{i}".encode(),
                0.0, 0, "residue")
    src.flush_cold()
    assert len(src.cold_entries) == n, f"冷层应 {n} 条，实际 {len(src.cold_entries)}"
    return path, src


def test_model_scale_monotonic():
    """S12 模型规格自适应：参数量越大 → memory_scale 越大，且始终落在 [0.1, 1.0]。"""
    small = make_dolphin(cfg=Config(d_model=32, n_layers=2, n_heads=2, block_size=64))
    large = make_dolphin(cfg=Config(d_model=256, n_layers=2, n_heads=4, block_size=64))
    s_small = small.memory_scale()
    s_large = large.memory_scale()
    assert 0.1 <= s_small <= 1.0
    assert 0.1 <= s_large <= 1.0
    assert s_large >= s_small


def test_sleep_cold_sample_by_mode(tmp_path, monkeypatch):
    """S12 四层睡眠冷层遍历预算：light=0；nrem/rem≥8 且 rem≥nrem；deep=冷层×(0.2+0.8×scale)。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    cold_path, _ = _fill_cold_entries(100, prefix="S12分档冷层")

    def _run_single_mode(prog):
        d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
        d.life_enabled = False
        d.memory.cold_path = cold_path
        monkeypatch.setattr("dolphin.life.compute_sleep_program",
                            lambda dolphin: dict(prog))
        monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
        for i in range(4):
            d.learn(("S12 分档冷层预算测试文本，用于触发睡眠周期。第%d条。" % i) * 3)
        report = run_cycle(d, feeding=True, steps=2)
        return d, report

    scale = make_dolphin(cfg=_SMALL_CFG, probe_path=probe).memory_scale()

    _, rep = _run_single_mode(
        {"deep": 0.0, "rem": 0.0, "nrem": 0.0, "light": 1.0, "total_steps": 120})
    assert rep["memory_cold_sample"] == 0  # 超浅度：跳过冷层，仅热层快速通道

    _, rep = _run_single_mode(
        {"deep": 0.0, "rem": 0.0, "nrem": 1.0, "light": 0.0, "total_steps": 120})
    nrem_sample = rep["memory_cold_sample"]
    assert nrem_sample == max(8, int(64 * scale))

    _, rep = _run_single_mode(
        {"deep": 0.0, "rem": 1.0, "nrem": 0.0, "light": 0.0, "total_steps": 120})
    rem_sample = rep["memory_cold_sample"]
    assert rem_sample == max(8, int(128 * scale))
    assert rem_sample >= nrem_sample

    _, rep = _run_single_mode(
        {"deep": 1.0, "rem": 0.0, "nrem": 0.0, "light": 0.0, "total_steps": 120})
    assert rep["memory_cold_sample"] == int(100 * (0.2 + 0.8 * scale))


def test_resurrect_light_cold_sample_zero_only_hot():
    """S12：cold_sample=0 时 resurrect_light 只处理热层达标条目，完全跳过冷层。"""
    cold_path, src = _fill_cold_entries(60, prefix="S12零抽样冷层")
    for e in src.cold_entries:
        e.hits = 3
    src.flush_cold()

    ms = MemoryStore(cap_entries=100, promote_hits=3, cold_path=_sandbox_cold_path())
    ms.add("热层达标记忆A，内容足够长以稳定参与选拔。".encode(), 0.0, 0, "residue")
    ms.add("热层达标记忆B，内容足够长以稳定参与选拔。".encode(), 0.0, 0, "residue")
    for e in ms.entries:
        e.hits = 3
    ms.cold_path = cold_path
    ms._ensure_cold()
    assert len(ms.cold_entries) == 60

    got = ms.resurrect_light(cold_sample=0)
    assert len(got) == 2                       # 只升回热层两条达标条目
    assert len(ms.cold_entries) == 60          # 冷层零触碰、零误删
    assert all(e.hits == 0 for e in ms.entries)  # 热层达标者已复位


def test_no_model_size_hallucination():
    """S12：代码与文档中不得残留大模型规模幻觉引用（57 后接 M/m/mb 或 5700 万形态）。"""
    import glob
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = []
    # dolphin/ 下所有 .py 与 .md（含 archive）
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "dolphin")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith((".py", ".md")):
                targets.append(os.path.join(dirpath, fn))
    targets.append(os.path.join(root, "feed.py"))
    for pattern in ("README.md", "架构设计.md", "维护文档.md", "部署手册.md",
                    "重构交付报告.md", "生产抗压测试报告.md", "验收报告_*.md"):
        targets.extend(glob.glob(os.path.join(root, pattern)))
    # .trae/specs/ 下所有 .md
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, ".trae", "specs")):
        for fn in filenames:
            if fn.endswith(".md"):
                targets.append(os.path.join(dirpath, fn))
    # 动态拼接模式，避免测试源码自身出现字面幻觉 token。
    _57 = "5" + "7"
    _M = "M"
    _m = "m"
    pat = re.compile(_57 + r"\s*" + _m + "b|" + _57 + _m + r"\b|" + _57 + "00万|" + _57 + _M)
    hits = []
    for p in sorted(set(targets)):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        hits.append(f"{p}:{i}:{line.strip()}")
        except OSError:
            continue
    assert not hits, "发现大模型规模幻觉引用：\n" + "\n".join(hits)


def test_sleep_program_frac_weighted_cold_sample(tmp_path, monkeypatch):
    """S12 多层组合睡眠：cold_sample = 各档预算按 sleep_program 占比加权。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    cold_path, _ = _fill_cold_entries(100, prefix="S12加权冷层")
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    d.life_enabled = False
    d.memory.cold_path = cold_path
    prog = {"deep": 0.5, "rem": 0.3, "nrem": 0.2, "light": 0.0, "total_steps": 120}
    monkeypatch.setattr("dolphin.life.compute_sleep_program", lambda dolphin: dict(prog))
    monkeypatch.setattr(d, "gate", lambda old, new: (True, _fake_gate_detail()))
    for i in range(4):
        d.learn(("S12 加权组合冷层预算测试文本，用于触发睡眠周期。第%d条。" % i) * 3)
    report = run_cycle(d, feeding=True, steps=2)
    scale = d.memory_scale()
    deep_b = int(100 * (0.2 + 0.8 * scale))
    rem_b = max(8, int(128 * scale))
    nrem_b = max(8, int(64 * scale))
    expected = int((0.5 * deep_b + 0.3 * rem_b + 0.2 * nrem_b) / 1.0)
    got = report["memory_cold_sample"]
    assert got == expected
    assert min(deep_b, rem_b, nrem_b) <= got <= max(deep_b, rem_b, nrem_b)


# ==================== F-03 / F-05 回归（2026-10-14） ====================

def test_serve_empty_input_no_crash(tmp_path):
    """F-05：serve('') 空输入不再崩溃（模型层空上下文自然省略生成）。"""
    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)
    resp, eid = d.serve("", max_new=8)
    assert isinstance(resp, str)
    # 空上下文无内容可生成：响应为空或任意字节均可，关键是不得抛异常。
    assert eid is not None


def test_multimodal_real_files_end_to_end(tmp_path):
    """F-03：真实小文件（WAV/PPM/二进制视频）端到端喂食不崩溃。

    修复前从未用真实文件跑过多模态链路；本用例生成真实 WAV、PPM 与二进制
    视频文件，经 feed_from(input_type=image/audio/video) 进入经验链路。
    """
    import wave
    from feed import feed_from

    wav_path = os.path.join(tmp_path, "test.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)

    ppm_path = os.path.join(tmp_path, "test.ppm")
    with open(ppm_path, "wb") as f:
        f.write(b"P6\n4 4\n255\n" + bytes(range(4 * 4 * 3)))

    vid_path = os.path.join(tmp_path, "test.bin")
    with open(vid_path, "wb") as f:
        f.write(bytes(range(256)) * 40)

    probe = os.path.join(tmp_path, "probe.txt")
    _write_synthetic_probe(probe, block_size=_SMALL_CFG.block_size)
    d = make_dolphin(cfg=_SMALL_CFG, probe_path=probe)

    for input_type, path in (("image", ppm_path), ("audio", wav_path), ("video", vid_path)):
        cursor = {"sources": [str(path)], "source_idx": 0, "record_idx": 0, "round": 0}
        seen = {}
        fed, limit_hit, _ = feed_from(d, [str(path)], cursor, seen, quota=1,
                                      input_type=input_type, loop=False)
        assert fed == 1, f"{input_type} 应喂入 1 条，实际 {fed}"
        assert limit_hit is True
        assert d.learn_total >= 1, f"{input_type} 应计入 learn_total"