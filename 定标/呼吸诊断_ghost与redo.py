# -*- coding: utf-8 -*-
"""呼吸实验·只读诊断（2026-10-05，实验专用，非生产管线）。

目的：主实验证明"手术排程通道打不开"之后，把两个关键机制的数字拿到手，
让报告的结论从"推断"升级为"实测"：

  ① ghost 增益实测（如果平台期阶梯打开，系统会怎么判？）
     ghost_scan 是纯测量函数（前向+反向+zero_grad 还原，不改权重），
     在重复数据（cmath 呼吸切片）与全新数据（gsm8k）上各扫一遍，
     对照 GHOST_GAIN_MIN=1e-4 给出"本会发生的判决"。

  ② ReDo 彩票定量（平台期计数器为什么被反复清零？）
     每次抽样：快照睡脑权重+账本 → 调 life.LifeController._redo（M4 真实
     回收路径）→ 测探测损失差 Δprobe。Δprobe 的分布 = 每周期强加在
     margin 上的结构噪声（EPS_PLATEAU=0.005 与之同尺度竞争）。

红线合规：本脚本**绝不 save**、不写任何生产文件；ghost_scan/零梯度还原；
ReDo 抽样的突变只发生在内存中的投掷实例上。生产资产指纹开工收工各核一次。

用法（项目根目录，须在主实验释放 GPU 后运行）：
  "C:/.../python.exe" 定标/呼吸诊断_ghost与redo.py
"""
import copy
import hashlib
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from dolphin.datasets import iter_source, serialize  # noqa: E402
from dolphin.dolphin import Dolphin  # noqa: E402
from dolphin.life import (EPS_PLATEAU, GHOST_GAIN_MIN, WITHER_BORN_FRAC,  # noqa: E402
                          WITHER_DORMANT_FRAC)
from dolphin.surgery import ghost_scan  # noqa: E402
from feed import FED_STATE  # noqa: E402

T0 = time.monotonic()


def log(msg):
    print(f"[+{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_of(hem):
    return sum(p.numel() for p in hem.model.parameters())


def main():
    log("=" * 20 + " 呼吸实验·只读诊断开工 " + "=" * 20)
    log(f"生产资产指纹（开工）：probe.txt={sha256(os.path.join(ROOT, 'probes', 'probe.txt'))[:16]}…  "
        f"memory_cold.jsonl={sha256(os.path.join(ROOT, 'dolphin', 'memory_cold.jsonl'))[:16]}…")
    d = Dolphin(device="cuda" if torch.cuda.is_available() else "cpu")
    d.load(FED_STATE)
    a, b = d.h[0], d.h[1]
    pa, pb = params_of(a), params_of(b)
    log(f"读档完成：cycle={d.cycle} 醒脑={d.awake().name} 状态机={d.life_ctl.state} "
        f"平台期={d.life_ctl.plateau}")
    log(f"[铁证|诊断起点] A={pa:,} B={pb:,} Σ={pa + pb:,}  d_model={b.model.cfg.d_model}")
    probe_a = d.probe_loss(a)
    probe_b = d.probe_loss(b)
    log(f"探测损失：醒脑{a.name}={probe_a:.4f}  睡脑{b.name}={probe_b:.4f}  "
        f"代际差={probe_b - probe_a:+.4f}（睡脑落后=换班后追赶项的来源）")

    # ---------- ① ghost 增益实测 ----------
    log("—— ① ghost 增益实测（纯测量，权重零改动）——")
    rep_slices = [serialize(rec) for rec in itertools.islice(iter_source("cmath"), 0, 80)]
    rep_slices = [t for t in rep_slices if t]
    stream_repeat = b"\n".join(t.encode("utf-8") for t in rep_slices)[:4096]
    fresh = []
    for rec in itertools.islice(iter_source("gsm8k"), 0, 20):
        t = serialize(rec)
        if t:
            fresh.append(t.encode("utf-8"))
        if sum(len(x) + 1 for x in fresh) > 4400:
            break
    stream_fresh = b"\n".join(fresh)[:4096]
    log(f"重复数据流（cmath 呼吸切片口径）={len(stream_repeat)}B  "
        f"全新数据流（gsm8k 前 {len(fresh)} 条）={len(stream_fresh)}B")
    h = d.sleeping()
    for tag, stream in (("重复(cmath切片)", stream_repeat), ("全新(gsm8k)", stream_fresh)):
        t0 = time.monotonic()
        gains = ghost_scan(h.model, stream, seed=d.cycle)
        best_key = max(gains, key=gains.get)
        best = gains[best_key]
        top = sorted(gains.items(), key=lambda kv: -kv[1])[:3]
        verdict = "≥1e-4 → 本会排 GROW_PLAN" if best >= GHOST_GAIN_MIN else "<1e-4 → 本会看休眠占比（<0.10 → 不动刀）"
        log(f"  [{tag}] 最佳增益 {best_key[0]}层/{best_key[1]} = {best:.6f} "
            f"（门槛 GHOST_GAIN_MIN={GHOST_GAIN_MIN}）→ {verdict}  耗时{time.monotonic()-t0:.1f}s")
        log(f"    前三：{[(f'b{k[0]}.{k[1]}', round(v, 6)) for k, v in top]}")

    # ---------- ② ReDo 彩票定量 ----------
    log("—— ② ReDo 每周期扰动定量（5 次抽样，投掷实例，绝不存档）——")
    v = d.life_ctl.vitals_for(d, h.name)
    pristine_v = copy.deepcopy(v.to_state())
    pristine_sd = {k: t.detach().cpu().clone() for k, t in h.model.state_dict().items()}
    base = d.probe_loss(h)
    log(f"  睡脑{h.name} 基线探测={base:.4f}")
    deltas = []
    for i in range(1, 6):
        h.model.load_state_dict({k: t.to(d.device) for k, t in pristine_sd.items()})
        v.from_state(copy.deepcopy(pristine_v))
        rep = d.life_ctl._redo(d, h, v)
        n_units = sum(len(x) for x in rep["recycled"].values())
        after = d.probe_loss(h)
        delta = after - base
        deltas.append(delta)
        log(f"  抽样{i}：回收 {n_units} 单元（{len(rep['recycled'])} 个site）  "
            f"Δprobe={delta:+.4f}  {'损伤' if delta > 0 else '改善'}"
            f"  {'|Δ|>EPS_PLATEAU(' + str(EPS_PLATEAU) + ')' if abs(delta) > EPS_PLATEAU else '|Δ|≤EPS'}")
    log(f"  Δprobe 分布：min={min(deltas):+.4f} max={max(deltas):+.4f} "
        f"均值={sum(deltas)/len(deltas):+.4f}  全部|Δ|>{EPS_PLATEAU}的比例="
        f"{sum(1 for x in deltas if abs(x) > EPS_PLATEAU)}/5")
    log(f"  对照：休眠占比={rep['dormant_frac']:.4f}  "
        f"（凋零门槛 decay≥{WITHER_DORMANT_FRAC} born≥{WITHER_BORN_FRAC}）")

    # ---------- 汇总 ----------
    log("========== 诊断总结 ==========")
    log(f"参数总量（诊断终点）：A={params_of(a):,} B={params_of(b):,} Σ={params_of(a) + params_of(b):,}"
        f"（与起点一致=纯测量未动身体）")
    log(f"生产资产指纹（收工）：probe.txt={sha256(os.path.join(ROOT, 'probes', 'probe.txt'))[:16]}…  "
        f"memory_cold.jsonl={sha256(os.path.join(ROOT, 'dolphin', 'memory_cold.jsonl'))[:16]}…")


if __name__ == "__main__":
    main()
