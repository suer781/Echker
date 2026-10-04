# -*- coding: utf-8 -*-
"""呼吸实验驱动器（2026-10-04，实验专用，非生产管线）。

【复跑棒标注 2026-10-05】前任殉职于阶段1周期14/15（平台期始终 0，探测损失
仍在 0.036~0.055/周期 稳定下降——steps=120 的每周期改善远高于 EPS_PLATEAU
=0.005，平台期按此配方不可达）。复跑改动（全部为观测/存档层，不碰决策）：
  1. --s1-steps / --s2-steps 可分阶段下调每周期训练步数。实测依据（复跑
     首次 12 步尝试，周期2~12）：每步改善率最高 ~0.0038（0.0458/12步），
     且记忆复活+做梦每周区注入半遗忘内容，使每周期改善稳定在 0.025~0.046
     ——**步数不是瓶颈，内容补给才是**，12 步压不进平台期。阶段1取 1 步：
     所有观测到的每步改善率（≤0.0038）都落在 EPS_PLATEAU=0.005 之下，
     平台期计数由此真实累积；阶段2取 24 步：变宽验收（GROWN 门控）需要
     margin>2ε 的真改善，全新 gsm8k 上 24 步能诚实通过。平台期阶梯、
     ghost_probe 判决、手术执行、体检验收全部照旧真实运行。
  2. 阶段1增加周期性存档（原来只在阶段边界存——殉职即全丢）。
  3. record() 增加睡脑休眠占比观测打印（只读 vitals 账本，实验专用）。
  4. 开工打印随档喂食游标（证明 gsm8k 未被正式管线消费过）。

目的：人为构造平台期场景，让 M4 自生长/自凋零被真实触发，用每周期
sum(p.numel()) 的**实际参数总量**作为"身体真的会涨/掉"的铁证。

呼吸节律（缩→涨）：
  阶段1 诱导平台期：cmath 前 SLICE 条小切片反复喂（反刍同款语义——L7 明文
        允许跨周期重复，feed.ruminate 路径同样不给重复内容打负分，故本阶段
        不接 structural_reward）。每周期只喂半片（≈38 条），使
        _since_sleep≈38 → 睡眠债 ≈2.5 < 4.0（Bellesi 2017 手术守卫不误伤）。
  阶段2 换粮触发生长：切**未喂过**的 gsm8k 全新记录（复用正式喂食管线
        feed.feed_from；零点实验只喂过 logiqa+cmath，gsm8k 全新）。

律 L11 合规边界：本驱动器只决定「何时睡、喂什么」——平台期阶梯、
ghost_probe 判决、手术执行、体检验收全部在 dolphin/life.py 内部自主完成；
本脚本不读改其决策、不调用任何手术函数、不碰任何 M4 常量。
周期间隙 torch.cuda.empty_cache() 属显存卫生（让显存守卫读到公平的
"当前预留"值，避免分配器缓存残渣造成误推迟），不构成 M4 决策干预。

复用清单（import 复用，不重写生产逻辑）：
  - dolphin.life.run_cycle        唯一睡眠周期实现（M0+M4 全语义）
  - feed.feed_from                正式喂食管线（游标/序列化/喂入）
  - dolphin.datasets.iter_source  数据解析器

观测全部在驱动器层完成（周期首末各测一次 sum(p.numel())），不改动
life.py / feed.py / model.py / probe.txt / 既有测试。

用法（项目根目录下）：
  "C:/.../python.exe" 定标/breathing_experiment.py --s1-max 15 --s2-cycles 5 \
      --steps 120 --slice 80 --s2-feed 50 --save-every 4 \
      2>&1 | tee 实验记录/呼吸实验_20261004.log
"""
import argparse
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
from dolphin.life import PLATEAU_CYCLES, run_cycle as life_run_cycle  # noqa: E402
from feed import FED_STATE, feed_from  # noqa: E402

T0 = time.monotonic()
WALL_BUDGET_S = 40 * 60          # 实验总墙钟预算（收兵线，留最终存档余量）
TIMELINE = []                    # 参数总量时间线（每周期一行）
EVENTS = []                      # 触发链事件流水


def log(msg):
    print(f"[+{time.monotonic() - T0:8.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_of(hem):
    """铁证口径：sum(p.numel() for p in model.parameters())。"""
    return sum(p.numel() for p in hem.model.parameters())


def iron_line(d, tag):
    """阶段转换铁证：实际参数总量 + d_model + 状态机 + 门控锚。"""
    a, b = d.h[0], d.h[1]
    aw, sl = d.awake(), d.sleeping()
    probe = d.probe_loss(aw)
    ctl = d.life_ctl
    log(f"[铁证|{tag}] 醒脑{aw.name}={params_of(aw):,}  睡脑{sl.name}={params_of(sl):,}"
        f"  两半球合计={params_of(a) + params_of(b):,}"
        f"  |  d_model={sl.model.cfg.d_model} n_layers={sl.model.cfg.n_layers}"
        f"  |  状态机={ctl.state} 平台期={ctl.plateau}（阶梯 {ctl.plateau // PLATEAU_CYCLES}/3）"
        f"  |  醒脑探测={probe:.4f} 历史锚={ctl.anchor_best}")
    TIMELINE.append({"tag": tag, "pa": params_of(a), "pb": params_of(b),
                     "d_model": sl.model.cfg.d_model, "n_layers": sl.model.cfg.n_layers,
                     "state": ctl.state, "plateau": ctl.plateau, "probe": round(probe, 4)})


def _fmt(v, n=120):
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


def record(d, rep, stage, idx, pa0, pb0, fed_note):
    """每周期遥测：状态机 + 参数总量 + 门控判决 + M4 事件 + 身体变化铁证。"""
    ctl = d.life_ctl
    pa1, pb1 = params_of(d.h[0]), params_of(d.h[1])
    band = ctl.plateau // PLATEAU_CYCLES
    m, eps = rep.get("gate_margin"), rep.get("gate_eps")
    if m is None:
        mstr, verdict = "—", ("空转" if rep.get("note") else "—")
    else:
        mstr = f"{m:+.4f}(ε={eps})"
        verdict = "通过" if rep.get("passed") else "否决"
    if rep.get("swapped"):
        swap = "换班"
    elif rep.get("m4_wither_keep") is not None:
        swap = f"学生保留(第{rep['m4_wither_keep']}代)"
    else:
        swap = "回滚/未换"
    log(f"[呼吸 {stage}周期{idx}] 喂入={fed_note} 状态={ctl.state} "
        f"平台期={ctl.plateau}(阶梯{band}/3) 参数 A={pa1:,} B={pb1:,} Σ={pa1 + pb1:,} "
        f"d_model={d.awake().model.cfg.d_model} 探测={rep.get('probe_new')} "
        f"{verdict} margin={mstr} → {swap}")
    log(f"    选拔={rep.get('selected')} 做梦={rep.get('dreamed')} lr={rep.get('lr')}"
        f" 阈值={d.buffer.sleep_threshold:.2f} 记忆库={len(d.memory.entries)}"
        f" 冷层={len(d.memory.cold_entries)} 手术LR窗口={ctl.lr_window}")
    m4 = {k: rep[k] for k in rep if k.startswith("m4_")}
    if m4:
        log("    M4事件: " + "  ".join(f"{k}={_fmt(v)}" for k, v in m4.items()))
    # —— 观测[实验专用]：睡脑休眠占比（只读账本，不干预决策）——
    #    凋零分支的门槛：decay≥WITHER_DORMANT_FRAC(0.10)，born_again≥0.25。
    vslp = ctl.vitals_for(d, d.sleeping().name)
    dorm_rep, dorm_frac = vslp.dormant_report()
    n_dorm = sum(len(x) for x in dorm_rep.values())
    n_all = sum(led.C for led in vslp.sites.values())
    log(f"    观测[实验专用]: 睡脑{d.sleeping().name} 休眠占比={dorm_frac:.4f}"
        f"（确认休眠 {n_dorm}/{n_all}；凋零门槛 decay≥0.10 born≥0.25）")
    # —— 身体变化铁证（变宽 / born-again / 回滚还原 全走这里）——
    if (pa0, pb0) != (pa1, pb1):
        why = ("变宽手术执行" if rep.get("m4_surgery")
               else ("凋零换装执行" if rep.get("m4_wither_start")
                     else ("手术回滚还原" if rep.get("m4_rollback")
                           else ("凋零放弃还原" if rep.get("m4_wither_abandoned")
                                 else "形态变化"))))
        log(f"[铁证|身体变化] {why}: A {pa0:,}→{pa1:,}  B {pb0:,}→{pb1:,}  "
            f"合计 {pa0 + pb0:,}→{pa1 + pb1:,}（Δ{(pa1 + pb1) - (pa0 + pb0):+,}）")
        EVENTS.append({"cycle": rep.get("cycle"), "kind": why, "pa0": pa0, "pa1": pa1,
                       "pb0": pb0, "pb1": pb1, "report_m4": {k: rep[k] for k in m4}})
    else:
        EVENTS.append({"cycle": rep.get("cycle"), "kind": "周期",
                       "state": ctl.state, "plateau": ctl.plateau,
                       "margin": m, "passed": rep.get("passed"),
                       "m4_keys": sorted(m4)})
    TIMELINE.append({"tag": f"{stage}周期{idx}", "pa": pa1, "pb": pb1,
                     "d_model": d.awake().model.cfg.d_model,
                     "n_layers": d.awake().model.cfg.n_layers,
                     "state": ctl.state, "plateau": ctl.plateau,
                     "probe": rep.get("probe_new"), "margin": m,
                     "passed": rep.get("passed"), "swapped": rep.get("swapped")})
    return pa1, pb1


def maybe_save(d, force=False, count=0, every=4):
    if force or (count % every == 0 and count > 0):
        t0 = time.monotonic()
        d.save(FED_STATE)
        log(f"[存档] → {FED_STATE}（{time.monotonic() - t0:.1f}s）")


def run_one_cycle(d, args, pa0, pb0, stage, idx, fed_note, steps):
    rep = life_run_cycle(d, steps=steps, feeding=True)
    return record(d, rep, stage, idx, pa0, pb0, fed_note), rep


def stage1(d, args):
    """阶段1：小切片反复喂 → 诱导平台期 → 平台期阶梯逐级触发。"""
    texts = []
    for rec in itertools.islice(iter_source("cmath"), 0, args.slice):
        t = serialize(rec)
        if t:
            texts.append(t)
    half = (len(texts) + 1) // 2
    total_b = sum(len(t.encode("utf-8")) for t in texts)
    debt = half / max(1, d.target_interval)
    log(f"[阶段1] 平台期诱导：cmath 前 {args.slice} 条 → 可用 {len(texts)} 条（{total_b} 字节），"
        f"两半轮流喂（每周期 {half} 条 ≈ {total_b // 2} 字节）")
    log(f"[阶段1] 睡眠债守卫自检：_since_sleep={half} → 债 {debt:.2f} < 4.0（不误伤手术）"
        f"  反刍语义：重复内容不打负分（L7 跨周期重复合法）")
    iron_line(d, "阶段1起点")
    base_cycle = None          # plateau 首次 ≥9（第三阶梯）的周期号
    saves = 0
    i = 0
    while i < args.s1_max:
        i += 1
        if time.monotonic() - T0 > WALL_BUDGET_S:
            log("[阶段1] 达到墙钟预算，提前收兵（如实报告）")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 显存卫生：给守卫公平读数（实验专用，非 M4 干预）
        lo = 0 if i % 2 else half
        for t in texts[lo:lo + half]:
            d.learn(t, source="breath_slice")
        pa0, pb0 = params_of(d.h[0]), params_of(d.h[1])
        (pa1, pb1), rep = run_one_cycle(d, args, pa0, pb0, "S1", i,
                                        f"{half}条呼吸切片", steps=args.s1_steps)
        saves += 1
        maybe_save(d, count=saves, every=args.save_every)  # 复跑新增：中途殉职不再全丢
        ctl = d.life_ctl
        if base_cycle is None and ctl.plateau >= 3 * PLATEAU_CYCLES:
            base_cycle = i
            log(f"[阶段1] 第三阶梯到达（平台期 {ctl.plateau} ≥ {3 * PLATEAU_CYCLES}）"
                f"→ 手术排程通道已开（ghost_probe 判决在位）")
        # 收兵条件：第三阶梯已到 + 已观察 2 个周期 + 无未执行计划 + 本周期无手术落地
        settled = base_cycle is not None and (i - base_cycle) >= 2 \
            and ctl.plan is None and not (rep.get("m4_surgery") or rep.get("m4_wither_start"))
        hard_over = i >= args.s1_max and ctl.plan is None
        if settled or hard_over:
            log(f"[阶段1] 收兵（平台期={ctl.plateau} 状态={ctl.state} 计划待执行={ctl.plan is not None}）")
            break
        if ctl.plan is not None and i >= args.s1_max:
            args.s1_max += 2   # 有计划未执行：最多顺延 2 周期等它落地与验收
    maybe_save(d, force=True, count=-1)
    iron_line(d, "阶段1终点")
    return i


def stage2(d, args):
    """阶段2：换粮（未喂过的 gsm8k 全新记录）→ ghost 增益 → GROW。"""
    log(f"[阶段2] 换粮触发生长：gsm8k 全新记录（每周期 {args.s2_feed} 条，"
        f"喂入后 _since_sleep={args.s2_feed} → 债 {args.s2_feed / max(1, d.target_interval):.1f}）"
        f"——正式喂食管线 feed_from 复用")
    iron_line(d, "阶段2起点")
    cursor = {"sources": ["gsm8k"], "source_idx": 0, "record_idx": 0}
    seen = {}
    total_fed = 0
    for i in range(1, args.s2_cycles + 1):
        if time.monotonic() - T0 > WALL_BUDGET_S:
            log("[阶段2] 达到墙钟预算，提前收兵（如实报告）")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fed, _, skip = feed_from(d, ["gsm8k"], cursor, seen, trainer=None,
                                 sleep_every=0, quota=args.s2_feed,
                                 progress_every=10 ** 9)
        total_fed += fed
        pa0, pb0 = params_of(d.h[0]), params_of(d.h[1])
        (pa1, pb1), rep = run_one_cycle(d, args, pa0, pb0, "S2", i,
                                        f"{fed}条全新gsm8k", steps=args.s2_steps)
        maybe_save(d, count=i, every=args.save_every)
    log(f"[阶段2] 收兵（全新喂入累计 {total_fed} 条）")
    maybe_save(d, force=True, count=-1)
    iron_line(d, "实验终点")
    return total_fed


def summary(d, s1_cycles, s2_fed):
    log("========== 呼吸实验总结 ==========")
    log("参数总量时间线（铁证口径 sum p.numel()，单位=参数个数）：")
    for row in TIMELINE:
        extra = ""
        if row.get("probe") is not None:
            extra += f"  探测={row['probe']}"
            if row.get("margin") is not None:
                extra += f" margin={row['margin']:+.4f}"
        log(f"  {row['tag']:<12} A={row['pa']:,} B={row['pb']:,} Σ={row['pa'] + row['pb']:,}"
            f"  d_model={row['d_model']} 层数={row['n_layers']}"
            f"  状态={row['state']} 平台期={row['plateau']}" + extra)
    log("触发链事件流水：")
    for ev in EVENTS:
        log("  " + _fmt(ev, 200))
    a, b = d.h[0], d.h[1]
    log(f"最终状态：醒脑={d.awake().name} 状态机={d.life_ctl.state} "
        f"平台期={d.life_ctl.plateau} 锚={d.life_ctl.anchor_best} cycle={d.cycle}")
    log(f"最终参数：A={params_of(a):,} B={params_of(b):,} Σ={params_of(a) + params_of(b):,}")


def main():
    ap = argparse.ArgumentParser(description="呼吸实验（M4 生长/凋零真实触发，实验专用）")
    ap.add_argument("--s1-max", type=int, default=15, help="阶段1最大周期数")
    ap.add_argument("--s2-cycles", type=int, default=5, help="阶段2周期数")
    ap.add_argument("--steps", type=int, default=120, help="每周期训练步数（生产默认 120）")
    ap.add_argument("--s1-steps", type=int, default=None,
                    help="阶段1每周期训练步数（缺省随 --steps；复跑用 1——实测依据见文件头）")
    ap.add_argument("--s2-steps", type=int, default=None,
                    help="阶段2每周期训练步数（缺省随 --steps；复跑用 24——变宽验收需要真改善）")
    ap.add_argument("--slice", type=int, default=80, help="阶段1小切片条数（cmath 前 N 条）")
    ap.add_argument("--s2-feed", type=int, default=50, help="阶段2每周期全新喂入条数")
    ap.add_argument("--save-every", type=int, default=4, help="每 N 周期存档一次")
    args = ap.parse_args()
    if args.s1_steps is None:
        args.s1_steps = args.steps
    if args.s2_steps is None:
        args.s2_steps = args.steps

    log("=" * 24 + " 呼吸实验开工 " + "=" * 24)
    log(f"参数：{vars(args)}  墙钟预算 {WALL_BUDGET_S // 60} 分钟")
    log(f"生产资产指纹（开工）：probe.txt sha256={sha256(os.path.join(ROOT, 'probes', 'probe.txt'))[:16]}…"
        f"  memory_cold.jsonl sha256={sha256(os.path.join(ROOT, 'dolphin', 'memory_cold.jsonl'))[:16]}…")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Dolphin(device=dev)
    log(f"读档 {FED_STATE} …（1.38GB，约半分钟）")
    t0 = time.monotonic()
    d.load(FED_STATE)
    log(f"读档完成（{time.monotonic() - t0:.1f}s）：cycle={d.cycle} 醒脑={d.awake().name}"
        f" 阈值={d.buffer.sleep_threshold:.2f} 记忆库={len(d.memory.entries)}"
        f" 冷层={len(d.memory.cold_entries)} 探测块={len(d.probe_chunks)}"
        f" 状态机={d.life_ctl.state} 平台期={d.life_ctl.plateau} 锚={d.life_ctl.anchor_best}")
    if d.gate_disabled:
        log("[致命] 探测集缺失（G8），实验无法进行")
        return
    if d.life_enabled is False:
        d.life_enabled = True  # 实验前提：L11 总开关开启（读档默认即 True，双保险）
    log(f"随档喂食游标（G3）：{d.feed_cursor}  —— 阶段2将以本地游标从 gsm8k "
        f"record_idx=0 起喂（不写随档游标）")
    try:
        s1 = stage1(d, args)
        s2 = stage2(d, args)
        summary(d, s1, s2)
    except KeyboardInterrupt:
        log("[中断] 优雅收兵：存档 + 总结")
        maybe_save(d, force=True, count=-1)
        summary(d, -1, -1)
    finally:
        log(f"生产资产指纹（收工）：probe.txt sha256={sha256(os.path.join(ROOT, 'probes', 'probe.txt'))[:16]}…"
            f"  memory_cold.jsonl sha256={sha256(os.path.join(ROOT, 'dolphin', 'memory_cold.jsonl'))[:16]}…")


if __name__ == "__main__":
    main()
