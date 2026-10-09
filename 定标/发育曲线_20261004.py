# -*- coding: utf-8 -*-
"""发育实验曲线绘制与证据分析（2026-10-05）。

输入：实验记录/发育时间线_20261004.jsonl（feed.py SleepTrainer._timeline 落盘）
输出：实验记录/发育曲线_20261004.png（生长曲线 + 学习曲线）+ stdout 摘要
证据：V1 双脑轮换序列 / 手术事件标注 / 门控判决统计。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.join(ROOT, "实验记录", "发育时间线_20261004.jsonl")
OUT = os.path.join(ROOT, "实验记录", "发育曲线_20261004.png")

rows = [json.loads(l) for l in open(TL, encoding="utf-8")]
# 只保留发育实验本体行（41,856=种子/42,376=首刀后）；排除：
#  a) 空选拔早退行（无 body 观测，params=None）
#  b) 终验测试套件运行时 SleepTrainer 测试写入的测试行（132,864 等）
rows = [r for r in rows if r.get("params") in (41856, 42376)]

cyc = [r["cycle"] for r in rows]
par = [r["params"] for r in rows]
runmax = []
m = 0
for p in par:
    m = max(m, p)
    runmax.append(m)
probe = [r.get("probe_new") for r in rows]

surg = [(r["cycle"], r["params"], r["m4_surgery"], r.get("probe_old"), r.get("probe_new"))
        for r in rows if r.get("m4_surgery")]
plans = [(r["cycle"], r.get("m4_plan")) for r in rows
         if r.get("m4_plan") and r["state"] in ("GROW_PLAN", "WITHER_PLAN")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
a1.step(cyc, par, where="post", color="#888", lw=0.8, alpha=0.6,
        label="在训半球参数（随换班交替）")
a1.step(cyc, runmax, where="post", color="#0a7", lw=2.2, label="体型前沿（历史最大）")
a1.axhline(41856, color="#bbb", ls=":", lw=1)
a1.text(1, 41950, "种子出生体重 41,856（0.042M）", fontsize=9, color="#666")
for c, p, s, *_ in surg:
    a1.annotate(f"生长: {s.get('axis')} L{s.get('layer')} +{s.get('delta')}通道\n"
                f"（ghost 需求过 1.5× 门槛）",
                xy=(c, p), xytext=(c - 30, p - 300), textcoords="data", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#a06"},
                color="#a06")
a1.set_ylim(41750, 42900)
a1.set_ylabel("参数总量")
a1.set_title("蓝蓟智能发育实验：参数总量生长曲线（最小种子 d32×2层×2头 = 41,856 → 经验驱动生长）")
a1.legend(loc="lower right", fontsize=9)
a1.grid(alpha=0.3)

a2.plot(cyc, probe, "-", color="#06c", lw=1.6, label="probe 损失（体检卷面，训练侧永不接触）")
a2.axhline(5.545, color="#c33", ls="--", lw=1)
a2.text(1, 5.56, "随机初始化理论基线 ln(256)≈5.545", fontsize=9, color="#c33")
for c, *_ in surg:
    a2.axvline(c, color="#a06", ls=":", lw=1.2)
a2.set_xlabel("睡眠周期")
a2.set_ylabel("probe NLL")
a2.set_title("学习曲线：边长身体边学习（每个生长事件=竖虚线）")
a2.legend(loc="upper right", fontsize=9)
a2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"曲线已存 {OUT}")

# —— 摘要统计 ——
n = len(rows)
swaps = sum(1 for r in rows if r.get("swapped"))
passed = sum(1 for r in rows if r.get("passed"))
print(f"\n== 发育实验摘要（{n} 个有效周期）==")
print(f"体检判决：{passed}/{n} 通过，换班 {swaps} 次")
print(f"probe：首测 {probe[0]} → 末测 {probe[-1]}（随机基线 5.545，降幅 "
      f"{(probe[0]-probe[-1])/5.545*100:.1f}%）")
print(f"手术事件 {len(surg)} 次：")
for c, p, s, po, pn in surg:
    print(f"  cycle={c}: {s} → 参数 {p:,}，体检 {po}→{pn} passed")
print(f"生长/凋零计划 {len(plans)} 次：")
for c, pl in plans:
    print(f"  cycle={c}: {pl}")
# V1 轮换序列
seq = []
for r in rows:
    nm = r.get("awake")
    if not seq or seq[-1][1] != nm:
        seq.append((r["cycle"], nm))
print(f"双脑轮换序列（连续同半球合并）：{' → '.join(f'{c}:{n}' for c, n in seq)}")
