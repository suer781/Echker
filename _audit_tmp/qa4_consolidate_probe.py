# -*- coding: utf-8 -*-
"""P0-1 配套探针: consolidate() 的 loss>2.0 跳过 + NaN 卫兵 + 容量只对真实训练样本记账。"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from byt5_dual import Byt5DualBrain, Seq2SeqSample

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "_audit_tmp" / "qa4_gate_probe_state"
brain = Byt5DualBrain(str(ROOT / "byt5_model"), STATE)
brain.load()
brain.ensure_active_gpu()

passive = brain.passive_h()
print("passive:", passive.name)

# 垃圾样本入被动侧债务（模拟被拒样本错误入账的情形——看巩固侧是否拒烧）
passive.admit_pressure(Seq2SeqSample(
    ids=list(b"garbage gibberish zxq wkjv bwq 0921 lkjhgf"),
    labels=list(b"zxq wkjv bwq 0921 lkjhgf qwert mnbv"),
    weight=1.0, source="stream:fake"))
cap_before = passive.debt.repayment_capacity
debt_before = len(passive.debt)
clean_before = passive.debt.total_clean_added
replay_before = len(passive.replay)

samples = passive.consolidation_samples()
brain.active_h().to_cpu()          # 模拟 consolidate_passive 的换出
result = passive.consolidate(samples, 1.0)
passive.to_cpu()
brain.ensure_active_gpu()
print("垃圾样本 consolidate:", result)
print(f"容量: {cap_before} -> {passive.debt.repayment_capacity}  债务: {debt_before} -> {len(passive.debt)}"
      f"  clean: {clean_before} -> {passive.debt.total_clean_added}  回放: {replay_before} -> {len(passive.replay)}")

# 正常样本对照
passive.admit_pressure(Seq2SeqSample(
    ids=list("你好".encode("utf-8")), labels=list("你好，很高兴见到你。".encode("utf-8")),
    weight=1.0, source="stream:normal"))
cap2 = passive.debt.repayment_capacity
samples2 = passive.consolidation_samples()
brain.active_h().to_cpu()
result2 = passive.consolidate(samples2, 1.0)
passive.to_cpu()
brain.ensure_active_gpu()
print("正常样本 consolidate:", result2)
print(f"容量: {cap2} -> {passive.debt.repayment_capacity} (期望 >0 且 1-3 量级)")
