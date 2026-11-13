# -*- coding: utf-8 -*-
import json, os, datetime
from collections import Counter

ROOT = r"O:\Documents\智能自学习AI（新）"
MD = os.path.join(ROOT, "AUDIT_ADAPTIVE_SWITCH.md")
MERGED = os.path.join(ROOT, "_audit_tmp", "merged_findings.json")

with open(MERGED, "r", encoding="utf-8-sig") as f:
    merged = json.load(f)

rows = merged["findings"]
SEV = {"阻断": 0, "高": 1, "中": 2, "低": 3}
rows.sort(key=lambda r: (SEV.get(r["severity"], 9), r["id"]))
cnt = Counter(r["severity"] for r in rows)
kind = Counter(r["kind"] for r in rows)
strength = Counter(r["strength"] for r in rows)

def loc(r):
    v = r.get("location") or []
    if isinstance(v, str):
        return v.replace("|", "/")
    return "<br>".join(x.replace("|", "/") for x in v[:2])

lines = []
lines.append("\n## 附录 A：全部发现索引（%d 条）\n" % len(rows))
lines.append("严重度分布：高 %d / 中 %d / 低 %d；性质：真实缺陷 %d / 验证盲区 %d / 设计边界 %d；证据强度：已实测 %d / 代码推断 %d / 尚未验证 %d。\n"
             % (cnt["高"], cnt["中"], cnt["低"], kind["真实缺陷"], kind["验证盲区"], kind["设计边界"],
                strength["已实测证实"], strength["代码推断"], strength["尚未验证"]))
lines.append("| ID | 严重度 | 性质 | 证据强度 | 位置 | 标题 |")
lines.append("|---|---|---|---|---|---|")
for r in rows:
    t = r["title"].replace("|", "/")
    lines.append("| `%s` | %s | %s | %s | %s | %s |" % (r["id"], r["severity"], r["kind"], r["strength"], loc(r), t))

lines.append("""
> 表中严重度沿用各分区原始定级。报告中 §3 的 B1–B7 是**针对「是否进入 10.164 GiB 全量训练」这一门禁**的重定级，不改变原表定级。

---

## 附录 B：证据与复现路径

所有实验均在 `O:\\Documents\\智能自学习AI（新）\\_audit_tmp\\` 下，源码只读，未改动 `life_agent/`、`tests/`、`validation_runs/`、`state_dataset_smoke/`、`O:\\数据集`。

| 分区 | 目录 | 主要产物 | 关键命令 |
|---|---|---|---|
| 并发与队列 | `_audit_tmp\\concurrency\\` | `findings.json`、`NOTES.md`、`audit_metrics.json`、`audit2_metrics.json` | `& .venv-life\\Scripts\\python.exe _audit_tmp\\concurrency\\run_audit.py`；`run_audit2.py`（64 轮混合 / 24+36 轮） |
| 切换数学 | `_audit_tmp\\math\\` | `findings.json`、`sim_results.json`、`derived_ratios.json`、`metrics_analysis.json`、`checkpoint_inspection.json`、`sync_cpu_check.json` | 纯 CPU 仿真 + 检查点只读解析 |
| 持久化与验证器 | `_audit_tmp\\persistence\\` | `findings.json`、`audit_roundtrip.py`、`audit_results.json`、`run1..3\\state` | `& .venv-life\\Scripts\\python.exe _audit_tmp\\persistence\\audit_roundtrip.py` |
| 训练准备与边界 | `_audit_tmp\\readiness\\` | `findings.json`、`boundary_experiment.py`、`experiments.json` | `& .venv-life\\Scripts\\python.exe -m unittest discover -s tests -v` |
| 重复/新颖度 | `_audit_tmp\\novelty\\` | `findings.json`、`section.md`、`probe_novelty.py`、`probe_results.json` | `& .venv-life\\Scripts\\python.exe -B _audit_tmp\\novelty\\probe_novelty.py`（importlib 直载，不拉起 torch） |

合并产物：`_audit_tmp\\merged_findings.json`（64 条原文，含 evidence / impact / recommendation / regression_test 完整字段）。

### 复现注意事项

- 全程 `PYTHONDONTWRITEBYTECODE=1`；未在源码目录留下 `__pycache__`。
- GPU 实验使用 GTX 1060 3GB、d_model=64 的小配置，未触发 OOM；**不能据此推断全尺寸配置的资源余量**。
- `pytest` 未安装，测试必须用 `unittest`；`pytest -q` 会直接失败。
- `math` 分区的切换回放使用合成步数，只能验证公式与 reason，不能代表真实训练动力学。

---

## 附录 C：本报告明确不声称什么

1. **不声称语义理解或通用智能**。200 次混合交互、20 MB 真实字节、38 项机制测试，只能证明「机制在运行」，不能证明「学到了语义」。
2. **不声称长期稳定性**。最长实测为 275.7 s；全量 41.8 h 是外推，不是测量。RSS 是否趋平、检查点是否会拖慢、日志是否会撑爆，均未观测。
3. **不把 61/200 定性为抖动**。可观测的 39 次间隔 min 3 / median 5 / max 12，无相邻连翻；22 次落在无逐轮记录的灌入阶段，证据不足以支持任一方向。因此也**不建议**加固定间隔定时器来压频率。
4. **不声称故障路径已验证**。worker 崩溃、future 永久挂起、GPU OOM、磁盘写满均未注入；`CONC-01` 等为代码推断。
5. **不声称「一切皆自适应」**。当前真正在线的只有学习率、结构宽度、半球切换三处；其余为常量或半自适应（§6）。
6. **不声称「厌恶重复」已是能力**。§9 全部为设计边界与最小实验建议，代码中没有任何习惯化/饱腹度实现。
7. **不声称修复没有代价**。`_drain_passive_consolidation()` 解决了被动饥饿，但引入/暴露了 pending 双残留、周期落盘边界不一致、RLock 内阻塞与 monitor 兜底失效（§2.4）。

---

## 附录 D：一句话总结

**这次修复治好了「被动侧被饿死」，但没有治好「判据算错」和「验收看不出算错」。** 先修 B1（基线角色）、B2（工作口径）、B3/B4（恢复与落盘），再把切换决策本身落盘成可复算的证据，才谈得上「数据压力驱动」这句话成立。
""")

with open(MD, "a", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("MD bytes:", os.path.getsize(MD))

# ---- audit JSON under validation_runs ----
BLOCKING = [
    {"id": "B1", "from": "MATH-01", "severity": "阻断", "title": "切换基线按半球名而非角色传参，active=right 时整体错位",
     "why": "note_switch(left.step, right.step) 被当作 (active_work, passive_work)；decide 按角色传参，二者错位，一半决策基于错误基线",
     "fix": "note_switch 显式传 state(self.active)/state(self.passive) 的步数，并在 to_dict 存角色名",
     "test": "test_switch_baseline_follows_role_not_hemisphere_name / test_switch_baseline_roundtrip_right_active"},
    {"id": "B2", "from": "MATH-02", "severity": "阻断", "title": "global_step 同时计 train/ttt/consolidate，work 判据混用两类工作量",
     "why": "worker.py:295 无 kind 分支；passive_delta 含巩固步数，active_delta>=passive_delta 可永久阻塞切换或反向放行",
     "fix": "按 kind 分计数，sufficient_work 改为同类工作量比较或按样本数加权",
     "test": "test_work_counter_separates_train_and_consolidate"},
    {"id": "B3", "from": "PERS-02", "severity": "阻断", "title": "恢复一致性只比单层 blocks.0.fc1.weight，且是磁盘到磁盘",
     "why": "debt/controller/structure/optimizer/history/last_decision/RNG 均未比对，还原错误不可见",
     "fix": "state_fingerprint() 全字段摘要，save 前与 load 后逐项 diff 并输出",
     "test": "test_checkpoint_roundtrip_full_fingerprint"},
    {"id": "B4", "from": "PERS-01 + CONC-02 + PERS-05", "severity": "阻断", "title": "周期 save 不等在途巩固；due() 先记账后训练",
     "why": "snapshot(2) 可先于 consolidate(4)；实测右半球 16 处不一致；检查点出现已记账未训练",
     "fix": "save 内先 _wait_consolidations；due() 拆为预选与提交记账，记账移到训练成功后",
     "test": "test_save_waits_for_inflight_consolidation / test_due_does_not_count_before_training"},
    {"id": "B5", "from": "PERS-06 + PERS-07 + PERS-04", "severity": "阻断", "title": "验收判据可假阳：skip 短路、计数型切换判据、决策样本缺失",
     "why": "skip-real 三项恒 true；adaptive_switch_without_fixed_countdown 只查旧键缺失 + switches>=8；report adaptive_switch=null",
     "fix": "skipped 语义；切换决策逐条落盘并可复算；duplicate_cleaning 改为注入重复验证；both_hemispheres_trained 改为窗口饥饿指标",
     "test": "test_skip_real_marks_skipped_not_passed / test_switch_decisions_are_logged_and_recomputable / test_duplicate_injection_merges_digest / test_hemisphere_starvation_window"},
    {"id": "B6", "from": "READY-01 + READY-05", "severity": "阻断", "title": "20 MB 只覆盖 LogiQA 8 文件（0.184% 字节），吞吐无法拆分",
     "why": "Parquet/Arrow/CSV/JSON/图片/zip 全未在真实数据验证；41.8h 为外推",
     "fix": "跨格式分层抽 2-4 GB（每格式≥100 帧，含边界），落盘每格式 frames/bytes/seconds/RSS/GPU/失败率",
     "test": "test_dataset_format_coverage_matrix / test_max_frame_boundary"},
    {"id": "B7", "from": "READY-03", "severity": "阻断", "title": "验证路径先写进度后存检查点，崩溃丢 1-3 帧",
     "why": "continuous_validation.py:422-423 与 ingest_dataset.py:250-264 顺序相反；丢帧不报错",
     "fix": "统一为「原子写检查点+offset，再提交 progress」，并校验末帧 payload hash",
     "test": "test_progress_offset_only_advances_after_checkpoint"},
]

CRITERIA = [
    ["cpu_inference_gpu_training", "读 training_device==cuda + 一个 CPU 参数在 CPU", "中：设备标签"],
    ["mixed_cycles_completed", "CLI 参数 mixed_cycles>=200", "中：自报参数"],
    ["both_hemispheres_trained", "终点两侧步数>=20", "中：955 步也通过，对饥饿不敏感"],
    ["automatic_switching", "switches>=8", "高：纯计数"],
    ["adaptive_switch_without_fixed_countdown", "无旧键属性 + switches>=8", "高：删键当机制"],
    ["ttt_weight_updates", "前 40 轮嵌入 L1>0", "中：混入 sync、后 160 轮无探针"],
    ["passive_cpu_isolation", "运行期 hook 计数，判据字段硬编码 True", "低-中：落盘未关联记录"],
    ["train_loss_finite", "终点 last_loss 有限", "中：非有限批次 continue 跳过"],
    ["debt_bounded / replay_bounded", "终点长度<=capacity", "低-中：瞬时采样"],
    ["duplicate_cleaning", "increase<=5 且 min_feed_weight<0.5", "高：429->430 仍通过"],
    ["automatic_structure_events", "独立模型合成 growth/prune", "中：非在线真实决策"],
    ["restarts_exact", ">=4 条且 5 字段 true", "高：单层权重+磁盘到磁盘"],
    ["real_dataset_volume", "skip 短路 / payload>=20MB", "高：skip 时零字节通过"],
    ["real_dataset_resume", "skip 时硬编码 true", "高"],
    ["source_files_unchanged", "默认 true，非 skip 才比对", "高：skip 时永不校验"],
    ["queue_drained", "终点 queued_jobs==0", "中：瞬时"],
    ["learning_rate_in_bounds", "全程 lr 采样落在 min/max", "低"],
    ["host_memory_bounded", "final-baseline<2048MB", "中：固定阈值，RSS 含共享页"],
    ["gpu_memory_bounded", "final<=baseline+512MB", "中：allocated 非 reserved"],
    ["no_runtime_errors", "len(run.errors)==0", "中-高：后台 future 异常只进 LOG"],
]

report = {
    "audit_name": "adaptive hemisphere switch audit after feed/consolidate starvation fix",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "project_root": ROOT,
    "code_baseline_time": "2026-11-13T20:47 (agent.py/worker.py/switching.py)",
    "reference_validation": r"validation_runs\continuous-adaptive-full-7",
    "scope": "Correctness, verification credibility and design boundaries after the _drain_passive_consolidation fix. No semantic/AGI/long-term-stability claim.",
    "method": {
        "reviewers": 5,
        "partitions": ["concurrency", "switch_math", "persistence_verifier", "readiness", "novelty"],
        "isolation_dir": "_audit_tmp\\<partition>\\",
        "source_modification": "none",
        "full_dataset_training_started": False,
    },
    "verdict": {
        "allow_full_dataset_training": False,
        "reason": "Two mathematical defects in the switch criterion (role-misaligned baselines, mixed work counters) plus a verifier whose core switch criterion is a count and leaves no decision samples. Full training would produce an unattributable and unprovable run record.",
        "blocking_count": len(BLOCKING),
        "recommended_next": "Close B1-B7, then run a 2-4 GB cross-format bounded measurement with per-stage timing before scheduling the 10.164 GiB run.",
    },
    "blocking_items": BLOCKING,
    "findings_totals": {
        "total": len(rows),
        "by_severity": dict(cnt),
        "by_kind": dict(kind),
        "by_evidence_strength": dict(strength),
    },
    "criteria_audit": [
        {"criterion": c[0], "real_evidence": c[1], "false_positive_risk": c[2]} for c in CRITERIA
    ],
    "measured_facts": {
        "mixed_interactions": 200,
        "switches": 61,
        "elapsed_seconds": 275.7164018,
        "left_steps": 5558,
        "right_steps": 5618,
        "ttt_samples": 40,
        "ttt_minimum_weight_l1_delta": 2.356353998184204,
        "rss_baseline_mb": 422.2890625,
        "rss_final_mb": 1225.69140625,
        "real_bytes": 20000000,
        "real_frames": 26,
        "report_adaptive_switch": None,
        "observed_switch_intervals_rounds": {"min": 3, "median": 5, "mean": 5.68, "max": 12, "interval_1": 0, "observed_count": 39, "unlogged_count": 22},
        "repayment_capacity_ratio": {"left": 2.633, "right": 1.548},
        "due_batches_observed": {"batches": 393, "draws": 3108, "mean_batch": 7.91},
        "exchange_pressure_share_upper_bound": {"simulated": 0.331, "measured_by_count": 0.00608},
        "drain_latency_ms": {"mean": 13.75, "max": 20.6},
        "status_latency_ms": {"mean": 40.6, "max": 166.6},
        "pending_double_key_rate": "23/64",
        "save_inconsistency_run1": "16 fields on right hemisphere",
        "baseline_pressure_reset_run3": "1.46875 -> 0",
        "cleaning_unique": {"before": 429, "after": 430},
        "dataset": {"files": 760, "bytes": 10913873703, "gib": 10.164, "real_sample_files": 8, "real_sample_byte_share": 0.00184},
        "tests": {"runner": "unittest", "collected": 38, "passed": 38, "failed": 0, "pytest_installed": False},
        "throughput_extrapolation_hours": 41.8,
    },
    "novelty_design_section": {
        "status": "not_implemented / design proposal only",
        "claim_boundary": "No emotion, motivation or satiety field exists. Current mechanisms are exact digest dedupe, intra-frame structural weighting, loss-outlier suppression.",
        "current_gap": "No signal answers whether marginal improvement tends to zero under repetition; repeated debt gains weight (0.001->0.06978 over 5x) and outranks novel debt in due().",
        "allowed_role": "down-weight sampling/consolidation priority, exploration trigger, additional gate only; never hard-reject or stop learning",
        "relationship_to_switch": "Current switch is pressure/capacity/work driven, orthogonal to novelty; must not be called boredom-driven",
        "phases": ["record only", "down-weight only", "additional gate"],
        "candidate_signals": [
            {"signal": "rolling byte entropy", "availability": "intra-frame exists; rolling needs bounded counter", "cost": "CPU O(n)", "bias": "high-entropy noise looks novel"},
            {"signal": "n-gram novelty", "availability": "not present; needs bounded recency table", "cost": "CPU hash O(n)", "bias": "domain terms/protocol headers misjudged as familiar"},
            {"signal": "marginal loss improvement", "availability": "stored.loss EWMA exists (worker.py:300-309)", "cost": "zero extra forward/backward", "bias": "LR/batch/weight scaling contamination"},
            {"signal": "gradient SNR", "availability": "not present", "cost": "extra backward/memory", "bias": "grad_clip/AdamW interference", "feasible_on_3gb": False},
            {"signal": "TTT before/after gain", "availability": "ttt_deltas is weight L1, not loss gain", "cost": "two CPU forwards", "bias": "probe distribution shift, n=40"},
        ],
    },
    "findings": rows,
}

out = os.path.join(ROOT, "validation_runs", "audit-adaptive-switch-2026-11-13.json")
if os.path.exists(out):
    raise SystemExit("refusing to overwrite existing report: " + out)
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("JSON bytes:", os.path.getsize(out))
print("JSON path:", out)
