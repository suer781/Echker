# 自适应半球切换审查报告（修复后）

> **2026-11-13 关闭说明**：本文件保留审查当时（修复前基线）的 64 条发现与结论。B1–B7 的代码路径随后已修复，最终 200 次混合交互 + 20 MB 真实数据验收 21/21、退出码 0；逐项关闭证据见 `AUDIT_CLOSURE.md`。本文件“不建议启动全量训练”的判断适用于审查当时，不适用于当前版本。全量训练已单独启动，尚未完成。

- **审查对象**：`O:\Documents\智能自学习AI（新）`，`life_agent` + `continuous_validation.py` + `tests`
- **代码基线**：2026-11-13 20:47 之后的 `agent.py` / `worker.py` / `switching.py` / `memory.py` / `store.py`
- **对照验收**：`validation_runs/continuous-adaptive-full-7`（200 次混合交互、61 次切换、21/21 true）
- **审查方式**：5 个互斥分区的独立只读评审，各自在 `_audit_tmp\<分区>\` 隔离运行，未修改源码、测试或既有验证产物
- **报告生成**：2026-11-13
- **本报告不声称**：语义理解、通用智能、长期稳定性。它只回答「这次修复之后，哪些地方仍然不可信」。

---

## 0. 结论先行

**不建议进入 `O:\数据集` 10.164 GiB 全量训练。** 不是因为模型能力不够，而是因为**切换判据本身存在两处数学错误，而验收器没有任何一条判据能独立证明「切换由数据压力驱动」**。在修正这些之前启动全量训练，产出的将是一份无法归因、无法复现、也无法证明正确性的长跑记录。

三条最硬的事实：

1. `agent.py:314-318` 把 `left.global_step` / `right.global_step` 按**半球名**传给 `note_switch(active_work, passive_work)`，而 `decide()` 在 `agent.py:336-337` 是按**角色**传的。当 `active == "right"` 时，基线整体错位。仿真下正确值应为 `active_delta=30 / passive_delta=10` 的场景被算成 `50 / -10`（钳 0 后为 `50 / 0`），决策理由变成「被动侧未工作」。
2. `worker.py:295` 的 `state.global_step += 1` 在 `_train` 内部，**ttt / train / consolidate 三类作业共用同一个计数器，没有 kind 分支**。而 `switching.py:86` 的 `sufficient_work = passive_delta > 0 and active_delta >= passive_delta` 把「被动侧巩固步数」和「主动侧学习步数」放在同一把尺子上比较。这不是「工作量加权」，是两个语义不同的量相减。
3. 验收中 `adaptive_switch_without_fixed_countdown` 的真实实现是：**配置里不存在 `switch_interactions` / `switch_interval_s` 属性，且 `switches >= 8`**（`continuous_validation.py:473-478`）。它证明的是「旧键被删掉」和「切换发生了 8 次以上」，不证明切换由压力触发。最终 `report.json` 的 `adaptive_switch` 字段是 `null`（`PERS-04`），即**一条切换决策样本都没留下**。

**结论的可信边界**：「61 次/200 次」目前既不能定性为抖动，也不能排除抖动。混合循环内部可观测的 39 次翻转间隔为 min 3 / median 5 / 平均 5.68 / max 12 轮，**没有一次相邻连翻（interval==1）**；但另外 22 次发生在循环之后的真实数据灌入阶段，该阶段没有逐轮时间戳与压力日志（`MATH-06`）。所以「高频切换=抖动」是一个尚未闭合的盲区，不应据此外推，也不应为了压频率去加固定间隔定时器。

---

## 1. 审查方法与证据分级

### 1.1 分区（互斥，合并后覆盖全部清单）

| 分区 | 负责 | 交付 |
|---|---|---|
| 并发与队列 | 优先级/饥饿、`_drain`、future 生命周期、锁、单写者 | `_audit_tmp\concurrency\findings.json`（10 条，2 组实测） |
| 切换数学与双半球 | 压力累计、偿还容量、work 基线、sync/resize/optimizer | `_audit_tmp\math\findings.json`（10 条，5 组仿真/实测） |
| 持久化与验证器 | save/load 一致性、基线迁移、恢复判据、21 条判据可信度 | `_audit_tmp\persistence\findings.json`（13 条，3 组实测） |
| 训练准备与边界 | 数据集覆盖、断点、吞吐、常量分类、测试覆盖 | `_audit_tmp\readiness\findings.json`（23 条，4 组实测） |
| 重复/新颖度信号 | 情绪宣称 vs 可测机制、候选信号、硬拒绝误伤 | `_audit_tmp\novelty\findings.json`（8 条，3 组探测） |

合计 **64 条**：高 15 / 中 38 / 低 11；真实缺陷 19 / 验证盲区 16 / 设计边界 29；证据强度 已实测证实 28 / 代码推断 32 / 尚未验证 4。

### 1.2 证据强度的定义（避免把推断写成事实）

- **已实测证实**：在隔离目录中实际运行并观察到该现象，命令与产物路径可复跑。**不代表生产负载下同样成立。**
- **代码推断**：由源码路径、锁顺序、队列语义推导，未做故障注入或未构造出确定性复现。
- **尚未验证**：指出了缺口，但没有证据支持任一方向的结论。

### 1.3 合规与局限

- 未修改 `life_agent/`、`tests/`、`validation_runs/`、`state_dataset_smoke/`、`O:\数据集`。
- 未启动 10.164 GiB 全量训练；实验最大规模为 64 轮混合流、d_model=64 的 CPU 小模型。
- 未做 worker 崩溃/挂起故障注入；`CONC-01`（无超时等待）因此是代码推断。
- 各评审的 GPU 实验使用 GTX 1060 3GB，未触发 OOM；因此**没有证据**说明 3GB 显存下更大批次是否可行。

---

## 2. 修复本身是否成立

### 2.1 原饥饿的机制

`submit()` 的优先级表（`worker.py:208`）是**数字越小越优先**：

```
ttt=0  sync=1  resize=1  snapshot=2  train=3  consolidate=4
```

持续 `feed_frame` 时，每帧先 `submit("train", wait=True)`，再 `_schedule_passive_consolidation()` 入队 `consolidate(4)`。train(3) 永远排在 consolidate(4) 前面，被动侧只能在队列空档被调度。真实数据阶段出现右半球 10,213 步 / 左 955 步，就是这条路径的结果。

### 2.2 修复做了什么

`feed_frame`（`agent.py:205-228`）在 `with self.lock` 内：

```python
result = worker.submit("train", active, samples, steps, wait=True)
... _remember ...
self._schedule_passive_consolidation()
self._drain_passive_consolidation()   # 新增：pop 当前 passive 的 future 并 result()
self._maybe_switch()
```

`_drain_passive_consolidation()`（`264-267`）只是把 `pending_consolidation[self.passive]` pop 出来等结果。它保证了「每次 feed 结束时，队列里那一笔被动巩固已经做完」。

### 2.3 修复有效的实测证据

`_audit_tmp\concurrency\` 的 64 轮混合流（18 次切换、左右 114/120 步）：

- 无队列积压，`wait=False` 的 future **122 个全部 resolve、0 异常**；
- `drain` 每次推进被动侧 1 步，均值 13.75 ms、最大 20.6 ms；
- 纯 `interact` 36 次触发 40 个 consolidate、114 个 ttt，队列峰值 1，**未复现被动饥饿**。

所以：「被动侧被饿死」这个具体症状，在小负载下已经消失。**这不等于修复没有代价**——代价在下一节。

### 2.4 修复引入或暴露的问题

| 问题 | 证据 | 说明 |
|---|---|---|
| `pending_consolidation` 用半球名做 key，切换后旧角色 future 双残留 | `CONC-03`，已实测：23/64 轮出现 `pending={right,left}` | `_drain` 只排当前 passive；切换前那一笔属于旧 passive 的 future 留到它下次成为 passive 或 close 才被 `result()`，其失败也被推迟暴露 |
| monitor 的周期 `save()` 不等待在途巩固 | `CONC-02` + `PERS-01`，两分区独立实测 | `snapshot` 优先级 2 高于 `consolidate` 4，快照先于巩固落盘；实测右半球 16 处不一致 |
| `interact` 路径只 schedule 不 drain | `CONC-05`，尚未验证 | 本负载未复现饥饿（ttt 0 优先仍会压 consolidate 4）；高频 ttt / 慢 GPU / 大批次下的情形未测量 |
| drain 在 agent RLock 内阻塞 | `CONC-07`，已实测 | 13.75 ms 均值把 monitor 与其他调用方串行化；外部 `status()` 观测到 40.6 ms 均值 / 166.6 ms 最大 |
| drain 之后不再补 schedule | `CONC-10`，已实测 | 当轮因上一笔未完成而 `return` 的巩固机会不会立即补排，靠 monitor 每秒兜底；样本不丢，但延迟 |
| monitor 的 `_maybe_switch` 实际不触发 | `CONC-08`，已实测 | 探针守卫 `interaction_count == last_switch_probe` 在正常流程下总是成立，monitor 4 次 tick 全部提前返回；「每秒巡检」并不提供独立的切换机会 |

---

## 3. 阻断项（进入全量训练前必须解决）

以下 7 条按「会让全量训练产出错误或不可证明的结果」重定级为**阻断**。子评审原始定级多为「高」，重定级理由逐条写明。

### B1｜切换基线的角色错位（`MATH-01`，真实缺陷，已实测证实）

**位置**：`life_agent/agent.py:314-318`、`agent.py:336-337`、`life_agent/switching.py:57-65`、`store.py:118-122`

`_switch_locked` 末尾：

```python
self.switch_controller.note_switch(
    clean_pressure,
    self.state("left").global_step,     # 位置参数 → active_work
    self.state("right").global_step,    # 位置参数 → passive_work
)
```

`note_switch(clean_pressure, active_work, passive_work)` 的签名是角色。只有 `active == "left"` 时左右恰好对上；`active == "right"` 时两个基线互换。随后 `decide()` 用 `self.state(active_name).global_step` 和 `self.state(passive_name).global_step` 减去这两个**错位**的基线。

**证据**：`_audit_tmp\math\sim_results.json` 的 C2 场景（右主动）：错误映射得 `active_delta=50`、`passive_delta=-10→0`（钳零），正确应为 `30 / 10`；`sufficient_work` 因此返回 false，理由被写成「passive side has not yet carried measurable work」。这不是数值噪声，是判据方向的翻转。

**影响**：一半的切换决策（active=right 期间）基于错误的基线；错位后的 `passive_delta` 可能为负被钳零 → 永久阻塞，或虚高 → 触发不该有的切换。61 次切换里有多少次落在这个区间，现有产物无法分解。

**修复**：`note_switch` 改为显式传 `self.state(self.active).global_step, self.state(self.passive).global_step`（或改成 `note_switch(clean_pressure, active_name, passive_name)` 内部取步数），并在 `to_dict()` 中同时存角色名，避免重载后再次歧义。

**回归测试**：`test_switch_baseline_follows_role_not_hemisphere_name`——构造 active=right、两侧步数已知（left 100 / right 130 → 切换、再让 left +5 / right +3），断言 `baseline_active_work=130`、`baseline_passive_work=100`、`active_delta=3`、`passive_delta=5`，且 `decide` 的 reason 不是「passive 未工作」。另加 `test_switch_baseline_roundtrip_right_active`：save→load 后 `active=="right"` 时基线仍按角色还原。

---

### B2｜工作判据把三类作业混为一个计数器（`MATH-02`，真实缺陷，已实测证实）

**位置**：`life_agent/worker.py:295`（唯一递增点）、`worker.py:417-424`（`_run` 分发）、`agent.py:336-337`、`switching.py:86`

`_run` 的 else 分支对 `ttt` / `train` / `consolidate` 统一调用 `_train(state, samples, steps, job.kind)`；`_train` 内第 295 行无条件 `state.global_step += 1`。因此：

- `active_delta` = 主动侧的 train + ttt + consolidate 步数；
- `passive_delta` = 被动侧的 consolidate + ttt + train 步数。

两边不是「同一类工作量」，而是「同一个计数器在不同角色下的增量」。`sufficient_work` 要求 `active_delta >= passive_delta`，当被动侧因 monitor 高频巩固而步数增长更快时，切换被永久抑制；反过来，若主动侧 ttt 密集，切换会被无理由放行。

**证据**：`_audit_tmp\math\sim_results.json` C1——被动 +5/轮、主动 +3/轮，12 轮全部被工作判据阻塞。另有代码行号核实：`worker.py:295` 无 kind 分支，`worker.py:421` 是三类共用入口。

**影响**：这是「用工作量加权」这一设计意图的实现错误。它不会被 `automatic_switching>=8` 发现，因为阻塞只会减少切换次数，不会报错。

**修复**：在 `HemisphereState` 增加分 kind 计数（`train_steps` / `ttt_steps` / `consolidate_steps`），`sufficient_work` 改为「主动侧学习步数 ≥ 被动侧巩固步数」或按每次作业的样本数加权；`global_step` 保留为总步数用于日志与学习率控制器，但**不再直接充当 work 判据**。

**回归测试**：`test_work_counter_separates_train_and_consolidate`——同样总步数下，构造 `ttt=10/train=2/consolidate=8` 与 `ttt=0/train=8/consolidate=2` 两种分布，断言 `decide()` 的 `work_share` 不同且不出现「被动侧已承担更多工作」的误判；再断言 `global_step` 仍等于三者之和，避免破坏现有检查点字段。

---

### B3｜恢复一致性只比了单层权重，而且是磁盘到磁盘（`PERS-02`，验证盲区，已实测证实）

**位置**：`continuous_validation.py:164-205`（尤其 `180-186`）、`life_agent/store.py:77-97`

`restart_records` 的字段只有 `loaded / active_match / switches_match / steps_match / weights_match`，其中 `weights_match` 比对的是 **`blocks.0.fc1.weight` 这一层**，且比对方式是「save 到磁盘 → 从同一个文件 load 回来 → 比较」，即自己和自己比。

**未比对**：debt 台账（items / total_clean_added / repayment_capacity / repayment_batches）、replay 内容与 cleaner 的 EWMA、optimizer state、lr 控制器、structure 控制器、switch 控制器的三个基线、history、`last_decision`、RNG。

**影响**：`restarts_exact` 判 true 只证明「文件能读回、active 与计数对、第一层权重没坏」。它不能证明其余 90% 的状态被正确还原。这与 `PERS-03`（缺字段静默默认）叠加后，任何还原错误都不可见。

**修复**：定义一个 `state_fingerprint()`，对上述每一项取摘要（数值用四舍五入后的元组、集合用排序后的 digest 列表），在 save 前与 load 后各算一次并逐字段 diff；报告里输出每个字段的 match 与差异摘要。至少覆盖 debt 五个计数器、controller 三个基线、structure 的 interval/warmup 计数、optimizer 的 `step` 与 `exp_avg` 摘要。

**回归测试**：`test_checkpoint_roundtrip_full_fingerprint`——训练 20 步、save、load，断言指纹逐字段相等；再故意把检查点中的 `baseline_active_work` 改成 0，断言测试失败（证明比对不是空转）。

---

### B4｜周期落盘不等在途巩固，且 `due()` 先记账后训练（`PERS-01` + `CONC-02` + `PERS-05`，真实缺陷）

**位置**：`agent.py:348-357`（monitor 调 `save()`）、`agent.py:359-365`（`save` 只拿锁）、`agent.py:269-280`（`close` 才 wait）、`worker.py:208`（snapshot 优先级 2）、`memory.py:206-219`（`due()` 内累计 `total_repaid` / `repayment_capacity`）、`agent.py:240-247`

两条独立的缺陷叠加：

1. **快照边界**：monitor 每 300 s 调 `save()`，`save()` 不调 `_wait_consolidations()`；`snapshot` 的优先级 2 高于 `consolidate` 的 4，所以快照可以插到排队中的巩固之前。实测（`PERS-01` run1）右半球 16 处不一致：步数 7→8、`total_trained_bytes` 440→508、cleaner count 12→14、4 条 debt 样本不同、`pending` 从 `['right']` 变为 `[]`。`CONC-02` 在另一个分区独立复现：两次 homeostasis save 时 `pending=["left"]`，left 巩固未完成即快照（L step32/debt26、L step74/debt25）。
2. **记账时点**：`due(limit)` 一被调用就 `total_repaid += len(selected)`、`repayment_batches += 1`、更新 `repayment_capacity`，但真正训练发生在之后的 `_train` 里。若快照落在两者之间，检查点里的「偿还历史」已经记了一笔尚未发生的训练；恢复后既可能重复记账，也可能让 `repayment_capacity` 所代表的阈值偏离实际能力。

**影响**：全量训练是 40 小时量级（见 §7.2），周期落盘会执行数百次，这类边界不一致是**必然出现**而非偶发。它不会立刻报错，只会让恢复后的债务/阈值与训练历史悄悄分叉。

**修复**：`save()` 内先 `_wait_consolidations()` 再快照（或把 `snapshot` 的优先级提到 1 以下并与 `_wait_consolidations` 共用同一把 state 锁）；`due()` 改为「预选」与「记账」两段，记账移到 `_train` 成功返回之后（或用一个 `pending_repayment` 临时字段，在 consolidate 的 `set_result` 前提交）。

**回归测试**：`test_save_waits_for_inflight_consolidation`——提交一个 `wait=False` 的 consolidate 后立刻 `save()`，断言 `pending_consolidation` 为空且检查点里的 debt 权重等于训练后的值；`test_due_does_not_count_before_training`——mock `_train` 抛异常，断言 `total_repaid` 未增加。

---

### B5｜验收器可假阳：skip-real 短路、计数型切换判据、决策样本缺失（`PERS-06` + `PERS-07` + `PERS-04`，验证盲区，已实测证实）

**位置**：`continuous_validation.py:503-506`、`:473-478`、`:492`、`:535`、`:507`、`:472`；`switching.py:115-120`；`store.py:111-113`

四条已在源码中确认的判据实现：

```python
"real_dataset_volume": args.skip_real or real["payload_bytes"] >= min(args.real_bytes, 20_000_000),
"real_dataset_resume": real["resume_verified"],          # skip 时初值为 True
"source_files_unchanged": real["source_unchanged"],      # skip 时初值为 True
"automatic_switching": status["switches"] >= 8,
"adaptive_switch_without_fixed_countdown": (
    not hasattr(config.runtime, "switch_interactions")
    and not hasattr(config.runtime, "switch_interval_s")
    and status["switches"] >= 8
),
"duplicate_cleaning": cleaning["increase"] <= 5 and minimum_feed_weight < 0.5,
"queue_drained": status["queued_jobs"] == 0,
"both_hemispheres_trained": left_steps >= 20 and right_steps >= 20,
```

后果：

- `--skip-real` 时，`real_dataset_volume` / `real_dataset_resume` / `source_files_unchanged` **无条件 true**。`continuous-adaptive-1/2` 的 `payload_bytes=0 / frames=0` 仍三项通过（`PERS-06` 已实测）。
- `adaptive_switch_without_fixed_countdown` 的全部证据是「配置没有旧键 + 切换 ≥8 次」。它把「删掉固定阈值」当成了「已经自适应」，而 `report.json` 的 `adaptive_switch` 是 `null`——**没有任何一条 `pressure_growth / learned_capacity / work_share` 落盘**（`PERS-04`：`last_decision` 不持久化，`metrics.jsonl` 205 行中决策字段 0 命中）。
- `duplicate_cleaning` 允许 `unique_after - unique_before <= 5`，即去重后唯一数**可以增加 5 条**。full-7 实测 `429 → 430`（+1）仍通过（`PERS-09`）；该判据没有校验任何重复 digest 是否真的被合并。
- `both_hemispheres_trained` 的门槛是 20 步。历史上曾出现 10,213 / 955，**955 也会通过**。也就是说，这条判据对本次修复所针对的饥饿问题完全不敏感。
- `queue_drained` 只在终点取 `queued_jobs==0`；`train_loss_finite` 只取终点 `last_loss`（非有限批次会被 `continue` 跳过，旧值残留）；`no_runtime_errors` 只看 `run.errors`，后台 future 的异常只进 `LOG`。

**影响**：`all_measurable_criteria_passed: true` 的证明力被这四条判据显著稀释。它证明「进程跑完、没有致命异常、计数达到阈值」，不证明「切换是压力驱动的」「去重生效了」「恢复正确」「内存有界」。

**修复**：
- skip-real 时把三项记为 `"skipped"` 而不是 `true`，并在顶层区分 `passed / skipped / failed`。
- 每次切换把 `SwitchDecision.to_dict()` 追加到 `metrics.jsonl`（含 `active/passive` 角色名、时间戳、两侧 `train/ttt/consolidate` 分 kind 步数），report 里输出「前 5 次 + 后 5 次」决策样本与间隔分布。
- `automatic_switching` 改为「至少 N 次切换的 `reason` 是 `new clean data exceeded learned repayment capacity`，且这些决策的 `pressure_growth >= learned_capacity` 复算一致」。
- `duplicate_cleaning` 改为「注入 k 条已知重复 digest，断言 `unique` 只增加 1 且 `total_repayment_batches` 对应的权重被合并」。
- `both_hemispheres_trained` 改为「任一侧任意 20 轮窗口内的步数增量 ≥ 另一侧的 30%」，直接覆盖饥饿。

**回归测试**：`test_skip_real_marks_skipped_not_passed`、`test_switch_decisions_are_logged_and_recomputable`、`test_duplicate_injection_merges_digest`、`test_hemisphere_starvation_window`。

---

### B6｜20 MB 真实数据只覆盖 8 个文件、0.184% 字节（`READY-01` + `READY-05`，验证盲区，已实测证实）

**位置**：`validation_runs/continuous-adaptive-full-7/dataset_progress.json:5-53`、`dataset_inventory.json:14-87`、`continuous_validation.py:414-428`

只读遍历 `O:\数据集`：**760 文件 / 10,913,873,703 B（10.164 GiB）**，格式分布 .jsonl 41（5.706 GiB）、.parquet 227（1.663 GiB）、.json 238（1.336 GiB）、.csv 136（0.746 GiB）、.arrow 3（0.618 GiB）、.txt 17、.zip 3、图片/PDF/ipynb 少量。

已跑的 20 MB 只触碰 `01_LogiQA` 下 **8 个文件**（.parquet 1 / .txt 4 / .md 2 / .jsonl 1），合计 20.12 MB = **全量字节的 0.184%、文件数的 1.05%**。Parquet / Arrow / CSV / JSON / 图片 / 压缩包**一个都没喂到**。

**吞吐也无法拆分**：275.7 s 是「200 次混合交互 + 20 MB 灌入 + 4 次重启」的合计，产物里没有分阶段计时。按混合速率 72.5 KB/s 外推全量约 **41.8 小时**，但这是外推，不是测量；README 提到的 79.3 KB/s（约 38.2 h）的段落在当前目录找不到产物，**无法复核**（`READY-05`）。

**影响**：全量训练的第一小时就会遇到从未验证过的格式与尺寸分布。单帧 1 MiB 上限、509 token 分块、Parquet 的列式解码、zip 的压缩比，全都没有真实样本支撑。

**修复**：先做一次**跨格式有界实测**：按扩展名与大小分层抽 2–4 GB（覆盖每种格式至少 100 帧，含 >1 MiB 与 ≤0 字节边界），记录每格式的 frames / bytes / 耗时 / 峰值 RSS / 峰值显存 / 失败率；把「每 GB 耗时」做成可复算的表，再决定全量预算。

**回归测试**：`test_dataset_format_coverage_matrix`——对每种扩展名断言至少 1 帧成功、记录 payload 字节与 chunk 数；`test_max_frame_boundary`——1 MiB / 1 MiB+1 / 0 B / 空 payload 四种输入的 accept/reject 行为固定。

---

### B7｜验证路径的进度提交顺序会丢帧（`READY-03`，真实缺陷，代码推断）

**位置**：`continuous_validation.py:418-428`、`ingest_dataset.py:250-264`、`ingest_dataset.py:268-270`

验证路径在 `feed_frame` 之后**先** `progress.mark(record, next_offset)`，**每 4 帧才** `agent.save()`。若进程在第 1–3 帧之间崩溃，进度已前进但模型没保存 → 这 1–3 帧在恢复后被跳过，**永久丢失**。`ingest_dataset.py` 的顺序相反（先存检查点、后提交偏移），两条路径的 durability 语义不一致。

**影响**：全量训练按 41.8 h 估算，进程崩溃/断电的概率不可忽略；丢帧不会报错，只会让最终模型少学一段数据，且无法从进度文件看出少了哪一段。

**修复**：统一为「先 save 检查点（含 offset），再原子提交 progress」（临时文件 + `os.replace`）；progress 里记录最后一帧的 `sha256(payload)`，恢复时校验该帧确实已训练。

**回归测试**：`test_progress_offset_only_advances_after_checkpoint`——在第 2 帧后模拟崩溃（抛出并重建 agent），断言恢复后 offset 等于第 1 帧结束位置，第 2 帧被重放且不重复计入 `total_trained_bytes`。

---

## 4. 真实缺陷（非阻断，按影响排序）

| ID | 严重度 | 证据 | 位置 | 问题 |
|---|---|---|---|---|
| `MATH-03` | 中 | 已实测 | `memory.py:206-218`、`agent.py:243-244` | `repayment_capacity` 是 top-weight 选样的累计算术平均，阈值系统性偏高：实测检查点左 1.1703 / oracle 0.4444 = **2.633x**，右 1.2326 / 0.7961 = **1.548x**；稀疏段批次缩小时平均值不回退（滞后比 1.424）。切换因此偏晚，而非偏早 |
| `MATH-04` | 中 | 已实测 | `memory.py:206-219`、`agent.py:240-247`、`agent.py:301` | `due()` 只克隆不移除，连续两次调用（`_schedule` 与 `_switch_locked` 各一次）把同一 top-weight 视图重复计为两个偿还批次：实测 393 批 / 3,108 次抽取，均批 7.91 ≈ limit 8，8/8 digest 重叠 |
| `MATH-05` | 中 | 代码推断 + 实测上界 | `agent.py:282-290`、`memory.py:192-201`、`agent.py:311-318` | `_exchange_switch_debt` 把两侧 `replay.latest(2)` 克隆进对方 debt，重复 digest 叠加权重并再次累加 `total_clean_added`；基线在注入**之后**取值（300 行先于 313 行），当次基线偏高。仿真上界 exchange 占压力 33.1%；按条数实测上界 ≤244/40,124 = **0.608%**，质量占比无法事后分解 |
| `PERS-05` | 高→已并入 B4 | 代码推断 | `worker.py:208/422-424`、`memory.py:206-219` | snapshot(2) 可先于 train(3)/consolidate(4)；`due()` 预选即记账 → 检查点里出现「已记账、未训练」 |
| `CONC-03` | 中 | 已实测 | `agent.py:56/249-262/265` | pending 以半球名做 key，切换后双残留 23/64 轮；旧角色 future 的失败被推迟到它下次成为 passive 或 close |
| `CONC-04` | 中 | 代码推断 | `agent.py:255-256/269-280/351-357` | monitor 路径的巩固失败只进日志，不进 `run.errors`；`_wait_consolidations` 只保留首个异常，后续丢失 |
| `SYNC-02` | 中 | 代码推断 | `worker.py:221-225/232-234/336-339` | resize 与每次 sync 都重建 AdamW → 动量清零。会话内每次切换都伴随一次动量重置，只有重载才恢复保存的动量。这与「切换是为了让另一侧接管学习」的意图冲突：接管方每次接手都从零动量开始 |
| `NOV-02` | 中 | 已实测 | `memory.py:60-79/27-31` | `structural_quality` 与序列长度、上下文包裹耦合：同一内容 32/64/128 长度 → 1.0/0.78125/0.390625；换前后缀 digest 不同且质量均 1.0；单调可预测序列 1.0 |
| `NOV-03` | 中 | 已实测 | `memory.py:197-198/206-219/180-186` | 债务对同 digest 叠加权重并 cap 4.0，`due()` 按原始权重降序偿还：重复 3 次的零债（0.5273 / quality 0.001）排在**新债**（0.1758 / quality 1.0）之前。方向与「重复应缓学」相反 |
| `READY-04` | 中 | 代码推断 | `ingest_dataset.py:43-49/158-160` | 断点只比对 `size + mtime_ns`，无内容校验和；源文件被 touch 或时间戳变动即整文件回零重读 |
| `READY-07` | 中 | 已实测 | `agent.py:119-125`、`byte_codec.py:132-137` | `max_frame_bytes=1 MiB` 是拒绝式上限；`ingest_dataset.py:169-183` 在 960 KiB 处硬切；`feed_file` 会整文件读入内存 |
| `READY-08` | 中 | 已实测 | `byte_codec.py:252-260` | `observation_chunks` 按 509 token 切分，块首每块重置 BOS/USER，跨块前驱上下文缺失 |
| `READY-23` | 低 | 已实测 | `dataset_inventory.json:14-87` | 清单与当前目录有少量不一致，全量前须重新生成并冻结 |
| `PERS-12` | 低 | 代码推断 | `agent.py:53-55`、`store.py:30-42` | Python RNG 与 torch generator 状态不持久化，重启后采样序列不可复现 |
| `PERS-13` | 低 | 代码推断 | `worker.py:359-374` | 快照中 optimizer `state_dict` 在 `state.lock` 之外读取，与权重/账本非同锁 |
| `CONC-01` | 高 | 代码推断 | `agent.py:256/267/273`、`worker.py:219` | 所有 `future.result()` 无超时；worker 崩溃或作业挂起时 `close()` 先 wait 后 stop，会永久阻塞。健康实验 122 个 future 全 resolve，故障路径未注入 |
| `CONC-06` | 低 | 代码推断 | `worker.py:206-219/432-441` | `submit` 的 stop 检查与 `jobs.put` 之间有竞态，停止后入队的 future 永久 pending |
| `CONC-05` | 中 | 尚未验证 | `agent.py:182-197` | interact 路径不 drain；本负载未复现，高频 ttt / 慢 GPU / 大批次未测 |

---

## 5. 验证盲区：21/21 不能当作证明

### 5.1 三条结构性假阳性模式

1. **skip 短路**：`--skip-real` 让真实数据三项恒 true。这不是「测过但放宽」，是**没测也记通过**。
2. **计数替代机制**：`automatic_switching` 与 `adaptive_switch_without_fixed_countdown` 都只看 `switches >= 8`；后者额外只看「旧键不存在」。任何能产生 8 次切换的路径（包括结构事件、monitor、手工调用 `switch()`）都能满足。
3. **终点瞬时采样**：`queue_drained`、`train_loss_finite`、`debt_bounded`、`replay_bounded`、`both_hemispheres_trained`、`no_runtime_errors` 都只取最后一次 `status()`。运行中途的积压、NaN、超容、饥饿全部不可见。

### 5.2 逐条判据的真实证据来源（关键项）

> `report.json` 共 21 条判据；下表 20 行，其中 `debt_bounded` 与 `replay_bounded` 合并为一行，两者实现与风险相同。

| 判据 | 真实证据 | 假阳性风险 |
|---|---|---|
| `cpu_inference_gpu_training` | 只读 `training_device=="cuda"` + 一个 CPU 参数在 CPU（`:469-470`） | 中：设备标签，未证明推理未走 GPU |
| `mixed_cycles_completed` | 取 CLI 参数 `mixed_cycles>=200`（`:471`） | 中：自报参数，非完成度测量 |
| `both_hemispheres_trained` | 终点两侧步数 ≥20（`:472`） | 中：955 步也会通过，对饥饿不敏感 |
| `automatic_switching` | `switches>=8`（`:473`） | **高**：纯计数 |
| `adaptive_switch_without_fixed_countdown` | 无旧键属性 + `switches>=8`（`:474-478`） | **高**：把删键当机制 |
| `ttt_weight_updates` | 前 40 轮嵌入向量 L1>0（`:479`、`:329-339`） | 中：混入 sync 插值；后 160 轮无探针；无禁用 TTT 对照 |
| `passive_cpu_isolation` | 循环内 hook 违例即抛（`:297-313`），判据字段硬编码 True（`:480`） | 低-中：落盘判据未关联运行期记录 |
| `train_loss_finite` | 终点两侧 `last_loss` 有限（`:481-483`） | 中：非有限批次被 `continue`，旧值残留 |
| `debt_bounded` / `replay_bounded` | 终点长度 ≤ capacity（`:484-491`） | 低-中：瞬时超容后驱逐不可见 |
| `duplicate_cleaning` | `increase<=5` 且最小喂入权重 <0.5（`:492`） | **高**：429→430 反增仍通过；不校验 digest 合并 |
| `automatic_structure_events` | 独立模型上合成 growth/prune（`:217-259`） | 中：验证控制器逻辑，非在线真实决策 |
| `restarts_exact` | ≥4 条记录且 5 个字段全 true（`:494-502`） | **高**：单层权重 + 磁盘到磁盘 |
| `real_dataset_volume` | skip 短路，否则 payload≥阈值（`:503-504`） | **高**：skip 时零字节也通过 |
| `real_dataset_resume` | skip 时硬编码 true（`:505`、`:455`） | **高** |
| `source_files_unchanged` | 默认 true，仅非 skip 时比对 size/mtime（`:506`、`:429-433`） | **高** |
| `queue_drained` | 终点 `queued_jobs==0`（`:507`） | 中：瞬时 |
| `learning_rate_in_bounds` | 全程 lr 采样集合（`:508-512`） | 低：基于全程集合 |
| `host_memory_bounded` | `final-baseline < 2048 MB`（`:513-517`） | 中：固定绝对阈值；Windows RSS 是工作集（含共享页），非私有字节 |
| `gpu_memory_bounded` | `final ≤ baseline+512 MB`（`:518`） | 中：固定余量，读 allocated 非 reserved |
| `no_runtime_errors` | `len(run.errors)==0`（`:519`） | 中-高：后台 future 异常只进 LOG |

### 5.3 缺失的关键证据（建议随报告一起落盘）

- 每次切换的 `SwitchDecision`（角色、pressure_growth、capacity、两侧分 kind 步数、时间戳）；
- 真实数据阶段的分阶段计时（frames / bytes / seconds 分开）；
- 全部字段的 save/load 指纹 diff；
- 每 30 s 的 RSS / 显存采样曲线，而不是首尾两点；
- 20 轮滑动窗口内的两侧步数增量（饥饿指标）。

---

## 6. 设计边界（不是缺陷，但也不是「已自适应」）

### 6.1 仍未自适应的常量：三分类

| 类别 | 参数 | 依据 |
|---|---|---|
| ① 资源安全上限（**应保持常量**） | `max_seq_len 512`、`max_frame_bytes 1 MiB`、`batch_size 4`、`consolidation_steps 2`、`max_hidden_dim 512`、`min_hidden_dim 128`、`replay_capacity 1024`、`debt_capacity 512` | 直接对应 3 GB 显存、CPU 在线推理与内存上限；自适应化会把「安全」变成「概率」 |
| ② 工程节流（**可半自适应，带硬上限**） | `checkpoint_interval_s 300`、`idle_sleep_s 0.1`、`feed_steps 1`、`structure_interval 32`、`structure_warmup 64`、`debt_decay 0.75`、`debt_retention 0.35` | 应由「每次落盘字节数 / 每秒作业数」等测量值驱动，但必须保留硬上限与最小间隔 |
| ③ 适合在线学习（**护栏内自适应**） | `learning_rate`（已在 1e-5..3e-3 内自适应）、`growth_size 16` / `prune_size 8`（可由 loss 曲率与参数利用率驱动）、`debt_min_weight 0.005`、`sync_mix 0.02`、`replay_probability 0.35` | 这些量的最优值随数据分布变化；但每一个都需要独立的稳定性判据与回滚阈值 |

### 6.2 结构生长/剪枝

`StructureController.consider` 由「loss 阈值 + 固定周期」双门控（`worker.py:83-107`）；`growth 16 / prune 8 / interval 32 / warmup 64` 全部是常量。full-7 里生长到 416/400、剪枝一次（`272 → 248`），但**没有验证「生长后立刻剪枝」的震荡**——200 轮太短，且剪枝事件只发生一次（`READY-11`）。

另一个耦合：`_train` 在**任何 kind**（包括 `ttt` 与 `consolidate`）之后都可能触发 `structure.consider` → `resize` → `_optimizer_reset`（`worker.py:310-312`、`221-225`）。也就是说，一次被动巩固也可能清空主动侧的 AdamW 动量。这与 `SYNC-02` 叠加，意味着「切换」与「结构事件」都会让动量归零，而验收里没有任何一条判据观察动量。

### 6.3 双半球同步

`_synchronize`（`worker.py:333-346`）先逐层对齐宽度再按模块顺序 `zip` 插值，`sync_mix=0.02`。实测（`SYNC-01`）在最小模型上未触发形状错误，但代码**没有显式校验名称/形状**，`zip` 的截断语义会在宽度不一致时静默少同步一部分参数；buffer（如 LayerNorm running stats，若未来引入）不参与同步。这属于「当前配置下安全、结构演化后不安全」的设计边界。

### 6.4 「一切皆自适应」的实际边界

当前**真正在线自适应**的只有三处：学习率（loss 比例升降）、结构宽度（loss 阈值 + 周期）、半球切换（压力 + 容量 + 步数）。其余全部是常量或半自适应。把这三处说成「一切皆自适应」是口径放大；`READY-10/13` 给出了分阶段清单。

---

## 7. 全量训练准备度

### 7.1 覆盖度

见 B6：760 文件 / 10.164 GiB，20 MB 只覆盖 8 个文件、0.184% 字节、1.05% 文件数，且只落在 `01_LogiQA`。**格式覆盖矩阵目前是空的**：Parquet 227 个文件（1.663 GiB）、Arrow 3 个（0.618 GiB）、CSV 136 个（0.746 GiB）、JSON 238 个（1.336 GiB）、zip 3 个，均在真实数据上未验证。

`1% 路径哈希保留集`（`READY-02`）是按**相对路径**哈希划分，不是内容去重，也不跨格式识别重复。它能保证「每次抽样的文件集合稳定」，不能保证「抽到的内容代表分布」。这与「非噪声数据密度」的说法是两件事。

### 7.2 吞吐与资源（全部为外推，不是测量）

- 275.7 s 包含 200 次混合交互 + 20 MB 灌入 + 4 次重启，**无法拆出纯真实数据耗时**（`READY-05`）。按混合速率 72.5 KB/s 外推 10.164 GiB ≈ **41.8 h**；不确定性来自三部分：混合交互占比未知、格式解码成本未知（Parquet/Arrow 与 txt 差一个量级）、GPU 队列等待未知。
- RSS 实测 422 → 1,226 MB（2.9 倍）。没有中间采样，**无法判断是趋于平稳还是线性增长**。
- 检查点 `latest.pt` 18.1 MB + `latest.backup.pt` 18.1 MB，每 300 s 覆盖写一次。按 41.8 h 算约 500 次落盘，每次 36 MB 写入 → 约 18 GB 累计写盘（覆盖写，不占额外空间，但 IO 压力真实存在）。
- `metrics.jsonl` 在 200 次交互后 47 KB；按线性外推全量可能到数十 MB 量级，**无轮转、无上限**。
- 没有磁盘配额检查、没有日志轮转、没有检查点代际保留（只有 latest + backup）。

### 7.3 全量前的前置条件清单

| # | 前置条件 | 现状 | 对应发现 |
|---|---|---|---|
| 1 | 跨格式分层抽样（每格式 ≥100 帧，含 >1 MiB 与 0 B 边界） | 未做 | `READY-01/07` |
| 2 | 分阶段计时（frames / bytes / seconds 分开落盘） | 未做 | `READY-05` |
| 3 | 每 GB 耗时与峰值资源的可复算表 | 未做 | `READY-05/06` |
| 4 | 进度文件原子写 + 内容校验和 + 「先检查点后偏移」 | 两条路径语义相反 | `READY-03/04` |
| 5 | 检查点/日志轮转与磁盘配额告警 | 未做 | `READY-06` |
| 6 | 清单冻结（重新生成 inventory 并记录 hash） | 清单有小幅漂移 | `READY-23` |
| 7 | 长程 RSS/显存采样曲线（≥1 h 空跑 + 1 h 真实数据） | 未做 | `READY-06` |
| 8 | 崩溃恢复演练（kill -9 后核对 offset 与已训练字节） | 未做 | `READY-03` |
| 9 | 切换决策日志（含角色与分 kind 步数） | 未做 | `PERS-04/07` |
| 10 | 全字段 save/load 指纹比对 | 未做 | `PERS-02` |

**结论**：不具备直接启动全量训练的条件。最小可行路径是先做 #1、#2、#4、#7 四项，拿到「每 GB 耗时 + 峰值资源 + 崩溃可恢复」三条实测数据，再讨论全量。

---

## 8. 测试覆盖

### 8.1 真实用例数（重新数，不引用历史数字）

`pytest` 在当前环境**未安装**（`.venv-life` 与系统 Python 均 `ModuleNotFoundError`；`.venv` 基础路径已损坏）。改用标准库：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& "O:\Documents\智能自学习AI（新）\.venv-life\Scripts\python.exe" -m unittest discover -s "O:\Documents\智能自学习AI（新）\tests" -v
```

结果：**collected 38 / passed 38 / failed 0 / skipped 0（4.218 s）**，两个 CUDA 用例因 CUDA 可用实际执行。

逐文件：`test_byte_codec` 5、`test_byte_frame` 8、`test_controllers` 3、`test_dataset_ingest` 2、`test_integration` 12、`test_memory` 5、`test_model` 3。

> 注意：README/VALIDATION 中的「38 项」与实际一致，但它不是 pytest 用例，而是 unittest。CI 若依赖 `pytest -q` 会直接失败（`READY-14`）。

### 8.2 缺失的关键场景（10 类）

| # | 场景 | 现有覆盖 | 风险 |
|---|---|---|---|
| 1 | 持续 feed 下 drain 的锁持有时间 | 无 | RLock 内阻塞，吞吐随批次线性恶化（`CONC-07`） |
| 2 | monitor 与 feed 的 `_maybe_switch` 竞争 | 无 | monitor 探针守卫总是提前返回，兜底失效（`CONC-08`） |
| 3 | save 期间存在 pending consolidate | 无（只测 close 路径） | 检查点边界不一致（B4） |
| 4 | 切换时 debt exchange 的重复训练 | 无 | 同一批样本在两侧各训一遍，压力重复计（`MATH-05`） |
| 5 | 重载后立即触发切换 | 无 | 旧检查点缺基线时行为未定（`PERS-03`） |
| 6 | 空债务 / 零容量 / 空样本巩固 | 无 | `required>0` 短路，静默不切换（`MATH-07` 判定为安全但未测） |
| 7 | resize 与 sync 并发/交错 | 无 | `zip` 静默截断、动量重置（`SYNC-01/02`） |
| 8 | pending future 异常传播（schedule/drain） | 仅 close 路径 | 失败被日志吞掉（`CONC-04`） |
| 9 | 恢复后全字段指纹 | 无 | 只比单层权重（B3） |
| 10 | skip-real 时的判据语义 | 无 | 零字节记通过（B5） |

### 8.3 新增测试清单（名字 + 断言）

1. `test_switch_baseline_follows_role_not_hemisphere_name`：active=right 时基线按角色对齐；`active_delta=3 / passive_delta=5` 不被误判。
2. `test_work_counter_separates_train_and_consolidate`：分 kind 计数之和等于 `global_step`；两种分布下 `work_share` 不同。
3. `test_checkpoint_roundtrip_full_fingerprint`：债务/控制器/结构/optimizer 逐字段相等；篡改基线后测试失败。
4. `test_save_waits_for_inflight_consolidation`：save 后 `pending` 为空、debt 权重为训练后值。
5. `test_due_does_not_count_before_training`：训练抛异常时 `total_repaid` 不增。
6. `test_skip_real_marks_skipped_not_passed`：skip 时三项为 `"skipped"`。
7. `test_duplicate_injection_merges_digest`：注入 k 条重复，`unique` 只 +1。
8. `test_hemisphere_starvation_window`：20 轮窗口内单侧增量 < 另一侧 30% 时判失败。
9. `test_switch_decisions_are_logged_and_recomputable`：每条决策可由日志复算 `pressure_growth >= capacity`。
10. `test_meta_switch_stays_inactive_or_logs`：为了收窄 `_maybe_switch` 的歧义，统一「决策日志」后无需单独测竞争。
11. `test_progress_offset_only_advances_after_checkpoint`：崩溃重放不重复计字节。
12. `test_max_frame_boundary`：1 MiB / +1 / 0 B / 空 payload 的 accept-reject 固定。

---

## 9. 「厌恶重复」：情绪宣称 vs 可测机制

> **本节为设计边界审查，尚未实现。** 下文没有任何一条是现有能力；所有建议均标注「尚未实现/设计建议」。实测数字来自 `_audit_tmp\novelty\probe_results.json` 的纯 Python 探测，**不代表真实训练动力学**。

### 9.1 界线

「像人一样厌恶烦躁重复」是情绪化类比。代码里不存在情绪、动机、饱食度字段。当前可观测的只有三类机制：字节指纹精确去重（`memory.py:27-31`）、单帧分布离散度的结构乘权（`memory.py:60-79`）、高损失离群抑制（`memory.py:81-95`）。三者都不回答「同一内容重复出现时，边际改善是否趋零」。把重复降权说成「厌恶」或「习惯化」，属于口径错误。习惯化若成立，必须落成可记录、可对照、可回归的指标。

### 9.2 现有清洗器覆盖了什么（已实测）

`structural_quality` 只在单条序列内部统计字节熵、最长游程、unique 占比；`digest` 是 `ids+labels` 的 SHA256，去重只认**完全相同**的字节序列。

| 输入 | 质量 | 说明 |
|---|---|---|
| 静音 PCM / 纯色灰 | 0.001（地板） | 合法低熵 |
| 随机 167 取值（7.19 bit） | 1.0 | 高熵 |
| 单调 0..255（完全可预测） | 1.0 | **可预测却满分** |
| 16 字节周期 | 0.78125 | |
| 两值交替 | 0.0977 | |
| 长度 8 短序列 | 1.0 | `n<16` 直接放行 |

反例（同一内容、不同包装）：32/64/128 长度 → `1.0 / 0.78125 / 0.390625`，仅长度不同，质量漂移；换前后缀后 digest 不同、质量均 1.0；每帧微差的周期流 digest 全唯一，8 帧使干净压力累计到 **1.1167**；抖动表格 1.0。

债务侧：同 digest 重复 5 次，权重 `0.001 → 0.06978`（**不降反升**）；`due()` 按原始权重降序偿还，重复 3 次的零债（`0.5273 / quality 0.001`）排在**新债**（`0.1758 / quality 1.0`）之前 —— 方向与「重复应缓学」相反（`NOV-03`）。

**结论**：现有机制只处理「表面重复」，没有处理「重复导致信息收益下降」。

### 9.3 候选信号对比（代码推断，尚未实现）

| 信号 | 可得性 | 成本 | 主要偏差 | 建议用途 |
|---|---|---|---|---|
| (a) 滚动字节熵 | 帧内熵已有；跨帧滚动计数需新增有界统计 | CPU O(n)，零前后向 | 只测离散度，高熵噪声被判「新颖」 | 辅助记录，不单独决策 |
| (b) n-gram 新颖度 | 代码无此结构，需近期 n-gram 计数/衰减 | CPU 哈希 O(n)，内存有界 | 领域词表、短语、协议帧头合法重复被误判熟悉 | 跨帧重复检测首选，先只记录 |
| (c) 边际损失改善 | `stored.loss` 已有 0.9/0.1 EWMA（`worker.py:300-309`），复访可算前后差 | 零额外 forward/backward | 受 LR、batch 组成、权重缩放、步数共用污染；首访无前值 | 弱代理，需配对复访 + 平滑 |
| (d) 梯度信噪比 | 当前无梯度统计；需累积梯度矩或额外反向 | 逐样本额外显存/反向 | 受 grad_clip、AdamW 归一化、resize 干扰 | 3 GB 显存 + CPU 在线推理下不可行，暂缓 |
| (e) TTT 前后收益差 | 现有 `ttt_deltas` 是权重 L1 位移（`:331,339`），**不是 loss 收益**；真收益差需探针集前后各一次 CPU forward | 两次 CPU 前向 | 探针集偏移、样本量小（full-7 仅 40 条） | 离线对照指标，不进在线门控 |

零成本可记录的只有 (c) 与 (a)；其余要么需要新结构，要么需要额外前向，要么在 3 GB 显存下不可行。

### 9.4 作用边界：只能降权，不能硬拒绝

现状没有硬拒绝：`add()` 始终入库，权重地板 0.001，注释明示纯色、静音、稀疏格式不得硬拒。**若未来对低新颖度信号直接丢弃或停止学习**：

- 静音、纯色、表格、重复日志、协议帧头等合法低熵数据被永久排除；
- 在 `debt_capacity=512`、`replay_capacity=1024` 下，被拒样本永久缺席 → 干净压力少计 → 切换更难触发；
- 清洗器的 loss 均值/方差 EWMA 只在入库样本上更新，选择偏置把 2.5σ 阈值带偏。

**建议（尚未实现）**：信号只允许「降采样 / 降巩固优先级 / 触发探索 / 作为附加门控」，**不允许丢弃样本或停止学习**。

### 9.5 与当前「数据挤压驱动」切换的关系：不可混称

当前判据（`switching.py:81-87`）：`pressure_growth >= repayment_capacity` 且 `passive_delta>0`、`active_delta>=passive_delta`。压力回答「积压待学的干净数据是否超过被动侧一次可消化量」；习惯化回答「当前流的边际改善是否趋零」。**二者正交，不能混称「无聊驱动切换」。**

纯 Python 回放（合成步数，非真实动力学）：全 0 重复流 12 轮切换 3 次、间隔 `[1,2,5]`；随机流 4 次、间隔 `[1,2,3,4]`；重复流下容量 EWMA 随压力收缩、阈值自校准到近零仍会切换 —— 说明现有机制并不以「无聊」为驱动。

若用新颖度**替换**压力项：高熵噪声流因「没见过」即触发切换，丢失已学量与偿还能力证据，造成振荡式频繁切换。**建议（尚未实现）**：新颖度只做附加门控或探索触发器，决策记录同时输出压力、容量、步数差与新颖度，便于归因。

### 9.6 分阶段落地（均为尚未实现/设计建议）

| 阶段 | 做什么 | 可验证判据 |
|---|---|---|
| 一 只记录 | 记录帧内/滚动熵、n-gram 新颖度、复访边际 loss 差，与 LR、步数、结构事件并列落盘 | 同种子两次运行序列一致；额外 CPU 开销不阻塞在线推理 |
| 二 仅降权 | 用边际改善/新颖度调整回放抽签与巩固优先级，样本仍入库 | 低熵合法类入库率不下降；权重变化与边际改善相关，而非与帧内熵单因子相关 |
| 三 附加门控 | 在「压力 AND 工作量」之外接低收益门控或探索请求，不替换压力项 | 高熵噪声流不触发额外切换；重复流边际趋零时出现探索倾向 |

### 9.7 最小实验与回归测试

对照实验：同一初始权重、同一 token 预算、固定种子 7；A 高重复流（固定块 + 微差周期），B 高新颖流（固定种子随机/异质帧）。测量：采样权重分布、切换次数/间隔与 reason、每字节边际 loss 改善曲线、`clean_pressure` 与 `repayment_capacity` 走势、TTT 权重位移与探针集 loss 前后差。

**排除假象**：边际趋零必须在「LR 未下调、隐藏维度未变、整体 loss 基线未下移」时成立，否则只是学习率或结构变化造成的错觉。

回归测试（种子 7）：
1. `test_repeated_digest_accumulates_not_familiarity`：重复 5 次 replay `unique=1`、debt 权重按叠加增长，无独立熟悉度字段更新。
2. `test_structural_quality_length_context_baseline`：锁定 32/64/128 长度质量基线与短序列放行。
3. `test_low_entropy_legal_frames_not_hard_rejected`：静音/纯色/表格/日志全部入库且权重 ≥0.001。
4. `test_novelty_is_additional_gating_not_pressure_replacement`：高熵零收益流压力未达容量时，附加门控不切换。
5. `test_marginal_delta_uses_existing_ewma_no_extra_backward`：边际差来自已有 EWMA 前后差，不增加前后向调用。

---

## 10. 分阶段路线

| 阶段 | 目标 | 必须满足的门禁 | 不应做的事 |
|---|---|---|---|
| **阶段 0（现在）** | 修正正确性 | B1–B7 全部关闭；64 条中的「高」项有回归测试 | 不要为了压 61/200 去加固定间隔定时器 |
| **阶段 1** | 让机制可观测 | 切换决策样本、分 kind 步数、全字段指纹、分阶段计时、30 s 资源曲线、饥饿窗口指标全部落盘；skip 语义修正 | 不要把「计数 ≥8」当机制证明 |
| **阶段 2** | 半自适应 | 结构生长/剪枝由 loss 曲率 + 参数利用率驱动但保留硬上限；清洗下限由边际收益驱动；每次改动带稳定性判据与回滚阈值 | 不要让 `consolidate` 触发结构事件并清空动量 |
| **阶段 3** | 护栏内在线控制 | 阶段 2 的指标在 ≥3 次长跑中稳定；有回滚机制；有独立对照实验 | 不要把情绪化口号（「厌烦」「无聊」）写进机制命名 |

---

## 附录 A：全部发现索引（64 条）

严重度分布：高 15 / 中 38 / 低 11；性质：真实缺陷 19 / 验证盲区 16 / 设计边界 29；证据强度：已实测 28 / 代码推断 32 / 尚未验证 4。

| ID | 严重度 | 性质 | 证据强度 | 位置 | 标题 |
|---|---|---|---|---|---|
| `CONC-01` | 高 | 真实缺陷 | 代码推断 | life_agent/agent.py:256<br>life_agent/agent.py:267 | 所有 future.result() 无超时：worker 崩溃或任务挂起时调用方与 close 永久阻塞 |
| `MATH-01` | 高 | 真实缺陷 | 已实测证实 | life_agent/agent.py:314-318<br>life_agent/agent.py:336-337 | note_switch 工作基线按 left/right 硬编码传参，与实际 active/passive 角色错位 |
| `MATH-02` | 高 | 真实缺陷 | 已实测证实 | life_agent/worker.py:295<br>life_agent/worker.py:417-424 | ttt/train/consolidate 三类 job 共用同一 global_step，被动侧巩固步数被计入 passive_delta |
| `NOV-05` | 高 | 设计边界 | 代码推断 | life_agent/memory.py:111-124<br>life_agent/memory.py:81-95 | 若未来对低新颖度信号做「硬拒绝/完全停止学习」，存在合法低熵数据误伤路径与缓冲/统计双重分布偏置；当前代码本身无硬拒绝路径 |
| `PERS-01` | 高 | 真实缺陷 | 已实测证实 | life_agent/agent.py:359-365<br>life_agent/agent.py:348-357 | agent.save() 不等待在途 consolidation，monitor 自动落盘的检查点与调用方看到的内存状态处于不同作业边界 |
| `PERS-02` | 高 | 验证盲区 | 已实测证实 | continuous_validation.py:164-205<br>continuous_validation.py:180-186 | 重启一致性比对只覆盖 active/switch_count/global_step/单层权重，债务、replay、cleaner、optimizer、lr、structure、历史均未参与；且权重比对是磁盘-磁盘循环验证 |
| `PERS-05` | 高 | 真实缺陷 | 代码推断 | life_agent/worker.py:208<br>life_agent/memory.py:206-219 | 队列优先级使 snapshot(2) 可先于排队中的 train(3)/consolidate(4) 执行；而 due() 在预选阶段就累计偿还计数，快照先于训练时计数与权重状态不一致 |
| `PERS-06` | 高 | 验证盲区 | 已实测证实 | continuous_validation.py:503-506<br>continuous_validation.py:407-413 | --skip-real 时 real_dataset_volume / real_dataset_resume / source_files_unchanged 全部短路为 true，无真实数据证据也判通过 |
| `READY-01` | 高 | 验证盲区 | 已实测证实 | validation_runs/continuous-adaptive-full-7/dataset_progress.json:5-53<br>dataset_inventory.json:14-87 | 20 MB 真实样本仅覆盖 LogiQA 单目录 8 个文件，格式分布不代表全量 |
| `READY-03` | 高 | 真实缺陷 | 代码推断 | continuous_validation.py:418-428<br>ingest_dataset.py:250-264 | 验证路径先写进度偏移、后存检查点，崩溃窗口可丢帧；ingest 路径顺序相反，两条路径 durability 语义不一致 |
| `READY-05` | 高 | 验证盲区 | 尚未验证 | validation_runs/continuous-adaptive-full-7/report.json:27-33<br>validation_runs/continuous-adaptive-full-7/metrics.jsonl | 吞吐无法从持久化产物中拆分出纯真实数据段，全量时长外推不确定性大 |
| `READY-06` | 高 | 验证盲区 | 代码推断 | validation_runs/continuous-adaptive-full-7/report.json:329-333<br>life_agent/store.py:27-48 | 长程资源趋势未验证：RSS 422→1226 MB、检查点 18.1 MB×2、无轮转、无日志轮转、无磁盘配额检查 |
| `READY-12` | 高 | 验证盲区 | 已实测证实 | validation_runs/continuous-adaptive-full-7/report.json:34-71<br>life_agent/worker.py:35-56 | LR 上限 3e-3 从未被真实数据触碰；spike 降速分支原因未持久化，边界行为未验证 |
| `READY-15` | 高 | 设计边界 | 代码推断 | life_agent/agent.py:205-228<br>life_agent/agent.py:264-267 | 缺失测试：持续 feed 下 drain 被动巩固的锁持有时间 |
| `READY-22` | 高 | 设计边界 | 代码推断 | life_agent/agent.py:249-267<br>life_agent/agent.py:269-280 | 缺失测试：pending future 异常在 schedule/drain 路径上的传播（现有覆盖仅限 close 路径） |
| `CONC-02` | 中 | 真实缺陷 | 已实测证实 | life_agent/agent.py:354-355<br>life_agent/agent.py:359-365 | monitor 触发的 save() 未等待待完成巩固，snapshot 优先级高于 consolidate，检查点先于被动巩固落盘 |
| `CONC-03` | 中 | 真实缺陷 | 已实测证实 | life_agent/agent.py:56<br>life_agent/agent.py:249-262 | pending_consolidation 以半球名为 key，切换后旧角色 future 残留双存留，失败异常被推迟暴露 |
| `CONC-04` | 中 | 真实缺陷 | 代码推断 | life_agent/agent.py:255-256<br>life_agent/agent.py:269-280 | monitor 路径的背景巩固失败被日志吞掉；_wait_consolidations 只保留首个异常，后续失败被丢弃 |
| `CONC-05` | 中 | 设计边界 | 尚未验证 | life_agent/agent.py:182-197<br>life_agent/worker.py:208 | interact 路径只 schedule 不 drain，被动侧巩固依赖队列间隙与 1s monitor 重排，存在残余不对称饥饿风险 |
| `MATH-03` | 中 | 真实缺陷 | 已实测证实 | life_agent/memory.py:206-218<br>life_agent/agent.py:243-244 | repayment_capacity 为累计算术平均：top-weight 选样上偏，批次缩小时阈值不回退 |
| `MATH-04` | 中 | 真实缺陷 | 已实测证实 | life_agent/memory.py:206-219<br>life_agent/agent.py:240-247 | due() 不移除样本，重复调用把同一 top-weight 视图重复计为还款批次 |
| `MATH-05` | 中 | 真实缺陷 | 代码推断 | life_agent/agent.py:282-290<br>life_agent/memory.py:192-201 | exchange 双向 replay 入债：重复 digest 叠加权重并再次累计 clean mass；基线在注入之后取值 |
| `MATH-06` | 中 | 验证盲区 | 尚未验证 | validation_runs/continuous-adaptive-full-7/metrics.jsonl<br>continuous_validation.py:365-370 | 22/61 次切换在混合周期之后的灌入阶段，缺逐轮时间戳与压力日志，该阶段自激环无法证实或排除 |
| `NOV-01` | 中 | 设计边界 | 已实测证实 | life_agent/memory.py:60-79<br>life_agent/memory.py:27-31 | 现有「重复处理」只覆盖字节级精确去重与单帧内部低熵惩罚，不建模「重复导致边际信息收益下降」；不存在习惯化/饱食度信号 |
| `NOV-02` | 中 | 真实缺陷 | 已实测证实 | life_agent/memory.py:60-79<br>life_agent/memory.py:27-31 | 结构质量因子与序列长度、上下文包裹耦合：同一内容不同长度质量漂移；不同上下文包裹产生不同 digest 且质量均为 1.0；可预测序列被判为高质量 |
| `NOV-03` | 中 | 真实缺陷 | 已实测证实 | life_agent/memory.py:197-198<br>life_agent/memory.py:206-219 | 债务账本对同 digest 重复样本叠加权重，且 due() 按原始权重降序偿还；低质量重复债务反而被优先选中，与「重复应降权/缓学」方向相反 |
| `NOV-04` | 中 | 设计边界 | 代码推断 | life_agent/worker.py:300-310<br>continuous_validation.py:125 | 五个候选「习惯化/饱食度」信号的可得性、成本与偏差评估；梯度 SNR 在当前 3GB 显存 + CPU 在线推理下不可行；现有 ttt_deltas 是权重 L1 位移而非收益差 |
| `NOV-06` | 中 | 设计边界 | 已实测证实 | life_agent/switching.py:81-92<br>life_agent/agent.py:323-346 | 当前切换是数据挤压/偿还能力/步数工作量驱动，不含任何新颖度项；「压力驱动切换」与「无聊驱动切换」不可混称 |
| `NOV-07` | 中 | 验证盲区 | 代码推断 | life_agent/worker.py:300-309<br>continuous_validation.py:521-536 | 现有验证链路没有边际收益/新颖度指标：stored.loss 不进 report，决策样本不落盘，事后无法判断「习惯化」是否发生 |
| `PERS-03` | 中 | 真实缺陷 | 已实测证实 | life_agent/memory.py:247-256<br>life_agent/memory.py:157-163 | 旧/缺字段检查点静默默认：debt 计数器、replay cleaner、切换基线被归零或用当前值重建，无任何告警 |
| `PERS-04` | 中 | 设计边界 | 已实测证实 | life_agent/switching.py:115-120<br>life_agent/store.py:111-113 | 切换决策 last_decision 不持久化，重载后 report.adaptive_switch 为 null；历史切换决策样本（pressure_growth/capacity/work_share）不落盘、不入 metrics |
| `PERS-07` | 中 | 验证盲区 | 代码推断 | continuous_validation.py:473-478<br>life_agent/switching.py:76-113 | 自动/自适应切换判据只统计 switches 次数 ≥8，不校验决策依据；报告缺少切换决策样本，存在计数型假阳性空间 |
| `PERS-08` | 中 | 验证盲区 | 代码推断 | continuous_validation.py:80-83<br>continuous_validation.py:291 | TTT 权重探针只在前 40 轮采样，未分离 sync 插值对嵌入向量的贡献；探针能排除形状重建，但不能严格归因于 TTT 梯度更新 |
| `PERS-09` | 中 | 验证盲区 | 已实测证实 | continuous_validation.py:380-402<br>continuous_validation.py:492 | 清洗判据不校验重复 digest 是否真正合并，仅要求净增量 ≤5 且最小喂入权重 <0.5；unique_after 反超 unique_before 仍可通过 |
| `PERS-10` | 中 | 设计边界 | 代码推断 | continuous_validation.py:513-518<br>continuous_validation.py:51-77 | 内存判据使用固定绝对阈值，不随模型规模/运行时长缩放；Windows RSS 取工作集（含共享页），非私有工作集 |
| `PERS-11` | 中 | 验证盲区 | 代码推断 | continuous_validation.py:507<br>continuous_validation.py:481-483 | queue_drained/train_loss_finite/both_hemispheres_trained/no_runtime_errors 均为终点瞬时采样；后台 future 异常只进日志不进 run.errors |
| `READY-02` | 中 | 设计边界 | 代码推断 | ingest_dataset.py:92-95<br>ingest_dataset.py:119-126 | 1% 保留集按相对路径哈希划分，不是内容去重，也不跨格式识别重复 |
| `READY-04` | 中 | 真实缺陷 | 代码推断 | ingest_dataset.py:43-49<br>ingest_dataset.py:158-160 | 断点续训只比对 size 与 mtime_ns，无文件内容校验和；源文件变动即整文件回零重读 |
| `READY-07` | 中 | 设计边界 | 已实测证实 | life_agent/agent.py:119-125<br>life_agent/byte_codec.py:132-137 | max_frame_bytes=1 MiB 为拒绝式上限；大二进制在 960 KiB 处硬切，feed_file 路径会整文件读入内存 |
| `READY-08` | 中 | 设计边界 | 已实测证实 | life_agent/byte_codec.py:252-260<br>tests/test_byte_frame.py:66-75 | observation_chunks 按 509 token 切分，跨块前驱上下文缺失，块头每块重置 BOS/USER |
| `READY-09` | 中 | 验证盲区 | 代码推断 | ingest_dataset.py:197-283<br>life_agent/agent.py:359-373 | 全量前工程前置项尚未成套落实：格式探测、清单冻结、校验和、磁盘配额、检查点/日志轮转、失败重试、进度原子写校验、断点幂等 |
| `READY-10` | 中 | 设计边界 | 代码推断 | config.json:2-53<br>life_agent/config.py:34-88 | 仍为常量的控制量分级：①资源安全上限保持常量 / ②工程节流可半自适应 / ③适合在线学习（护栏内） |
| `READY-11` | 中 | 验证盲区 | 代码推断 | life_agent/worker.py:83-107<br>life_agent/model.py:61-110 | 结构生长/剪枝由损失阈值 + 固定周期双门控；同模型“生长后立刻剪枝”的震荡未在真实长跑中验证 |
| `READY-13` | 中 | 设计边界 | 代码推断 | life_agent/worker.py:27-114<br>life_agent/switching.py:67-113 | 自适应分阶段路线：阶段1 常量+可观测 → 阶段2 半自适应带硬上限 → 阶段3 在线控制，各阶段附可验证判据 |
| `READY-14` | 中 | 验证盲区 | 已实测证实 | tests/<br>.venv-life/Scripts/python.exe | 指定 pytest 命令在当前环境无法运行（pytest 未安装）；改用标准库 unittest 实测 38 个用例全部通过 |
| `READY-16` | 中 | 设计边界 | 代码推断 | life_agent/agent.py:323-346<br>life_agent/agent.py:348-357 | 缺失测试：monitor 线程与 feed/interact 的 _maybe_switch 竞争 |
| `READY-17` | 中 | 设计边界 | 代码推断 | life_agent/agent.py:359-365<br>life_agent/store.py:27-48 | 缺失测试：save 期间存在 pending consolidate 时的快照一致性 |
| `READY-18` | 中 | 设计边界 | 代码推断 | life_agent/agent.py:282-290<br>life_agent/memory.py:192-204 | 缺失测试：切换时 debt exchange 造成的重复训练与权重叠加 |
| `READY-19` | 中 | 设计边界 | 代码推断 | life_agent/store.py:111-122<br>life_agent/switching.py:57-65 | 缺失测试：重载检查点后立即触发自适应切换的基线恢复 |
| `READY-20` | 中 | 设计边界 | 代码推断 | life_agent/agent.py:240-247<br>life_agent/worker.py:258-260 | 缺失测试：空债务 / 零容量 / 空样本巩固路径 |
| `READY-21` | 中 | 设计边界 | 代码推断 | life_agent/worker.py:227-242<br>life_agent/worker.py:333-355 | 缺失测试：resize 结构变更与 sync 同步的并发/交错 |
| `SYNC-01` | 中 | 设计边界 | 已实测证实 | life_agent/worker.py:333-346<br>life_agent/model.py:150-152 | _synchronize 先逐层对齐宽度再按模块顺序 zip 插值；名称/形状未显式校验，buffer 不参与同步 |
| `SYNC-02` | 中 | 真实缺陷 | 代码推断 | life_agent/worker.py:221-225<br>life_agent/worker.py:232-234 | resize/sync 均重建 AdamW 清空动量；会话内每次切换伴随动量重置，重载才恢复保存动量 |
| `CONC-06` | 低 | 真实缺陷 | 代码推断 | life_agent/worker.py:206-219<br>life_agent/worker.py:432-441 | submit 的 stop_event 检查与 jobs.put 之间存在竞态，停止后入队的 job future 永久 pending |
| `CONC-07` | 低 | 设计边界 | 已实测证实 | life_agent/agent.py:205-228<br>life_agent/agent.py:264-267 | drain 在持有 agent RLock 期间阻塞等待被动巩固，monitor 调度与其他调用方在临界区内被串行化 |
| `CONC-08` | 低 | 验证盲区 | 已实测证实 | life_agent/agent.py:323-326<br>life_agent/agent.py:348-353 | monitor 的 _maybe_switch 探针守卫在正常流程下总是提前返回，monitor 侧每秒兜底切换评估实际不触发 |
| `CONC-09` | 低 | 设计边界 | 已实测证实 | life_agent/worker.py:258-327<br>life_agent/worker.py:333-355 | 单写者假设、CPU/GPU 更新窗口与快照拷贝复核：未见缺陷（安全项记录） |
| `CONC-10` | 低 | 设计边界 | 已实测证实 | life_agent/agent.py:227<br>life_agent/agent.py:249-262 | drain 完成当轮巩固后不补排剩余 due 样本；同半球 pending 在跑时新巩固延迟合并，样本不丢弃 |
| `MATH-07` | 低 | 设计边界 | 已实测证实 | life_agent/switching.py:76,85<br>life_agent/memory.py:256 | 空债务/零容量/基线高于当前压力/重载基线重建边界均安全；债务先于基线恢复 |
| `NOV-08` | 低 | 设计边界 | 尚未验证 | life_agent/worker.py:258-328<br>life_agent/switching.py:67-113 | 分阶段落地建议与最小对照实验设计（仅设计建议，尚未实现） |
| `PERS-12` | 低 | 设计边界 | 代码推断 | life_agent/agent.py:53-55<br>life_agent/store.py:30-42 | Python RNG 与 torch 生成器状态不持久化，重启后采样/采样序列不可复现 |
| `PERS-13` | 低 | 设计边界 | 代码推断 | life_agent/worker.py:359-374 | 快照中 optimizer state_dict 在 state.lock 之外读取，与权重/账本快照非同锁 |
| `READY-23` | 低 | 真实缺陷 | 已实测证实 | dataset_inventory.json:14-87<br>O:\数据集（只读遍历） | dataset_inventory.json 与当前目录存在小幅不一致，全量前应重新生成冻结清单 |
| `SYNC-03` | 低 | 设计边界 | 代码推断 | life_agent/agent.py:298-306<br>life_agent/worker.py:305-309 | 切换顺序 exchange→巩固新主动→同步：无样本丢失/乱序，但首批巩固混入刚注入的对侧样本 |

> 表中严重度沿用各分区原始定级。报告中 §3 的 B1–B7 是**针对「是否进入 10.164 GiB 全量训练」这一门禁**的重定级，不改变原表定级。

---

## 附录 B：证据与复现路径

所有实验均在 `O:\Documents\智能自学习AI（新）\_audit_tmp\` 下，源码只读，未改动 `life_agent/`、`tests/`、`validation_runs/`、`state_dataset_smoke/`、`O:\数据集`。

| 分区 | 目录 | 主要产物 | 关键命令 |
|---|---|---|---|
| 并发与队列 | `_audit_tmp\concurrency\` | `findings.json`、`NOTES.md`、`audit_metrics.json`、`audit2_metrics.json` | `& .venv-life\Scripts\python.exe _audit_tmp\concurrency\run_audit.py`；`run_audit2.py`（64 轮混合 / 24+36 轮） |
| 切换数学 | `_audit_tmp\math\` | `findings.json`、`sim_results.json`、`derived_ratios.json`、`metrics_analysis.json`、`checkpoint_inspection.json`、`sync_cpu_check.json` | 纯 CPU 仿真 + 检查点只读解析 |
| 持久化与验证器 | `_audit_tmp\persistence\` | `findings.json`、`audit_roundtrip.py`、`audit_results.json`、`run1..3\state` | `& .venv-life\Scripts\python.exe _audit_tmp\persistence\audit_roundtrip.py` |
| 训练准备与边界 | `_audit_tmp\readiness\` | `findings.json`、`boundary_experiment.py`、`experiments.json` | `& .venv-life\Scripts\python.exe -m unittest discover -s tests -v` |
| 重复/新颖度 | `_audit_tmp\novelty\` | `findings.json`、`section.md`、`probe_novelty.py`、`probe_results.json` | `& .venv-life\Scripts\python.exe -B _audit_tmp\novelty\probe_novelty.py`（importlib 直载，不拉起 torch） |

合并产物：`_audit_tmp\merged_findings.json`（64 条原文，含 evidence / impact / recommendation / regression_test 完整字段）。

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
