# 自适应切换审查：阻断项关闭与残余边界

审查基线：`AUDIT_ADAPTIVE_SWITCH.md`，2026-11-13。  
关闭验证：`validation_runs/continuous-adaptive-final-1/report.json`，2026-11-13。  
版本：`0.5.0`。

> 本文件不覆盖原审查的 64 条历史发现，只回答一个问题：原报告列出的 7 个阻断项，在代码、测试和最终运行中分别关闭到什么程度，哪些仍然没有证明。

## 0. 结论

**B1–B7 的阻断代码路径已关闭，最终 200 次混合交互 + 20 MB 真实数据运行 21/21 通过、退出码 0。** 这不等于全量数据训练完成，也不证明语义理解、长期稳定性或 6 GB 长跑。全量 `O:\数据集` 训练曾启动，但在 28 帧后停止，尚未完成。

| 阻断项 | 状态 | 代码/证据 | 剩余边界 |
|---|---|---|---|
| B1 基线角色错位 | 关闭 | 角色索引基线；`SwitchRoleTests`；46 条决策均为角色一致 | 旧 v2 检查点若只有数值基线，加载时 `rebase`，不猜测角色 |
| B2 工作口径混用 | 关闭 | `train_steps/ttt_steps/consolidate_steps` 分 kind；主动侧 `train+ttt` 对被动侧 `consolidate`；“分 kind 计数和等于 global_step”测试 | 两种操作仍是不同语义，`active_delta >= passive_delta` 是工程判据，不是理论最优 |
| B3 恢复只比单层权重 | 关闭 | 全字段指纹；5 次恢复全一致；篡改检测测试 | 指纹不证明长期数值稳定性 |
| B4 周期落盘不等巩固 + `due()` 先记账 | 关闭 | `save()` / `_switch_locked()` 先等 consolidation；`due()` 预选不记账；成功后 commit；失败 release | 真实故障注入未覆盖磁盘满、断电等极端路径 |
| B5 验收器假阳 + 决策样本缺失 | 关闭 | `--skip-real` 为 skipped；决策样本与复算；饥饿窗口；峰值 RSS；重复只允许新增 ≤1 | 仍有“新增 1 个 digest”这一宽松边界；不是零容差 |
| B6 20 MB 覆盖 0.184% | 部分关闭 | 跨格式 20/20、131 万字节；20 MB 真实数据 26 帧 | 10.164 GiB 全量尚未完成；不能宣称覆盖全部字节分布 |
| B7 先写偏移后保存丢帧 | 关闭 | `ingest_dataset` 先 save 后 mark；块哈希/长度校验；真实段先 save 后 mark | 崩溃注入未做端到端强制断电测试 |

## 1. B1：基线角色错位

原缺陷：`note_switch` 按 `left/right` 名称传 `(active_work, passive_work)`，`decide` 按角色传；`active=="right"` 时基线互换。

现在的实现：

- `switching.py` 保存 `baseline_active_name` / `baseline_passive_name`；
- `note_switch(clean_pressure, active_work, passive_work, active_name, passive_name)`；
- `decide(..., active_name, passive_name)` 先检查角色匹配；
- 不匹配时返回 `switch baseline role mismatch; rebase required`，由 `agent._maybe_switch()` 调 `rebase()`，不在错误基线上切换；
- 基线数值统一 `float()`，避免 int→float 指纹误报。

证据：

- `tests/test_adaptive_contracts.py::SwitchRoleTests` 两个用例；
- `smoke-b1-b7-2` 与 `continuous-adaptive-final-1` 的 5 次恢复；
- 最终报告 46 条决策的 `active_name` / `passive_name` / `active_after` 均存在，可复算。

## 2. B2：工作口径混用

原缺陷：`global_step` 在 `_train` 内无条件递增，`ttt/train/consolidate` 共用；控制器却把“主动侧学习步数”和“被动侧巩固步数”相减。

现在：

- `HemisphereState.train_steps`、`ttt_steps`、`consolidate_steps` 分别递增；
- `global_step` 只是三类之和；
- `online_learning_steps = train_steps + ttt_steps`；
- `passive_consolidation_steps = consolidate_steps`；
- `SwitchDecision` 明确记录 `active_work_kind="train+ttt"`、`passive_work_kind="consolidate"`；
- 最终 `step_accounting`：左 `4,860 + 244 + 416 = 5,520`，右 `5,045 + 292 + 364 = 5,701`，与总步数一致。

剩余边界：两者仍是不同操作，不能用“步数相等”证明学习收益等价。该判据只防止拿总量冒充分工。

## 3. B3：恢复一致性

原缺陷：只比较 `blocks.0.fc1.weight`，且是磁盘→同一文件重载的自比对。

现在 `StateStore.fingerprint_agent()` 覆盖：

- 两侧 `hidden_dims`、`model`、`optimizer`、`replay`、`debt`、`lr`、`structure`、`metrics`；
- `active`、`history`、`interaction_count`、`observed_frames`、`switch_count`、`adaptive_switch`、`switch_decisions`；
- `format_version==1` 的旧检查点跳过指纹，不伪装成验证通过。

证据：

- 最终运行 5 次恢复：`loaded=true`、`active_match=true`、`switches_match=true`、`steps_match=true`、`step_kinds_match=true`、`fingerprint_match=true`、`fingerprint_differences=[]`；
- `CheckpointFingerprintTests` 覆盖往返与篡改检测。

剩余边界：指纹只证明“读回的状态与写入时一致”，不证明权重在长时间训练中不会漂移或损坏。

## 4. B4：落盘顺序与债务记账

原缺陷：

- monitor 每 300 秒 `save()` 不等在途 consolidate；
- `snapshot(2)` 优先级高于 `consolidate(4)`；
- `due()` 一调用就增加 `total_repaid` / `repayment_batches` / `repayment_capacity`，训练尚未发生。

现在：

- `agent.save()` 在锁内先 `_wait_consolidations()`，再 `StateStore.save()`；
- `_switch_locked()` 先 `_wait_consolidations()`，再交换债务/同步/换角色；
- `DebtLedger.due()` 只预选，`_reserved` 防止同一 digest 连续被选中；
- `commit_repayment(samples, trained_steps)` 只在训练成功后记 `total_repaid`、`repayment_batches` 和容量；
- 失败或零步时 `release_reservation()`；
- 容量取最近 32 批中位数，不再用可能被重复批抬高的累计平均。

证据：最终 5 次恢复全字段一致；`DebtReservationTests` 覆盖“due 不记账、未 commit 不计数、重复不制造压力”。

剩余边界：没有做强制断电/磁盘满故障注入；`CONC-01` 已于 2026-11-14 修复（见第 8 节）。

## 5. B5：验收器可证伪性

原缺陷：

- `--skip-real` 时 `real_dataset_volume` / `resume` / `source_unchanged` 恒 true；
- `adaptive_switch_without_fixed_countdown` 只查“配置无旧键 + switches>=8”；
- `report.json` 的 `adaptive_switch` 为 null；
- `duplicate_cleaning` 允许唯一数增加 5；
- `both_hemispheres_trained` 门槛仅 20 步。

现在：

- 跳过项为 `"skipped"`，顶层 passed/failed/skipped 分开；
- `automatic_switching` 要求至少 3 条自动切换且两个活动角色都出现；
- `adaptive_switch_without_fixed_countdown` 要求至少 3 条 `pressure_growth >= learned_capacity` 可复算样本，且工作类型齐全；
- `adaptive_switch` 非 null，记录最后一条决策；`switch_decisions` 记录最后 12 条，`switch_decisions_total` 记录总数；
- `duplicate_cleaning` 要求新增 digest ≤1；
- `both_hemispheres_trained` 增加 20 轮饥饿窗口，单侧增量不低于另一侧 30%；
- `host_memory_bounded` 使用包含最终读数的峰值 RSS；`gpu_memory_bounded` 使用总显存。

证据：最终 21/21、0 skipped；46 条自动切换全部可复算；`starvation_window_fraction=0.8571`；RSS 峰值 1,297.66 MB。

剩余边界：`duplicate_cleaning` 仍允许新增 1 个 digest；这不是“零新增”的强判据。饥饿窗口是统计阈值，不是形式证明。

## 6. B6：真实数据覆盖

原缺陷：20 MB 只覆盖 LogiQA 下 8 个文件、0.184% 字节；Parquet/Arrow/CSV/JSON/图片/zip 未在真实数据验证；41.8 h 是外推。

现在：

- `cross_format_probe.py` 只读采样 20 个格式桶，每桶最多 131,072 字节：
  - `.jsonl`、`.parquet`、`.json`、`.csv`、`.arrow`、`.txt`、`.zip`、`.ipynb`、`.png`、`.pdf`、`.jpeg`、`.py`、`.md`、`.sh`、`.yaml`、`.yml`、`.example`、`<none>`、`.gitignore`、`.gitattributes`；
  - 20/20 无失败；采样载荷 1,310,870 字节；耗时 67.14 秒；RSS 562.99→1,165.36 MB；GPU 2.92→35.70 MB；
- 真实 20 MB 仍为 26 帧、`resume_offset=16,724,230`、源文件未改。

剩余边界：

- 20 MB 仍只覆盖约 0.184% 的 10.164 GiB；
- 跨格式每桶只有约 128 KB，且一篇内只取一个文件，不能代替全量分布；
- 41.8 h 是混合速率外推；全量启动后前 16 帧为 10.95 MB / 约 10 分钟（约 1 MB/min），按当前速率更接近 100 小时以上，仍不是保证；
- 全量训练曾启动，但在 28 帧后停止，未完成，不能把启动写成完成。

## 7. B7：进度与检查点顺序

原缺陷：`continuous_validation.py` 先 `progress.mark`，每 4 帧才 `agent.save()`；崩溃可跳过 1–3 帧。`ingest_dataset.py` 顺序相反。

现在：

- `continuous_validation.py` 真实段先 `agent.save()`，再 `progress.mark()`；
- `ingest_dataset.py` 同样先 save 后 mark；
- `ProgressStore.offset()` 校验 size、mtime、`last_chunk_sha256` 和 `last_chunk_bytes`；不匹配从 0 重读；
- `mark()` 支持可选块哈希；
- 最终真实段 26 帧、resume 通过。

剩余边界：没有用强制杀进程/断电验证“保存后、mark 前崩溃”的恢复路径；代码顺序和单测覆盖了逻辑，不是故障注入实测。

## 8. 未关闭的非阻断项

2026-11-14 修复三项（55 项测试全过 + verify_architecture 16/16 + `validation_runs/continuous-adaptive-postfix-1` 验收 **21/21、0 failed、0 skipped**，42 次切换全部可复算，5 次恢复指纹全对，RSS 峰值 1,230 MB）：

- `CONC-01`（已修）：`TrainingWorker.submit(wait=True)` 按 job 种类带 300–900 秒超时（`_JOB_TIMEOUT_S`），agent 侧 `_await_consolidation()` 对后台巩固 future 同样带 900 秒死线；挂死不再无声冻结，调用方收到 `RuntimeError` 后由进程级看门狗重启。
- `SYNC-02`（已修）：`_optimizer_rebuild` 携带动量重建 AdamW——形状不变的参数原样搬运 `exp_avg/exp_avg_sq/step`；生长神经元按 `row_map` 继承其分裂源神经元的动量；剪枝按 `keep_map` 重映射保留神经元的动量。新神经元从零动量开始。
- `MATH-05`（已修）：切换交换样本打 `exchange:` 来源标签并以 `count_pressure=False` 入账——已学材料不再计入新净压力，也不再污染 `capacity_history`（混合批只计新数据质量，纯交换批不记账）。

原审查的其余高/中非阻断项仍在，不能因为 21/21 通过就宣称全部修完：

- `CONC-03/04/05/06/07/08/10`：队列/重建/锁粒度的残余风险（`_wait_consolidations` 仍在 agent RLock 内等待，但现在有死线）；
- `MATH-03`：容量偏高（历史实测左 2.633x、右 1.548x）；MATH-05 修复移除了交换污染这一来源，剩余偏差来自 top-weight 选择偏置，需长跑数据再校准；
- `MATH-04`：`due()` 重复计批的旧证据已被“预选不记账”修掉，但容量中位数与批边界仍需更多实验；
- `NOV-02/03`：结构质量与长度耦合已改为前 32 字节，但重复债务优先级仍需实验；
- `READY-04/07/08`、`PERS-12/13`：数据集漂移、文档/版本、长期恢复；
- `--skip-real` 的 skipped 语义已修，但跨格式探针不进入正式 21 项判据，是独立证据。

## 9. 明确的“不声称”

- 不声称 21/21 等于语义理解、通用智能或“数字生命”已完成；
- 不声称 46 次/200 次是最优切换频率；
- 不声称 1.29 GB RSS 能保证 40 小时不超 6 GB；
- 不声称 20 MB / 1.31 MB 跨格式样本覆盖了 10.164 GiB；
- 不声称“厌恶重复”是情绪能力；它只是 `StatisticalCleaner` 的熟悉度降权，且不硬拒绝；
- 不声称全量训练已经完成；它曾启动，但在 28 帧后停止，当前不是运行中。
