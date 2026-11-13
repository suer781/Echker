# 持续验证报告：可观测机制实跑

验证日期：2026-11-13  
版本：`0.5.0`  
最终运行：`validation_runs/continuous-adaptive-final-1`  
状态：**21 项可观测判据全部通过，0 failed，0 skipped，退出码 0**

> 本报告只验证可观测的工程机制，不证明语义理解、意识、通用智能，也不代表 `O:\数据集` 10.164 GiB 全量训练已经完成。全量训练曾单独启动，但在 28 帧后停止，当前不是运行中，也不是完成。

## 1. 最终运行

运行目录：`validation_runs/continuous-adaptive-final-1`  
报告：`validation_runs/continuous-adaptive-final-1/report.json`

| 项目 | 实测值 |
|---|---:|
| 混合模态在线循环 | 200 |
| 自适应半球切换 | 46 |
| 可复算切换决策 | 46 / 46 |
| 左半球训练步骤 | 5,520 |
| 右半球训练步骤 | 5,701 |
| 左侧 train / ttt / consolidate | 4,860 / 244 / 416 |
| 右侧 train / ttt / consolidate | 5,045 / 292 / 364 |
| TTT 权重变化采样 | 40 |
| TTT 最小权重 L1 变化 | 1.3143 |
| 双半球饥饿窗口比例 | 0.8571 |
| 真实数据集读取字节 | 20,000,000 |
| 真实数据帧数 | 26 |
| 真实数据保留偏移 | 16,724,230 |
| 中途关闭/恢复 | 4 次周期性 + 1 次最终恢复 |
| 恢复全字段指纹一致 | 5 / 5 |
| 总耗时 | 1,134.84 秒 |
| 主机 RSS 基线 / 峰值 / 最终 | 423.37 / 1,297.66 / 1,297.66 MB |
| GPU 已分配显存 基线 / 峰值 / 最终 | 2.92 / 47.56 / 44.83 MB |
| GPU 总显存 | 3,221,094,400 字节 |
| 内存预算 | 6,144 MB |

峰值 RSS 已包含最终读数，不再出现“最终高于峰值”的统计错误；1,297.66 MB 远低于 6,144 MB。但它只覆盖约 19 分钟运行，不能证明 40 小时量级无泄漏。

## 2. 判据与真实语义

1. `cpu_inference_gpu_training`：CPU 模型推理，CUDA 反向传播和优化器更新。
2. `mixed_cycles_completed`：200 次 RAW、UTF-8、RGB、PCM、未知 codec 交替完成。
3. `both_hemispheres_trained`：两侧均不少于 20 步，且 20 轮饥饿窗口内单侧增量不低于另一侧 30%。
4. `automatic_switching`：至少 3 条自动切换，且左右两个活动角色都出现过；本次 46 条。
5. `adaptive_switch_without_fixed_countdown`：配置无旧 `switch_interactions` / `switch_interval_s`；至少 3 条可复算决策；工作类型必须为主动侧 `train+ttt`、被动侧 `consolidate`；本次 46 条全部满足。
6. `ttt_weight_updates`：40 次采样中，每次 TTT 后活动侧 CPU 权重都变化。
7. `passive_cpu_isolation`：逐轮检查中，被动侧 CPU 前向计数未增加。
8. `train_loss_finite`：两侧最终训练损失有限。
9. `debt_bounded`：债务数量不超过容量 512。
10. `replay_bounded`：回放数量不超过容量 1,024。
11. `duplicate_cleaning`：重复低熵帧只降权、不硬拒绝；新增 digest 不超过 1；本次 `429→430`，新增 1 个不同 digest，不是把同一内容重复计为新样本。
12. `automatic_structure_events`：生产结构控制器真实给出生长与剪枝决策，并应用到独立模型。
13. `restarts_exact`：4 次周期恢复及最终恢复的加载状态、活动侧、切换次数、两侧步数、分 kind 计数和全字段指纹全部一致。
14. `real_dataset_volume`：真实数据集完整学习 20,000,000 字节，共 26 帧。
15. `real_dataset_resume`：从已保存偏移继续，而非从头重读；`resume_offset=16,724,230`。
16. `source_files_unchanged`：源文件大小与 mtime 未改变。
17. `queue_drained`：结束时 GPU 队列已清空。
18. `learning_rate_in_bounds`：两侧学习率始终位于 `1e-5..3e-3`。
19. `host_memory_bounded`：峰值 RSS 不超过 `memory_budget_mb=6144`。
20. `gpu_memory_bounded`：峰值总显存不超过 3,221,094,400 字节。
21. `no_runtime_errors`：运行期无未处理异常。

`--skip-real` 时，第 14–16 项现在是 `"skipped"`，不是 `true`；顶层分别统计 passed / failed / skipped。旧版把跳过项判真的假阳性已移除。

## 3. 自适应切换的实际输入

控制器不使用固定交互计数或固定秒数。每个交互/观察后读取：

- **累计干净压力**：自上次切换以来进入债务的非噪声质量，按 `debt_weight * structural_quality(ids)` 累计。
- **已学偿还容量**：债务账本最近 32 批成功偿还批次的单批干净质量中位数。`due()` 只预选，训练成功后 `commit_repayment()` 才记账；失败时释放预选。
- **工作量**：主动侧 `train + ttt` 步数增量与被动侧 `consolidate` 步数增量分别计数；`global_step` 仅作总步数，不参与判据。
- **角色基线**：控制器保存 `baseline_active_name` / `baseline_passive_name`；角色不匹配时拒绝决策并要求 `rebase`，避免 `active==right` 时左右基线互换。

最后一条自动决策：

```text
clean_pressure=553.5363
baseline_pressure=550.2649
pressure_growth=3.2715
learned_capacity=0.2504
work_share=0.8049
active=right  passive=left
active_work_kind=train+ttt
passive_work_kind=consolidate
active_after=left
```

46 条切换全部可按 `pressure_growth >= learned_capacity` 复算。报告正文保留最后 12 条明细，`switch_decisions_total=46` 和 `switch_decisions_recomputable=46` 分别记录总数与复算通过数；这不是只验证 12 条。

46 次/200 次是当前数据流密度与偿还容量共同产生的频率，不是目标值；它既不能证明抖动，也不能排除抖动。切换开销与长期稳定性仍未证明。

## 4. 双半球、TTT 与结构

- 只有活动侧执行在线生成；被动侧仍离线巩固。`interact_frame()` / `feed_frame()` 后等待当次被动巩固，避免高优先级 `train` 饿死 `consolidate`。
- 两侧 `train`、`ttt`、`consolidate` 分 kind 计数，且三者和等于 `global_step`。
- 40 次 TTT 采样最小 L1 为 1.3143；TTT 不是仅打印状态，实际改变了活动侧权重。
- 左半球自然生长到 `[352, 352]`，右半球到 `[352, 336]`；不是固定宽度。
- 独立生产控制器同时给出生长 `256→272` 与剪枝 `256→248`，随后通过 `DynamicByteLM.resize_hidden()` 应用。

## 5. 恢复与持久化

每次恢复均比较：

- 活动侧、切换次数、总步数；
- `train` / `ttt` / `consolidate` 分 kind 步数；
- 两侧模型、优化器、回放池、债务账本、学习率、结构控制器、指标、切换控制器基线、历史与决策记录的全字段指纹。

5 次恢复全部 `fingerprint_match=true`、`fingerprint_differences=[]`。这取代了旧版只比较 `blocks.0.fc1.weight` 的单层自比对。

真实数据段采用“先 `agent.save()`，再 `progress.mark()`”的顺序；检查点落盘后才提交偏移，崩溃最多重学最近未提交帧，不会跳过未训练帧。进度文件同时校验文件大小、mtime、块长度和块 SHA-256。

## 6. 跨格式覆盖

`cross_format_probe.py` 以只读方式对 `O:\数据集` 的 20 个格式桶各采样最多 131,072 字节，全部走同一个 `feed_frame()`：

- 覆盖 `.jsonl`、`.parquet`、`.json`、`.csv`、`.arrow`、`.txt`、`.zip`、`.ipynb`、`.png`、`.pdf`、`.jpeg`、`.py`、`.md`、`.sh`、`.yaml`、`.yml`、`.example`、`<none>`、`.gitignore`、`.gitattributes`；
- 20 / 20 无失败，采样载荷合计 1,310,870 字节，耗时 67.14 秒；
- RSS 562.99→1,165.36 MB，GPU 2.92→35.70 MB，仍低于预算；
- 这不是格式解析、语义理解或质量评估，只证明这些扩展名的原始字节能进入同一路径。

20,000,000 字节真实数据仍只覆盖约 0.184% 的 10.164 GiB，且主要落在 LogiQA 的少数文件；这是当前验收的真实边界。

## 7. 这轮修复的代码

- `switching.py`：角色索引基线；`note_switch` 显式接角色名；`decide` 接 `active_name/passive_name`；角色不匹配拒绝并 `rebase`；`SwitchDecision` 记录角色、工作类型、时间戳。
- `worker.py`：`train_steps` / `ttt_steps` / `consolidate_steps` 分 kind 计数；`online_learning_steps` / `passive_consolidation_steps`；巩固成功后才 `commit_repayment`，失败释放预选；optimizer 快照在锁内读取。
- `memory.py`：`DebtLedger.due()` 只预选不记账；`commit_repayment()` 训练成功后记容量；容量取最近 32 批中位数；重复 digest 不制造压力；`StatisticalCleaner` 内置有界 4-gram 熟悉度，重复只降权到 0.2 倍，不硬拒绝；结构质量只看前 32 字节。
- `agent.py`：切换前等待在途巩固；切换决策记录；`interact_frame` / `feed_frame` 末尾 drain 被动巩固；`save()` 先等待巩固；`status()` 输出 `memory_signals`、`step_accounting`、`background_errors`。
- `store.py`：全字段指纹与逐字段 diff；恢复分 kind 步数和决策；旧 v1 检查点跳过指纹；int/float 数值归一。
- `ingest_dataset.py`：进度块哈希/长度校验；先检查点后提交偏移。
- `continuous_validation.py`：可证伪判据、全字段恢复、资源曲线、真实数据保存顺序、决策样本、饥饿窗口、skip 语义；峰值 RSS/GPU 包含最终读数。
- `cross_format_probe.py`：只读跨格式准入探针，含隐藏 dotfile。

## 8. 回归测试

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv-life\Scripts\python.exe -m unittest discover -s .\tests -v
```

结果：**47 项通过，0 失败，0 错误**（4.826 秒）。

## 9. 明确未证明

1. 没有证明语义理解、事实正确性或通用智能。
2. 没有证明 40 小时量级的内存稳定性；6 GB 预算只在本轮约 19 分钟运行中被观测。
3. 20 MB 真实数据只覆盖 0.184% 字节；跨格式探针每桶约 128 KB，不覆盖大规模真实分布。
4. 全量 `O:\数据集` 10.164 GiB 训练曾启动，但在 28 帧后停止，尚未完成；`state_dataset` 的结果不能在本报告中宣称。
5. 46 次切换是高密度数据流入下的观测结果，不证明频率好坏，也不证明长期不抖动。
6. 熟悉度信号只证明降权且不硬拒绝；不证明“厌烦”、情绪或长期学习收益提高。
