# 交接文档：统一字节级在线学习原型

最后更新：2026-11-13  
项目根目录：`O:\Documents\智能自学习AI（新）`  
版本：`0.5.0`  
当前源码包：`ContinualLifeAgent.zip`（38 文件，已包含本文件新增前的当前代码与文档）

---

## 0. 一分钟接手

1. 进入项目根目录，激活/使用 `.venv-life`，不要用系统 Python。
2. 先跑回归：`python -m unittest discover -s .\tests -v`，应为 **47 项全过**。
3. 先读 `AUDIT_CLOSURE.md` 和 `CONTINUOUS_VALIDATION.md`，再读本文件。
4. **不要碰 `state_dataset_smoke`**：它是真实 JSONL 冒烟状态，最终偏移 2,263,168、5 帧，是历史证据。
5. **全量训练当前已停止，不是运行中**。最后一次观察是 28 帧、21,318,772 载荷字节、10,487 步；恢复命令见第 4 节，恢复前先读第 2.2 节。

---

## 1. 项目定位与不可破坏约束

这是一个**统一字节级持续学习原型**，不是按模态分别建模型的工具箱。

不可破坏的约束：

- **统一字节级**：词表固定 262；`0..255` 是原始字节，`256..261` 是 BOS/EOS/USER/ASSISTANT/SEP/PAD。没有 BPE、SentencePiece、子词或文本 tokenizer。
- **一套模型、一个目标**：文本、RGB、PCM、未知二进制都通过同一个 BLF1 自描述帧，进入同一个 `feed_frame()` / `interact_frame()`，共用同一个下一字节交叉熵。
- **双半球**：左右各自独立模型、优化器、回放、债务、学习率、结构控制器和检查点状态；同一时刻只有一侧在线生成，另一侧离线巩固。
- **TTT 必须保留**：CPU 在线推理时做测试时下一字节反向传播，更新后发布到活动侧 CPU；不能为了“更稳”取消 TTT。
- **CPU 推理、GPU 训练**：`inference_device=cpu`，`training_device=cuda`；运行时 resize 必须走 `TrainingWorker.resize()`。
- **自适应切换**：不得恢复固定交互计数、固定秒数或替代性阈值。
- **资源硬约束**：`memory_budget_mb=6144`；GPU 使用全部可用显存。

用户明确拒绝：

- 文本/图片/音频分别建专用模型、专用目标或能力管线；
- 固定切换计数/时间阈值；
- 为了“更好”牺牲在线学习、字节模型、双半球或 TTT；
- 独立的“心理学层”或硬盖式补丁；
- 把“机制成立”包装成“已经实现完整智能”。

---

## 2. 当前可运行状态（事实）

### 2.1 最终短程验收：已通过

运行目录：`validation_runs/continuous-adaptive-final-1`

报告：`validation_runs/continuous-adaptive-final-1/report.json`

| 项目 | 实测 |
|---|---:|
| 判据 | **21 / 21 通过，0 failed，0 skipped** |
| 退出码 | 0 |
| 混合在线交互 | 200 |
| 自适应切换 | 46，46 条全部可复算 |
| 左 / 右训练步数 | 5,520 / 5,701 |
| 左侧 train / ttt / consolidate | 4,860 / 244 / 416 |
| 右侧 train / ttt / consolidate | 5,045 / 292 / 364 |
| TTT 最小权重 L1 变化 | 1.3143 |
| 真实数据 | 20,000,000 字节、26 帧，`resume_offset=16,724,230` |
| 恢复 | 4 次周期 + 1 次最终，5/5 全字段指纹一致 |
| RSS 基线 / 峰值 / 最终 | 423.37 / 1,297.66 / 1,297.66 MB |
| GPU 基线 / 峰值 / 最终 | 2.92 / 47.56 / 44.83 MB |
| GPU 总显存 | 3,221,094,400 字节 |
| 耗时 | 1,134.84 秒 |

结构：左自然生长到 `[352,352]`，右到 `[352,336]`；生产控制器真实给出生长 `256→272` 与剪枝 `256→248`。

切换没有固定计数：运行时只看累计非噪声压力、自己学到的单批偿还容量、主动侧 `train+ttt` 对被动侧 `consolidate` 的步数增量。验收器要求“至少 3 条可复算决策”，这是**验收判据**，不是运行时阈值。

### 2.2 全量训练：已停止，不能称为完成

启动：

- 时间：2026-11-13 23:04:35
- 初始化：`validation_runs/continuous-adaptive-final-1/state/latest.pt`
- 命令：`ingest_dataset.py --root O:\数据集 --train --state-dir .\state_dataset --save-every 16`
- 父/子 PID：15188 / 27040（现在进程表里都已不存在）

最后一次 `state_dataset.stdout.log`：

```json
{"progress": {"frames": 28, "payload_bytes": 21318772, "encoded_bytes": 21325964, "chunks": 41911, "trained_steps": 10487, "last_file": "01_LogiQA/LogiQA-2.0/LogiQA2.0-main/logiqa/DATA/LOGIQA/dev_zh.txt", "offset": 1202597}}
```

停止时的状态：

- `stderr.log` 为空；`last_run.json` 不存在，说明进程没有走正常收尾的 `finally` 汇总分支。
- 没有 Python 异常记录，也没有退出码；进程消失原因**未确认**。不能排除父 PowerShell/会话生命周期结束、系统 job 或外部终止。
- `dataset_progress.json` 停在 **16 帧**，时间 23:14:35。
- `latest.pt` 时间 23:19:51，大小 44.06 MB；`latest.backup.pt` 时间 23:14:35。
- 最后 stdout 到 28 帧，但进度只提交到 16 帧：`latest.pt` 可能已经包含 17–28 帧的训练，而进度仍从 16 帧恢复。**恢复会重放第 17–28 帧**，不会跳过未训练数据，但会造成这段数据的重复训练。这是当前需要先处理的交接风险。

全量规模：

- 只读清单：760 文件、10,913,873,703 字节（10.164 GiB）
- 默认训练范围：754 文件、10,910,693,561 字节
- 28 帧的 21,318,772 字节约覆盖 **0.195%**，不是全量完成。

### 2.3 跨格式准入：已跑，但只是准入

`validation_runs/cross-format-probe/report.json`

- 20 个格式桶：`.jsonl`、`.parquet`、`.json`、`.csv`、`.arrow`、`.txt`、`.zip`、`.ipynb`、`.png`、`.pdf`、`.jpeg`、`.py`、`.md`、`.sh`、`.yaml`、`.yml`、`.example`、`<none>`、`.gitignore`、`.gitattributes`
- 20/20 无失败，采样 1,310,870 字节，耗时 67.142 秒
- RSS 562.99→1,165.36 MB，GPU 2.92→35.70 MB
- 证明原始字节能进入同一路径，**不证明**格式解析、语义理解或质量评估；每桶约 128 KB，不能代替全量分布。

### 2.4 测试

```powershell
Set-Location -LiteralPath "O:\Documents\智能自学习AI（新）"
$env:PYTHONPATH = (Get-Location).Path
& ".\.venv-life\Scripts\python.exe" -m unittest discover -s .\tests -v
```

最近结果：**47 项通过，0 失败，0 错误**（约 4.8 秒）。测试输出里可能出现一次预期的 “failed to restore checkpoint” 指纹篡改测试日志，那是负向测试，不代表回归失败。

---

## 3. 代码地图

```text
run_agent.py                      CLI 入口，调用 life_agent.cli.main
start_agent.ps1                   固定使用 .venv-life 和 config.json 的启动脚本
config.json                      模型/运行参数；memory_budget_mb=6144
life_agent/
  byte_codec.py                  BLF1 帧、codec、流式观察、CRC32
  model.py                       2 层因果 Transformer，可生长/剪枝
  memory.py                      回放池、StatisticalCleaner、债务账本
  switching.py                   AdaptiveSwitchController、角色基线、决策记录
  worker.py                      GPU 单写者队列、train/ttt/consolidate 计数
  agent.py                       feed_frame()/interact_frame()、双半球调度、保存
  store.py                       v2 检查点、全字段指纹、v1 迁移
  cli.py                         交互终端
ingest_dataset.py                只读流式数据集训练、进度与恢复
continuous_validation.py         200 轮 + 20 MB 可证伪验收器
cross_format_probe.py            20 格式只读准入探针
tests/                           47 项单元/集成/CUDA 测试
```

关键语义：

- `TrainingWorker` 队列优先级：TTT 0，sync/resize 1，snapshot 2，train 3，consolidate 4。
- `global_step` 只是总步数；`train_steps` / `ttt_steps` / `consolidate_steps` 才是分工口径。
- 所有运行时 resize 必须 `TrainingWorker.resize()`；`LifeAgent.load()` 在 worker 启动后调用会抛 `RuntimeError`。
- 被动巩固异常必须显式传播，不能吞掉。
- `DebtLedger.due()` 只预选；`commit_repayment()` 训练成功后记账；失败 `release_reservation()`。

---

## 4. 运行命令

### 4.1 交互式运行

```powershell
Set-Location -LiteralPath "O:\Documents\智能自学习AI（新）"
.\start_agent.ps1
```

或直接：

```powershell
& ".\.venv-life\Scripts\python.exe" ".\run_agent.py" `
  --config ".\config.json" --state-dir ".\state"
```

### 4.2 只生成数据集清单

```powershell
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" `
  --root "O:\数据集"
```

### 4.3 续跑全量训练（当前已停止）

目标状态已有 `latest.pt`，不要加 `--initialize-from`，也不要加 `--fresh`：

```powershell
Set-Location -LiteralPath "O:\Documents\智能自学习AI（新）"
$env:PYTHONPATH = (Get-Location).Path
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" `
  --root "O:\数据集" --train --state-dir ".\state_dataset" --save-every 16
```

注意：

- 由于 `dataset_progress.json` 停在 16 帧，而 `latest.pt` 时间更新到 23:19，恢复会从第 17 帧开始重放；这是可恢复的重复训练，不是数据丢失。
- 如果要从一个与进度严格对齐的点恢复，优先用 `latest.backup.pt`（23:14:35，与进度 16 帧同一时刻）替换 `latest.pt`，或先把 `state_dataset` 备份后重建；不要在未备份时手工改 `dataset_progress.json`。
- 运行前先用 `--max-bytes` 做一次小段续跑，确认进度和检查点行为。
- 如果再次异常停止，先确认 `last_run.json` 是否存在，再决定是否重跑；没有 `last_run.json` 表示没有正常收尾。

有界测试示例：

```powershell
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" `
  --root "O:\数据集" --train --state-dir ".\state_dataset" `
  --max-bytes 2000000 --save-every 16
```

### 4.4 复跑持续验收

```powershell
& ".\.venv-life\Scripts\python.exe" ".\continuous_validation.py" `
  --mixed-cycles 200 --real-bytes 20000000 --restart-every 50 `
  --dataset-root "O:\数据集" `
  --output-dir ".\validation_runs\continuous-adaptive-handover-1"
```

- 默认输出目录不能已存在且非空，否则会拒绝复用。
- 需要 GPU；运行约 19 分钟。
- `--skip-real` 时真实数据三项为 `skipped`，不能用跳过结果证明真实覆盖。

### 4.5 跨格式探针

```powershell
& ".\.venv-life\Scripts\python.exe" ".\cross_format_probe.py" `
  --root "O:\数据集" --output-dir ".\validation_runs\cross-format-probe"
```

### 4.6 CUDA 架构验证

```powershell
& ".\.venv-life\Scripts\python.exe" ".\verify_architecture.py"
```

历史结果：16 项通过。最近一轮没有在改动后重跑；交接后如修改 worker/resize/切换，必须重跑。

---

## 5. 关键机制速览

### 5.1 统一帧

`BLF1` 自描述帧包含 codec、JSON 元数据、载荷长度、载荷、CRC32。所有格式都按原始字节读入，外部 codec 可以不同，但模型入口只有一个。

### 5.2 双半球

- 初始活动侧 `left`，被动侧由 active 反推。
- 只有活动侧做 CPU 生成；两侧都可做 GPU 反向。
- 切换时做 2% 总量对称插值、债务交换、宽度对齐；这些是现有设计，不是完全隔离。
- 切换前先等待在途巩固；切换后重置压力/步数基线，并随 v2 检查点保存。

### 5.3 自适应切换

输入：

1. 自上次切换以来进入债务的非噪声干净质量；
2. 债务账本成功偿还批次的单批容量中位数；
3. 主动侧 `train+ttt` 步数增量对被动侧 `consolidate` 步数增量；
4. 角色名匹配的基线；不匹配要求 `rebase`。

不存在 `switch_interactions` / `switch_interval_s`。不要为了“稳定”恢复固定 16 次或 900 秒。

### 5.4 TTT

- CPU 活动侧每批做 `ttt_steps=1` 的下一字节反向传播；
- 更新后发布到活动侧 CPU；
- 不改变“在线学习必须存在”的约束；
- 探针比较权重前后 L1，最终运行最小值为 1.3143。

### 5.5 习惯化/重复降权

没有 `life_agent/psychology.py`，没有独立心理学层。熟悉度在 `StatisticalCleaner` 内：

- 有界 4-gram 熟悉度，`familiarity_order=4096`；
- `observe(ids)` 返回 `novelty/familiarity/sampling_weight`；
- 回放权重下限 0.001，熟悉度下限 0.2；
- 重复 digest 的存储权重向输入权重收敛，不每次抬高；
- 结构质量只看前 32 个原始字节；
- 静音/纯色只降权，不拒绝；
- **不进入切换判据**。

它只证明“降权且保留”，不证明“厌烦”、情绪或长期学习收益提高。

---

## 6. 验收器怎么读

最终报告关键字段：

- `criteria_counts.passed/failed/skipped`
- `adaptive_switch`
- `switch_decisions`：最后 12 条明细
- `switch_decisions_total`：自动切换总数（本次 46）
- `switch_decisions_recomputable`：可复算数（本次 46）
- `step_accounting`：两侧 train/ttt/consolidate
- `memory.rss_peak_mb` / `memory.rss_final_mb`
- `memory.gpu_peak_bytes` / `gpu_total_bytes`
- `restarts[]`：每次恢复的 `fingerprint_match` 与差异
- `real_dataset.resume_offset`

重要：`switch_decisions` 只列最后 12 条，不等于只验证 12 条；总数看 `switch_decisions_total`，复算数看 `switch_decisions_recomputable`。

---

## 7. 已完成 / 未完成

### 已完成并有证据

- B1–B7 的阻断代码路径：角色基线、分 kind 步数、全字段指纹、债务提交、验收器可证伪、跨格式准入、先保存后提交偏移。
- 47 项回归。
- final-1 短程 21/21。
- 跨格式 20/20 准入。
- 全量训练已启动过一次，并从 `state_dataset` 产生 28 帧真实训练记录。

### 未完成 / 不能声称

- 全量 10.164 GiB **没有完成**；当前进程已停止，28 帧约 0.195%。
- 6 GB 长跑没有实测：final-1 约 19 分钟峰值 1,297.66 MB；`state_dataset` 停止前约 1.20 GB；都不能证明 40 小时量级不超 6 GB。
- 20 MB 真实数据仅覆盖 0.184% 字节；跨格式每桶约 128 KB。
- 没有证明语义理解、事实正确性、意识或通用智能。
- 46 次/200 次只是观测频率，不证明频率最优或长期不抖动。
- `AUDIT_CLOSURE.md` 第 8 节列出的非阻断项多数仍未修：`CONC-01`（无超时等待）、`CONC-03/04/05/06/07/08/10`、`MATH-03/04/05`、`SYNC-02`、`NOV-02/03`、`READY-04/07/08`、`PERS-12/13`。
- 1% 路径哈希保留集不是语义去污染；已知重复版本需要 `--exclude` 或人工清单。
- `ARCHITECTURE_AUDIT.md` / `ARCHITECTURE_VALIDATION.json` 的数字是旧版；以本文件和 final-1 报告为准。

---

## 8. 交接检查清单

按顺序做：

1. `git status` 或至少确认工作区文件时间；不要覆盖 `state_dataset_smoke`。
2. 跑 47 项测试。
3. 读 `AUDIT_CLOSURE.md`，确认 B6 仍是“部分关闭”。
4. 备份 `state_dataset` 目录（尤其 `latest.pt`、`latest.backup.pt`、`dataset_progress.json`）。
5. 解决 16/28 帧的检查点/进度错位，再续跑。
6. 先 `--max-bytes 2000000` 有界续跑，记录 `last_run.json`、RSS、GPU、吞吐。
7. 再决定是否启动长跑；启动后至少每 30 分钟检查 `stdout`、`stderr`、RSS、GPU。
8. 若改 `worker.py`、`switching.py`、`store.py`、`agent.py`，必须重跑 47 项测试和 `verify_architecture.py`。
9. 修改任何验收判据前，先确认不是为了“通过”而放宽。

---

## 9. 不要做的事

- 不要新增 `life_agent/psychology.py` 或“心理学层”。
- 不要把熟悉度信号塞进切换公式；它会让高熵噪声凭“没见过”触发振荡。
- 不要恢复固定 16 次/900 秒切换。
- 不要为了降低切换频率而加固定最小间隔。
- 不要取消 TTT、把 CPU 推理改成 GPU、或把双半球合并成单模型。
- 不要用文本 tokenizer/BPE 替换字节词表。
- 不要在没有备份时手工改 `dataset_progress.json`。
- 不要删除或覆盖 `state_dataset_smoke`。
- 不要把 `21/21` 写成“语义理解/数字生命已完成”。
- 不要把 cross-format 20/20 写成“全量格式覆盖”。

---

## 10. 文档与产物索引

| 文件 | 用途 |
|---|---|
| `README.md` | 项目说明、运行方式、当前数字 |
| `VALIDATION.md` | 历史与当前实测记录 |
| `CONTINUOUS_VALIDATION.md` | final-1 的逐项判据与边界 |
| `AUDIT_ADAPTIVE_SWITCH.md` | 独立审查原文（保留审查当时结论） |
| `AUDIT_CLOSURE.md` | B1–B7 关闭映射、残余风险、不声称清单 |
| `PSYCHOLOGY_MECHANISM_ANALYSIS.md` | 心理学机制映射，不是独立层 |
| `ARCHITECTURE.md` | 架构细节（部分数字为旧版） |
| `ARCHITECTURE_AUDIT.md` / `ARCHITECTURE_VALIDATION.json` | 旧版双半球架构审计与 16 项 CUDA 结果 |
| `validation_runs/continuous-adaptive-final-1/report.json` | 最终 21/21 报告 |
| `validation_runs/cross-format-probe/report.json` | 20 格式准入报告 |
| `state_dataset.stdout.log` / `stderr.log` | 全量训练最后记录（当前已停止） |
| `state_dataset/dataset_progress.json` | 全量进度（当前 16 帧） |
| `state_dataset/latest.pt` | 全量检查点（当前 23:19:51，44.06 MB） |
| `ContinualLifeAgent.zip` | 当前源码包 |
