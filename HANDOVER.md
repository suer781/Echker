# 交接文档：统一字节级在线学习原型

最后更新：2026-11-14  
项目根目录：`O:\Documents\智能自学习AI（新）`  
版本：`0.5.0`  
当前源码包：`ContinualLifeAgent.zip`（38 文件，2026-11-13 快照；此后新增 `watchdog.ps1` / `status_check.ps1`，见第 4.7 节）

---

## 0. 一分钟接手

1. 进入项目根目录，激活/使用 `.venv-life`，不要用系统 Python。
2. 先跑回归：`python -m unittest discover -s .\tests -v`，应为 **47 项全过**。
3. 先读 `AUDIT_CLOSURE.md` 和 `CONTINUOUS_VALIDATION.md`，再读本文件。
4. **不要碰 `state_dataset_smoke`**：它是真实 JSONL 冒烟状态，最终偏移 2,263,168、5 帧，是历史证据。
5. **全量训练当前运行中**（2026-11-14 00:24 起，看门狗 + WMI 分离式启动，与本会话解耦）。查看状态：`powershell -NoProfile -ExecutionPolicy Bypass -File .\status_check.ps1`。停止方式见第 4.7 节，历史与错位处理见第 2.2 节。

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

### 2.2 全量训练：运行中（2026-11-14 起，看门狗守护）

#### 2026-11-13 第一次启动与异常停止（已处理）

- 时间：2026-11-13 23:04:35 启动，初始化自 `validation_runs/continuous-adaptive-final-1/state/latest.pt`，命令 `ingest_dataset.py --root O:\数据集 --train --state-dir .\state_dataset --save-every 16`。
- 停止：28 帧后进程消失，`stderr.log` 为空、无 `last_run.json`、无异常栈、无退出码。结合 `agent.close()` 会先 `save()` 再由 `finally` 提交进度的顺序推断：**进程在收尾保存（23:19 的 `latest.pt`）之后、进度提交之前被硬杀**，最可能是父 PowerShell/控制台会话结束连带进程树被终止。
- 遗留错位：`dataset_progress.json` 停在 16 帧，`latest.pt`（23:19）可能已含 17–28 帧训练。

#### 2026-11-14 错位处理（已完成）

1. 三个文件已备份到 `state_dataset_backup_20261114/`：`latest.pt.misaligned28f`（23:19 检查点）、`latest.backup.pt.frame16`、`dataset_progress.json.frame16`。
2. 按文档推荐用 `latest.backup.pt`（23:14，与进度 16 帧严格对齐）替换 `latest.pt`，两文件 sha256 一致（`5abe2630…`）。
3. `--max-bytes 2000000` 有界续跑验证：从偏移 7,680,000 精确恢复，3 帧 / 984 步 / 106 秒，退出码 0，`last_run.json` 正常生成，进度与检查点重新严格对齐（19 帧）。

#### 当前运行（分离式 + 看门狗）

- 2026-11-14 00:24 起：`watchdog.ps1` 经 WMI `Win32_Process.Create` 启动（父进程为 WmiPrvSE，不随任何会话结束），由它拉起并守护 `ingest_dataset.py`（`-u -X utf8`）。
- 实测吞吐约 **1.6 MB/min**（~1.6 帧/分钟），全量 10.66 GiB 剩余对应 **约 5 天** 量级；按当前速率 6 GB 内存约束的 40 小时级风险需持续观察 RSS（看门狗每 5 分钟记录进程树 RSS）。
- worker 实测 RSS ≈ 1.17 GB（venv `python.exe` 只是启动器桩，真实解释器在其子进程，内存/CPU 必须按进程树口径看）。

全量规模（不变）：

- 只读清单：760 文件、10,913,873,703 字节（10.164 GiB）
- 默认训练范围：754 文件、10,910,693,561 字节
- 当前实时进度用 `status_check.ps1` 查看；**不能把"运行中"写成"完成"**。

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

### 4.7 长跑运维（watchdog.ps1 / status_check.ps1）

启动守护（WMI 分离式，关掉所有终端也不影响）：

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "O:\Documents\智能自学习AI（新）\watchdog.ps1"'
}
```

看状态：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "O:\Documents\智能自学习AI（新）\status_check.ps1"
```

行为说明：

- 看门狗循环启动 `python -u -X utf8 ingest_dataset.py --root O:\数据集 --train --state-dir .\state_dataset --save-every 16`，每次运行的 stdout/stderr 独立成 `state_dataset\stdout_<时间戳>.log` / `stderr_<时间戳>.log`。
- **崩溃自动重启**（上限 200 次，间隔 15 秒）；退出码 0 且刚写完 `last_run.json` 判定为优雅完成（数据集全部消费完），看门狗随之退出。
- **停滞检测**：stdout 超过 20 分钟无新输出，且 60 秒进程树 CPU 增量 < 2 秒（区分"活着但慢/休眠唤醒"与"真挂死"，避免休眠唤醒误杀）才 `taskkill /T /F` 并重启。
- 每 5 分钟向 `state_dataset\watchdog.log` 记录进程树 RSS（6 GB 预算的长期证据）。注意 venv `python.exe` 是启动器桩，真实解释器在子进程，内存/CPU 按进程树合计。
- **优雅停止**：在 `state_dataset` 下创建空文件 `watchdog.stop`，看门狗会在 ≤30 秒内硬杀当前训练并退出（硬杀会损失自上次进度提交以来的 ≤16 帧训练，恢复时重放，不丢数据）。
- 直接删除 `watchdog.stop` 后用上面的命令即可重新启动守护。
- 修改这两个 `.ps1` 时注意：文件必须带 UTF-8 BOM（PowerShell 5.1 对无 BOM 的 UTF-8 按 ANSI 解析，中文会破坏语法）；改完用解析器检查语法再上线。

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
3. 读 `AUDIT_CLOSURE.md`，确认 B6 仍是”部分关闭”。
4. ~~备份 `state_dataset` 目录~~ → 2026-11-14 已做：`state_dataset_backup_20261114/`。
5. ~~解决 16/28 帧的检查点/进度错位~~ → 2026-11-14 已做：对齐到 16 帧后有界续跑验证，进度/检查点严格对齐。
6. ~~先 `--max-bytes 2000000` 有界续跑~~ → 已做：3 帧 / 984 步 / 106 秒 / 退出码 0。
7. 长跑已由 `watchdog.ps1` 守护运行（第 4.7 节）；用 `status_check.ps1` 常规检查，重点盯 `watchdog.log` 里的 RSS 序列。
8. 若改 `worker.py`、`switching.py`、`store.py`、`agent.py`，必须重跑 47 项测试和 `verify_architecture.py`。
9. 修改任何验收判据前，先确认不是为了”通过”而放宽。

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
| `state_dataset.stdout.log` / `stderr.log` | 2026-11-13 第一次全量启动的最后记录（历史证据） |
| `state_dataset/stdout_<时间戳>.log` / `stderr_<时间戳>.log` | 看门狗每次运行的独立输出（当前证据） |
| `state_dataset/watchdog.log` | 看门狗动作与进程树 RSS 序列 |
| `state_dataset/dataset_progress.json` | 全量进度（实时，`status_check.ps1` 可读） |
| `state_dataset/latest.pt` / `latest.backup.pt` | 全量检查点对（save-then-mark，崩溃只重放不跳过） |
| `state_dataset_backup_20261114/` | 2026-11-13 停止时三个文件的原始备份 |
| `watchdog.ps1` / `status_check.ps1` | 长跑守护与状态查看（第 4.7 节） |
| `ContinualLifeAgent.zip` | 2026-11-13 源码快照（不含本节新增脚本） |
