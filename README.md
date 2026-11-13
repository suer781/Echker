# 通用字节帧双半球持续学习智能体

当前版本：`0.5.0`。

这是一个已落地运行的 PyTorch 研究原型。文本、图片、音频和未知格式都通过**同一个自描述字节帧接口**进入同一套模型、TTT、回放、训练债和神经元调节机制，没有模态专用模型或模态专用损失。

## 启动

本机依赖已安装在独立 `.venv-life`，原有 `.venv` 未被修改。

```powershell
powershell -ExecutionPolicy Bypass -File .\start_agent.ps1
```

或者：

```powershell
.\.venv-life\Scripts\python.exe .\run_agent.py --config .\config.json --state-dir .\state
```

普通文字直接输入即可。可选命令：

- `/feed <文件路径>`：把任意文件作为原始二进制字节帧完整观察并学习，不生成回复。
- `/status`：查看活动半球、接口版本、观察帧数、权重、学习率、损失和债务。
- `/switch`：立即双向留债并切换半球。
- `/save`：保存检查点。
- `/quit`：等待训练队列完成、保存并退出。

`/feed` 不依赖文件扩展名，也不会把文件转成文本。PNG/JPEG 等文件通过该命令输入时是**原始压缩文件字节**；要输入 RGB 像素，请使用下面的统一 API 构造 RGB 帧。

## 统一 Python 接口

### 任意二进制

```python
from pathlib import Path
from life_agent import ByteFrame

frame = ByteFrame.from_file(Path("任意文件.bin"))
metrics = agent.feed_frame(frame)
```

### 原始 RGB 图片

```python
from life_agent import ByteFrame

# pixels 必须是 width * height * 3 个 uint8 字节，按 HWC 排列。
stimulus = ByteFrame.rgb(pixels, width=64, height=64)
agent.feed_frame(stimulus)

# 响应规格只描述格式和结构，不包含载荷。
response_spec = stimulus.empty()
response = agent.interact_frame(stimulus, response_spec)
assert len(response.payload) == 64 * 64 * 3
```

### 文本到指定结构 RGB

```python
response_spec = ByteFrame.rgb(bytes(64 * 64 * 3), 64, 64).empty()
response = agent.interact_frame(ByteFrame.text("生成一张 64x64 图片"), response_spec)
```

`max_new_tokens` 必须不小于声明的输出字节数，否则会明确报错而不是返回长度错误的图片。PCM 同样通过元数据携带采样率、采样宽度、通道数和采样帧数。未知格式可以使用自定义 codec 与 `payload_length` 元数据。

### 文本只是统一接口的便捷包装

```python
text = agent.interact("你好")
```

其内部仍然执行 `ByteFrame.text()`、`interact_frame()` 和 UTF-8 载荷解码，不存在另一套语言学习通道。

## BLF1 自描述帧

每帧包含：

```text
magic "BLF1" | codec 1 字节 | flags 1 字节
metadata 长度 | payload 长度 | CRC32
JSON 元数据字节 | 任意 payload 字节
```

内置 codec：

- `0 RAW`：任意二进制。
- `1 UTF8`：UTF-8 文本。
- `2 RGB`：原始 HWC uint8 像素及宽高通道元数据。
- `3 PCM`：原始音频采样及格式元数据。
- `4..255`：可扩展自定义格式，学习核心无需修改。

模型看到的是帧和载荷字节；格式解析只负责封装、校验和解释，不参与智能决策。

## 当前机器验证环境

- NVIDIA GeForce GTX 1060 3GB，`sm_61`。
- Python 3.13.18。
- PyTorch 2.8.0+cu126。
- 每个初始半球 364,288 参数。
- CPU 推理，CUDA GPU 反向传播和优化器更新。

重建依赖：

```powershell
python -m venv .venv-life
.\.venv-life\Scripts\python.exe -m pip install --upgrade pip
.\.venv-life\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-life\Scripts\python.exe -m pip install numpy
```

## 大规模数据集流式训练

`ingest_dataset.py` 面向 `O:\数据集` 这类大目录：源文件只用 `rb` 读取，按默认 960,000 字节分块，每块封装成带相对路径、偏移和块 SHA-256 的 BLF1 RAW 帧。JSONL、Parquet、CSV、JSON、Arrow、图片等都按原始文件字节学习，不增加格式专用模型、损失或文本分词器。

只生成清单，不训练：

```powershell
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" --root "O:\数据集"
```

正式全量训练使用独立状态，避免覆盖交互或冒烟检查点：

```powershell
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" `
  --root "O:\数据集" --train --state-dir ".\state_dataset"
```

重复同一命令即可续训。模型每 10 帧自动保存一次，检查点成功后才提交对应文件偏移；异常退出可能重学最近未提交的少量帧，但不会把尚未落盘的训练标为完成。`dataset_progress.json` 记录文件大小、修改时间和下一偏移，`last_run.json` 记录本轮用量和吞吐。

常用限制：

```powershell
# 只训练 JSONL，最多读取 2,000,000 字节
& ".\.venv-life\Scripts\python.exe" ".\ingest_dataset.py" `
  --root "O:\数据集" --train --state-dir ".\state_dataset_smoke" `
  --include "*.jsonl" --max-bytes 2000000

# 也可用 --max-files、--max-seconds、--exclude 和 --holdout 控制范围
```

默认按相对路径哈希保留约 1% 文件不参与训练。它不是跨来源语义去污染，也不代表这些文件与训练集没有重复内容。不同格式中的同源数据不会因原始字节不同而自动去重；已知重复版本应通过 `--exclude` 排除，或另行制定数据选择清单。

当前只读清单为 760 个文件、10,913,873,703 字节（10.164 GiB）；默认训练 754 个文件、10,910,693,561 字节，保留 6 个文件、3,180,142 字节。详见 `dataset_inventory.json`。

真实 JSONL 冒烟已完成三段续训，最终源偏移 2,263,168；较稳定的 2,000,000 字节段实测 79.3 KB/s、39.0 GPU 步/秒。早期按该速度外推全量约 38–50 小时；但 2026-11-13 全量启动后的前 16 帧为 10.95 MB / 约 10 分钟，约 1 MB/min，明显慢于早期外推，按当前速率更接近 100 小时以上。该数字只是当前速率外推，不是完成时间承诺，实际会受结构生长、巩固、文件类型和保存开销影响。

2026-11-13 23:04 已从 `validation_runs/continuous-adaptive-final-1/state/latest.pt` 初始化，启动 `state_dataset` 全量训练；**该进程在 23:23:56 左右停止，最后为 28 帧 / 21,318,772 字节 / 10,487 步，未正常收尾，当前不是运行中，也不是完成。** 启动扫描确认 760 文件 / 10,913,873,703 字节（10.164 GiB），训练范围 754 文件 / 10,910,693,561 字节；恢复步骤、检查点与进度错位风险见 `HANDOVER.md`。全量完成前不宣称 6 GB 长跑、语义理解或全部格式覆盖已成立。

`--fresh` 会忽略目标状态中的旧模型和旧进度，应只在明确需要重新开始时使用。新状态也可通过 `--initialize-from <检查点>` 从一个可信检查点初始化；目标已有检查点时不会覆盖它。

## 文件布局

```text
run_agent.py / start_agent.ps1   启动入口
verify_learning.py               原始字节在线学习对照实验
verify_architecture.py           双半球、TTT、债务与恢复 CUDA 验证
ingest_dataset.py                只读流式数据集训练、断点与吞吐统计
cross_format_probe.py            20 个扩展名的只读跨格式准入探针
dataset_inventory.json           当前数据集清单与训练/保留范围
config.json                     模型、帧大小和运行限制
life_agent/
  byte_codec.py                 BLF1 协议、统一样本和流式观察
  agent.py                      通用帧交互、双半球与后台调度
  switching.py                  数据压力/偿还容量/梯度步自适应切换
  worker.py                     GPU 单写者队列、学习率和结构控制
  model.py                      可生长/剪枝的共享字节 Transformer
  memory.py                     通用字节样本、清洗、回放和训练债
  store.py                      v2 检查点及 v1 文本历史迁移
  cli.py                        文本与任意文件交互终端
tests/                          47 项单元、集成和 CUDA 测试
CONTINUOUS_VALIDATION.md        200 次混合交互与 20 MB 数据集持续验收报告
continuous_validation.py        可重复的持续验收器
ARCHITECTURE.md                  详细架构
ARCHITECTURE_AUDIT.md            双半球架构审计与并发修复
ARCHITECTURE_VALIDATION.json     16 项正式 CUDA 架构检查原始结果
AUDIT_ADAPTIVE_SWITCH.md         独立审查原文（保留历史发现）
AUDIT_CLOSURE.md                 B1–B7 关闭映射与残余边界
HANDOVER.md                      交接文档：当前状态、恢复步骤、风险
PSYCHOLOGY_MECHANISM_ANALYSIS.md 心理学机制映射与“不声称”边界
VALIDATION.md                    实测记录
state/                          交互运行后生成的权重、日志与备份
state_dataset/                  全量数据集训练状态（当前已停止，独立于交互状态）
```

单个编码帧默认上限为 1 MiB，由 `max_frame_bytes` 控制。`feed_frame()` 按上下文窗口流式覆盖完整帧，不是只学习末尾 512 字节。

## 检查点

- `state/latest.pt`：当前完整状态。
- `state/latest.backup.pt`：上一次成功保存的状态。
- `state/life.log`：滚动日志。

当前保存格式为 v2，历史记录以通用字节帧保存；读取 v1 时会把旧文本历史自动迁移为 UTF-8 帧，不需要清空权重。只加载自己生成的可信 `.pt` 文件。

## 可重复的在线学习验证

```powershell
.\.venv-life\Scripts\python.exe .\verify_learning.py --steps 48
```

该脚本使用非法 UTF-8 的原始字节载荷，固定同一半球，比较 `feed_frame()` 学习前后的同一帧损失和实际权重差。当前实测损失由 5.4817 降至 1.8011，下降 67.14%。它证明在线参数学习，不把这一结果包装成通用智能。

模型严格按单字节 ID 处理载荷：`0..255` 对应原始字节，`256..261` 仅用于 BOS/EOS/角色等协议控制。没有 BPE、SentencePiece、子词或文本分词器；BLF1 元数据同样以字节进入模型。

## 双半球架构复验

```powershell
& ".\.venv-life\Scripts\python.exe" ".\verify_architecture.py"
```

该脚本通过生产 CUDA 路径验证左右独立模型与优化器、单侧在线生成、另一侧离线巩固、TTT、完整切换周期、双向训练债和双侧检查点恢复。当前 16 项检查全部通过；完整周期后左右各 5 个训练步骤。它使用临时状态，不覆盖任何正式训练检查点。

“只有一个半球工作”的准确含义是只有一侧在线生成，被动侧仍会离线巩固。架构审计还修复了主线程结构调节与后台训练竞争的问题：现在所有运行时 resize 都进入 GPU 单写者队列，后台异常不再静默吞掉。详见 `ARCHITECTURE_AUDIT.md`。

## 测试

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv-life\Scripts\python.exe -m unittest discover -s .\tests -v
```

当前 **47 项测试全部通过**，另已完成同一正式 GPU 实例依次学习全 256 字节二进制文件和处理文本的冒烟验证、真实 JSONL 分块续训、16 项双半球 CUDA 架构检查，以及 200 次混合模态交互与 20 MB 真实数据的持续验收。

## 持续机制验收

```powershell
& ".\.venv-life\Scripts\python.exe" ".\continuous_validation.py" `
  --mixed-cycles 200 --real-bytes 20000000 --restart-every 50 `
  --output-dir ".\validation_runs\continuous-final"
```

该验收器轮流输入 RAW、UTF-8、RGB、PCM 和未知 codec，逐轮核对只有活动侧 CPU 生成、TTT 后权重变化、两侧训练和自动切换；每 50 轮关闭并从检查点恢复；最后读取 20 MB 真实数据并验证偏移续读与源文件未改。**只有全部判据为 true 才返回 0。**

2026-11-13 最终运行 `validation_runs/continuous-adaptive-final-1`：**21 项判据全部通过，退出码 0**。200 次混合交互发生 46 次自适应切换（46 条决策均可按 `pressure_growth >= learned_capacity` 复算，不是固定计数），左右各 5,520/5,701 个训练步骤；`step_accounting` 左侧 `train=4,860 / ttt=244 / consolidate=416`，右侧 `train=5,045 / ttt=292 / consolidate=364`；40 次 TTT 权重变化采样最小 L1 为 1.3143；两侧自然生长到 352/352，显式结构判定仍产生生长 `256→272` 与剪枝 `256→248`；真实 `O:\数据集` 读取 20,000,000 字节、26 帧，偏移续读通过，源文件大小与 mtime 未改；4 次周期恢复加末尾恢复全部为全字段指纹一致；RSS 423.37→1,297.66 MB（峰值等于最终值），GPU 2.92→47.56 MB，总显存 3,221,094,400 字节，均低于 6,144 MB 预算。

自适应切换由三项可测量共同决定：被动侧累计接收的非噪声数据质量、它自己学到的单批偿还容量、两侧真实梯度更新步数。没有 `switch_interactions`、`switch_interval_s` 或替代性的固定阈值。高密度合成输入下的切换频率由数据流入速度产生，不是调参目标；切换开销、长期稳定性与全量数据上的行为仍未证明。

该轮同时保留了对真实吞吐缺陷的修复：连续 `feed_frame()` 时，活动侧训练任务持续压过低优先级被动巩固，导致旧运行真实数据阶段左半球停在 955 步、右半球增长到 10,213 步。现在每次观察后显式排空当次被动巩固任务，`starvation_window_fraction=0.8571`，两侧均持续学习。

**边界不变：** 上述结果证明机制在可观测范围内真实运行，不证明语义理解、通用智能，也不等于全量数据集训练已经完成。


1. **统一接口已经实现，不等于通用理解已经形成。** 模型仍需从数据中学习各模态结构。
2. 冷启动输出接近随机字节，当前不是预训练多模态模型。
3. 原始 RGB/PCM 序列很长，统一接口不消除计算和上下文成本。
4. 统计清洗只降权，不再因长重复段直接拒绝样本；纯色图片和静音仍可入库。
5. 自训练可能学到错误规律，CRC 只保证容器完整，不保证语义真实。
6. “数字生命”是工程目标称谓，不代表意识或无限能力。

详见 [ARCHITECTURE.md](ARCHITECTURE.md) 与 [VALIDATION.md](VALIDATION.md)。
