# 全面架构重构与代码重写 Spec（v7 · 值自成 + 原理→代码映射，防未实现幻觉）

## Why

项目被叙事包装与幻觉实现污染（endogenous 影子模式、amygdala 死方法、硬编码清洗、未接线组件）。用户理想：海豚式换班在线学习、婴儿式噪声学习（数据无需清洗）、金发姑娘效应、杏仁核完整原理、参数自生长+理想容量、四档睡眠自适应组合、多模态输入打穿、按大脑解剖映射架构。**本版要点**：
1. 每个机制给出"原理→代码映射"（对应什么原理、落到哪个类/函数）。
2. 所有数值遵循"律固定，值自成"——**只定义律定域（物理上下界），域内值由反馈自调（autotune），不写死固定权重**。用户明确指定的数值（157.1%、10%→30%、5%→15%~5%、50%压力差）作为律定值保留。
3. 每个机制必须有调用方+测试，杜绝未实现幻觉。

## 铁律

1. **律固定，值自成**：设计只定"物理域/机制"，数值初值全部纳入 autotune 反馈自调；禁止在代码里写死无反馈的权重/阈值。
2. **每个机制必须有调用方 + 测试**，禁止"影子模式/预留接口/后续接入"。
3. 保持 M0 语义等价：既有 tests/ 断言只增不降；旧档 fed_state.pt 可加载。
4. 零外部重依赖（仅 torch + Python 标准库）。

---

## 具体实现方案（实施必须照此执行；【律定】=用户/物理指定，【初值】=纳入 autotune 自调）

### S1. 数据无需清洗（删硬编码过滤）
**原理→代码映射**：婴儿统计学习（Saffran 1996）——噪声靠暴露自动滤除；映射为删除人工清洗规则，噪声交给金发姑娘门控+带通+语义饱和。
**现状**：`datasets.py` `MIN_CHARS=30` 过滤短样本；`feed.py` 有碎片过滤分支；`tools/verify_new_sources.py` import `MIN_CHARS`。
**改法**：
1. 删除 `dolphin/datasets.py` 的 `MIN_CHARS` 常量及所有 `len(text) < MIN_CHARS` 过滤分支；`serialize()` 对任何非空样本原样返回 bytes。
2. 删除 `feed.py` 中"碎片过滤：序列化后短于 30 字符"整段。
3. `tools/verify_new_sources.py` 去掉 `MIN_CHARS` import，`--count` 只统计非空样本。
4. 短碎片/噪声完全由 `Dolphin.learn()` 的 `goldilocks_gate` 三态分流（太熟→skip、太怪→suspend 进冷层、中间带→pass）。
5. 清理语料白名单的运行时读取依赖（若有），`tools/verify_new_sources.py` 保留为人工工具。

### S2. 惊讶值链路（金发姑娘核心计算值）
**原理→代码映射**：金发姑娘效应（Kidd 2012）——惊讶值是"信息量"度量；映射为 `mean_nll → goldilocks_gate → band_w → score` 全链路。
**现状**：`Dolphin.learn()` 已接线 `mean_nll → goldilocks_gate`；`ExperienceBuffer.select()` 已用 `band()` 高斯带通。
**改法**：链路已存在，只做验证与补测试，不重写。
1. 新增单元测试：短文本/噪声/重复文本喂入，断言三态分流正确、带通权重单调（中间带>边缘）。
2. `dolphin.py` docstring 明确惊讶值链路。

### S3. 睡眠压力（多因素 + 对方训练进度换算）
**原理→代码映射**：下丘脑稳态调节——睡眠压力是稳态偏差信号；映射为 `Dolphin.hemi_pressure`（每半球一个压力值），由多因素综合并经对方训练进度换算。
**现状**：`ExperienceBuffer.pressure()`=最近 k 条 surprise 均值；`sleep_threshold` 初值 4.0；`_since_sleep` 计数；`hemi_trained_steps=[0,0]`。
**改法**：
1. `Dolphin.__init__` 新增 `self.hemi_pressure = [0.0, 0.0]`（随档持久化）。
2. 新增 `Dolphin.update_hemi_pressure()`：
   - 输入：`buffer.pressure()`（数据量/惊讶度积累）、`debt_ledger.debts`（债务）、`hemi_trained_steps`（对方训练进度）、`_since_sleep`。
   - 公式（**初值**，权重/系数纳入 autotune 慢调域，律定域 [0,1]）：
     `raw_i = buf_p*w1 + debt_i*w2 + since_sleep*w3`，`pressure_i = raw_i * (1 - k * opp_progress_i)`。
   - `opp_progress_i = trained[1-i] / max(1, sum(trained))`——对方训练进度占比，进度高→本侧压力小。
   - 初值 `w1=0.5, w2=0.3, w3=0.2, k=0.3`，全部注册进 autotune 慢调（律定域内值自成）。
3. 触发：`maybe_sleep()` 除 `buffer.pressure() >= sleep_threshold` 外，新增 `max(hemi_pressure) >= threshold`。
4. 存档：`save()` 增加 `"hemi_pressure"`；`load()` 用 `ck.get("hemi_pressure", [0.0,0.0])`。
5. 测试：对方进度高→本侧压力下降（单调性断言）。

### S4. 双半球压力比较与临时中断换班
**原理→代码映射**：海豚 USWS 两侧交替的"需求侧优先"——压力失衡时打破固定节奏，资源让给高压力方。
**现状**：`life.run_cycle(feeding=True)` 开头做 swap-first。
**改法**：
1. feeding 路径 swap-first 前加判定（**律定阈值**：50%）：`interrupt = sleep_p > awake_p * 1.5`。
2. 若 `interrupt`：跳过 swap-first，直接以高压力方为睡脑训练；训练前调用 `debt_ledger.settle()` 提前结算债务。
3. 报告写 `report["pressure_interrupt"]=True` 与两半球压力值。
4. 仅 `feeding=True` 生效，`feeding=False` 保持 M0 语义（测试等价性不破）。

### S5. 双半球债务账本 + 胼胝体通道（互为蒸馏防分裂）
**原理→代码映射**：胼胝体是两半球信息传递通道（裂脑→认知割裂）；映射为 `DebtLedger`——债务互相结算（耦合）+ 蒸馏方向交替（互为师生）+ KL 健康度观测。
**新增** `dolphin/debt.py`：
```python
class DebtLedger:
    def __init__(self):
        self.debts = [0.0, 0.0]     # 每半球欠对方的训练债务（字节当量）
        self.notes = [[], []]       # 蒸馏笔记 eid 列表（胼胝体通道载荷）
        self.callosum_kl = None      # 最近一次两半球输出分布 KL（通道健康度）
    def to_state(self): ...
    def from_state(self, st): ...
    def settle(self): ...           # 结算：合并/清零未消化债，返回结算记录
```
**接线**：
1. `Dolphin.__init__` 挂载 `self.debt_ledger = DebtLedger()`。
2. 换班记账（`life.run_cycle` 换班分支）：
   - 新半球记"未消化旧数据"债：`debt += sum(len(e.data) for _,e in sel) * ALPHA_LOSS`，`ALPHA_LOSS`【初值 0.4，纳入 autotune 慢调】。
   - 旧半球记"服务期新数据"债：`debt += _since_sleep 期间 learn/serve 的近似字节量`。
3. 还债：训练步数 `steps = base_steps + int(debt * BETA_REPAY)`【初值 0.05，autotune 自调】；训练后 `debt *= (1 - GAMMA_REPAY)`【初值 0.5，autotune 自调】。
4. **互为蒸馏**：当前 KD"睡脑←醒脑"，换班后角色互换天然交替；`run_cycle` 记录 `report["teacher"]` 并 docstring 声明交替性。
5. **防分裂观测**：换班后计算两半球对 probe 卷面前 8 块输出的 softmax 分布 KL 散度均值，写 `report["callosum_kl"]` 与 `debt_ledger.callosum_kl`。
6. 存档：`save()` 增加 `"debt"`；`load()` 用 `ck.get("debt")` 缺省兼容。
7. 测试：记账/还债/持久化/旧档兼容/KL 输出。

### S6. 四档睡眠 + 睡眠编制计算 + 调度
**原理→代码映射**：人类睡眠分期（N1/N2/N3/REM）——不同阶段不同职能；映射为 `run_cycle` 四档 mode + `compute_sleep_program` 编制函数（下丘脑调度）。
**现状**：`run_cycle(mode="deep"|"light"|"rem")`；`sleep_steps_light=16`、`sleep_steps_rem=40`、`sleep_steps=120`。
**改法**：
1. 新增第四档 `mode="nrem"`（N2 浅睡）：`Dolphin.sleep_steps_nrem=80`【初值，autotune 域 (40,120)】；`autotune._DOMAINS` 增加 `"sleep_steps_nrem":(40,120)` 并同步回属性。
2. 四档语义（职责对齐用户设计）：
   - `light`（N1 轻浅睡）：步数域 [8,40]；KD 每 4 步一次；跳过 ReDo；快通道 top-1；恒温器 observe_only；param_adaptive 不介入。职责=快速过数据+微调理想容量+平衡债务。
   - `nrem`（N2 浅睡）：步数域 [40,120]；KD 正常；恒温器允许容量调整（hold/decay/micro_tune，禁 born_again 重手术）。职责=轻训练+较大块调整理想容量。
   - `rem`（REM）：步数域 [20,80]；增强做梦注入（dream_k=min(8,dream_k+2)）；恒温器允许生长判决。职责=记忆回访+巩固+生长真实参数。
   - `deep`（N3 深度）：步数域 [120,400]；全量训练+ReDo 回收/软衰减（神经裁切）+恒温器全判决。职责=大量数据整理+降本增效。
3. **睡眠编制计算**：`life.py` 新增模块级函数 `compute_sleep_program(dolphin)`：
   - 输入：`hemi_pressure`（归一化压力 p）、`debt_ledger.debts`（债务比）、`buffer` 数据量。
   - 输出：`{"deep":..., "rem":..., "nrem":..., "light":..., "total_steps":...}`。
   - 单调性要求（**律定约束**）：压力 p 越大 → `deep` 占比单调上升、`total_steps` 单调变长；轻浅睡占比用于平衡债务。
   - 具体系数为**初值**，纳入 autotune 慢调（域内值自成），但必须满足上述单调性约束。
4. 调度器：`run_cycle` 开头调用 `compute_sleep_program`，按占比轮盘选本轮 mode；`total_steps` 作为步数上限并经债务加权；`report["sleep_program"]` 输出编制结果。
5. 测试：压力升高→deep 占比/总时长上升（单调性断言）。

### S7. 杏仁核记忆优先化 + 岛叶内省信号
**原理→代码映射**：杏仁核调制记忆巩固（McGaugh 2013）+ 岛叶内省——情绪唤醒增强显著记忆；映射为 `Amygdala.salience_score`（显著性信号）接入选拔/记忆。
**现状**：`amygdala.py` 死方法 `read_tone/salience/emotion_tag`；采样托管已接线。
**改法**：
1. 删除死方法。
2. 新增 `salience_score(surprise)`（维护 `surprise_ema`）：
   - `base = surprise_ema or surprise`；`score = clamp(|surprise-base|/max(|base|,eps)/GAIN, 0, 1)`，`GAIN`【初值 1.0，autotune 自调】。
   - 这是"岛叶内省信号"：模型对自身预测偏差越大越显著。
3. 接线（`life.run_cycle` 训练前）：对 `sel` 中每条经验用 `salience_score(e.surprise)` 重打分 `score *= (1 + BOOST * salience)`【BOOST 初值 0.5，autotune 自调】后重排。
4. 记忆优先化：高显著（sal > THRESH【初值 0.7，autotune 自调】）且未入选预算的经验 `memory.add(..., kind="note")` 优先保留。
5. 采样托管保留（temperature/top_k/top_p）。
6. 测试：显著性上调选拔权重、高显著未入选入 note。

### S8. 多模态输入适配（modal_input.py 新增，零重依赖）
**原理→代码映射**：婴儿统计学习对声音流/视觉场景用同一机制（Saffran 1996；Fiser & Aslin 2002）——多模态以原始字节进入同一条链路，金发姑娘统一去噪。
**新增** `dolphin/modal_input.py`（仅标准库）：
```python
def encode_image_bytes(raw: bytes) -> bytes: return raw   # 原始字节直通
def encode_audio_bytes(raw: bytes) -> bytes: return raw   # 原始字节直通
def sample_video_chunks(raw: bytes, chunk=4096, max_chunks=64) -> list[bytes]:
    # 零依赖"自适应抽帧"：分块后由金发姑娘带通在链路中自然筛选信息量大的块
    return [raw[i:i+chunk] for i in range(0, len(raw), chunk)][:max_chunks]
```
**关键设计**（修正此前幻觉）：零重依赖下不做像素/PCM 解码承诺。图片/音频文件以**原始字节**进入链路（模型靠暴露自学统计结构）；视频"自适应抽帧"落地为**分块 + 金发姑娘带通**（信息量大的块自然高分、重复/静止块低分）。`feed.py` 新增 `--input-type text|image|audio|video`，读取文件后按类型调用 `modal_input`。`Dolphin.learn()` 已接收 bytes 无需改签名。

### S9. 参数自生长 + 理想容量三档下调
**原理→代码映射**：工程机制（无直接生物对应）——利用率/幽灵增益驱动生长（widen）+ 显存压力触发收缩；映射为 `param_adaptive._schedule_shrink` 按显存占用分档。
**现状**：`PARAM_LIGHT_LIMIT=1.571`；`PARAM_STAGE_RATIOS=(0.30,0.10,0.03)`；`detect_pressure()` 用显存/容量压力。
**改法**：
1. 理想容量=真实参数×157.1%（**律定**，用户指定）作为生长上限目标。
2. 三档下调（**律定档位**，用户指定）：显存占用 `>0.90`（可用<10%）→ 理想容量 30%；`>0.95`（可用<5%）→ 理想参数 10%（落在 15%~5%）。修改 `_schedule_shrink` 目标计算按 `mem_ratio` 分档。
3. 保留 `MIN_BODY_FRAC` 容量下限与体检门控验收。
4. 测试：mock `mem_ratio=0.92→target≈0.30`；`0.97→≈0.10`。

### S10. 幻觉堆清理
1. 删除 `dolphin/endogenous.py`。
2. `dolphin.py`：删 `from .endogenous import Endogenous`、`self.endogenous`、`save/load` 的 `"endogenous"` 字段、`endo_alpha/beta/gamma` 属性及同步。
3. `life.py`：删 `from .endogenous import compute_signals, select_old_memory` 及 run_cycle 中"内生健康度（影子模式）"整段。
4. `autotune.py`：删 `endo_alpha/beta/gamma` tunables、`_DOMAINS` 条目、`endo_zero_rate` 信号及调节分支。
5. `amygdala.py`：删死方法。

### S11. 反功能吹捧扫描（作为测试断言）
新增 `tests/test_no_dead_code.py`：用 `ast` 遍历 `dolphin/` 收集公共函数/方法名，检查每个公共方法至少有一个调用方（排除 `__init__`/`to_state`/`from_state` 等反射调用），断言死方法列表为空。纳入常驻套件。

---

## 设计依据（联网科学查证，摘要）

见 v6 表 1（海豚 USWS/婴儿统计学习/金发姑娘/睡眠分期/突触稳态/REM 整合/语义饱和/杏仁核记忆调制）与表 2（枕叶/颞叶/角回/岛叶/胼胝体/丘脑/下丘脑/上丘/下丘/RAS 映射）。

## ADDED Requirements（验收场景，均绑定 S 编号）

- **S3**：多因素睡眠压力（数据量/债务/积累+对方训练进度换算）；压力越高深度占比越多、睡眠越长。
  - WHEN 对方半球训练进度高 → THEN 本侧换算压力下降。
- **S4**：睡侧压力 > 醒侧 ×1.5 → 临时中断默认轮换、settle 结算债务、报告写 `pressure_interrupt`。
- **S8**：图片/音频原始字节进入链路；视频分块+金发姑娘带通自适应抽帧；金发姑娘对所有模态同一套逻辑。
- **S7**：杏仁核显著性信号上调选拔/记忆权重；高显著未入选经验入 note 优先保留。
- **S9**：理想容量=真实参数×157.1%；显存 >0.90→30%、>0.95→10%。
- **S6**：四档睡眠职责对齐；`compute_sleep_program` 计算档位占比/总时长并写入报告；压力升高→deep 占比/时长上升。
- **S5**：蒸馏方向交替（互为师生）、债务互相结算、换班后输出 `callosum_kl`。
- **S1**：移除硬编码过滤，脏数据/短碎片由金发姑娘门控自然分流。
- **S11**：`tests/test_no_dead_code.py` 断言死方法列表为空。

## MODIFIED Requirements

- **life.run_cycle**：保留 M0 语义，扩展四档睡眠+债务记账+压力比较与临时中断+睡眠编制+防分裂观测+杏仁核显著性加权。
- **goldilocks.py**：保留三态门控与 autotune 阈值；惊讶值为核心计算值；承担噪声处理。
- **amygdala.py**：保留采样托管；删死方法；新增 `salience_score` 记忆优先化。
- **param_adaptive.py**：保留三段式与手术复用；档位对齐 S9。

## REMOVED Requirements

- 硬编码碎片过滤（MIN_CHARS=30）：删除，由金发姑娘门控+带通+语义饱和处理。
- endogenous.py 影子模式：删除文件及全部接入。
- 语料白名单运行时依赖：移出链路，tools 保留人工工具。
- 叙事性文档内容：去叙事化，脑区映射作为附录。
- 任何未接线的"影子/预留/后续"组件：反功能吹捧扫描确保不存在。

---

## 追加订正：记忆库自适应睡眠机制重构（S12）

本追加订正记录"记忆库模块睡眠机制全面重构"（S12）的设计要点与参数幻觉清理说明。

### 设计要点

1. **模型规格自适应（`Dolphin.memory_scale()`）**
   - 位置：`dolphin/dolphin.py`。
   - 公式：`s = (log10(max(param_count, 1e5)) - 5.0) / 3.0`，钳位到 `[0.1, 1.0]`。
   - 语义：小模型（≈1e5 参数）→ 0.1，避免记忆处理过载导致上下文溢出；大模型（≈1e8 参数）→ 1.0，充分利用处理能力。

2. **四层睡眠冷层遍历预算（`life.run_cycle` feeding 路径内实现）**
   - `light`（超浅度）：`0` = 跳过全量冷层遍历，仅热层快速通道。
   - `nrem`（普通浅度）：`max(8, int(64×scale))` = 极小部分冷层。
   - `rem`（RAM 回放）：`max(8, int(128×scale))` = 自适应小幅遍历。
   - `deep`（深度）：`int(cold_len × (0.2 + 0.8×scale))` = 深度遍历。

3. **多层组合睡眠（加权组合）**
   - `cold_sample = Σ(frac_i × budget_i) / Σfrac_i`，按 `compute_sleep_program` 返回的四档占比（>0 者）加权。
   - 一个周期内的记忆处理规模 = 各档位占比加权组合，与 `sleep_program` 占比一致。
   - 输出写 `report["memory_scale"]`（4 位小数）与 `report["memory_cold_sample"]`。

4. **统一抽样版记忆处理（feeding 生产路径）**
   - 所有档位统一走 `resurrect_light(cold_sample=...)` / `dream_light(queries, k=..., cold_sample=...)`。
   - `feeding=False`（M0 单线程路径）保持原全量 `resurrect()` / `dream()` 语义完全不变（既有测试依赖）。
   - `cold_sample=0` 时 `resurrect_light`/`dream_light` 内部抽样集合为空 → 仅热层，实现超浅度语义。

5. **参数幻觉清理（历史「≈57 百万参数」引用）**
   - 项目历史中「≈57 百万参数」是对 `Config(d_model=768, n_layers=8, n_heads=8, block_size=256)` 的**估算幻觉引用**，并非真实容量上限；该配置本体保留不动（它是历史默认出生档/大模型档）。
   - 已清理 `feed.py`（--model 帮助文本与 make_cfg 注释）、`dolphin/life.py`（R1 注释）、`tests/test_conformance.py`、`架构设计.md`、`验收报告_用户体验.md` 中的全部该幻觉引用，统一替换为中性表述（`大模型` / `历史默认出生档（d_model=768×8 层）`）。
   - 最终 Grep 检查（57 后接 M/m/mb 或 5700 万等形态）在代码与文档中零命中（排除 `.git` / `实验记录/` / `datasets/` / `__pycache__` / `*.pt` / `*.jsonl`）。

6. **测试**
   - `tests/test_acceptance_scenarios.py` 追加：`test_model_scale_monotonic`、`test_sleep_cold_sample_by_mode`、`test_resurrect_light_cold_sample_zero_only_hot`、`test_no_model_size_hallucination`、`test_sleep_program_frac_weighted_cold_sample`。
   - 验证命令：`python -m pytest tests/ -q` 全量通过；`python smoke_test.py` 通过。