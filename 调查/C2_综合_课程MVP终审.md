# C2 综合终审：课程 MVP 算法选型（RHO-LOSS / ProCuRL / PACED）

> 终审 Agent：C2（代码+数据调查）｜依据：本组 `调查/C2_代码_课程化喂食可行性.md` 的数据源实测事实，
> 对网络组 `调查/N1_网络_最近发展区与脚手架.md`、`调查/N2_网络_睡眠重放与记忆整合.md` 推荐的三个算法做可行性终审。
> 只读代码与数据；未运行任何训练/喂食/测试；未触碰生产资产。

---

## 一、结论先行

**MVP 最终选型：ProCuRL 式"能力-难度匹配"，但用「元数据难度档位」做难度轴、「免费惊讶度 EMA」做能力轴、「L4 带通」做目标带——即"Band-Tier Matching（带通×档位匹配）"。**

三算法终审一句话：

| 算法 | 网络组主张 | 数据源事实裁决 | 是否进 MVP |
| --- | --- | --- | --- |
| **RHO-LOSS**（模型 NLL 定义难度，选"可学×值得学×未学"） | N1 候选 B：喂前用 `mean_nll` 对候选样本算 NLL，只送中间带 | 完整版需**喂前预打分**，成本放大 `1+K/B` 倍（C2 实测估算：GTX 1060 3GB / 57M / block_size=256 下，单样本 1-8 块前向 ≈ 15-240ms）；但其"中间带优先"判据已被 L4 `band()` 承载 | **取其判据，不取其成本**——不进 MVP 的预打分部分 |
| **ProCuRL**（能力-难度匹配，选"既不太难也不太易"的任务） | N1：与 L4 同构，选样判据=能力差带 | 数据源**恰好有现成难度档位**（CMATH `grade`、GSM8K 步骤数、SynLogic easy/hard、LogiConBench 语句数），能力轴可用 `learn()` 已免费算出的 surprise EMA——匹配成本为零 | **✅ MVP 核心**（做廉价化改造） |
| **PACED**（两阶段前向-反向 KL，按学生通过率 p(1−p) 加权） | N1 候选 C：教师退出调度 | 需要**多次 rollout 统计通过率**，对字节级生成模型成本高；且它调度的是**训练侧 KD 权重**，不是喂食侧选料 | **不进 MVP**；留作后续 KD 退火实验（与 N1 候选 C 合并） |

**为什么是 ProCuRL 式而非 RHO-LOSS 完整版**：RHO-LOSS 的"难度=模型 NLL"在喂前需要对候选样本逐一前向。
而我的数据源调查（C2 报告 §三）证明：**四个训练源已经自带或可廉价派生出难度标签**——
CMATH 每行有 `grade`（1-6 各 100 条，人工标注）、GSM8K 英文 `answer` 字段的 `<<...>>` 步骤数可数（实测 1-9 步）、
SynLogic 目录本身就是 easy/hard、LogiConBench 文件名就是 2/3/4/5 语句。
这些标签比模型 NLL 更便宜、更可解释、更稳定（NLL 会被生僻字符/格式噪声干扰，N1 自己也承认）。
而模型能力轴完全不需要额外前向：`d.learn()` 在 `dolphin/dolphin.py:205` 已经为每条样本算了 `mean_nll`，
调度器读这个免费信号做 EMA 即可。

**MVP 定义**：`feed_from` 增加可选 `curriculum=None` 参数；启用时按「当前能力 EMA 相对 `buffer.band()` 中心±宽度的位置」决定下一批从哪个难度档取记录。默认 `None` 时行为与现在完全一致（现有测试全绿）。

---

## 二、三算法终审对比（结合数据源事实）

### 1. RHO-LOSS：判据正确，成本不必要

- 原理：用模型当前 loss 给候选样本打分，选"可学（reducible）× 值得学（loss 高）× 未学（novelty）"。
- 数据源事实：`awake().model.mean_nll(data, device)`（`dolphin/model.py:119-137`）就是现成的 NLL 难度信号，每次喂食已经在算（`dolphin.py:205`）。
- 裁决：
  - 若**喂前**对候选样本逐条算 NLL（N1 候选 B 的做法），成本 = 每条候选一次 `mean_nll`。C2 实测估算：单块 (1,256) 前向 ≈ 29 GFLOP，1060 上约 15-30ms/块，最多 8 块 → 长样本最坏 120-240ms。若每 B 条喂食决策一次、每次打 K 个候选，开销放大 `1+K/B`：K=8、B=64 → 1.125 倍；K=8、B=8 → 2 倍。这会与睡脑训练抢 GPU（1060 3GB 上双半球+优化器已占 ~1.0GB）。
  - 但 **L4 带通已经是 RHO-LOSS 的"中间带优先"简化版**（N2 也确认这一点），`experience.band()`（`experience.py:56-72`）免费提供中心±宽度。
  - **结论**：MVP 不实现 RHO-LOSS 的喂前预打分；只借用其"聚焦中间带"判据，由 L4 带通 + 元数据档位共同实现。

### 2. ProCuRL：能力-难度匹配，与数据源天然契合 ✅

- 原理：把任务按难度分档，按智能体当前能力只选"跳一跳够得着"（ZPD）的任务，最大化学习进度。
- 数据源事实：**难度档位是现成的**：
  - CMATH：`grade` 1-6 人工标注（`cmath_dev.jsonl` 实测每档 100 条，共 600），另含 `reasoning_step` 1-5、`num_digits` 可作档内细分。
  - GSM8K：英文 `answer` 的 `<<...>>` 步骤数实测分布良好（train 7,473 条：1 步 404 / 2 步 2175 / 3 步 2137 / 4 步 1424 / 5 步 785…），可按步骤数分桶。
  - SynLogic：`synlogic_easy`（15,837）/ `synlogic_hard`（32,840）目录即难度档。
  - LogiConBench：`2/3/4/5statements.jsonl` 文件名即推理链长度档。
- 裁决：ProCuRL 的"能力-难度匹配"落到海豚 = **能力轴（surprise EMA，免费）× 难度轴（元数据档位）× 目标带（L4 band）**。三者全部现成或零成本可得，是三个算法里唯一不需要新增前向计算/rollout 的。
- **结论**：MVP 采用 ProCuRL 式匹配的廉价实现。

### 3. PACED：训练侧 KD 调度，不是喂食侧选料

- 原理：按学生通过率 p 加权 w=p(1−p)，把蒸馏聚焦在 ZPD 前沿；两阶段前向-反向 KL。
- 数据源事实：通过率需要模型对题目多次 rollout 判对错——字节级生成模型没有现成"对/错"信号（CMATH/GSM8K 有 golden 答案，但 rollout 判分成本高且不稳定）。
- 裁决：PACED 调度的是 `kd_alpha`（蒸馏权重），属于**训练侧**机制（N1 候选 C：教师退出调度），与"喂什么数据"正交。可作为后续实验，但**不是课程 MVP**。
- **结论**：不进 MVP；若后续做 KD 退火，可参考 PACED 的"聚焦 ZPD 前沿"思想，但需要用 NLL 阈值近似通过率（N1 自己也承认需简化）。

---

## 三、MVP 课程调度器最终选型

### 算法：Band-Tier Matching（带通×档位匹配）

```
每批喂食决策（例如每 32 条）：
  1. 能力估计：ability_ema ← α·recent_surprise + (1-α)·ability_ema   # surprise 来自 d.learn() 免费输出
  2. 目标带：center, width ← d.buffer.band()                          # 现有 L4，免费
  3. 档位选择：
       if ability_ema < center - width:  当前档太简单 → 难度档 +1
       elif ability_ema > center + width: 当前档太难  → 难度档 -1
       else:                             维持当前档
  4. 从当前 (source, difficulty) 档位的记录游标处取下一批，喂入 d.learn()
```

- **难度轴**：数据源自带档位（见 §四 解析器改造）。
- **能力轴**：surprise EMA（零额外前向，因为 `learn()` 已在算）。
- **目标带**：`buffer.band()`（`experience.py:56-72`）——这就是 ProCuRL 的"ZPD 区间"的工程形态，也是 RHO-LOSS "中间带优先"的判据。
- **与 L4 的关系**：课程调度是**事前选料**（喂什么候选），L4 带通是**事后选拔**（睡眠周期选什么训练）。两层都聚焦中间带，但课程层管"投喂分布"，L4 管"训练分布"，不冲突。调度器不绕过带通（带通仍会过滤）。
- **防塌缩**：每 N 批注入 5% 随机档位样本（带外探索），防止缓冲惊讶度分布变窄导致 band 塌缩（C2 报告 §七 风险清单）。

### 为什么这个选型（一句话）

**因为数据源已经有难度标签、模型已经免费产出惊讶度、L4 已经给出目标带——三个算法里只有 ProCuRL 式匹配能把这三者零成本拼起来；RHO-LOSS 的预打分和 PACED 的 rollout 都是在为"其实已经有标签"的东西付额外前向成本。**

---

## 四、解析器改造清单（`dolphin/datasets.py`）

> 原则：每个解析器 yield 的 dict 增加一个 `"difficulty"` 键；`serialize()`（`datasets.py:252-255`）只拼 prompt/answer，**自动忽略多余键，输出文本不变**——因此对现有喂食/测试零破坏。

| 解析器 | 位置 | 新增 `difficulty` 字段 | 来源 |
| --- | --- | --- | --- |
| `read_cmath` | `datasets.py:129-135` | `rec.get("grade")`（1-6 整数） | CMATH `cmath_dev.jsonl` 自带 `grade` 字段（人工标注年级） |
| `read_gsm8k` | `datasets.py:138-149` | 步骤数 = 英文 `answer` 中 `<<...>>` 的计数（可再分桶：1-2 易 / 3-5 中 / 6+ 难） | 英文 `answer` 字段（文件里本来就有，当前被丢弃；`answer_zh` 已无 `<<>>` 标记） |
| `read_synlogic` | `datasets.py:203-233` | `0`（synlogic_easy）/ `1`（synlogic_hard） | 子目录名（解析器已按 easy→hard 顺序遍历，`datasets.py:214`） |
| `read_logiconbench` | `datasets.py:108-124` | 语句数 = 文件名首字符（`2/3/4/5`） | 文件名（解析器已按 2→5 顺序遍历，`datasets.py:114-115`） |
| `read_metamath` | `datasets.py:152-160` | **暂不加**（或 `None`） | AI 生成增强，难度标签不可靠；留作后续实验 |
| `read_logiqa` / `read_logiqa2` | `datasets.py:78-89` / `92-105` | **暂不加**（或 `None`） | 无显式难度标注；可用题干长度做启发式，但 MVP 不做 |
| `read_distil` / `read_coig` | `datasets.py:165-179` / `182-192` | **不加**（或 `None`） | 无难度标注；distil 的 reasoning 长度可做启发式，MVP 不做 |
| `serialize` | `datasets.py:252-255` | **不改** | 只拼 prompt/answer，多余键自动忽略 |
| `iter_source` | `datasets.py:258-262` | **不改** | 按名取解析器 |

**MVP 只用四个源做档位调度**：`cmath`（人工年级最可靠）、`gsm8k`（步骤数可派生）、`synlogic`（程序化 easy/hard）、`logiconbench`（语句数）。其余源在课程启用时按 `difficulty=None` 处理（不参与档位切换，按原顺序喂）。

---

## 五、需要先做实验验证的开放问题（最多 3 个）

### 开放问题 1：元数据档位与模型惊讶度是否单调相关？（决定档位轴是否可信）

- **为什么**：Band-Tier Matching 假设"档位越高 → 模型惊讶度越高"。若不成立（如 CMATH grade 1 的题因生僻词反而惊讶度高），档位轴就需要用惊讶度分位数重标定。
- **最小实验设计**：
  - 数据：`O:/数据集/03_数学推理/CMATH/cmath_dev.jsonl`（600 条，grade 1-6 各 100）。
  - 跑法（未来实验，本次未运行）：用 `small` 或 `seed` 模型（d128 或 d32，CPU 可跑），对 600 条逐条算 `awake().model.mean_nll`，按 grade 分组求均值。
  - 判据：Spearman 秩相关 ρ(grade, 平均 surprise)。**ρ ≥ 0.3** → 档位轴可信，直接采用；**ρ ≈ 0** → 放弃元数据档位，改用"惊讶度分位数分桶"作难度轴。

### 开放问题 2：Band-Tier Matching 是否真的优于线性喂食？（A/B 验证）

- **为什么**：N1 引用的 `When Do Curricula Work?`（ICLR 2021）指出显式课程在标准基准上收益微弱，海豚的带通选拔本身已是"动态训练集"——需要验证排序层的边际收益。
- **最小实验设计**：
  - 数据：CMATH dev（有 grade，可做档位调度）600 条。
  - 跑法（未来实验）：两条手臂，各用 `seed` 模型 + 隔离冷层/存档：
    - A 臂（对照）：`python feed.py --sources cmath --model seed --limit 600 --no-guard`（线性 grade 1→6）。
    - B 臂（课程）：同参数但启用 `curriculum`（Band-Tier Matching）。
    - 注意：两条手臂需**独立存档路径**（当前 `FED_STATE` 硬编码，实验前需复制/备份 `dolphin/fed_state.pt`，或后续给 feed.py 加 `--state` 参数）。
  - 判据：跑相同睡眠周期数后比较 `d.probe_loss(awake())`（`dolphin.py:116`）与体检通过换班次数；另记录"入选训练样本的惊讶度落在 band 内比例"，课程组应显著更高。

### 开放问题 3：课程调度会不会导致缓冲多样性塌缩（band collapse）？

- **为什么**：若调度器只喂单档，缓冲惊讶度分布变窄 → `band()` 宽度塌缩 → L4 选拔失衡（C2 报告 §七 风险）。需要实测塌缩速度并验证 5% 随机注入缓解有效。
- **最小实验设计**：
  - 数据：CMATH dev + SynLogic easy/hard 各抽 200 条（混合档位）。
  - 跑法（未来实验）：用 `seed` 模型跑 50 个睡眠周期，每周期记录 `buffer.band()` 的 center/width（`experience.py:56-72`）与入选样本的档位分布熵。
  - 判据：50 周期后 band width **不低于初始值的 50%**，且入选档位熵 > 1.0（至少两个档位在训练）。若塌缩，开启 5% 随机注入后重复，验证 width 恢复。

---

## 六、机制映射表（简版）

| 人类机制 | 工程对应物（终审后） | 证据分级 | 落地成本 |
| --- | --- | --- | --- |
| 父母选"跳一跳够得着"的教材 | Band-Tier Matching：元数据难度档 × 免费 surprise EMA × L4 带通 | 同构（ProCuRL 能力-难度匹配 + RHO-LOSS 中间带判据，两者在文献中同向） | 小（MVP） |
| 难度渐进（一年级→六年级） | CMATH grade / GSM8K 步骤数 / SynLogic easy-hard / LogiConBench 语句数 | 同构（数据自带难度信号） | 小（解析器加键） |
| 教师逐步退出 | PACED 式 KD 退火（`kd_alpha` 反馈化） | 类比（训练侧，与喂食侧正交） | 中（后续实验） |
| 孩子能力在变化 | surprise EMA 作为模型能力估计，随喂食免费更新 | 同构（`learn()` 已算 mean_nll） | 零 |

---

## 七、来源

- 本组数据调查：`调查/C2_代码_课程化喂食可行性.md`（数据源难度信号清单、成本估算、接入点、测试设计）
- 网络组：`调查/N1_网络_最近发展区与脚手架.md`（RHO-LOSS/ProCuRL/PACED 三候选）、`调查/N2_网络_睡眠重放与记忆整合.md`（L4≈RHO-LOSS 简化版、learnability 第三因子）
- 代码：`feed.py:269-321`（feed_from）、`dolphin/datasets.py:78-262`（各解析器/serialize/iter_source）、`dolphin/dolphin.py:194-212`（learn）、`dolphin/dolphin.py:205`（mean_nll 调用）、`dolphin/model.py:119-137`（mean_nll）、`dolphin/experience.py:56-72`（band）
- 数据实测：`O:/数据集/03_数学推理/CMATH/cmath_dev.jsonl`（grade 1-6 各 100）、`O:/数据集/03_数学推理/GSM8K_zh/GSM8K_zh.json`（train 7,473 条，英文 answer `<<>>` 步骤 1-9 分布）、`O:/数据集/05_推理指令/SynLogic/`（easy 15,837 / hard 32,840）、`O:/数据集/06_LogiConBench/LogiConBench-main/`（2/3/4/5 statements）