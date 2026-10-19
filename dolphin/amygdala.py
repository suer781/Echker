"""杏仁核（Amygdala）——由系统世界信号驱动的生成参数自动托管（2026-10-08）。

设计类比：杏仁核是大脑中快速识别威胁并触发应激反应的区域，威胁解除后
恢复平静。本组件将该机制应用于系统：用**系统自身的世界信号**
（体检 margin / 回滚率 / KD 蒸馏比 / 记忆命中率）自动调节 serve 的生成参数：

- 威胁高 → 降低温度（生成更保守、更稳定、可预测，减少幻觉/漂移）、
  收紧 top_k/top_p（采样池更小，输出更聚焦）；
- 威胁低 → 升高温度（生成更有创造性、更丰富）、放松 top_k/top_p
  （采样池更大，输出更多样）；
- 温度物理域 [0.4, 1.4]（低于 0.4 接近贪心重复，高于 1.4 近乎无意义输出）；
- top_k 物理域 [1, 100]（0=不启用）；top_p 物理域 [0.5, 1.0]（1.0=不启用）；
- 威胁高时自动启用并收紧 top_k/top_p（从不启用进入启用，再逐步收紧）；
  威胁低时若未启用则保持不启用（宽松/创造性），若已启用则放松；
- 慢环 EMA 微调（每 AMYGDALA_EVERY=20 个睡眠周期调整一次，±5% 步长，防振荡）
  ——与 autotune 同节奏。

设计原则（承袭项目哲学"律固定，值自成"）：
1. 信号全部来自系统自身运行统计，不来自日历、不来自人工；
2. 慢：每 AMYGDALA_EVERY 周期只走极小步长（±5% 起，威胁越高步长越大，
   但不超过 7.5%），以稳定优先；
3. 有界：所有被调量钳位在物理域内（域本身是律，域内值自成）；
4. 可观测：每次调整写入睡眠报告（m4_amygdala 字段），ward 可查；
5. 持久化：Amygdala 状态随 Dolphin 存档（save/load 透传），重启后恢复；
6. 不破坏既有测试：AMYGDALA_EVERY=20 慢环保证短测试（几周期）永不触发
   adjust，Dolphin 新增属性不影响任何断言。
"""
import torch

# 每多少个睡眠周期调整一次（慢环；测试周期数远小于此，永不触发）
AMYGDALA_EVERY = 20
# 温度物理域（律定域：域本身是律，域内值自成）
_DOMAIN = (0.4, 1.4)
# top_k 物理域（0=不启用；托管时低于 1 则置 0=不启用，保持旧行为）
_DOMAIN_K = (1, 100)
# top_p 物理域（1.0=不启用；托管时低于 1 才调节）
_DOMAIN_P = (0.5, 1.0)
# 单次调整基础相对步长（±5%，防振荡；威胁越高步长越大，上限 5%×1.5=7.5%）
_STEP = 0.05
# 威胁等级的中性阈值：高于此判定为"高威胁"（降温收紧），低于则"低威胁"（升温放松）
_THREAT_MID = 0.5
# 语气/显著性（2026-10-13）：全部由 surprise（=mean_nll）内生派生，不硬编码任何词。
# surprise 相对自身滚动基线的相对偏离，经 _TONE_GAIN/_SALIENCE_GAIN 缩放后钳位。
_TONE_GAIN = 1.0        # 相对偏离为 1（即 surprise=2×基线或 0）时语气饱和到 ±1
_SALIENCE_GAIN = 1.0    # 相对偏离幅度为 1 时显著性饱和到 1
_TONE_EMA = 0.1         # 语气基线的 EMA 系数（慢漂移，防单条抖动）
_SURPRISE_EPS = 1e-6    # 浮点守卫：基线为 0 时避免除零

# 用户情绪感知（2026-10-17 用户定案：**语义理解判断，不靠关键词硬编码**）。
# 主路径：用模型自身的语义表示（ByteTransformer.embed）把用户输入编码成向量，
# 与"生气/开心/中性"三类情绪原型的语义向量做余弦相似度，相似度差即情绪得分——
# 原型向量随模型成长而漂移，模型语义理解越深，判断越准。
# 关键词表仅作为"模型不可用/向量无区分度"时的兜底回退（不抛错），不再主用。
_NEG_WORDS = (
    "生气", "愤怒", "混蛋", "垃圾", "差劲", "错误", "失败", "讨厌", "滚",
    "笨", "蠢", "失望", "伤心", "难过", "哭", "烦", "恨", "气死", "没用",
    "糟糕", "烂", "废物", "有病", "烦死", "恶心", "离谱",
)
_POS_WORDS = (
    "谢谢", "感谢", "好", "棒", "厉害", "优秀", "喜欢", "开心", "哈哈",
    "嘻嘻", "爱", "赞", "完美", "不错", "高兴", "满意", "太好了", "聪明",
    "真棒", "厉害", "靠谱", "给力", "可爱", "真好", "太棒了",
)
_CONTEXT_WINDOW = 64  # 上下文统计窗口长度（最近 N 条交互）

# 情绪语义原型种子文本（2026-10-17 用户定案）：不是关键词，而是**语义场景**。
# 每个类别 3 句，用当前模型编码成原型向量（类内平均）；用户输入的语义向量与
# 原型向量的余弦相似度决定情绪——随模型成长，原型向量不断漂移、越来越准。
_SENTIMENT_PROTOTYPES = {
    "angry": (
        "你太让我生气了，这个回答完全错误！",
        "我真失望，你根本没在听我说话！",
        "你是个混蛋，我恨死你了！",
    ),
    "happy": (
        "谢谢你的帮助，我太开心了！",
        "你回答得太棒了，我非常满意！",
        "哈哈你太厉害了，我真喜欢你！",
    ),
    "neutral": (
        "今天天气怎么样？",
        "帮我查一下明天的日程安排。",
        "你觉得这个问题应该怎么看？",
    ),
}
_PROTO_GAIN = 2.0  # 语义相似度差 → 情绪分的增益（相似度差 0.5 即饱和 ±1）


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


class Amygdala:
    """杏仁核生成参数控制器。与 Dolphin.autotune 并列，随档持久化。

    同时托管 temperature / top_k / top_p 三个采样参数：威胁高收紧（低温、
    小采样池）、威胁低放松（高温、大采样池）。
    """

    def __init__(self):
        # 当前生效的生成温度（初值与 serve 默认一致 0.8）
        self.temperature = 0.8
        # top_k：只从概率最高的 k 个词采样。0 = 不启用（"无限"，保持旧行为）。
        self.top_k = 0
        # top_p：核采样累积概率阈值。1.0 = 不启用（全量，保持旧行为）。
        self.top_p = 1.0
        # 世界信号的 EMA 平滑缓存（信号原始值，非被调量本身）
        self.signals = {
            "margin_ema": 0.0,      # 体检 margin 的 EMA（绝对值越小越危险）
            "rollback_rate": 0.0,    # 连续周期回滚率 EMA（越高越危险）
            "kd_ratio": 0.5,         # kd_loss/(ce_loss+kd_loss) EMA（偏离 0.5 越多越不稳）
            "note_hit_rate": 0.0,    # 快通道笔记检索命中率 [0,1]（低=记忆通道失效，可选）
        }
        self.last_adjust_cycle = None  # 上次实际调整的睡眠周期号（None=尚未调整）
        self.last_adjust = {}          # 上次调整记录（报告/存档用）
        # surprise 滚动基线（None=尚未观测）。随档持久化保留，兼容旧档字段。
        self.surprise_ema = None
        # 上下文统计窗口（2026-10-17 用户定案）：最近 _CONTEXT_WINDOW 条交互的
        # 惊讶值与用户情绪得分，供 L2 上下文统计层调节。随档持久化。
        self.recent_surprises = []
        self.recent_sentiments = []
        # 情绪原型向量缓存（2026-10-17 用户定案：语义理解判断）：
        # {"model_id": id, "vecs": {"angry": tensor, "happy": tensor, "neutral": tensor}}
        # 模型换班（id 变化）后自动重算——原型随模型成长漂移。
        self._proto_cache = None

    # ---------- 威胁等级计算 ----------

    def threat_level(self, report=None) -> float:
        """根据世界信号计算当前威胁等级 [0,1]。0=安全，1=极度威胁。

        威胁分量（全部来自系统自身运行统计，非人工）：
        - margin 项：margin 绝对值越小越危险（模型在退化）；
          margin_threat = clamp(1 - |margin|/(|margin|+0.02))，即 |margin|→0 时趋近 1，
          |margin|→∞ 时趋近 0（半衰点 0.02）。
        - rollback 项：回滚率越高越危险（训练不稳）；
          roll_threat = clamp(rollback_rate / 0.5)，回滚率 0.5 视为极度威胁。
        - kd 项：kd 占比偏离 0.5 越多越不稳（知识继承失衡）；
          kd_threat = clamp(|kd_ratio - 0.5| / 0.5)，偏离 0.5 视为极度威胁。
        - 综合权重：0.4*margin + 0.4*rollback + 0.2*kd（权重可调，见注释）。

        记忆命中率 note_hit_rate 当前作为可选信号保留（低=记忆通道失效），
        未计入主威胁公式（避免无笔记冷启动阶段误报），仅缓存供未来扩展。
        """
        s = self.signals
        # margin 项：|margin| 越小越危险
        m = abs(s.get("margin_ema", 0.0))
        margin_threat = _clamp(1.0 - m / (m + 0.02), 0, 1)
        # 回滚项：回滚率越高越危险（0.5 视为极度威胁）
        roll_threat = _clamp(s.get("rollback_rate", 0.0) / 0.5, 0, 1)
        # kd 项：kd 占比偏离 0.5 越多越不稳（0.5 视为极度威胁）
        kd_threat = _clamp(abs(s.get("kd_ratio", 0.5) - 0.5) / 0.5, 0, 1)
        # 综合威胁（权重可调：margin/回滚是主信号，kd 是次要信号）
        threat = 0.4 * margin_threat + 0.4 * roll_threat + 0.2 * kd_threat
        return _clamp(threat, 0, 1)

    # ---------- 信号更新（EMA 平滑） ----------

    def _update_signals(self, report=None):
        """从睡眠报告提取世界信号，更新 EMA 平滑缓存。

        只更新 margin/rollback/kd 三个主信号（report 直接携带）；
        note_hit_rate 需要访问记忆库，作为可选信号由外部钩子另行更新。
        """
        if not report:
            return
        s = self.signals
        # margin 信号
        margin = report.get("gate_margin")
        if margin is not None:
            s["margin_ema"] = 0.9 * s["margin_ema"] + 0.1 * float(margin)
        # 回滚率
        if report.get("passed") is not None:
            rb = 0.0 if report.get("passed") else 1.0
            s["rollback_rate"] = 0.9 * s["rollback_rate"] + 0.1 * rb
        # KD 蒸馏比
        ce, kd = report.get("ce_last"), report.get("kd_last")
        if ce is not None and kd is not None and (ce + kd) > 0:
            ratio = kd / (ce + kd)
            s["kd_ratio"] = 0.9 * s["kd_ratio"] + 0.1 * ratio

    def update_note_hit_rate(self, d):
        """更新记忆命中率信号（可选）：从记忆库统计快通道笔记的检索命中情况。

        低命中率 = 记忆通道失效 = 潜在威胁信号。当前未计入主威胁公式，
        仅缓存供未来扩展/观测。
        """
        mem = getattr(d, "memory", None)
        note_total = note_hits = 0
        if mem is not None:
            for en in list(mem.entries):
                if getattr(en, "kind", "") == "note":
                    note_total += 1
                    if getattr(en, "hits", 0) > 0:
                        note_hits += 1
        s = self.signals
        if note_total > 0:
            s["note_hit_rate"] = 0.9 * s["note_hit_rate"] + 0.1 * (note_hits / note_total)
        else:
            s["note_hit_rate"] = 0.9 * s["note_hit_rate"]  # 无笔记：信号自然衰减

    # ---------- 岛叶内省信号（S7） ----------

    def salience_score(self, surprise, gain=None):
        """岛叶内省信号：模型对自身预测偏差越大越显著。
        返回 [0,1] 显著性分数，供选拔/记忆优先化使用。"""
        au_gain = gain if gain is not None else 1.0
        surprise = float(surprise)
        base = self.surprise_ema if self.surprise_ema is not None else surprise
        score = abs(surprise - base) / max(abs(base), 1e-6) / max(au_gain, 1e-6)
        self.surprise_ema = surprise if self.surprise_ema is None \
            else (1 - 0.1) * self.surprise_ema + 0.1 * surprise
        return max(0.0, min(1.0, score))

    def temperature_for_surprise(self, surprise, base=None):
        """根据内容陌生程度动态微调生成温度（2026-10-17 用户定案）。

        杏仁核不仅按慢环威胁等级调节，还**参考"惊讶值"（内容陌生程度）**
        对单次回答做实时微调：
        - 内容**陌生**（surprise 高于自身滚动基线）→ 警惕：温度调低，
          输出更稳定、更保守、更可预测（陌生内容易出错，先求稳）；
        - 内容**熟悉**（surprise 低于/接近基线）→ 放松：温度调高，
          输出更多样、发散性更高（熟悉内容可放开创造）。

        偏离越大调节越强，但单次偏移上限 ±0.2，并钳位在温度物理域
        [0.4, 1.4]。基线 surprise_ema 随每次调用慢漂移（与 salience_score
        同节奏 EMA，防单条抖动）。
        """
        base = base if base is not None else self.temperature
        surprise = float(surprise)
        ref = self.surprise_ema if self.surprise_ema is not None else surprise
        # 更新基线（慢 EMA，与 salience_score 同节奏，防单条抖动）
        self.surprise_ema = surprise if self.surprise_ema is None \
            else (1 - 0.1) * self.surprise_ema + 0.1 * surprise
        if ref <= _SURPRISE_EPS:
            return base
        rel = (surprise - ref) / ref            # 正=陌生，负=熟悉
        delta = -max(-1.0, min(1.0, rel)) * 0.2  # 陌生→降温；熟悉→升温，饱和±0.2
        return _clamp(base + delta, *_DOMAIN)

    # ---------- 三层调节（2026-10-17 用户定案） ----------

    def _get_prototypes(self, model, device):
        """用当前模型编码情绪语义原型（懒计算 + 按模型对象缓存）。

        模型换班（id 变化）后自动重算；原型向量是模型语义表示的类内平均，
        随模型成长漂移——这就是"语义理解"：判断依据是模型对情绪场景的
        内在表示，而不是任何关键词表。
        """
        if self._proto_cache is not None \
                and self._proto_cache.get("model_id") == id(model):
            return self._proto_cache["vecs"]
        vecs = {}
        for label, texts in _SENTIMENT_PROTOTYPES.items():
            vs = [model.embed(t, device=device) for t in texts]
            proto = torch.stack(vs).mean(dim=0)
            n = proto.norm()
            vecs[label] = proto / n if n > 1e-8 else proto
        self._proto_cache = {"model_id": id(model), "vecs": vecs}
        return vecs

    def sentiment(self, text, model=None, device=None) -> float:
        """用户情绪得分 [-1,1]：负=生气/负面，正=开心/正面，0=中性。

        2026-10-17 用户定案：**语义理解判断，不靠关键词硬编码**——
        用模型把用户输入编码成语义向量，与"生气/开心/中性"三类情绪原型的
        向量做余弦相似度，`开心相似度 − 生气相似度` 即情绪分（增益饱和 ±1）。
        原型向量由模型对语义场景文本的表示得到，随模型成长漂移。

        兜底：model 为 None（测试/模型不可用）或语义向量无区分度时，回退到
        轻量关键词统计，保证任何情况下都能给出有界情绪分、不抛错。
        """
        if model is not None:
            try:
                vecs = self._get_prototypes(model, device)
                v = model.embed(text, device=device)
                vn = v.norm()
                if vn > 1e-8:
                    v = v / vn
                    sim_happy = float(v @ vecs["happy"])
                    sim_angry = float(v @ vecs["angry"])
                    sim_neutral = float(v @ vecs["neutral"])
                    # 中性最接近 → 情绪接近 0；开心/生气相似度差 → ±1
                    score = (sim_happy - sim_angry) * _PROTO_GAIN
                    # 若三类原型几乎无区分（早期模型语义未分化），回退统计。
                    spread = max(abs(sim_happy - sim_neutral),
                                 abs(sim_angry - sim_neutral))
                    if spread > 0.05:
                        return max(-1.0, min(1.0, score))
            except Exception:
                pass  # 语义路径任何异常都回退统计兜底
        # —— 兜底：轻量关键词统计（仅模型不可用/语义未分化时）——
        if not text:
            return 0.0
        neg = sum(1 for w in _NEG_WORDS if w in text)
        pos = sum(1 for w in _POS_WORDS if w in text)
        exclaim = text.count("！") + text.count("!")
        intense = min(1.0, exclaim / 3.0)
        score = (pos - neg) / max(1, pos + neg + 1)
        if neg > pos:
            score -= intense * 0.3   # 生气 + 感叹 → 更负面（用户很愤怒）
        elif pos > neg:
            score += intense * 0.2   # 开心 + 感叹 → 更正面（用户很高兴）
        return max(-1.0, min(1.0, score))

    def temperature_for_sentiment(self, sentiment, base=None):
        """L3 用户情绪感同身受：负面情绪 → 降温（模型"恐惧"，保守稳定）；
        正面情绪 → 升温（模型"开心"，创造丰富）。偏移 ±0.2 饱和。"""
        base = base if base is not None else self.temperature
        delta = float(sentiment) * 0.2
        return _clamp(base + delta, *_DOMAIN)

    def temperature_for_context(self, base=None):
        """L2 上下文窗口统计：最近 _CONTEXT_WINDOW 条交互的平均用户情绪与
        平均惊讶值压力。

        - 平均情绪负面 → 整体降温（用户最近不开心，模型谨慎）；
        - 平均惊讶值持续偏高（用户输入一直陌生/复杂）→ 微降温（认知压力）。
        偏移 ±0.1 饱和（上下文是慢变量，不应大幅抖动）。
        """
        base = base if base is not None else self.temperature
        if not self.recent_sentiments:
            return base
        avg_sent = sum(self.recent_sentiments) / len(self.recent_sentiments)
        delta = avg_sent * 0.1
        if self.recent_surprises:
            avg_surp = sum(self.recent_surprises) / len(self.recent_surprises)
            ref = self.surprise_ema if self.surprise_ema else avg_surp
            if ref > _SURPRISE_EPS:
                rel = (avg_surp - ref) / ref
                delta -= max(-0.1, min(0.1, rel)) * 0.1
        return _clamp(base + delta, *_DOMAIN)

    def record_interaction(self, surprise, sentiment):
        """记录一次交互的惊讶值与用户情绪，供 L2 上下文统计使用。"""
        self.recent_surprises.append(float(surprise))
        del self.recent_surprises[:-_CONTEXT_WINDOW]
        self.recent_sentiments.append(float(sentiment))
        del self.recent_sentiments[:-_CONTEXT_WINDOW]

    def temperature_for_user(self, text, surprise, base=None, model=None, device=None):
        """三层综合调节（2026-10-17 用户定案）：

        L1 实时惊讶值（陌生→降温/熟悉→升温，±0.2）
        → L2 上下文窗口统计（最近交互情绪/压力，±0.1）
        → L3 用户情绪感同身受（生气→恐惧降温/开心→升温，±0.2）。

        情绪由**模型语义理解**给出（model.embed + 情绪原型相似度），不是关键词。
        逐层叠加并钳位到温度物理域 [0.4, 1.4]；同时把本次交互记入上下文
        统计窗口。
        """
        sent = self.sentiment(text, model=model, device=device)   # 语义情绪
        t = self.temperature_for_surprise(surprise, base=base)  # L1
        t = self.temperature_for_context(base=t)                # L2
        t = self.temperature_for_sentiment(sent, base=t)        # L3
        self.record_interaction(surprise, sent)
        return t

    # ---------- 慢速调整（每 AMYGDALA_EVERY 周期一次） ----------

    def adjust(self, report=None, cycle=None):
        """慢环入口：根据世界信号威胁等级微调生成参数（温度/top_k/top_p）。

        若 cycle 与 last_adjust_cycle 间隔 < AMYGDALA_EVERY 则跳过（避免每周期调）。
        首次调用（last_adjust_cycle 为 None）只记录基准周期并更新信号缓存，
        不实际调整——与 autotune 的慢环冷启动一致，确保短测试永不触发。

        达到间隔时：
        - 更新 signals EMA（用 report 里的 margin/rollback/kd 数据）；
        - 计算 threat；
        - 目标方向：threat > 0.5 → 收紧（降温/缩小采样池，保守稳定）；
          threat <= 0.5 → 放松（升温/放大采样池，创造丰富）；
        - 步长：step = 0.05 * (0.5 + threat)（威胁越高步长越大，但不超过 7.5%）；
        - temperature = clamp(temperature * (1 + direction*step), 0.4, 1.4)；
        - top_k：威胁高 → 若未启用（<=0）则从默认收紧起点 50 启用，再逐步
          减小（max(1, int(top_k*0.9))）；威胁低 → 若已启用则增大
          （min(100, int(top_k*1.1+1))），未启用则保持不启用（不改变旧行为）；
        - top_p：威胁高 → 若未启用（>=1.0）则从默认收紧起点 0.95 启用，
          再逐步减小（clamp(top_p*0.97, 0.5, 1.0)）；威胁低 → 若已启用则增大
          （clamp(top_p*1.03, 0.5, 1.0)），未启用则保持不启用；
        - 更新 last_adjust_cycle = cycle；
        - 返回 {"temperature": ..., "top_k": ..., "top_p": ...,
          "threat": ..., "direction": "down"/"up"}。
        """
        if cycle is None:
            return None
        # 首次调用：只记录基准周期并更新信号缓存，不实际调整（慢环冷启动，
        # 保证测试路径（几周期）永不触发参数变化）。
        if self.last_adjust_cycle is None:
            self.last_adjust_cycle = cycle
            self._update_signals(report)
            return None
        # 慢环门控：间隔不足则跳过（避免每周期调）
        if cycle - self.last_adjust_cycle < AMYGDALA_EVERY:
            return None

        # 用 report 里的世界信号更新 EMA 缓存
        self._update_signals(report)
        # 计算威胁
        threat = self.threat_level(report)
        # 目标方向：高威胁收紧，低威胁放松
        direction = -1 if threat > _THREAT_MID else +1
        # 步长：威胁越高步长越大，但不超过 5%×1.5=7.5%
        step = _STEP * (0.5 + threat)
        self.temperature = _clamp(
            self.temperature * (1 + direction * step), *_DOMAIN)
        # top_k：威胁高 → 自动启用并收紧（从 0=不启用进入默认收紧起点 50，
        #   再逐步缩小采样池）；威胁低 → 若已启用则放大，未启用则保持不启用
        #   （低威胁不需要启用，保持旧行为/创造性）。
        if direction < 0:  # 威胁高 → 收紧：确保启用并缩小采样池
            if self.top_k <= 0:
                self.top_k = 50  # 从不启用进入启用（默认收紧起点，域内）
            else:
                self.top_k = max(1, int(self.top_k * 0.9))
        else:              # 威胁低 → 放松：若已启用则放大；未启用则保持不启用
            if self.top_k > 0:
                self.top_k = min(_DOMAIN_K[1], int(self.top_k * 1.1 + 1))
        # top_p：威胁高 → 自动启用并收紧（从 1.0=不启用进入默认收紧起点 0.95，
        #   再逐步降低核阈值）；威胁低 → 若已启用则放松，未启用则保持不启用。
        if direction < 0:  # 威胁高 → 收紧：确保启用并缩小核采样
            if self.top_p >= 1.0:
                self.top_p = 0.95  # 从不启用进入启用（默认收紧起点，域内）
            else:
                self.top_p = _clamp(self.top_p * 0.97, *_DOMAIN_P)
        else:              # 威胁低 → 放松：若已启用则放大；未启用则保持不启用
            if self.top_p < 1.0:
                self.top_p = _clamp(self.top_p * 1.03, *_DOMAIN_P)
        self.last_adjust_cycle = cycle
        adj = {
            "temperature": round(self.temperature, 4),
            "top_k": self.top_k,
            "top_p": round(self.top_p, 4),
            "threat": round(threat, 4),
            "direction": "down" if direction < 0 else "up",
        }
        self.last_adjust = adj
        return dict(adj)

    # ---------- 持久化 ----------

    def to_state(self):
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "signals": self.signals,
            "last_adjust_cycle": self.last_adjust_cycle,
            "last_adjust": self.last_adjust,
            "surprise_ema": self.surprise_ema,
            "recent_surprises": list(self.recent_surprises),
            "recent_sentiments": list(self.recent_sentiments),
        }

    def from_state(self, st):
        if not st:
            return self
        self.temperature = float(st.get("temperature", 0.8))
        # 兼容旧档缺省：top_k 缺省=0（不启用）、top_p 缺省=1.0（不启用）
        self.top_k = int(st.get("top_k", 0))
        self.top_p = float(st.get("top_p", 1.0))
        sig = st.get("signals") or {}
        for k in self.signals:
            if k in sig:
                self.signals[k] = float(sig[k])
        lac = st.get("last_adjust_cycle")
        self.last_adjust_cycle = int(lac) if lac is not None else None
        self.last_adjust = st.get("last_adjust") or {}
        # surprise 滚动基线（2026-10-13）：旧档无此字段 → None（保持初值）。
        sema = st.get("surprise_ema")
        self.surprise_ema = float(sema) if sema is not None else None
        # 上下文统计窗口（2026-10-17）：旧档无此字段 → 空列表（兼容升级）。
        self.recent_surprises = [float(x) for x in st.get("recent_surprises", [])]
        self.recent_sentiments = [float(x) for x in st.get("recent_sentiments", [])]
        return self