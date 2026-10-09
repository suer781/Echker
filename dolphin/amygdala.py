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
        return self