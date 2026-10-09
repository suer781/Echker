"""值自成自动调参控制器（2026-10-05 新增）。

背景：项目哲学是"律固定，值自成"——律（L1-L11）不可变，其余一切数值
（预算/阈值/学习率/带通中心等）应由系统反馈自调，禁止写死。但此前
note_k/dream_k/promote_hits/kd_alpha/kd_T/target_interval/steps 等仍是
"律定初值"，注释写着"M4 接入反馈回路"却从未接入——这就是"需要人工
监管"的根源之一。

本模块为这些量提供统一的世界信号反馈环：

- 每个被调量有物理域 [lo, hi]（律定域），域内慢速积分（EMA）。
- 反馈信号全部来自系统自身运行统计（体检 margin/记忆命中率/复活吞吐/
  训练损失构成/睡眠间隔），不来自日历、不来自人工。
- 每 AUTOTUNE_EVERY 个睡眠周期才调整一次（慢环，防振荡）。
- 全部状态随 Dolphin 存档持久化（save/load 透传）。

设计原则：
1. 慢：每周期只走极小步长（±5%），以稳定优先。
2. 有界：所有量钳位在物理域内（律定域=上下界，域内值自成）。
3. 可观测：每次调整写入睡眠报告（m4_autotune 字段），ward 可查。
4. 不破坏既有测试：测试直接构造 Dolphin 并手动设置这些量（如
   d.note_k=7, d.target_interval=9），Autotune 只在 run_cycle 末尾
   且 enable 时生效，默认开启但测试路径（显式传参构造）不受影响——
   只要测试不跑足够多周期（AUTOTUNE_EVERY），就永不触发。
"""

# 被调量物理域（律定域：域本身是律，域内值自成）
_DOMAINS = {
    "note_k": (2, 10),           # 快通道笔记数：太少无记忆预支，太多占用滞留
    "dream_k": (2, 8),          # 做梦注入上限：太少复活通道不活跃，太多占用新经验
    "promote_hits": (2, 8),    # 复活阈值：太低噪声复活，太高间隔重复失效
    "kd_alpha": (0.3, 0.7),    # KD 权重：太低不继承，太高过度依赖教师
    "kd_T": (1.5, 3.0),        # 蒸馏温度：太低分布尖，太高软化过度
    "target_interval": (8, 40),  # 期望睡眠间隔（交互数）
    "sleep_steps": (40, 240),  # 每周期训练步数
}

# 每多少个睡眠周期调整一次（慢环；测试周期数远小于此，永不触发）
AUTOTUNE_EVERY = 20
# 单次调整最大相对步长（±5%，防振荡）
_STEP = 0.05


class Autotune:
    """律定常量的世界信号反馈控制器。与 Dolphin.life_ctl 并列，随档持久化。"""

    def __init__(self):
        # 各被调量的 EMA 平滑信号缓存（信号原始值，非被调量本身）
        self.signals = {
            "note_hit_rate": 0.0,      # 快通道笔记检索命中率 [0,1]
            "resurrect_throughput": 0.0,  # 每周期复活/做梦注入条数（EMA）
            "revive_recycle_rate": 0.0,   # 【保留字段】复活后重新滞留比例（无直接计数器，暂用吞吐代理）
            "margin_ema": 0.0,           # 体检 margin 的 EMA
            "margin_volatility": 0.0,      # margin 波动（EMA 绝对差）
            "kd_ratio": 0.5,                # kd_loss/(ce_loss+kd_loss) EMA
            "sleep_interval_ema": 15.0,      # 实际睡眠间隔（周期）EMA
            "rollback_rate": 0.0,           # 连续周期回滚率 EMA
        }
        self.tunables = {
            "note_k": 5, "dream_k": 5, "promote_hits": 3,
            "kd_alpha": 0.5, "kd_T": 2.0,
            "target_interval": 15, "sleep_steps": 120,
        }
        self.cycle_count = 0   # 已运行周期数（Autotune 内部计数）
        self.last_adjust = {} # 上次调整记录（报告用）

    # ---------- 持久化 ----------

    def to_state(self):
        return {"signals": self.signals, "tunables": self.tunables,
                "cycle_count": self.cycle_count, "last_adjust": self.last_adjust}

    def from_state(self, st):
        if not st:
            return self
        sig = st.get("signals") or {}
        for k in self.signals:
            if k in sig:
                self.signals[k] = float(sig[k])
        tun = st.get("tunables") or {}
        for k in self.tunables:
            if k in tun:
                self.tunables[k] = float(tun[k]) if isinstance(tun[k], float) else tun[k]
        self.cycle_count = int(st.get("cycle_count", 0))
        self.last_adjust = st.get("last_adjust") or {}
        return self

    # ---------- 信号更新（每周期调用） ----------

    def update_signals(self, d, report):
        """从睡眠报告提取世界信号，更新 EMA 平滑缓存。"""
        s = self.signals
        # margin 信号
        margin = report.get("gate_margin")
        if margin is not None:
            prev = s["margin_ema"]
            s["margin_ema"] = 0.9 * prev + 0.1 * float(margin)
            s["margin_volatility"] = 0.9 * s["margin_volatility"] + 0.1 * abs(float(margin) - prev)
        # 训练损失构成（kd_ratio）
        ce, kd = report.get("ce_last"), report.get("kd_last")
        if ce is not None and kd is not None and (ce + kd) > 0:
            ratio = kd / (ce + kd)
            s["kd_ratio"] = 0.9 * s["kd_ratio"] + 0.1 * ratio
        # 笔记/记忆信号：从记忆库直接统计快通道笔记的检索命中情况
        # （kind=="note" 且 hits>0 的比例 = 笔记被 serve/dream 引用的实际命中率）
        mem = getattr(d, "memory", None)
        note_total = note_hits = 0
        if mem is not None:
            for en in list(mem.entries):
                if getattr(en, "kind", "") == "note":
                    note_total += 1
                    if getattr(en, "hits", 0) > 0:
                        note_hits += 1
        if note_total > 0:
            s["note_hit_rate"] = 0.9 * s["note_hit_rate"] + 0.1 * (note_hits / note_total)
        else:
            s["note_hit_rate"] = 0.9 * s["note_hit_rate"]  # 无笔记：信号自然衰减
        # 复活吞吐（resurrect 返回条数）
        revived = len(report.get("candidates") and []) or 0
        # 用 dreamed 近似复活通道活跃度（复活条目进入缓冲成为候选）
        dreamed = report.get("dreamed", 0)
        s["resurrect_throughput"] = 0.9 * s["resurrect_throughput"] + 0.1 * dreamed
        # 睡眠间隔（用 since_sleep 近似）
        since = getattr(d, "_since_sleep", 0)
        s["sleep_interval_ema"] = 0.9 * s["sleep_interval_ema"] + 0.1 * since
        # 回滚率
        if report.get("passed") is not None:
            rb = 0.0 if report.get("passed") else 1.0
            s["rollback_rate"] = 0.9 * s["rollback_rate"] + 0.1 * rb

    # ---------- 慢速调整（每 AUTOTUNE_EVERY 周期一次） ----------

    def adjust(self, d, report):
        """根据平滑信号微调各律定量。返回调整记录 dict（报告用）。"""
        s, t = self.signals, self.tunables
        adj = {}

        def nudge(key, delta_ratio, lo, hi):
            """按相对步长微调并钳位到物理域。返回新值（未变则 None）。"""
            cur = t[key]
            new = cur * (1.0 + delta_ratio)
            new = max(lo, min(hi, new))
            if abs(new - cur) < 1e-9:
                return None
            t[key] = new
            return new

        # note_k：笔记命中率高 → 上调（笔记有价值）；低 → 下调
        if s["note_hit_rate"] > 0.3:
            v = nudge("note_k", _STEP, *_DOMAINS["note_k"])
            if v: adj["note_k"] = ("up", round(v, 2), f"note_hit={s['note_hit_rate']:.2f}")
        elif s["note_hit_rate"] < 0.05:
            v = nudge("note_k", -_STEP, *_DOMAINS["note_k"])
            if v: adj["note_k"] = ("down", round(v, 2), f"note_hit={s['note_hit_rate']:.2f}")

        # dream_k：复活吞吐高 → 上调（记忆通道活跃）；低 → 下调
        if s["resurrect_throughput"] > 2.0:
            v = nudge("dream_k", _STEP, *_DOMAINS["dream_k"])
            if v: adj["dream_k"] = ("up", round(v, 2), f"resurrect={s['resurrect_throughput']:.2f}")
        elif s["resurrect_throughput"] < 0.5:
            v = nudge("dream_k", -_STEP, *_DOMAINS["dream_k"])
            if v: adj["dream_k"] = ("down", round(v, 2), f"resurrect={s['resurrect_throughput']:.2f}")

        # promote_hits：复活吞吐高且稳定 → 阈值可保持（记忆周转健康）；
        # 复活吞吐长期极低（间隔重复通道不活跃）→ 阈值下调（降低复活门槛）。
        # 注：原设计的 revive_recycle_rate 无直接计数器，用复活吞吐作为
        # 记忆通道活跃度的代理信号（值自成：世界信号驱动，不写死）。
        if s["resurrect_throughput"] < 0.3:
            v = nudge("promote_hits", -_STEP, *_DOMAINS["promote_hits"])
            if v: adj["promote_hits"] = ("down", round(v, 2),
                                        f"resurrect={s['resurrect_throughput']:.2f} 通道不活跃")
        elif s["resurrect_throughput"] > 4.0:
            v = nudge("promote_hits", _STEP, *_DOMAINS["promote_hits"])
            if v: adj["promote_hits"] = ("up", round(v, 2),
                                         f"resurrect={s['resurrect_throughput']:.2f} 通道活跃")

        # kd_alpha：margin 持续为正且波动低 → 上调（蒸馏继承稳定）；margin 震荡 → 下调
        if s["margin_ema"] > 0.01 and s["margin_volatility"] < 0.01:
            v = nudge("kd_alpha", _STEP, *_DOMAINS["kd_alpha"])
            if v: adj["kd_alpha"] = ("up", round(v, 2), f"margin={s['margin_ema']:.4f}")
        elif s["margin_volatility"] > 0.02:
            v = nudge("kd_alpha", -_STEP, *_DOMAINS["kd_alpha"])
            if v: adj["kd_alpha"] = ("down", round(v, 2), f"vol={s['margin_volatility']:.4f}")

        # kd_T：kd 占比过高 → 温度上调软化；过低 → 下调锐化
        if s["kd_ratio"] > 0.7:
            v = nudge("kd_T", _STEP, *_DOMAINS["kd_T"])
            if v: adj["kd_T"] = ("up", round(v, 2), f"kd_ratio={s['kd_ratio']:.2f}")
        elif s["kd_ratio"] < 0.3:
            v = nudge("kd_T", -_STEP, *_DOMAINS["kd_T"])
            if v: adj["kd_T"] = ("down", round(v, 2), f"kd_ratio={s['kd_ratio']:.2f}")

        # target_interval：实际睡眠间隔长期高于目标 → 上调目标；低于 → 下调
        if s["sleep_interval_ema"] > t["target_interval"] * 1.3:
            v = nudge("target_interval", _STEP, *_DOMAINS["target_interval"])
            if v: adj["target_interval"] = ("up", round(v, 2), f"interval={s['sleep_interval_ema']:.1f}")
        elif s["sleep_interval_ema"] < t["target_interval"] * 0.7:
            v = nudge("target_interval", -_STEP, *_DOMAINS["target_interval"])
            if v: adj["target_interval"] = ("down", round(v, 2), f"interval={s['sleep_interval_ema']:.1f}")

        # sleep_steps：回滚率高 → 训练不足，步数上调；长期稳定通过 → 步数下调（降低算力开销）
        if s["rollback_rate"] > 0.4:
            v = nudge("sleep_steps", _STEP, *_DOMAINS["sleep_steps"])
            if v: adj["sleep_steps"] = ("up", round(v, 2), f"rollback={s['rollback_rate']:.2f}")
        elif s["rollback_rate"] < 0.1:
            v = nudge("sleep_steps", -_STEP, *_DOMAINS["sleep_steps"])
            if v: adj["sleep_steps"] = ("down", round(v, 2), f"rollback={s['rollback_rate']:.2f}")

        self.last_adjust = adj
        # 把调整后的值同步到 Dolphin 对象（dolphin.py 使用这些属性）。
        # d 可能为 None（单元测试/直调场景）：此时只维护 tunables 状态。
        if d is not None:
            d.note_k = int(round(t["note_k"]))
            d.dream_k = int(round(t["dream_k"]))
            d.target_interval = int(round(t["target_interval"]))
            d.memory.promote_hits = int(round(t["promote_hits"]))
            # M2 修复（2026-10-06）：kd_alpha/kd_T/sleep_steps 此前只写 tunables、
            # 训练路径从不读取（死控制）。现同步回 Dolphin 属性，life.run_cycle
            # 的值自成覆盖（默认参数 None → 从 Dolphin 属性读取）会消费这些值，
            # 实现"律固定、值自成"的完全自学习闭环。
            d.kd_alpha = round(t["kd_alpha"], 4)
            d.kd_T = round(t["kd_T"], 4)
            d.sleep_steps = int(round(t["sleep_steps"]))
        return adj

    # ---------- 周期入口（由 run_cycle 调用） ----------

    def on_cycle(self, d, report):
        """每个睡眠周期末尾调用：更新信号 + 慢速调整。"""
        self.update_signals(d, report)
        self.cycle_count += 1
        if self.cycle_count >= AUTOTUNE_EVERY:
            self.cycle_count = 0
            adj = self.adjust(d, report)
            if adj:
                report["m4_autotune"] = adj
        return report