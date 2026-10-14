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
    "sleep_steps": (120, 400),  # 每周期训练步数（S6 deep 律定域 [120,400]）
    # 三档睡眠（2026-10-11）：light/rem 各自独立的自适应步数域，调节律与
    # sleep_steps 相同（AUTOTUNE_EVERY=20 周期、±5%、回滚率高上调/稳定通过下调）。
    "sleep_steps_light": (8, 40),   # 浅睡：低强度快速巩固
    "sleep_steps_rem": (20, 80),    # REM：中等强度记忆巩固
    # 判决带传导系数（2026-10-11）：K_MARGIN 真正接入反馈回路——域 [1.0,3.0]，
    # 域本身是律（1.0=1SE 判决下限，3.0≈单侧 99.9% 上限，越界会破坏门控语义）。
    "k_margin": (1.0, 3.0),
    # 金发姑娘带通阈值（2026-10-13）：太熟/太怪的相对偏离阈值域。
    # 域本身是律：LO 下限 2%（基本不过滤）上限 10%（全跳过）；HI 下限 10%
    # （大量挂起）上限 50%（几乎不挂起）。域内值由跳过率/挂起率反馈自调。
    "goldilocks_lo": (0.02, 0.10),
    "goldilocks_hi": (0.10, 0.50),
    # S6 四档睡眠（2026-10-13）：nrem（N2 浅睡）独立自适应步数域。
    "sleep_steps_nrem": (40, 120),
    # S3 睡眠压力权重（2026-10-13）：Dolphin.update_hemi_pressure 通过
    # autotune.get 读取，域 [0,1] 内值自成。
    "pressure_w1": (0.0, 1.0),
    "pressure_w2": (0.0, 1.0),
    "pressure_w3": (0.0, 1.0),
    "pressure_k": (0.0, 1.0),
    # S5 债务/还债系数（2026-10-13）：life.py 换班记账/还债通过
    # autotune.get 读取，域 [0,1] 内值自成。
    "debt_alpha_loss": (0.0, 1.0),
    "debt_beta_repay": (0.0, 1.0),
    "debt_gamma_repay": (0.0, 1.0),
    # S7 杏仁核显著性（2026-10-13）：salience_score 的增益/增强/阈值。
    "salience_gain": (0.1, 3.0),
    "salience_boost": (0.0, 2.0),
    "salience_thresh": (0.0, 1.0),
    # F-07 睡眠编制系数（2026-10-14）：compute_sleep_program 的可调系数。
    # 域设计保证单调性约束恒成立：deep=p（p 单调）、total_steps=base×(1+p)
    # （base>0 时随 p 单调不减）；rem/nrem 系数只影响占比分配。
    "sleep_prog_rem_base": (0.10, 0.30),
    "sleep_prog_rem_debt": (0.00, 0.20),
    "sleep_prog_nrem_scale": (0.30, 0.70),
    "sleep_prog_total_base": (120, 240),
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
            "goldilocks_skip_rate": 0.0,    # 金发姑娘太熟跳过率 EMA（[0,1]）
            "goldilocks_suspend_rate": 0.0,  # 金发姑娘太怪挂起率 EMA（[0,1]）
        }
        self.tunables = {
            "note_k": 5, "dream_k": 5, "promote_hits": 3,
            "kd_alpha": 0.5, "kd_T": 2.0,
            "target_interval": 15, "sleep_steps": 120,
            # 三档睡眠（2026-10-11）：light/rem 独立步数域初值（域内）。
            "sleep_steps_light": 16, "sleep_steps_rem": 40,
            # 判决带传导系数（2026-10-11）：初值 2.0 = probe.K_MARGIN（域内）。
            "k_margin": 2.0,
            # 金发姑娘带通阈值（2026-10-13）：初值 0.05/0.20 与模块常量一致。
            "goldilocks_lo": 0.05,
            "goldilocks_hi": 0.20,
            # S6 四档睡眠（2026-10-13）：nrem（N2 浅睡）步数初值 80（域内）。
            "sleep_steps_nrem": 80,
            # S3 睡眠压力权重（2026-10-13）：初值 w1=0.5/w2=0.3/w3=0.2/k=0.3。
            "pressure_w1": 0.5,
            "pressure_w2": 0.3,
            "pressure_w3": 0.2,
            "pressure_k": 0.3,
            # S5 债务/还债系数（2026-10-13）：初值 ALPHA_LOSS/BETA_REPAY/GAMMA_REPAY。
            "debt_alpha_loss": 0.4,
            "debt_beta_repay": 0.05,
            "debt_gamma_repay": 0.5,
            # S7 杏仁核显著性（2026-10-13）：初值 GAIN/BOOST/THRESH。
            "salience_gain": 1.0,
            "salience_boost": 0.5,
            "salience_thresh": 0.7,
            # F-07 睡眠编制系数（2026-10-14）：初值与 compute_sleep_program
            # 既有硬编码一致（rem_base=0.2/rem_debt=0.1/nrem_scale=0.5/total_base=120）。
            "sleep_prog_rem_base": 0.2,
            "sleep_prog_rem_debt": 0.1,
            "sleep_prog_nrem_scale": 0.5,
            "sleep_prog_total_base": 120,
        }
        self.cycle_count = 0   # 已运行周期数（Autotune 内部计数）
        self.last_adjust = {} # 上次调整记录（报告用）
        # 门控历史（2026-10-11，K_MARGIN 调节用）：最近一个 AUTOTUNE_EVERY
        # 窗口内的 (passed, gate_margin, gate_eps) 样本，只收集 deep/rem 模式
        # （light 浅睡不参与惩罚/调节）。adjust 时消费后清空。不随档持久化
        # （短期调节缓存，丢失无害，重启后从空窗口重新累积）。
        self.gate_history = []
        # 金发姑娘计数增量跟踪（2026-10-13）：记录上次消费的 skipped/suspended
        # 累计计数，用于在 update_signals 中计算本周期增量。不随档持久化。
        # None = 尚未初始化（on_cycle 首次遇到 Dolphin 时以当前计数为基准，
        # 避免把重启前历史累计当成增量）。
        self._last_skipped = None
        self._last_suspended = None

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

    # ---------- 被调量读取（调用点消费） ----------

    def get(self, key, default=None):
        """读取被调量的当前值。旧档/测试没有该量时返回 default。"""
        return self.tunables.get(key, default)

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
        # P1 修复（验收）：candidates 是整数（len(buf.items)），旧表达式
        # `len(report.get("candidates") and [])` 在 candidates==0 时对整数 0 调
        # len() 抛 TypeError。改为 `int(report.get("candidates") or 0)`：
        # candidates 为 0/None/缺省时安全回落 0。
        revived = int(report.get("candidates") or 0)
        # 用 dreamed 近似复活通道活跃度（复活条目进入缓冲成为候选）
        dreamed = report.get("dreamed", 0) or 0
        s["resurrect_throughput"] = 0.9 * s["resurrect_throughput"] + 0.1 * dreamed
        # 睡眠间隔（用 since_sleep 近似）
        since = getattr(d, "_since_sleep", 0)
        s["sleep_interval_ema"] = 0.9 * s["sleep_interval_ema"] + 0.1 * since
        # 回滚率
        # 2026-10-11：light 浅睡失败不惩罚——不更新 rollback_rate 信号
        # （浅睡步数少、几乎不可能正改善，若计入会让浅睡被过度惩罚）。
        mode = report.get("mode")
        if report.get("passed") is not None and mode != "light":
            rb = 0.0 if report.get("passed") else 1.0
            s["rollback_rate"] = 0.9 * s["rollback_rate"] + 0.1 * rb
        # K_MARGIN 调节历史（仅 deep/rem 模式；light 不参与门控尺度调节）
        if mode != "light" and report.get("passed") is not None:
            m, e = report.get("gate_margin"), report.get("gate_eps")
            if m is not None and e is not None and e > 0:
                self.gate_history.append((bool(report["passed"]), float(m), float(e)))
        # 金发姑娘带通反馈（2026-10-13）：由 learn() 的 skipped/suspended 累计
        # 计数增量（dolphin.py 属性随档已有）驱动，用 EMA 平滑成 [0,1] 速率。
        # 每个睡眠周期只消费一次增量，避免同一批数据被重复计数。
        if d is not None:
            sk = int(getattr(d, "skipped_too_easy", 0))
            su = int(getattr(d, "suspended_too_hard", 0))
            if self._last_skipped is None:  # 首次：以当前计数为基准，不算增量
                self._last_skipped = sk
                self._last_suspended = su
            sk_inc = max(0, sk - self._last_skipped)
            su_inc = max(0, su - self._last_suspended)
            total_inc = sk_inc + su_inc
            if total_inc > 0:
                s["goldilocks_skip_rate"] = 0.9 * s["goldilocks_skip_rate"] + 0.1 * (sk_inc / total_inc)
                s["goldilocks_suspend_rate"] = 0.9 * s["goldilocks_suspend_rate"] + 0.1 * (su_inc / total_inc)
            else:
                # 无新门控样本：信号自然衰减，防止旧速率卡死。
                s["goldilocks_skip_rate"] *= 0.9
                s["goldilocks_suspend_rate"] *= 0.9
            self._last_skipped = sk
            self._last_suspended = su

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

        # lr 回升（2026-10-11）：连续高回滚 + lr 已退火到过低（如 1e-05 卡死）时，
        # 优先小幅回升 lr，而不是只加 sleep_steps（避免步数无限上涨）。
        # 这是 autotune 的慢速干预（每 AUTOTUNE_EVERY=20 周期一次）；Plasticity
        # 仍控制 lr 的日常退火/回升，此处只在「高回滚 + lr 低于阈值」时兜底抬升。
        lr_boosted = False
        if d is not None and s["rollback_rate"] > 0.4:
            plasticity = getattr(d, "plasticity", None)
            lr_max = getattr(plasticity, "lr_max", 2e-4) if plasticity else 2e-4
            lr_boost_threshold = 3e-5   # 低于此阈值视为退火过低（lr_min=1e-5 卡死）
            lr_boost_cap = min(lr_max, 1e-4)  # 回升封顶：≤1e-4 且不越物理域上限
            cur_lr = d.h[0].opt.param_groups[0]["lr"]
            if cur_lr < lr_boost_threshold:
                new_lr = min(cur_lr * 1.15, lr_boost_cap)
                if new_lr > cur_lr + 1e-12:
                    for hh in d.h:
                        hh.opt.param_groups[0]["lr"] = new_lr
                    adj["lr_recover"] = ("up", round(new_lr, 8),
                                         f"rollback={s['rollback_rate']:.2f} lr过低回升")
                    if report is not None:
                        report["lr"] = new_lr  # 报告与实际 opt lr 保持一致
                    lr_boosted = True

        # sleep_steps：回滚率高 → 训练不足，步数上调；长期稳定通过 → 步数下调（降低算力开销）
        # 本轮已回升 lr 时不再上调步数（lr 优先；避免步数无限上涨）。
        if s["rollback_rate"] > 0.4 and not lr_boosted:
            v = nudge("sleep_steps", _STEP, *_DOMAINS["sleep_steps"])
            if v: adj["sleep_steps"] = ("up", round(v, 2), f"rollback={s['rollback_rate']:.2f}")
        elif s["rollback_rate"] < 0.1:
            v = nudge("sleep_steps", -_STEP, *_DOMAINS["sleep_steps"])
            if v: adj["sleep_steps"] = ("down", round(v, 2), f"rollback={s['rollback_rate']:.2f}")

        # 三档睡眠（2026-10-11）：light/rem 步数域与 sleep_steps 同律调节但各自独立。
        # light 浅睡：回滚率高 → 上调（浅睡也不能无效空转）；稳定通过 → 下调（保持低强度）。
        # 本轮已回升 lr 时不再上调（与 sleep_steps 同律）。
        if s["rollback_rate"] > 0.4 and not lr_boosted:
            v = nudge("sleep_steps_light", _STEP, *_DOMAINS["sleep_steps_light"])
            if v: adj["sleep_steps_light"] = ("up", round(v, 2), f"rollback={s['rollback_rate']:.2f}")
        elif s["rollback_rate"] < 0.1:
            v = nudge("sleep_steps_light", -_STEP, *_DOMAINS["sleep_steps_light"])
            if v: adj["sleep_steps_light"] = ("down", round(v, 2), f"rollback={s['rollback_rate']:.2f}")

        # rem：回滚率高 → 上调；稳定通过 → 下调（REM 是中等强度巩固，步数域更宽）。
        # 本轮已回升 lr 时不再上调（与 sleep_steps 同律）。
        if s["rollback_rate"] > 0.4 and not lr_boosted:
            v = nudge("sleep_steps_rem", _STEP, *_DOMAINS["sleep_steps_rem"])
            if v: adj["sleep_steps_rem"] = ("up", round(v, 2), f"rollback={s['rollback_rate']:.2f}")
        elif s["rollback_rate"] < 0.1:
            v = nudge("sleep_steps_rem", -_STEP, *_DOMAINS["sleep_steps_rem"])
            if v: adj["sleep_steps_rem"] = ("down", round(v, 2), f"rollback={s['rollback_rate']:.2f}")

        # k_margin：判决带尺度自适应（2026-10-11，K_MARGIN 真正接入反馈回路）。
        # 与 sleep_steps 的调节并行且独立——sleep_steps 管「训练量」，k_margin 管
        # 「判决带尺度」，两者不冲突。
        # - 高回滚（rollback_rate>0.6）且失败样本 margin 中位接近 ε（|margin-ε|<0.5ε，
        #   说明是判决带边缘的抖动失败）→ 下调 k_margin（放宽带，域下限 1.0）。
        # - 高通过（rollback_rate<0.15）且通过样本 margin 中位远大于 ε（>3ε，
        #   说明改善幅度远超判决带，带过松）→ 上调 k_margin（收紧带，域上限 3.0）。
        gh = self.gate_history
        if gh:
            fails = [(m, e) for (p, m, e) in gh if not p]
            passes = [(m, e) for (p, m, e) in gh if p]
            if s["rollback_rate"] > 0.6 and fails:
                fm = sorted(m for m, _ in fails)
                fe = sorted(e for _, e in fails)
                med_m = fm[len(fm) // 2]
                med_e = max(fe[len(fe) // 2], 1e-12)
                if abs(med_m - med_e) < 0.5 * med_e:
                    v = nudge("k_margin", -_STEP, *_DOMAINS["k_margin"])
                    if v:
                        adj["k_margin"] = ("down", round(v, 3),
                                           f"边缘失败 margin≈ε rollback={s['rollback_rate']:.2f}")
            elif s["rollback_rate"] < 0.15 and passes:
                pm = sorted(m for m, _ in passes)
                pe = sorted(e for _, e in passes)
                med_m = pm[len(pm) // 2]
                med_e = max(pe[len(pe) // 2], 1e-12)
                if med_m > 3 * med_e:
                    v = nudge("k_margin", _STEP, *_DOMAINS["k_margin"])
                    if v:
                        adj["k_margin"] = ("up", round(v, 3),
                                           f"通过 margin>{3*med_e:.4f} rollback={s['rollback_rate']:.2f}")
        # 消费门控历史（每 AUTOTUNE_EVERY 周期一次）
        self.gate_history = []

        # 金发姑娘带通阈值（2026-10-13）：
        # - 跳过率偏高（>0.5 一半数据被当太熟）→ 调大 goldilocks_lo（放宽「太熟」
        #   判定，少跳过）；跳过率偏低（<0.05 几乎没跳过）→ 调小 goldilocks_lo
        #   （收紧「已会」判定）。目标：跳过率稳定在 ~0.2-0.3。
        # - 挂起率偏高（>0.3 三成数据太怪）→ 调大 goldilocks_hi（放宽「太怪」
        #   判定）；挂起率偏低（<0.05）→ 调小 goldilocks_hi。目标：挂起率稳定
        #   在 ~0.1-0.2。
        if s["goldilocks_skip_rate"] > 0.5:
            v = nudge("goldilocks_lo", _STEP, *_DOMAINS["goldilocks_lo"])
            if v: adj["goldilocks_lo"] = ("up", round(v, 4),
                                    f"skip_rate={s['goldilocks_skip_rate']:.2f}")
        elif s["goldilocks_skip_rate"] < 0.05:
            v = nudge("goldilocks_lo", -_STEP, *_DOMAINS["goldilocks_lo"])
            if v: adj["goldilocks_lo"] = ("down", round(v, 4),
                                          f"skip_rate={s['goldilocks_skip_rate']:.2f}")
        if s["goldilocks_suspend_rate"] > 0.3:
            v = nudge("goldilocks_hi", _STEP, *_DOMAINS["goldilocks_hi"])
            if v: adj["goldilocks_hi"] = ("up", round(v, 4),
                                        f"suspend_rate={s['goldilocks_suspend_rate']:.2f}")
        elif s["goldilocks_suspend_rate"] < 0.05:
            v = nudge("goldilocks_hi", -_STEP, *_DOMAINS["goldilocks_hi"])
            if v: adj["goldilocks_hi"] = ("down", round(v, 4),
                                          f"suspend_rate={s['goldilocks_suspend_rate']:.2f}")

        # —— F-01 修复：S3 睡眠压力权重反馈（值自成）——
        # 睡眠间隔长期短于目标 → 加大 w3（时间积累权重），让压力积累更平缓；
        # 回滚率高 → 加大 w1（数据惊讶度权重），让睡眠触发更及时。
        if s["sleep_interval_ema"] < t["target_interval"] * 0.7:
            v = nudge("pressure_w3", _STEP, *_DOMAINS["pressure_w3"])
            if v: adj["pressure_w3"] = ("up", round(v, 3),
                                        f"interval={s['sleep_interval_ema']:.1f}")
        if s["rollback_rate"] > 0.4:
            v = nudge("pressure_w1", _STEP, *_DOMAINS["pressure_w1"])
            if v: adj["pressure_w1"] = ("up", round(v, 3),
                                        f"rollback={s['rollback_rate']:.2f}")

        # —— F-01 修复：S5 债务系数反馈（值自成）——
        # 胼胝体 KL 偏大（两半球输出分布分化）→ 加大 beta_repay（债务加权步数），
        # 促进蒸馏对齐；记忆通道活跃 → 加大 gamma_repay（每轮还债比例），防滚雪球。
        if d is not None:
            kl = getattr(getattr(d, "debt_ledger", None), "callosum_kl", None)
            if kl is not None and kl > 0.01:
                v = nudge("debt_beta_repay", _STEP, *_DOMAINS["debt_beta_repay"])
                if v: adj["debt_beta_repay"] = ("up", round(v, 4),
                                                f"callosum_kl={kl:.4f}")
        if s["resurrect_throughput"] > 2.0:
            v = nudge("debt_gamma_repay", _STEP, *_DOMAINS["debt_gamma_repay"])
            if v: adj["debt_gamma_repay"] = ("up", round(v, 4),
                                             f"resurrect={s['resurrect_throughput']:.2f}")

        # —— F-01 修复：S7 杏仁核显著性系数反馈（值自成）——
        # 笔记命中率低 → 下调显著性阈值（更多经验可优先入 note）；
        # 命中率高 → 上调显著性增益（显著记忆获得更高选拔权重）。
        if s["note_hit_rate"] < 0.05:
            v = nudge("salience_thresh", -_STEP, *_DOMAINS["salience_thresh"])
            if v: adj["salience_thresh"] = ("down", round(v, 3),
                                            f"note_hit={s['note_hit_rate']:.2f}")
        elif s["note_hit_rate"] > 0.3:
            v = nudge("salience_boost", _STEP, *_DOMAINS["salience_boost"])
            if v: adj["salience_boost"] = ("up", round(v, 3),
                                           f"note_hit={s['note_hit_rate']:.2f}")

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
            # 三档睡眠（2026-10-11）：light/rem 步数域同步回 Dolphin 属性，
            # life.run_cycle 按 mode 读取对应属性。
            d.sleep_steps_light = int(round(t["sleep_steps_light"]))
            d.sleep_steps_rem = int(round(t["sleep_steps_rem"]))
            # S6 四档睡眠（2026-10-13）：nrem 步数域同步回 Dolphin 属性，
            # life.run_cycle 按 mode 读取对应属性。
            d.sleep_steps_nrem = round(t["sleep_steps_nrem"])
            # 判决带传导系数（2026-10-11）：同步回 Dolphin 属性，Dolphin.gate 消费。
            d.k_margin = round(t["k_margin"], 4)
            # 金发姑娘带通阈值（2026-10-13）：同步回 Dolphin 属性，learn() 的
            # goldilocks_gate 调用点会读取（见 dolphin.py learn()）。
            d.goldilocks_lo = round(t["goldilocks_lo"], 4)
            d.goldilocks_hi = round(t["goldilocks_hi"], 4)
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