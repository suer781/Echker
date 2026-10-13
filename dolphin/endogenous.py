"""内生稳态（Endogenous Homeostasis）——由系统自身信号驱动的健康度 h（影子模式）。

总纲「律固定，值自成」：不依赖外部探测集，健康度由系统自身的四个内生信号
合成，各信号先相对自身历史滚动基线做比值归一化（复用项目「相对自身基线」
惯例，免疫量级漂移）：

- 学习增益 g = mean(awake_nll(sel)) − mean(sleeping_nll(sel))
  正 = 新脑对刚学数据 NLL 下降（真学会）。
- 遗忘 f = mean(sleeping_nll(mem_old)) − mean(awake_nll(mem_old))
  正 = 新脑对较旧记忆 NLL 上升（忘旧知）。mem_old = 记忆库较旧条目（内生）。
- 发散 d = ‖θ_sleeping‖ / ‖θ_awake‖（权重范数比，含 NaN → 硬拦截）。
- 坍缩 c = H(softmax(sleeping_logits)) − H(softmax(awake_logits))
  负 = 输出熵塌（固定点退化）；合成时用坍缩深度 c_pen = max(−c, 0)。

健康度合成（g/f/c 相对各自历史中位基线比值归一化，d 以比值基线归一化）：
  h = clip( g_norm − 0.5·f_norm − 0.5·(d_norm−1) − 1.0·c_norm, 0, 1 )

NaN/发散硬拦截：d 非有限（‖θ‖ 含 NaN）或超过自身基线 10× → h = 0（这是
物理性保护，不是能力判断，保留）。

本次为「影子模式」：只在睡眠周期体检之后并行计算 h 并写日志
（report["m4_endogenous_h"] / report["m4_endogenous_signals"]），
不替换 gate()/post_exam 的 swap/rollback 判决、不退役探测集——切换是后续阶段。
"""
import math
from collections import deque

import torch
import torch.nn.functional as F

# 律定（域本身是律，值自成）
_ALPHA = 0.5        # 遗忘 f 的合成权重
_BETA = 0.5         # 发散 d 的合成权重
_GAMMA = 1.0        # 坍缩 c 的合成权重
_HISTORY_LEN = 200  # 滚动历史长度（归一化基线窗口）
_HARD_D_MULT = 10.0  # d 硬阈值：‖θ‖ 比值超过自身基线 10× → 硬拦截
_MEM_OLD_CAP = 32    # 遗忘检测用的「较旧记忆」抽样上限
_ENTROPY_SAMPLE = 8  # 熵（坍缩 c）抽样条数上限
_ENTROPY_CHUNKS = 4  # 熵抽样每条的最大窗口数
_EPS = 1e-9


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _median(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def weight_norm_ratio(sleeping_model, awake_model):
    """d = ‖θ_sleeping‖ / ‖θ_awake‖（权重范数比）。返回 (d, has_nan)。"""
    sn = an = 0.0
    has_nan = False
    for p in sleeping_model.parameters():
        if torch.isnan(p).any():
            has_nan = True
        sn += float(p.pow(2).sum())
    for p in awake_model.parameters():
        if torch.isnan(p).any():
            has_nan = True
        an += float(p.pow(2).sum())
    sn = sn ** 0.5
    an = an ** 0.5
    d = (sn / an) if an > 0 else float("inf")
    if has_nan:
        d = float("nan")
    return d, has_nan


@torch.no_grad()
def mean_entropy(model, datas, device, max_chunks=_ENTROPY_CHUNKS):
    """平均输出熵 H(softmax(logits))（每条若干窗口、逐 token 熵的全局均值）。"""
    model.eval()
    bs = model.cfg.block_size
    ents = []
    for data in datas:
        b = torch.tensor(list(data), dtype=torch.long, device=device)
        if b.numel() < 2:
            continue
        step = max(1, (b.numel() - 1) // max_chunks)
        n = 0
        for s in range(0, b.numel() - 1, step):
            e = min(s + bs, b.numel())
            if e - s < 2:
                break
            logits, _ = model(b[s:e - 1].unsqueeze(0))
            p = F.softmax(logits, dim=-1)
            ent = -(p * (p + 1e-12).log()).sum(-1).mean().item()
            ents.append(ent)
            n += 1
            if n >= max_chunks:
                break
    return _mean(ents)


def select_old_memory(memory, current_cycle, cap=_MEM_OLD_CAP):
    """较旧记忆条目（已内化、非本轮新入）的字节数据，用于内生遗忘检测。

    只读热层 entries、不触发冷层加载、不改任何状态；按 created 取最旧 cap 条。
    """
    olds = [en for en in memory.entries if en.cycle < current_cycle]
    olds.sort(key=lambda en: en.created)
    return [en.text.encode("utf-8", errors="replace") for en in olds[:cap]]


@torch.no_grad()
def compute_signals(awake_model, sleeping_model, sel_data, mem_old_data, device):
    """计算四内生信号 g/f/d/c（含 d 的 NaN 标志）。纯观测，不改任何权重。

    awake = 冻结服务脑（参考），sleeping = 本轮刚训练完的睡脑（新脑）。
    """
    if sel_data:
        g = _mean(awake_model.mean_nll_batch(sel_data, device)) \
            - _mean(sleeping_model.mean_nll_batch(sel_data, device))
        c = mean_entropy(sleeping_model, sel_data[:_ENTROPY_SAMPLE], device) \
            - mean_entropy(awake_model, sel_data[:_ENTROPY_SAMPLE], device)
    else:
        g = 0.0
        c = 0.0
    if mem_old_data:
        f = _mean(sleeping_model.mean_nll_batch(mem_old_data, device)) \
            - _mean(awake_model.mean_nll_batch(mem_old_data, device))
    else:
        f = 0.0
    d, has_nan = weight_norm_ratio(sleeping_model, awake_model)
    return {"g": g, "f": f, "d": d, "c": c, "has_nan": has_nan}


class Endogenous:
    """内生健康度 h 的滚动归一化 + 合成 + 持久化。

    只持有四个信号的滚动历史（用于相对自身基线的比值归一化）。随 Dolphin
    存档持久化（save/load 用 ck.get(key, 默认) 兼容旧档）。
    """

    def __init__(self):
        self.history = {
            "g": deque(maxlen=_HISTORY_LEN),
            "f": deque(maxlen=_HISTORY_LEN),
            "d": deque(maxlen=_HISTORY_LEN),
            "c": deque(maxlen=_HISTORY_LEN),  # 存坍缩深度 c_pen（非负）
        }

    def _baseline(self, key):
        """信号自身历史的中位基线；空/零/非有限 → None（由调用方兜底）。"""
        m = _median(self.history[key])
        if m is None or not math.isfinite(m) or m == 0.0:
            return None
        return m

    def _ratio(self, key, x, default=1.0):
        """比值归一化：x / 滚动中位基线。首观测以自身绝对值定基线（norm=±1）。"""
        base = self._baseline(key)
        if base is None:
            base = abs(x) if x != 0.0 else default
        if not math.isfinite(base) or base == 0.0:
            base = default
        return x / base

    def update(self, g, f, d, c):
        """喂入一轮四信号，返回 {h, hard_block, g_norm, f_norm, d_norm, c_norm}。"""
        g, f, d, c = float(g), float(f), float(d), float(c)
        # 非有限信号（除 d）按无信号处理，避免污染合成；d 非有限走硬拦截。
        if not math.isfinite(g):
            g = 0.0
        if not math.isfinite(f):
            f = 0.0
        if not math.isfinite(c):
            c = 0.0
        c_pen = max(-c, 0.0)  # 坍缩深度（负=坍缩 → 正=惩罚）
        g_norm = self._ratio("g", g)
        f_norm = self._ratio("f", f)
        c_norm = self._ratio("c", c_pen)
        d_base = self._baseline("d")
        if d_base is None:
            d_base = 1.0
        d_norm = d / d_base if math.isfinite(d) else float("inf")
        hard = (not math.isfinite(d)) or (d_norm > _HARD_D_MULT)
        if hard:
            h = 0.0
        else:
            h = _clamp(g_norm - _ALPHA * f_norm - _BETA * (d_norm - 1.0)
                       - _GAMMA * c_norm, 0.0, 1.0)
        # 滚动历史：d 非有限时以基线值补位（不污染归一化）
        self.history["g"].append(g)
        self.history["f"].append(f)
        self.history["c"].append(c_pen)
        self.history["d"].append(d if math.isfinite(d) else d_base)
        return {"h": h, "hard_block": hard,
                "g_norm": g_norm, "f_norm": f_norm,
                "d_norm": d_norm, "c_norm": c_norm}

    def to_state(self):
        return {k: list(v) for k, v in self.history.items()}

    def from_state(self, st):
        if not st:
            return self
        for k in self.history:
            vals = st.get(k) or []
            self.history[k] = deque(
                [float(x) for x in vals if math.isfinite(float(x))],
                maxlen=_HISTORY_LEN)
        return self
