"""成年期灵敏度控制——项目杀手公式的替代实现。

历史背景：旧实现存在三项缺陷——增益随裸步数日历衰减、稳态正比于年龄漂移、
钳位边界异常。本实现遵循以下原则（律固定，值自成）：

- 关键期不在这里——它由胎教承担（pretrain.py 的 warmup+cosine，声明式有限调度）。
  出生后即为成年状态；成年期可塑性永不为零，但必须限制在带通范围内。
- 增益来自世界（惊讶度），不来自日历：世界越陌生越允许大胆学，平静期回落。
- 一个变量一个控制器：本类是睡眠学习率的唯一持有者（含体检否决后的退火）。
- 钳位来自域：学习率的物理上下界是律的一部分，不是任意的 ±10000。
"""


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Plasticity:
    def __init__(self, lr_base=5e-5, lr_min=1e-5, lr_max=2e-4, kappa=0.5, anneal=0.9,
                 band_edge=2.0):
        self.lr_base = lr_base   # 平静期的锚点
        self.lr_min = lr_min     # 律：永不归零（可塑性为零 = 系统僵死）
        self.lr_max = lr_max     # 律：永不越界
        self.kappa = kappa       # 值：自成——惊讶度到灵敏度的传导系数
        self.anneal = anneal     # 体检否决时的退火系数
        self.band_edge = band_edge  # 值：带通边缘（超过视为异常输入，LR 回落）

    def next_lr(self, lr, mean_surprise, band_center, band_width, probe_passed):
        """根据本轮经验惊讶度与体检结果，给下一轮睡眠定灵敏度。

        - 体检否决 → 无条件退火（从当前值出发，不由 base 重来，避免震荡）。
        - 通过 → base × (1 + κ·|惊讶 − 常态中心|/带宽)：世界偏离自身常态越远，
          越允许大胆学；世界如常则回落到 base。惊讶永远相对常态度量，
          绝对 NLL 高不等于惊讶（那是模型的正常读字成本）。

        2026-10-05 自动闭环（挂账裁决）：原实现为"高通"——ratio 单调上升，
        异常输入反而获得最高学习率（达到 lr_max），与律 L4"异常输入挂起"字面冲突。
        现改为**真带通**：ratio 超过 BAND_EDGE 后 LR 回落（异常输入挂起），
        中间带优先。保留"越偏离常态越敢学"的合理内核（带内单调上升），
        但超出带边缘后学习率衰减——L4 的字面语义与"值自成"哲学同时满足。
        """
        if not probe_passed:
            return clamp(lr * self.anneal, self.lr_min, self.lr_max)
        ratio = abs(mean_surprise - band_center) / max(band_width, 1e-6)
        # 带通边缘（值：自成，律定域内）：超过该倍带宽视为异常输入，学习率回落
        band_edge = getattr(self, "band_edge", 2.0)
        if ratio <= band_edge:
            # 带内：偏离常态越远学习率越高（原高通内核，保留）
            return clamp(self.lr_base * (1.0 + self.kappa * ratio), self.lr_min, self.lr_max)
        # 带外（异常输入挂起）：LR 随偏离程度衰减回 base——L4 字面语义
        over = ratio - band_edge
        decay = 1.0 / (1.0 + over)
        return clamp(self.lr_base * (1.0 + self.kappa * band_edge) * decay,
                     self.lr_min, self.lr_max)
