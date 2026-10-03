"""成年期灵敏度控制——"项目杀手"公式的转世体。

前世死因（验尸报告）：增益随裸步数日历衰减、稳态正比于年龄漂移、钳位变磁铁。
转世原则（律固定，值自成）：

- 关键期不在这里——它由胎教承担（pretrain.py 的 warmup+cosine，声明式有限调度）。
  出生即已成年；成年大脑可塑性永不为零，但必须关在带通里。
- 增益来自世界（惊讶度），不来自日历：世界越陌生越允许大胆学，平静期回落。
- 一个变量一个控制器：本类是睡眠学习率的唯一持有者（含体检否决后的退火）。
- 钳位来自域：学习率的物理上下界是律的一部分，不是任意的 ±10000。
"""


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Plasticity:
    def __init__(self, lr_base=5e-5, lr_min=1e-5, lr_max=2e-4, kappa=0.5, anneal=0.9):
        self.lr_base = lr_base   # 平静期的锚点
        self.lr_min = lr_min     # 律：永不归零（可塑性死亡 = 系统僵死）
        self.lr_max = lr_max     # 律：永不爆表
        self.kappa = kappa       # 值：自成——惊讶度到灵敏度的传导系数
        self.anneal = anneal     # 体检否决时的退火系数

    def next_lr(self, lr, mean_surprise, band_center, band_width, probe_passed):
        """根据本轮经验惊讶度与体检结果，给下一轮睡眠定灵敏度。

        - 体检否决 → 无条件退火（当前值出发，不由 base 重来，避免震荡）。
        - 通过 → base × (1 + κ·|惊讶 − 常态中心|/带宽)：世界偏离自身常态越远，
          越允许大胆学；世界如常则回落到 base。惊讶永远相对常态度量，
          绝对 NLL 高不等于惊讶（那是模型的正常读字成本）。
        """
        if not probe_passed:
            return clamp(lr * self.anneal, self.lr_min, self.lr_max)
        ratio = abs(mean_surprise - band_center) / max(band_width, 1e-6)
        return clamp(self.lr_base * (1.0 + self.kappa * ratio), self.lr_min, self.lr_max)
