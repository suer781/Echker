"""经验缓冲区（海马体）。

- 惊讶度门控：带通滤波（律 L4）——过于熟悉的经验跳过、异常经验挂起、中间带优先。
- 窗口去重带惩罚（律 L7）：窗口内密集重复 = 负价值（语义饱和）。
- 睡眠压力：近期惊讶度积累，驱动睡眠时机（值自成）。
"""
import hashlib
import math
import time


class Experience:
    __slots__ = ("id", "data", "surprise", "reward", "ts")

    def __init__(self, eid, data: bytes, surprise: float):
        self.id = eid
        self.data = data
        self.surprise = float(surprise)
        self.reward = 0.0
        self.ts = time.time()


class ExperienceBuffer:
    def __init__(self, cap_bytes=262144, satiation_window=200):
        self.items = []
        self.cap = cap_bytes
        self.satiation_window = satiation_window
        self.sleep_threshold = 4.0  # 值：自成，由睡眠间隔反馈调整
        self._next_id = 0
        self.bytes = 0

    def add(self, data: bytes, surprise: float) -> int:
        e = Experience(self._next_id, bytes(data), surprise)
        self._next_id += 1
        self.items.append(e)
        self.bytes += len(e.data)
        while self.bytes > self.cap and self.items:
            old = self.items.pop(0)
            self.bytes -= len(old.data)
        return e.id

    def feedback(self, eid: int, reward: float) -> bool:
        for e in self.items:
            if e.id == eid:
                e.reward = float(reward)
                return True
        return False

    def pressure(self, k=20) -> float:
        recent = [e.surprise for e in self.items[-k:]]
        return sum(recent) / len(recent) if recent else 0.0

    PRIOR_CENTER = math.log(256)  # 随机初始化模型对任意字节流的理论惊讶度 ln(256)≈5.545
    PRIOR_WIDTH = 2.0

    def band(self):
        """带通中心与宽度：经验带向先验带收缩，样本越少越靠先验。

        修复审计低危项：自引用带在冷启动/异质缓冲下只是弱排序（异常经验仅衰减到 0.61）。
        小样本时混入宽先验带，门控冷启动即有效；n≥64 后纯经验。
        """
        ss = sorted(e.surprise for e in self.items)
        if not ss:
            return self.PRIOR_CENTER, self.PRIOR_WIDTH

        def pct(p):
            return ss[max(0, min(len(ss) - 1, int(p * len(ss))))]

        center, width = pct(0.5), max(0.25, pct(0.75) - pct(0.25))
        w = min(1.0, max(0.0, (len(ss) - 16) / 48.0))  # 16→64 条线性过渡到纯经验带
        return (w * center + (1 - w) * self.PRIOR_CENTER,
                w * width + (1 - w) * self.PRIOR_WIDTH)

    def select(self, budget_frac: float):
        """按价值选拔，预算耗尽即停。返回 ((分, 经验)...入选) 与 (分, 经验)...滞留。"""
        center, width = self.band()
        w0 = len(self.items) - self.satiation_window
        dup_count = {}  # 窗口内此前出现过的重复次数：原件保留最强，重复副本逐次衰减（L7）
        scored = []
        for idx, e in enumerate(self.items):
            h = hashlib.sha256(e.data).hexdigest()[:16]
            prior = 0
            if idx >= w0:  # L9：选拔域=全量缓冲，窗外经验进滞留不丢失；L7 的重复计数只看窗口
                prior = dup_count.get(h, 0)
                dup_count[h] = prior + 1
            band_w = math.exp(-0.5 * ((e.surprise - center) / width) ** 2)
            rw = max(-0.9, min(e.reward, 9.0))  # 负奖励会压低分数，但分数恒正（下限 0.1）
            score = band_w * (1.0 + rw) / (1.0 + prior)
            scored.append((score, e))

        scored.sort(key=lambda t: -t[0])
        budget = int(self.bytes * budget_frac)
        used, sel, rest = 0, [], []
        for s, e in scored:
            if used + len(e.data) <= budget:
                sel.append((s, e))
                used += len(e.data)
            else:
                rest.append((s, e))
        return sel, rest

    def clear(self):
        self.items = []
        self.bytes = 0
