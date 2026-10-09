"""参数自适应控制律（2026-10-06 用户定案）。

控制律（用户定案）：
- 轻限制上限 = 当前参数量 × 1.571（高出 57.1% 是轻限制，允许正常微调）；
- 压力触发（显存或容量任一"不够"）→ 不再增长，三段式下跌：30% → 10% → 3%；
- 正常态每睡眠周期用最小步长（±8 单元）微调，消除"上限误差"；
- 微调全程看显存，不够自动裁剪。

本控制器**完全独立于 LifeController 的判决主路径**（_decide/thermo_cycle）。
它只做排计划：复用 life.py 的现有手术计划机制（设置 life_ctl.plan /
life_ctl.state），由 run_cycle 的既有 pre_train/post_exam 执行与验收。
不修改任何恒温器主路径方法（由 M4-T 系列测试约束）。

设计：
- ParamAdaptive 持有独立的三段式下跌状态机（param_regime 0/1/2/3），
  不依赖 LifeController 内部字段，随 Dolphin 存档持久化。
- 每周期入口 on_cycle 由 run_cycle 末尾钩子调用（仅在
  dolphin.param_adaptive_enabled=True 时）。
- 只在恒温器处于 NORMAL 且无 pending 计划时介入，绝不覆盖恒温器已排计划。
"""
import torch

from .life import (GAP_BORN, MEM_PRESSURE_HI, MEM_PRESSURE_WINDOW,
                   MIN_BODY_FRAC, PARAM_STAGE_RATIOS, STATE_NORMAL,
                   STATE_WITHER_PLAN)
from .surgery import MEM_BUDGET


class ParamAdaptive:
    """参数自适应控制律：独立排计划控制器（随档持久化）。"""

    def __init__(self):
        self.param_regime = 0       # 0=正常 1=跌到30% 2=跌到10% 3=跌到3%
        self.last_param_reset = None
        self.micro_total = 0
        self.mem_ratio_hist = []    # 显存压力历史 [[cycle, ratio], ...]

    # ---------- 持久化 ----------

    def to_state(self):
        return {
            "param_regime": self.param_regime,
            "last_param_reset": self.last_param_reset,
            "micro_total": self.micro_total,
            "mem_ratio_hist": self.mem_ratio_hist,
        }

    def from_state(self, st):
        if not st:
            return self
        self.param_regime = int(st.get("param_regime", 0))
        self.last_param_reset = st.get("last_param_reset")
        self.micro_total = int(st.get("micro_total", 0))
        self.mem_ratio_hist = [[int(c), float(r)]
                               for c, r in (st.get("mem_ratio_hist") or [])]
        return self

    # ---------- 压力检测 ----------

    def _mem_pressure_ratio(self, d):
        """当前显存压力比率 = torch.cuda.memory_reserved() / MEM_BUDGET。
        CPU 无显存约束 → None（不参与收缩）。"""
        if d.device != "cuda":
            return None
        try:
            used = torch.cuda.memory_reserved()
        except Exception:
            return None
        return used / MEM_BUDGET

    def detect_pressure(self, d, gap_bar, mem_ratio=None):
        """压力检测：显存或容量任一"不够"即为压力。

        返回 (压力类型, 说明) 或 (None, "")。
        - 显存压力：mem_ratio 连续 MEM_PRESSURE_WINDOW 周期 > 0.90
        - 容量压力：gap_bar 超过浮动设定点带（用 life_ctl._xi_hi_eff() +
          h_eff()）
        """
        cycle = getattr(d, "cycle", 0)
        # 显存压力（Schmitt 确认窗：连续 MEM_PRESSURE_WINDOW 周期超阈值才触发收缩）
        if mem_ratio is None:
            mem_ratio = self._mem_pressure_ratio(d)
        if mem_ratio is not None:
            self.mem_ratio_hist.append([cycle, round(mem_ratio, 4)])
            del self.mem_ratio_hist[:-(MEM_PRESSURE_WINDOW * 2)]
            if len(self.mem_ratio_hist) >= MEM_PRESSURE_WINDOW and all(
                    r > MEM_PRESSURE_HI
                    for _, r in self.mem_ratio_hist[-MEM_PRESSURE_WINDOW:]):
                return ("mem",
                        f"显存 {mem_ratio:.2f} 连续超预算 {MEM_PRESSURE_HI:.0%}")
        # 容量压力：gap_bar 越浮动设定点带（空位过多=容量过剩，应触发收缩）
        ctl = getattr(d, "life_ctl", None)
        if ctl is not None:
            xi_eff = ctl._xi_hi_eff()
            h = ctl.h_eff()
            if gap_bar > max(GAP_BORN, xi_eff) + h:
                return "cap", f"gap̄={gap_bar:.3f} 越带（空位过多）"
        return None, ""

    # ---------- 计划排程 ----------

    def _schedule_shrink(self, d, report, ratio):
        """排一个 born-again 收缩计划（复用 life_ctl 的手术计划机制）。

        H1/H2：与恒温器主路径 _plan_born_again 一致，排程前做 MIN_BODY_FRAC
        容量下限保护（守卫⑥：防无限萎缩）——目标比例低于 MIN_BODY_FRAC /
        body_frac 时截断到该下限，并在报告中标注"容量下限拦截"；同时把
        target 钳位到 surgery.shrink_config 的合法域 [0.1, 0.95]，确保
        _execute_wither → born_again_student 永不抛 ValueError。
        """
        ctl = d.life_ctl
        cycle = getattr(d, "cycle", 0)
        body_frac = ctl._body_frac(d, d.sleeping().name)
        target = float(ratio)
        capped = False
        if body_frac * target < MIN_BODY_FRAC:
            target = max(target, MIN_BODY_FRAC / body_frac)
            capped = True
        target = max(0.1, min(0.95, target))
        ctl.plan = {"kind": "wither", "mode": "born_again",
                    "target": round(target, 4), "source": "param_adaptive"}
        ctl.state = STATE_WITHER_PLAN
        ctl.last_shrink_cycle = cycle
        ctl.last_born_again = cycle
        report["m4_plan"] = ["wither", "born_again", round(target, 3)]
        report["m4_param_adaptive"] = {
            "regime": self.param_regime,
            "target_ratio": round(target, 4),
            "source": "param_adaptive",
        }
        if capped:
            report["m4_param_adaptive"]["capacity_floor"] = True
            report["m4_param_adaptive"]["note"] = "容量下限拦截"
        return {"kind": "born_again", "target": round(target, 4),
                "class": "参数自适应（压力三段式下跌）",
                "reason": f"跌到 {target:.0%} 参数"
                          + ("（容量下限拦截）" if capped else "")}

    def _advance_regime(self, d, report):
        """上一段收缩完成 → 推进下一段。

        1→2→3 各排一次 born-again 计划；三段全完成回 0（下跌结束，恢复正常增长）。

        H1：区分"上一段验收通过"与"abandoned/回滚"。post_exam 在战役通过
        （体检 passed 换班）的同周期返回 "swap"，run_cycle 会在本钩子之前置
        report["swapped"]=True；而 gen 耗尽放弃时置 m4_wither_abandoned=True
        且 swapped=False。因此：
        - 上一段验收通过（swapped=True 且非 abandoned）→ 推进下一段；
        - 失败/回滚/放弃 → 回退 param_regime 一段（至少不推进），等待压力
          重新确认，避免在无法验收的档位上无限循环。
        """
        cycle = getattr(d, "cycle", 0)
        swapped = bool(report.get("swapped"))
        abandoned = bool(report.get("m4_wither_abandoned"))
        if not swapped or abandoned:
            # 上一段未真正验收通过（回滚/放弃）→ 回退一段，不推进
            self.param_regime = max(0, self.param_regime - 1)
            report["m4_param_adaptive"] = {
                "regime": self.param_regime,
                "rollback": True,
                "note": "上一段收缩未验收通过，回退参数档位",
            }
            return
        if self.param_regime >= len(PARAM_STAGE_RATIOS):
            # 三段全完成 → 回正常态
            self.param_regime = 0
            self.last_param_reset = cycle
            report["m4_param_adaptive"] = {"regime": 0, "stage_done": True}
            return
        self.param_regime += 1
        ratio = PARAM_STAGE_RATIOS[self.param_regime - 1]
        self._schedule_shrink(d, report, ratio)

    # ---------- 微调 ----------

    def _build_gaps(self, ctl):
        """从恒温器 thermo_hist 重建逐位 gap 字典（run_cycle 钩子不传 gaps）。"""
        gaps = {}
        for key, hist in (ctl.thermo_hist or {}).items():
            if hist:
                gaps[key] = hist[-1][1]
        return gaps

    def _micro_tune(self, d, report, gaps, gap_bar):
        """正常态微调：gap_bar 偏高→微缩，headroom 过热→微长（±8 单元）。

        复用 LifeController._micro_tune 的完整实现（该实现已内置全部守卫：
        surgery_allowed / 供给门 / 任何 site 冷却即限频 / 成熟刹车）。
        """
        ctl = getattr(d, "life_ctl", None)
        if ctl is None:
            return None
        if gaps is None:
            gaps = self._build_gaps(ctl)
        return ctl._micro_tune(d, report, gaps, gap_bar)

    # ---------- 每周期入口（run_cycle 末尾调用） ----------

    def on_cycle(self, d, report, gaps=None, gap_bar=None, mem_ratio=None):
        """每周期入口：三段式推进 → 压力检测 → 正常微调。

        只在恒温器处于 NORMAL 且无 pending 计划时介入（不与主路径冲突）。
        若恒温器本周期已排 grow/wither 计划，本控制器不介入。
        """
        ctl = getattr(d, "life_ctl", None)
        if ctl is None:
            return
        # 恒温器已有计划 / 战役进行中 → 不介入（不覆盖）
        if ctl.state != STATE_NORMAL or ctl.plan is not None:
            return
        # gap_bar：优先用恒温器本周期记录的全局过剩指数（最准确）
        if gap_bar is None or (gap_bar == 0.0 and ctl.gapbar_hist):
            gap_bar = ctl.gapbar_hist[-1][1] if ctl.gapbar_hist else 0.0
        # 1) 三段式下跌推进（上一段收缩完成 → 下一段）
        if self.param_regime > 0:
            self._advance_regime(d, report)
            if self.param_regime > 0:
                return  # 已排下一段计划，本周期不再微调
        # 2) 压力检测 → 触发三段式第一段
        pressure_kind, pressure_why = self.detect_pressure(d, gap_bar, mem_ratio)
        if pressure_kind is not None:
            self.param_regime = 1
            self.last_param_reset = getattr(d, "cycle", 0)
            self._schedule_shrink(d, report, PARAM_STAGE_RATIOS[0])
            return
        # 3) 正常微调（最小步长 ±8 单元）
        self._micro_tune(d, report, gaps, gap_bar)