"""生命节律控制器（M4）：睡眠周期的**唯一**实现 + 自生长/自凋零状态机。

取代路线（交接 §6-C）：
  - life.run_cycle 是唯一睡眠周期实现：完整复刻 sleep.run_cycle 的 M0 全语义
    （选拔/滞留入记忆/变异重放/KD 双损失/门控/换班或回滚/阈值反馈/Plasticity），
    feeding=True 时额外复刻 feed.trainer_cycle 的线程安全语义（做梦注入、
    feed_lock 摘除式快照、learn_busy 等待、锁内换班）——trainer_cycle 自此删除。
  - dolphin/sleep.py 文件本体冻结为历史件；M0 等价性由 tests 钉死：同种子同
    输入下 life.run_cycle 与 sleep.run_cycle 行为一致（迁移安全网）。
  - M4 钩子（d.life_enabled 总开关一键全关，L11 人工安全阀）：
      周期开头 pre_train（GROW 执行 widen / WITHER 换装或继续战役）
      → NREM 样训练（与 M0 同一循环；born-again 验收周期用手术验收 kd_alpha）
      → REM 样子相位（ReDo 回收 / 软衰减 / 手术排程）
      → 体检（与 M0 同一 gate）→ post_exam（锚/平台期/战役验收）。

状态机（规格 C）：
  NORMAL → GROW_PLAN → GROWN → 体检 → NORMAL / ROLLBACK
  NORMAL → WITHER_PLAN → WITHERING（多周期分代；学生保留标志：体检失败不回滚
  睡脑学生，逐周期续训直到验收或放弃——回滚会撤销凋零/学生进度，战役永不收敛）

触发信号（规格 C）：体检平台期（连续 N 周期改善<ε）、ghost_probe 增益（区分
优化/容量平台期）、显存余量、记忆库压力、睡眠债守卫（睡眠债高禁结构手术——
Bellesi 2017：慢性睡眠剥夺损害突触发生）。

固定干预次序（规格 C，不许跳级）：ReDo 回收 → LR 退火 → 手术。以平台期深度
为阶梯：plateau∈[N,2N) 第一阶梯（ReDo 代谢常开）、[2N,3N) 第二阶梯追加
LR 退火、≥3N 第三阶梯排程手术。

锚：历史最优 probe（anchor_best，持久化进 save/load；规格 C）。

常量来源身份（律固定，值自成）：
  PLATEAU_CYCLES=3       【值自成】平台期阶梯步长（周期数）
  EPS_PLATEAU=0.005      【定标】Q6：ε=2×配对SE 上界 0.0056 NLL，低于此算无改善
  GHOST_GAIN_MIN=1e-4    【值自成】容量平台判据初值（在线自校准通道，Q6 §6.3）
  WITHER_DORMANT_FRAC=0.10   【值自成】凋零排程的休眠占比下限
  WITHER_BORN_FRAC=0.25      【值自成】休眠占比高到直接 born-again 换装
  WITHER_TARGET=0.7          【值自成】born-again 学生体型比例
  WITHER_KD_ALPHA=0.75       【定标】规格带 0.7–0.8 取中（仅手术验收周期）
  WITHER_MAX_GEN=3           【值自成】凋零战役分代上限（规格"多周期分代"）
  REDO_MAX_FRAC=0.05         【值自成】单周期回收上限（审计：5% 上限改纯反馈量）
  DECAY_GAMMA=0.7            【值自成】软衰减单代系数
  SURGERY_LR_WINDOW=3        【值自成】规格带 2–3 周期取上，之后交还 Plasticity
  SURGERY_LR_MULT=2.0        【律定承袭规格 B】base×2，≤ lr_max
  SLEEP_DEBT_GUARD=4.0       【值自成】_since_sleep > target_interval×4 禁手术
  ROLLBACK_GUARD=3           【值自成】连续回滚 ≥3 次禁手术（脑在挣扎，先养）
  GROW_DELTA_FRAC=0.05       【值自成】变宽幅度 = site 宽度的 5%
  GROW_COOLDOWN=6            【值自成】连续手术回滚后的冷却周期
  MAX_GROW_ATTEMPTS=2        【值自成】冷却前允许的手术回滚次数
"""
import time

import torch
import torch.nn.functional as F

from .sleep import varied_replay  # 律 L6 唯一实现，复用不复制
from .surgery import (born_again_student, break_symmetry_dmodel, delta_params_attn_v,
                      delta_params_dmodel, delta_params_mlp, ghost_scan,
                      mem_budget_ok, morphology_of, rebuild_optimizer, widen)
from .vitals import Vitals

# —— 常量（来源身份见模块 docstring）——
PLATEAU_CYCLES = 3
EPS_PLATEAU = 0.005
GHOST_GAIN_MIN = 1e-4
WITHER_DORMANT_FRAC = 0.10
WITHER_BORN_FRAC = 0.25
WITHER_TARGET = 0.7
WITHER_KD_ALPHA = 0.75
WITHER_MAX_GEN = 3
REDO_MAX_FRAC = 0.05
DECAY_GAMMA = 0.7
SURGERY_LR_WINDOW = 3
SURGERY_LR_MULT = 2.0
SLEEP_DEBT_GUARD = 4.0
ROLLBACK_GUARD = 3
GROW_DELTA_FRAC = 0.05
GROW_COOLDOWN = 6
MAX_GROW_ATTEMPTS = 2

STATE_NORMAL = "NORMAL"
STATE_GROW_PLAN = "GROW_PLAN"
STATE_GROWN = "GROWN"
STATE_WITHER_PLAN = "WITHER_PLAN"
STATE_WITHERING = "WITHERING"


def _threshold_feedback(d):
    """值自成：睡眠频率反馈（G9：乘法结果必须过 clamp_threshold 域钳位）。

    模块级唯一实现：life.run_cycle(feeding=True) 内部与 SleepTrainer.run 的异常
    兜底共用（原 feed.trainer_cycle 闭包版随取代一并消亡）。
    钳位函数懒加载：dolphin.py ↔ life.py 各为对方模块级依赖，环内只允许
    函数级引用（G9：钳位唯一实现，不许出现第二份乘法）。
    """
    from .dolphin import clamp_threshold
    with d.feed_lock:
        if d._since_sleep < d.target_interval // 2:
            d.buffer.sleep_threshold = clamp_threshold(d.buffer.sleep_threshold * 1.10)
        elif d._since_sleep > d.target_interval * 2:
            d.buffer.sleep_threshold = clamp_threshold(d.buffer.sleep_threshold * 0.90)
        d._since_sleep = 0


class LifeController:
    """自生长/自凋零状态机 + 双半球营养账本 + 形态记录。

    由 Dolphin 持有（d.life_ctl）；save/load 经 to_state/from_state 持久化
    （锚、状态机、形态、账本——规格 C：锚=历史最优 probe，持久化进 save/load）。
    """

    def __init__(self):
        self.state = STATE_NORMAL
        self.plateau = 0            # 连续平台期周期数（margin < EPS_PLATEAU）
        self.anchor_best = None     # 锚：历史最优 probe NLL（持久化）
        self.plan = None            # 待执行手术计划（下一周期 pre_train 执行）
        self.campaign = None        # 战役 dict（grow/wither 进行时）
        self.lr_window = 0          # 手术 LR 重启窗口剩余周期（之后交还 Plasticity）
        self.anneal_pending = False  # 第二阶梯 LR 退火（一次性）
        self.anneal_band = 0        # 已退火到的阶梯号（不重复退火）
        self.consec_rollback = 0
        self.grow_attempts = 0
        self.cooldown = 0           # 手术冷却剩余周期
        self.morphology = {}        # 半球名 → morphology dict（持久化）
        self.vitals = {}            # 半球名 → Vitals（持久化）

    # ---------- 持久化 ----------

    def to_state(self):
        return {"state": self.state, "plateau": self.plateau,
                "anchor_best": self.anchor_best, "plan": self.plan,
                "campaign": self.campaign, "lr_window": self.lr_window,
                "anneal_band": self.anneal_band,
                "consec_rollback": self.consec_rollback,
                "grow_attempts": self.grow_attempts, "cooldown": self.cooldown,
                "morphology": self.morphology,
                "vitals": {k: v.to_state() for k, v in self.vitals.items()}}

    def from_state(self, st):
        self.state = st.get("state", STATE_NORMAL)
        self.plateau = int(st.get("plateau", 0))
        self.anchor_best = st.get("anchor_best")
        self.plan = st.get("plan")
        self.campaign = st.get("campaign")
        self.lr_window = int(st.get("lr_window", 0))
        self.anneal_band = int(st.get("anneal_band", 0))
        self.consec_rollback = int(st.get("consec_rollback", 0))
        self.grow_attempts = int(st.get("grow_attempts", 0))
        self.cooldown = int(st.get("cooldown", 0))
        self.morphology = st.get("morphology") or {}
        self.vitals = {k: Vitals().from_state(v)
                       for k, v in (st.get("vitals") or {}).items()}
        return self

    # ---------- 半球账本 / 形态 ----------

    def vitals_for(self, d, hname):
        v = self.vitals.get(hname)
        if v is None:
            v = self.vitals[hname] = Vitals()
        return v

    def morph_for(self, d, hname):
        m = self.morphology.get(hname)
        if m is None:
            m = self.morphology[hname] = {"d_model_k": 1, "mlp": {}, "attn_v": {}}
        return m

    # ---------- 守卫 ----------

    def sleep_debt(self, d):
        """睡眠债（Bellesi 2017 守卫）：距上次睡眠的交互数 / 期望间隔。"""
        return d._since_sleep / max(1, d.target_interval)

    def surgery_allowed(self, d):
        """结构手术守卫：冷却 / 睡眠债高 / 连续回滚 → 禁手术（规格 C 触发信号）。"""
        if self.cooldown > 0:
            return False, f"手术冷却中（还剩 {self.cooldown} 周期）"
        if self.sleep_debt(d) > SLEEP_DEBT_GUARD:
            return False, (f"睡眠债 {self.sleep_debt(d):.1f} > {SLEEP_DEBT_GUARD}"
                           f"（Bellesi 2017 守卫：剥夺期禁突触发生）")
        if self.consec_rollback >= ROLLBACK_GUARD:
            return False, f"连续回滚 {self.consec_rollback} 次，先养脑不动刀"
        return True, "ok"

    # ---------- 快照 / 还原（战役与手术的回退点） ----------

    def _snapshot(self, d, h):
        return {"sd": {k: t.detach().cpu().clone()
                       for k, t in h.model.state_dict().items()},
                "morph": {k: (dict(v) if isinstance(v, dict) else v)
                          for k, v in self.morph_for(d, h.name).items()},
                "vitals": self.vitals_for(d, h.name).to_state(),
                "lr": h.opt.param_groups[0]["lr"]}

    def _restore(self, d, h, snap):
        """按快照形态重建睡脑并装入快照权重；账本同步恢复；优化器重建为新鲜
        AdamW（动量不跨战役保留——如实标注的简化）。"""
        from .surgery import apply_morphology
        model, _ = apply_morphology(d.cfg, snap.get("morph"), d.device)
        sd = {k: t.to(d.device) for k, t in (snap.get("sd") or {}).items()}
        if sd:
            model.load_state_dict(sd)
        h.model = model
        h.opt = torch.optim.AdamW(model.parameters(), lr=snap.get("lr", 5e-5),
                                  weight_decay=0.01)
        self.morphology[h.name] = snap.get("morph") or {"d_model_k": 1, "mlp": {}, "attn_v": {}}
        self.vitals[h.name] = Vitals().from_state(snap.get("vitals") or {})

    # ---------- M4 钩子①：周期开头（GROW 执行 / WITHER 换装或续期） ----------

    def pre_train(self, d, report, stream):
        h = d.sleeping()
        v = self.vitals_for(d, h.name)
        if self.state == STATE_GROW_PLAN and self.plan:
            self._execute_grow(d, report, v)
        elif self.state == STATE_WITHER_PLAN and self.plan:
            self._execute_wither(d, report, v)
        elif self.state == STATE_WITHERING and self.campaign:
            report["m4_wither_gen"] = self.campaign.get("gen")

    def _execute_grow(self, d, report, v):
        h = d.sleeping()
        plan = self.plan
        ok, why = self.surgery_allowed(d)
        if not ok:  # 守卫在排程后恶化：推迟一周期（计划保留）
            report["m4_defer"] = why
            return
        old_model, old_opt = h.model, h.opt  # widen 前抓引用（opt.state 以参数对象为键）
        lr = old_opt.param_groups[0]["lr"]
        snap = self._snapshot(d, h)
        try:
            rec = widen(old_model, plan["layer"], plan["delta"], axis=plan["axis"],
                        seed=plan.get("seed", 0), device=d.device)
        except ValueError as e:  # 轴不可行（如③遇移植体）：放弃计划退回 NORMAL
            report["m4_grow_failed"] = str(e)
            self.state = STATE_NORMAL
            self.plan = None
            return
        if rec["axis"] == "d_model":
            h.model = rec["new_model"]
            v.remap_split(rec["delta"])  # 复制分裂：子通道继承父账本值/k
            break_symmetry_dmodel(h.model, rec["delta"], seed=plan.get("seed", 0) + 1)
            report["m4_break_symmetry"] = True
        # 手术后：受影响参数优化器状态重置（同名同形状迁移动量）+ LR 重启（规格 B）
        h.opt, optstat = rebuild_optimizer(old_model, old_opt, h.model, lr)
        self.lr_window = SURGERY_LR_WINDOW
        morph = morphology_of(h.model)
        prev = self.morph_for(d, h.name)
        if rec["axis"] == "d_model":
            # morphology_of 只记录移植体、把 d_model_k 重置为 1——平铺史必须累乘
            # 保留，否则存档 load 按基座 cfg 重建 → 形状不匹配崩溃。
            morph["d_model_k"] = int(prev.get("d_model_k", 1) or 1) * int(rec["delta"])
        if prev.get("shrink"):  # 在学生脑上动刀：学生体型记录延续（load 重建依赖）
            morph["shrink"] = prev["shrink"]
        self.morphology[h.name] = morph
        self.campaign = {"kind": "grow", "gen": 1, "snap": snap}
        self.state = STATE_GROWN
        self.plan = None
        report["m4_surgery"] = {k2: rec[k2] for k2 in rec if k2 != "new_model"}
        report["m4_opt"] = optstat
        report["m4_lr_window"] = SURGERY_LR_WINDOW

    def _execute_wither(self, d, report, v):
        h = d.sleeping()
        plan = self.plan
        ok, why = self.surgery_allowed(d)
        if not ok:  # 守卫在排程后恶化：推迟一周期（计划保留，与 _execute_grow 对称）
            report["m4_defer"] = why
            return
        mode = plan.get("mode", "decay")
        snap = self._snapshot(d, h)
        rec = {"mode": mode}
        if mode == "born_again":
            # 教师=现脑（规格 B）：学生体型从睡脑**当前**形态收缩——若现脑已平铺
            # （d_model≠基座），d.cfg 的基座不是教师真身。账本按当前形态供逐层重要性。
            student, opt, info = born_again_student(
                h.model.cfg, v, WITHER_TARGET, d.device, seed=d.cycle)
            h.model = student
            h.opt = opt
            morph = morphology_of(student)
            morph["d_model_k"] = 1  # 学生从头出生：平铺史已折进 shrink 记录，不再延续
            # 学生形态记录：没有它，存档 load 按基座 cfg 重建 → 形状不匹配崩溃
            morph["shrink"] = {"d_model": student.cfg.d_model,
                               "n_layers": student.cfg.n_layers}
            self.morphology[h.name] = morph
            self.vitals[h.name] = Vitals()  # 新脑新账本（账本随脑重置）
            rec["born_again"] = info
        else:
            rec["targets"] = {k: len(c) for k, c in (plan.get("targets") or {}).items()}
        self.campaign = {"kind": "wither", "mode": mode, "gen": 1,
                         "max_gen": WITHER_MAX_GEN, "snap": snap,
                         "targets": plan.get("targets") or {}}
        self.state = STATE_WITHERING
        self.plan = None
        report["m4_wither_start"] = rec

    # ---------- M4 钩子②：REM 样子相位（训练后、体检前） ----------

    def rem_phase(self, d, report, stream):
        h = d.sleeping()
        v = self.vitals_for(d, h.name)
        # ① ReDo 回收（常规代谢，每周期；固定干预次序的第一级）
        report["m4_redo"] = self._redo(d, h, v)
        # ② 凋零战役的软衰减分代（decay 模式的固定动作；born-again 模式无需衰减）
        if self.state == STATE_WITHERING and self.campaign.get("mode") == "decay":
            report["m4_decay"] = self._decay_generation(d, h, v)
        # ③ 手术排程（第三阶梯；仅 NORMAL 评估）
        if self.state == STATE_NORMAL:
            self._schedule(d, report, v, stream)

    def _bypass_handles(self, h, li, key):
        """旁路 (旁路Linear, 输出权重) 句柄。key 形如 b3.mlp_hidden.new.0。"""
        bidx = int(key.rsplit(".", 1)[1])
        if "mlp_hidden" in key:
            lin = h.model.blocks[li].mlp.bypass_fcs[bidx]
            pw = h.model.blocks[li].mlp.proj_new_ws[bidx]
        else:
            lin = h.model.blocks[li].attn.bypass_qkvs[bidx]
            pw = h.model.blocks[li].attn.proj_new_ws[bidx]
        return lin, pw

    def _redo(self, d, h, v):
        """ReDo 回收：确认休眠的 LN 免疫位单元（bottom-5% × 2 周期确认、非观察期）
        权重重置 + 账本重置 + probation（规格 D/A）。单周期上限 REDO_MAX_FRAC
        （审计：5% 上限改纯反馈量）。残差流位（ln1_out/ln2_out）不回收——残差流
        通道是全场共享的插件引脚，动它=动全身（对质红线同源）。"""
        dorm, frac = v.dormant_report()
        did = {}
        for key, chans in dorm.items():
            if ".ln1_out" in key or ".ln2_out" in key:
                continue
            li = int(key.split(".")[0][1:])
            led = v.sites[key]
            cap = max(1, int(round(REDO_MAX_FRAC * len(chans))))
            picked = sorted(chans, key=lambda c: led.stable_tay[c])[:cap]  # 最沉睡优先
            g = torch.Generator().manual_seed(hash((key, d.cycle)) & 0x7FFFFFFF)
            with torch.no_grad():
                if ".new." in key:  # 旁路新单元
                    lin, pw = self._bypass_handles(h, li, key)
                    for c in picked:
                        lin.weight[c].normal_(0, 0.02, generator=g)
                        lin.bias[c].zero_()
                        pw[:, c].zero_()
                elif key.endswith(".mlp_hidden"):
                    m = h.model.blocks[li].mlp
                    for c in picked:
                        m.fc.weight[c].normal_(0, 0.02, generator=g)
                        m.fc.bias[c].zero_()
                        m.proj.weight[:, c].zero_()
                elif key.endswith(".attn_out"):
                    a = h.model.blocks[li].attn
                    C = a.qkv.in_features
                    for c in picked:
                        a.qkv.weight[2 * C + c].normal_(0, 0.02, generator=g)
                        a.qkv.bias[2 * C + c].zero_()
                        a.proj.weight[:, c].zero_()
                else:
                    continue
            v.sites[key].remap_reset(picked)
            did[key] = picked
        return {"recycled": did, "dormant_frac": round(frac, 4)}

    def _decay_generation(self, d, h, v):
        """软衰减分代（凋零战役 decay 模式）：目标休眠单元整体 ×DECAY_GAMMA
        （输入行+输出列一起衰减——单元从功能与账本两条线上一起淡出）。"""
        targets = self.campaign.get("targets") or {}
        did = {}
        with torch.no_grad():
            for key, chans in targets.items():
                if key not in v.sites:
                    continue
                li = int(key.split(".")[0][1:])
                gamma = DECAY_GAMMA
                touched = 0
                if ".new." in key:
                    lin, pw = self._bypass_handles(h, li, key)
                    for c in chans:
                        if c < lin.weight.shape[0]:
                            lin.weight[c] *= gamma
                            lin.bias[c] *= gamma
                            pw[:, c] *= gamma
                            touched += 1
                elif key.endswith(".mlp_hidden"):
                    m = h.model.blocks[li].mlp
                    for c in chans:
                        if c < m.fc.weight.shape[0]:
                            m.fc.weight[c] *= gamma
                            m.fc.bias[c] *= gamma
                            m.proj.weight[:, c] *= gamma
                            touched += 1
                elif key.endswith(".attn_out"):
                    a = h.model.blocks[li].attn
                    C = a.qkv.in_features
                    for c in chans:
                        if c < C:
                            a.qkv.weight[2 * C + c] *= gamma
                            a.qkv.bias[2 * C + c] *= gamma
                            a.proj.weight[:, c] *= gamma
                            touched += 1
                if touched:
                    did[key] = touched
        # 分代计数唯一归 post_exam（监督审计 P1-1a）：此处曾再 +1，与 post_exam
        # 双重递增——名义 WITHER_MAX_GEN=3 实际只走 2 代，分代上限字面失真。
        return {"decayed": did, "gen": self.campaign.get("gen", 1)}

    def _schedule(self, d, report, v, stream):
        """手术排程（固定干预次序的阶梯；只有平台期够深才动刀）。"""
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        band = self.plateau // PLATEAU_CYCLES
        if band < 1:
            return
        if band >= 2 and self.anneal_band < 2:
            self.anneal_pending = True  # 第二阶梯：LR 退火（一次性，Plasticity 基线上）
            self.anneal_band = 2
            report["m4_lr_anneal"] = True
        if band < 3 or self.plan:
            return
        ok, why = self.surgery_allowed(d)
        if not ok:
            report["m4_surgery_blocked"] = why
            return
        # ghost 扫描：区分优化平台期（有增益→加单元）与容量过剩平台期（无增益+休眠多→凋零）
        gains = {}
        if stream and len(stream) > d.cfg.block_size + 2:
            try:
                gains = ghost_scan(d.sleeping().model, stream[:4096], seed=d.cycle)
            except (ValueError, RuntimeError):
                gains = {}
        best_gain = max(gains.values(), default=0.0)
        best_key = max(gains, key=gains.get) if gains else None
        dorm, frac = v.dormant_report()
        mem_pressure = len(d.memory.entries) >= d.memory.cap * 0.9
        report["m4_ghost_best"] = ([best_key[0], best_key[1]], round(best_gain, 6)) \
            if best_key else None
        report["m4_dormant_frac"] = round(frac, 4)
        if best_gain >= GHOST_GAIN_MIN and best_key is not None:
            li, site = best_key
            axis = {"mlp_hidden": "mlp", "attn_v": "attn_v"}.get(site)
            if axis is None:  # 幽灵探测位 ≠ 变宽轴（残差流位已被探测层拒绝，双保险）
                report["m4_note"] = f"探测位 {site} 无对应变宽轴，不排程"
                return
            delta = self._grow_delta(d, axis)
            okm, why = self._mem_ok(d, axis, delta)
            if not okm:
                report["m4_defer"] = f"显存预算不过：{why}"  # 规格 B：超 1.9GB 推迟
                return
            self.plan = {"kind": "grow", "axis": axis, "layer": li, "delta": delta,
                         "seed": d.cycle}
            self.state = STATE_GROW_PLAN
            report["m4_plan"] = ["grow", axis, li, delta]
        elif frac >= WITHER_BORN_FRAC:
            self.plan = {"kind": "wither", "mode": "born_again"}
            self.state = STATE_WITHER_PLAN
            report["m4_plan"] = ["wither", "born_again", round(frac, 3)]
        elif frac >= WITHER_DORMANT_FRAC:
            self.plan = {"kind": "wither", "mode": "decay", "targets": dorm}
            self.state = STATE_WITHER_PLAN
            report["m4_plan"] = ["wither", "decay", round(frac, 3)]
        elif mem_pressure:
            # 记忆库压力但幽灵无增益：瓶颈不在容量，不手术（如实报告）
            report["m4_note"] = "记忆库压力大但幽灵无增益：瓶颈不在容量"

    def _grow_delta(self, d, site):
        # 监督审计 P0-2：幅度必须按睡脑**真实**脑形计（d.sleeping().model.cfg），
        # 不能用基座 d.cfg——born-again 学生脑上名义 5% 会实际排成 7%+，轴③
        # 平铺脑上会排成 2.5%（morphology 含 shrink / d_model_k 记录的脑）。
        cfg = d.sleeping().model.cfg
        base = cfg.d_model if site != "mlp" else 4 * cfg.d_model
        return max(8, int(round(GROW_DELTA_FRAC * base)))

    def _mem_ok(self, d, site, delta):
        # 同 P0-2：Δ参数核算也按真实脑形的通道数——显存账按真实 C 才对得上。
        cfg = d.sleeping().model.cfg
        C = cfg.d_model
        if site == "mlp":
            dp = delta_params_mlp(C, delta)
        elif site == "attn_v":
            dp = delta_params_attn_v(C, delta)
        else:
            dp = delta_params_dmodel(cfg, max(2, delta))
        return mem_budget_ok(d.device, dp)

    # ---------- M4 钩子③：体检后（锚 / 平台期 / 战役验收） ----------

    def post_exam(self, d, report, passed):
        """返回换班指令："swap" | "rollback" | "keep"。

        NORMAL：与 M0 逐字等价（passed→swap / fail→rollback）。
        GROWN：fail → ROLLBACK（还原形态+装回醒脑权重，规格 C 状态机）。
        WITHERING：fail → keep（学生保留标志，规格 C）；分代耗尽 → 放弃还原。
        """
        probe_new = report.get("probe_new")
        margin = report.get("gate_margin")
        # 平台期计数（规格 C：连续 N 周期改善<ε）
        if margin is not None and margin < EPS_PLATEAU:
            self.plateau += 1
        else:
            self.plateau = 0
        # 锚更新（规格 C：历史最优 probe）
        if passed and probe_new is not None:
            if self.anchor_best is None or probe_new < self.anchor_best:
                self.anchor_best = probe_new
                report["m4_anchor_new"] = self.anchor_best
        h = d.sleeping()

        if self.state == STATE_GROWN:
            if passed:
                self.state = STATE_NORMAL
                self.campaign = None
                self.grow_attempts = 0
                self.consec_rollback = 0
                return "swap"
            # ROLLBACK：还原手术前形态 + 装回醒脑权重（M0 回滚语义"与醒脑同源"；
            # 醒脑形状不同时——如born-again学生上岗而睡脑曾生长——保持快照权重，
            # 克隆只对同构半球成立）
            snap = (self.campaign or {}).get("snap")
            if snap:
                self._restore(d, h, snap)
                sd_s, sd_a = h.model.state_dict(), d.awake().model.state_dict()
                if sd_s.keys() == sd_a.keys() and all(
                        sd_s[k].shape == sd_a[k].shape for k in sd_s):
                    h.model.load_state_dict(d.awake().model.state_dict())
            self.state = STATE_NORMAL
            self.campaign = None
            self.plan = None
            self.consec_rollback += 1
            self.grow_attempts += 1
            if self.grow_attempts >= MAX_GROW_ATTEMPTS:
                self.cooldown = GROW_COOLDOWN
                self.grow_attempts = 0
            report["m4_rollback"] = "grow"
            return "rollback"

        if self.state == STATE_WITHERING:
            if passed:
                self.state = STATE_NORMAL
                self.campaign = None
                self.consec_rollback = 0
                return "swap"
            camp = self.campaign or {}
            camp["gen"] = camp.get("gen", 1) + 1
            # 学生保留标志（规格 C）：born-again 与 decay 同一多周期分代语义——
            # 从头初始化的学生不可能一个周期内赢过教师，"失败即放弃"会让
            # born-again 战役永不收敛；分代上限到点才放弃还原。
            if camp.get("gen", 0) > camp.get("max_gen", WITHER_MAX_GEN):
                self._restore(d, h, camp.get("snap") or {})
                self.state = STATE_NORMAL
                self.campaign = None
                self.consec_rollback += 1
                report["m4_wither_abandoned"] = True
                return "rollback"
            report["m4_wither_keep"] = camp.get("gen")  # 学生保留标志：不回滚
            # keep 不计入 consec_rollback（监督审计 P1-1c）：keep 是"学生保留续训"，
            # 不是回滚——误递增会让 ROLLBACK_GUARD 把正常的多周期战役读成"脑在挣扎"
            return "keep"

        # NORMAL：与 M0 逐字等价
        if passed:
            self.consec_rollback = 0
            return "swap"
        self.consec_rollback += 1
        return "rollback"


# ---------------------------------------------------------------------------
# 睡眠周期：唯一实现（M0 全语义 + feeding 线程安全语义 + M4 钩子）
# ---------------------------------------------------------------------------

def run_cycle(dolphin, steps=40, kd_alpha=0.5, kd_T=2.0, verbose=True, feeding=False):
    """睡眠周期全流程。

    feeding=False：与 sleep.run_cycle 逐字同语义（M0 等价性，tests 钉死）；
    feeding=True：与原 feed.trainer_cycle 同语义（做梦注入、摘除式快照、
    learn_busy 等待、锁内换班、内部阈值反馈）。
    M4 钩子仅在 d.life_enabled 且 life_ctl 在场时生效；NORMAL 空账本下全部
    为无训练副作用的观察者（等价性测试钉死）。
    """
    ctl = getattr(dolphin, "life_ctl", None)
    m4 = bool(ctl) and bool(getattr(dolphin, "life_enabled", True))
    # G8 fail-fast：探测集缺失时抛 GateDisabled 拒绝开睡（唯一入口不变）
    dolphin.ensure_gate_ready()
    buf, rng, dev = dolphin.buffer, dolphin.rng, dolphin.device
    awake, sleeping = dolphin.awake(), dolphin.sleeping()
    lock = dolphin.feed_lock
    report = {"cycle": dolphin.cycle}

    if feeding:
        # ① 锁内快照（毫秒级）：复活 + 做梦 + 选拔 + 摘除（原 trainer_cycle 语义；
        #    带通在摘除前对完整缓冲取定）
        with lock:
            report["candidates"] = len(buf.items)
            center, width = buf.band()
            for rb in dolphin.memory.resurrect():
                buf.add(rb, surprise=center)
            queries = [e.data for e in buf.items[-8:]]
            dreamed = dolphin.memory.dream(queries, k=dolphin.dream_k)
            for en in dreamed:
                buf.add(en.text.encode("utf-8", errors="replace"), surprise=center)
            report["dreamed"] = len(dreamed)
            sel, rest = buf.select(dolphin.budget)
            report["selected"], report["residue"] = len(sel), len(rest)
            buf.clear()  # 摘除：名单已处置，缓冲即刻腾空（主线程喂数不丢）
    else:
        report["candidates"] = len(buf.items)
        # 间隔重复：命中达阈值的记忆复活，以带通中心惊讶度重新入选拔
        center, _ = buf.band()
        for rb in dolphin.memory.resurrect():
            buf.add(rb, surprise=center)
        sel, rest = buf.select(dolphin.budget)
        report["selected"], report["residue"] = len(sel), len(rest)

    # L9 周期层：落选/未选拔经验一律降级进记忆库（空选拔周期 rest 不得蒸发）
    for s, e in rest:
        dolphin.memory.add(e.data, s, dolphin.cycle, "residue")

    if not sel:
        if feeding:
            _threshold_feedback(dolphin)
        else:
            buf.clear()
        dolphin.cycle += 1
        report["note"] = "无可训经验（滞留已全部入记忆库）"
        return report

    # 律 L6：变异重放，拼成一条字节流
    donors = [e.data for _, e in sel]
    parts = []
    for i, (_, e) in enumerate(sel):
        donor = donors[(i + 1) % len(donors)] if len(donors) > 1 else None
        parts.append(varied_replay(e.data, rng, donor))
    stream = b"\n".join(parts)

    blk = dolphin.cfg.block_size
    bt = torch.tensor(list(stream), dtype=torch.long, device=dev)
    if bt.numel() < blk + 2:
        for s, e in sel:  # L9：训不了的选拔经验同样降级记忆库
            dolphin.memory.add(e.data, s, dolphin.cycle, "residue")
        if feeding:
            _threshold_feedback(dolphin)
        else:
            buf.clear()
        dolphin.cycle += 1
        report["note"] = "样本过短"
        return report

    if feeding:
        # 动睡脑前等在途 learn 清零（L3：刚退休的醒脑可能还有 learn 在读）
        while dolphin.learn_busy > 0:
            time.sleep(0.001)

    # —— M4 钩子①：周期开头（GROW widen / WITHER 换装）——
    _pre_train_sd = None
    if m4:
        ctl.pre_train(dolphin, report, stream)
        sleeping = dolphin.sleeping()  # 手术可能换了模型/优化器
        awake = dolphin.awake()
        # 形态分叉保护（M4 换班后两半球各自合法、形状可不同）：M0 回滚的
        # "睡脑←醒脑克隆"只在同构时可行。分叉时以"本轮训练前的睡脑自身快照"
        # 承担"只丢弃本轮训练"的回滚语义（同构路径不变，M4-E 等价性钉死）。
        sd_s, sd_a = sleeping.model.state_dict(), awake.model.state_dict()
        if sd_s.keys() != sd_a.keys() or any(sd_s[k].shape != sd_a[k].shape for k in sd_s):
            _pre_train_sd = {k: t.detach().cpu().clone() for k, t in sd_s.items()}

    # 训练：律 L5——蒸馏锚定真实数据。硬标签=真实字节，软标签=醒脑对同批真实字节
    sleeping.model.train()
    awake.model.eval()
    opt = sleeping.opt
    # born-again 验收周期：kd_alpha 提到手术验收档（规格 B，定标带 0.7–0.8 取中）
    alpha = kd_alpha
    if m4 and ctl.state == STATE_WITHERING \
            and (ctl.campaign or {}).get("mode") == "born_again":
        alpha = WITHER_KD_ALPHA
        report["m4_kd_alpha"] = alpha

    vit = ctl.vitals_for(dolphin, sleeping.name) if m4 else None
    if vit is not None:
        vit.attach(sleeping.model)
        vit.enable()  # 采集只在睡脑训练时启用（规格 A）

    ce_hist, kd_hist = [], []
    N = bt.numel()
    try:
        for _ in range(steps):
            s0 = rng.randrange(0, N - blk - 1)
            x = bt[s0:s0 + blk].unsqueeze(0)
            y = bt[s0 + 1:s0 + blk + 1].unsqueeze(0)
            with torch.no_grad():
                t_logits, _ = awake.model(x)
            s_logits, _ = sleeping.model(x)
            ce = F.cross_entropy(s_logits.reshape(-1, dolphin.cfg.vocab), y.reshape(-1))
            p_t = F.softmax(t_logits / kd_T, dim=-1)
            kd = (kd_T ** 2) * F.kl_div(
                F.log_softmax(s_logits / kd_T, dim=-1), p_t, reduction="batchmean")
            loss = (1 - alpha) * ce + alpha * kd
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sleeping.model.parameters(), 1.0)
            opt.step()
            ce_hist.append(ce.item())
            kd_hist.append(kd.item())
    finally:
        if vit is not None:
            vit.disable()
            vit.detach()

    # —— M4 钩子②：REM 样子相位（ReDo 回收 / 软衰减 / 手术排程）——
    if m4:
        ctl.rem_phase(dolphin, report, stream)

    # 体检（律 L8）：全卷逐块评分 + margin 判决带，判决唯一实现 Dolphin.gate
    passed, gate = dolphin.gate(awake, sleeping)
    report.update(gate)
    report["passed"] = passed
    report["ce_last"] = round(ce_hist[-1], 4)
    report["kd_last"] = round(kd_hist[-1], 4)
    if vit is not None:
        vit.end_cycle(steps)

    # —— M4 钩子③：体检后验收（锚/平台期/战役；NORMAL 下与 M0 逐字等价）——
    if m4:
        directive = ctl.post_exam(dolphin, report, passed)
    else:
        directive = "swap" if passed else "rollback"

    if directive == "swap":
        if feeding:
            with lock:
                dolphin.swap()  # awake_idx 只在锁内翻（一次赋值）
        else:
            dolphin.swap()  # 睡脑上岗，旧醒脑转睡
        report["swapped"] = True
        # 律 L10 快通道：新醒脑最自信的片段作为蒸馏笔记入记忆库（预支）。
        # 显式给 key：并列 NLL 时次级键用稳定 Experience.id（禁止随机）
        ranked = sorted(
            ((dolphin.awake().model.mean_nll(e.data, dev), e) for _, e in sel),
            key=lambda t: (t[0], t[1].id),
        )
        for nll, e in ranked[: dolphin.note_k]:
            dolphin.memory.add(e.data, nll, dolphin.cycle, "note")
        dolphin.budget = min(0.60, dolphin.budget * 1.05)  # 值自成：通过 → 预算放宽
    elif directive == "rollback":
        report["swapped"] = False
        if m4 and report.get("m4_rollback") == "grow":
            pass  # grow 回滚已在 post_exam 还原形态+装回醒脑权重
        elif m4 and report.get("m4_wither_abandoned"):
            pass  # 放弃凋零战役：post_exam 已按战役快照还原形态与权重
        elif _pre_train_sd is not None:
            # 形态分叉（born-again/生长换班后）：不能克隆醒脑——恢复本轮训练前
            # 的睡脑自身快照（语义仍是"只丢弃本轮训练"，不碰已上岗的醒脑）
            sleeping.model.load_state_dict(
                {k: v.to(dolphin.device) for k, v in _pre_train_sd.items()})
        else:
            # M0 语义：同构半球，只 load_state_dict（动量跨周期保留）
            sleeping.model.load_state_dict(awake.model.state_dict())
        dolphin.budget = max(0.25, dolphin.budget * 0.90)  # 值自成：收紧预算
    else:  # keep：WITHERING 战役学生保留（规格 C），不回滚权重
        report["swapped"] = False
        dolphin.budget = max(0.25, dolphin.budget * 0.90)

    # 值自成：学习率反馈（Plasticity 唯一控制器）+ 手术 LR 窗口 / 第二阶梯退火
    mean_surp = sum(e.surprise for _, e in sel) / len(sel)
    if feeding:
        _threshold_feedback(dolphin)
        lr_center, lr_width = center, width  # 带通在快照时取定（trainer_cycle 语义）
    else:
        lr_center, lr_width = buf.band()  # 带通在周期末重取（M0 语义；buf 尚未清）
    cur_lr = sleeping.opt.param_groups[0]["lr"]
    new_lr = dolphin.plasticity.next_lr(cur_lr, mean_surp, lr_center, lr_width, passed)
    if m4:
        if ctl.lr_window > 0:  # 规格 B：LR 重启 base×2，≤lr_max，窗口后交还
            new_lr = min(dolphin.plasticity.lr_base * SURGERY_LR_MULT,
                         dolphin.plasticity.lr_max)
            ctl.lr_window -= 1
            report["m4_lr_restart"] = new_lr
        elif ctl.anneal_pending:  # 第二阶梯：LR 退火（一次性）
            new_lr = max(new_lr * 0.5, dolphin.plasticity.lr_min)
            ctl.anneal_pending = False
            report["m4_lr_annealed"] = new_lr
    for hh in dolphin.h:
        hh.opt.param_groups[0]["lr"] = new_lr
    report["lr"] = new_lr

    if feeding:
        pass  # 缓冲已在快照时摘除
    else:
        buf.clear()
    dolphin.cycle += 1
    if verbose:
        print(f"[睡眠 {report['cycle']}] 选拔 {report['selected']} / 滞留 {report['residue']}"
              f"  体检 {report.get('probe_old')} → {report.get('probe_new')}"
              f"  lr={new_lr:.2e}"
              f"  {'✓ 换班' if report.get('swapped') else '✗ 作废回滚'}")
    return report
