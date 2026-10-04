"""生命节律控制器（M4）：睡眠周期的**唯一**实现 + 连续结构恒温器（自生长/自凋零）。

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
      → REM 样子相位（ReDo 回收 / 软衰减 / **连续结构恒温器**）
      → 体检（与 M0 同一 gate）→ post_exam（锚/平台期/战役验收/二阶阻尼反馈）。

═══════════════════════════════════════════════════════════════════════════
连续结构恒温器（2026-10-05 落地，取代平台期阶梯的活性路径）
═══════════════════════════════════════════════════════════════════════════
规格书：研究报告_自适应原理.md（Butz & van Ooyen 2013 突触元件框架的工程移植：
设定点带负反馈，NEST growth curve + update interval 同型）。控制形态：

  每周期末（REM 相位）：双信号"钙" → 对设定点带求偏差 → 按 λ_g·W·e 乘性律
  调整容量（长/缩/持有）→ 五重阻尼 → 受影响参数优化器状态重置（沿用
  gate/换班机制）。每周期至多一次结构动作；执行点仍在睡眠周期（律 L11）。

双信号"钙"（研究报告 §3.1 定案）：
  1. 幅值（主）：每 site 的 utilization_gap = (med − bottom-5%均值)/med
     （vitals.SiteLedger.utilization_indices，stable 口径）——过剩信号；
     headroom_hot = top-10%均值/med——过热信号（生长的利用率前提）。
  2. 方向（生长货币）：ghost_scan 逐位增益，对**自身滚动基线**求相对需求
     （细胞自主；L4"带通中心=滚动中位数"同款手术——不设绝对设定点，
     免疫量级漂移）。Q6/呼吸实验实测：基线稳定 ~1.0×，全新数据跳 1.6–1.8×。

外环分类（审计用，不是判决门；2026-10-05 监督审计 P2 对齐）：数据耗尽
（供给门关）/ 容量过剩（site 级·越浮动带 | 全局级·gap̄ 超带 | 持续过剩升级）/
容量受限（ghost 需求 × headroom 双确认）。原"五分类"中的"学习推进中"与
"瓶颈不在容量（记忆库压力）"两类**未实现、如实降级删除**——记忆库压力不是
容量信号（研究报告 §3.1 定案），不冒充在册。

**呼吸实验诊断（2026-10-05，触发链三断点实测）已吸收**：
  - 平台期阶梯（≥3 记账/≥6 退火/≥9 排程）**数学不可达**：120 步/周期下
    margin 恒在 +0.025~+0.055（AdamW 整步副作用 + 记忆复活/做梦每周期注入
    半遗忘内容——系统被设计成永不停学），9 连击期望 ~1300 周期。故阶梯
    **不再作为 fallback 挂在活性路径上**（留着是死代码+虚假安全感）；
    `_schedule` 方法体仅为 tests 钉住的历史件保留（退役身份，生产零调用）。
  - 凋零触发改"相对自身历史趋势"与连续带（休眠占比上限 ~5%，原绝对门槛
    10%/25% 构造性不可达）。
  - 恒温器判决**不设平台期前置门**——双信号钙每周期连续驱动（分工律的
    "权重通道先尽力"由生长的多重 AND 门槛与限频承担）。

**监督审计 R 级修复（2026-10-05，第四任修复工程师）**：
  - R1 设定点带"稳态即越带"：XI_HI=0.20 低于 57M 真实身体实测稳态 gap̄
    ≈0.45（16 个可动刀位全部 > 0.20）且带不可自校准 → born_sustained
    12 周期重锤在健康系统几乎必触发。带改浮动：max(XI_HI, gap̄ 滞后滚动
    p90)（_xi_hi_eff / _site_xi_eff）；滞后窗（XI_LAG=BORN_PERSIST）保证
    带慢于信号——升级路不被带当场吸收；稳态分布定标工件归档
    定标/gapbar_定标结果.json（兑现"实测"引用的可复核义务）。
  - R2 成熟刹车/死区收窄挂在永不流动的信号上：m 的原料从"平台期计数"
    （margin<绝对 ε）改为 margin 滚动分布分位（_mature_input）；死区收窄
    门同步从 plateau≥1 改为 m≥MATURITY_NARROW。
  - R3 生长回滚账目死代码：post_exam 回滚扣账先取 delta 再清 campaign。

四条守卫（研究报告 §3.5 全集，本实现）：
  ③ 数据供给门：近 SUPPLY_WINDOW 周期新鲜内容摄入不足 → 结构冻结
     （反刍期只消化不动刀；新鲜度=经验内容指纹，复训旧记忆不算新供给）；
  ④ 预算/供给比：累计生长通道 ≤ β×(新鲜字节/512B 通道当量)——结构投资须有
     证据流支撑，防反刍循环吹脑；
  ⑤ 选址禁用探测集：ghost_scan 只读真实回放流（训练侧），手术层选址永不
     读 probe（L8 红线：选址用探测=偷看考卷）；
  ⑥ 成熟刹车：可塑性永不归零（λ_g_eff = λ_g·(1−0.5m) ≥ λ_g·0.5 > 0，
     m=改善率（gate margin）跌破自身滚动分布低分位的 EMA——世界信号非日历；
     R2 修复：原"margin<绝对 ε 的平台期占比"口径在真实系统结构性不可达，
     实测 margin 恒 +0.025~0.055 ≫ ε=0.005）+ 容量下限非零（累计收缩后
     体型占比 < MIN_BODY_FRAC 禁再缩，防无限萎缩）。

五重阻尼（研究报告 §3.4）：①测量阻尼（stable EMA，禁 tag）②Schmitt 确认窗
（连续 CONFIRM_M 窗口越界才动作，1 窗口回带内即退出）③限频（每周期至多
1 动作 + 同 site 间隔 ≥6 周期 + 手术 LR 重启窗内不排新刀）④不对称增益
（长需双信号 AND + 供给守卫，缩单信号 OR 即可）⑤增益自适应（回滚 →
λ_g×0.5+冷却；验收通过且捕获成功 → λ_g×1.1；振荡熔断 → 死区放宽）。

战役语义（规格 C，不变）：NORMAL → GROW_PLAN → GROWN → 体检 → NORMAL /
ROLLBACK；NORMAL → WITHER_PLAN → WITHERING（学生保留标志：体检失败不回滚，
逐周期续训直到验收或分代耗尽放弃还原）。

锚：历史最优 probe（anchor_best，持久化进 save/load；规格 C）。

常量来源身份（律固定，值自成；2026-10-05 实测定标项标注实测）：
  —— 平台期阶梯（退役件身份：仅 tests 钉住的历史路径使用）——
  PLATEAU_CYCLES=3       【值自成】阶梯步长（周期数；随 `_schedule` 退役）
  EPS_PLATEAU=0.005      【退役件原料】仅平台期计数（退役阶梯的历史路径）使用；
                         活性成熟度已改 margin 滚动分位（R2：实测真实系统
                         margin 恒 +0.025~0.055 ≫ 0.005，绝对 ε 口径在真实
                         系统结构性不可达——平台期计数恒 0）
  GHOST_GAIN_MIN=1e-4    【值自成】退役件阈值（呼吸实验实测：全新数据
                         1.06e-4 恰可越线——但相对自身基线的恒温器口径
                         已取代它）
  WITHER_DORMANT_FRAC=0.10 / WITHER_BORN_FRAC=0.25
                         【值自成】退役件阈值（休眠占比实测上限 ~5%，
                         构造性不可达——恒温器用趋势+连续带取代）
  WITHER_TARGET=0.7      【退役件身份】born-again 学生体型比例——仅手工
                         计划/退役阶梯使用；恒温器目标由 gap̄ 导出
  WITHER_KD_ALPHA=0.75   【定标】规格带 0.7–0.8 取中（仅手术验收周期）
  WITHER_MAX_GEN=3       【值自成】凋零战役分代上限（规格"多周期分代"）
  DECAY_GAMMA=0.7        【值自成】软衰减单代系数（乘性、保相对——
                         Turrigiano 结构化身）
  REDO_MAX_FRAC=0.05     【值自成】单周期回收上限（审计：5% 上限改纯反馈量）
  SURGERY_LR_WINDOW=3    【值自成】手术后 LR 重启窗，窗内不排新刀（限频③）
  SURGERY_LR_MULT=2.0    【律定承袭规格 B】base×2，≤ lr_max
  SLEEP_DEBT_GUARD=4.0   【值自成】_since_sleep > target_interval×4 禁手术
  ROLLBACK_GUARD=3       【值自成】连续回滚 ≥3 次禁手术（脑在挣扎，先养）
  GROW_COOLDOWN=6        【值自成】生长回滚后的冷却周期
  MAX_GROW_ATTEMPTS=2    【值自成】冷却前允许的手术回滚次数
  GROW_DELTA_FRAC=0.05   【退役件身份】固定步长 5%——恒温器的乘性律
                         （λ_g·W·e）取代它；常量仅为退役 _grow_delta 保留
  —— 连续结构恒温器（研究报告 §3；2026-10-05 实测定标）——
  XI_HI=0.20             【值自成·带下限】过剩设定点带的冷启动下限（研究报告：
                         健康周转余量 5–20% 类比）。实际带 = max(XI_HI,
                         gap̄ 滞后滚动 p90)（_xi_hi_eff；R1 修复：57M 真实
                         身体稳态 gap̄≈0.45 ≫ 0.20，带不可自校准则稳态即
                         越带——带随系统自身稳态分布上浮，定标工件
                         定标/gapbar_定标结果.json）
  XI_LO_HOT=1.2          【值自成→Q7 定标】过热设定点（top-10%均值/中位）。
                         实测：mlp_hidden 过热带 1.21–1.37——规格草案 1.5
                         在真实账本上数学不可达（阶梯覆辙），按实测降到 1.2
  HOT_TOP_P=0.10         【值自成】过热 top-p 分位
  GHOST_K=2.0            【值自成→Q7 定标】ghost 需求门槛 = 1 + K×SE_REL
  GHOST_SE_REL=0.25      【值自成→Q7 定标】ghost 相对自身基线的噪声当量。
                         实测：基线期 0.99–1.12×，需求跳变期 1.6–1.8×，
                         门槛 1.5× 落在两分布之间（分辨力实证）
  GHOST_BASE_WINDOW=4    【值自成】ghost 自身基线滚动窗（扫描次数）
  LAMBDA_G=0.04          【值自成】乘性生长率/周期（Butz ν 的移植）
  LAMBDA_MIN=0.01        【律定承袭】可塑性永不归零（adapt.py 同款教训）
  LAMBDA_MAX=0.08        【值自成】λ_g 自适应上限
  GROW_DELTA_CAP_FRAC=0.10 【值自成】单次生长上限（当前宽度的 10%）
  CONFIRM_M=2            【定标承袭】DORMANT_CONFIRM 同款：连续 2 窗口越界
  H_SHRINK=0.06          【定标承袭 Q6】判决死区 ≈ 2×通道判决噪声（6% 相对）
  GAP_BORN=0.45          【值自成】全局结构错配阈下限；实际阈 = max(GAP_BORN,
                         浮动带 xi_hi_eff)（R1：随带上浮，稳态高于阈下限的
                         系统不被重锤误读）
  BORN_PERSIST=12        【值自成】持续过剩升级：gap̄ 连续 12 周期 > 带上缘
                         → born-again（重锤需要耐心；衰减通道先行）。
                         XI_LAG=带滞后视野与它等值（带慢过最慢判决——R1）
  SHRINK_COEF=0.6        【值自成】born-again 目标收缩系数 T=1−coef·gap̄
  MIN_BODY_FRAC=0.25     【值自成】成熟刹车容量下限（相对出生体型的
                         d_model×n_layers 当量占比；防无限萎缩）
  MIN_GAP_SAME_SITE=6    【值自成】同 site 两次动作最小间隔（周期，限频③）
  OSC_WINDOW=6           【值自成】振荡判定窗（周期）
  OSC_FUSE_CYCLES=12     【值自成】振荡熔断时长（周期）
  MATURITY_TAU=10        【值自成】成熟度 EMA 时间常数（周期；输入是 margin
                         滚动分位判据——世界信号，非日历；R2）
  MARGIN_HIST_CAP=16     【值自成】margin 滚动窗（周期；成熟度分布原料，R2）
  MATURITY_Q=0.25        【值自成】成熟判据分位（改善率近窗中位数对自身滚动
                         分布 q25；R2）
  MATURITY_RECENT=4      【值自成】成熟判据近窗（周期，滑动中位数；R2）
  MATURITY_NARROW=0.3    【值自成】死区收窄的成熟度门（R2）
  XI_P=0.90 / XI_HIST_MIN=4 / XI_LAG=12 / XI_HIST_CAP=64
                         【值自成】设定点带上浮四参数：分位 / 最小滞后样本 /
                         滞后视野（=BORN_PERSIST）/ 滚动窗深（R1）
  SUPPLY_WINDOW=3        【值自成】数据供给观察窗（周期；守卫③）
  SUPPLY_MIN_BYTES=512   【值自成】窗口内新鲜字节下限（守卫③）
  BETA_SUPPLY=0.01       【值自成】预算/供给比 β（研究报告 §3.5；守卫④）
  SUPPLY_BYTES_PER_UNIT=512 【值自成】通道当量字节数（守卫④）
  SEEN_CAP=8192          【值自成】新鲜度指纹集容量（条）
  CAPTURE_RATE_MIN=0.5   【值自成】战役级捕获成功占比（probation 期满，
                         新单元 stable_act ≥ 老单元中位一半的占比）
  DEADBAND_FLOOR=0.5     【律定承袭】死区自校准下限（可塑性不归零对偶）
  DEADBAND_CAL_EVERY=20  【值自成】死区收窄的静默周期数（×0.9）；门=成熟度
                         m ≥ MATURITY_NARROW（R2：原 plateau≥1 门在真实系统
                         结构性不可达，收窄路径死）
"""
import hashlib
import time

import torch
import torch.nn.functional as F

from .sleep import varied_replay  # 律 L6 唯一实现，复用不复制
from .surgery import (born_again_student, break_symmetry_dmodel, delta_params_attn_v,
                      delta_params_dmodel, delta_params_mlp, ghost_scan,
                      mem_budget_ok, morphology_of, rebuild_optimizer, widen)
from .vitals import PROBATION_CYCLES, SiteLedger, Vitals

# —— 常量（来源身份见模块 docstring）——
# 平台期阶梯（退役件身份）
PLATEAU_CYCLES = 3
EPS_PLATEAU = 0.005
GHOST_GAIN_MIN = 1e-4
WITHER_DORMANT_FRAC = 0.10
WITHER_BORN_FRAC = 0.25
WITHER_TARGET = 0.7
WITHER_KD_ALPHA = 0.75
WITHER_MAX_GEN = 3
DECAY_GAMMA = 0.7
REDO_MAX_FRAC = 0.05
SURGERY_LR_WINDOW = 3
SURGERY_LR_MULT = 2.0
SLEEP_DEBT_GUARD = 4.0
ROLLBACK_GUARD = 3
GROW_COOLDOWN = 6
MAX_GROW_ATTEMPTS = 2
GROW_DELTA_FRAC = 0.05
# 连续结构恒温器
XI_HI = 0.20             # 带下限（实际带 = max(XI_HI, gap̄ 滞后滚动 p90)，见 _xi_hi_eff）
XI_P = 0.90              # 【值自成】带上浮分位：带浮在自身 gap̄ 稳态分布的高分位（R1）
XI_HIST_MIN = 4          # 【值自成】带上浮所需最小滞后样本（不足→用下限 XI_HI；R1）
XI_LAG = 12              # 【值自成】带滞后视野= BORN_PERSIST：p90 只读确认视野之前的
                         # 历史——带必须慢过最慢判决，否则持续越带若干周期后会被带
                         # 当场吸收，born_sustained 升级路重蹈阶梯数学不可达（R1）
XI_HIST_CAP = 64         # 【值自成】gap̄/逐位 gap 滚动窗深度（周期；R1）
XI_LO_HOT = 1.2
HOT_TOP_P = 0.10
GHOST_K = 2.0
GHOST_SE_REL = 0.25
GHOST_BASE_WINDOW = 4
LAMBDA_G = 0.04
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.08
GROW_DELTA_CAP_FRAC = 0.10
CONFIRM_M = 2
H_SHRINK = 0.06
GAP_BORN = 0.45
BORN_PERSIST = 12
SHRINK_COEF = 0.6
MIN_BODY_FRAC = 0.25
MIN_GAP_SAME_SITE = 6
OSC_WINDOW = 6
OSC_FUSE_CYCLES = 12
MATURITY_TAU = 10
MARGIN_HIST_CAP = 16     # 【值自成】margin 滚动窗（周期；成熟度分布原料，R2）
MATURITY_Q = 0.25        # 【值自成】成熟判据分位：改善率近窗中位数跌破自身滚动
                         # 分布 q25 → 成熟输入 1（世界信号；R2）
MATURITY_RECENT = 4      # 【值自成】成熟判据近窗（周期，滑动中位数窗口；R2）
MATURITY_NARROW = 0.3    # 【值自成】死区收窄的成熟度门（m≥0.3 才收窄；R2）
SUPPLY_WINDOW = 3
SUPPLY_MIN_BYTES = 512
BETA_SUPPLY = 0.01
SUPPLY_BYTES_PER_UNIT = 512
SEEN_CAP = 8192
CAPTURE_RATE_MIN = 0.5
DEADBAND_FLOOR = 0.5
DEADBAND_CAL_EVERY = 20

STATE_NORMAL = "NORMAL"
STATE_GROW_PLAN = "GROW_PLAN"
STATE_GROWN = "GROWN"
STATE_WITHER_PLAN = "WITHER_PLAN"
STATE_WITHERING = "WITHERING"

# 生长轴 → 选址账本位的映射：轴② attn_v 新 v 通道的利用率读数在 attn_out 位
# （proj 输入 = v 路径输出；账本通道口径 = Linear 输入激活维，Q6 定标）。
# 振荡检测/同 site 限频都用账本位 id（轴②生长与 attn_out 收缩共享 id，
# 反向操作才会被识别为振荡）。
_AXIS_LEDGER = {"mlp": "mlp_hidden", "attn_v": "attn_out"}
_LEDGER_AXIS = {"mlp_hidden": "mlp", "attn_out": "attn_v"}
# 可动刀的主账本位（ln1_out/ln2_out 是残差流插件引脚——对质红线同源，不回收）
ACTIONABLE_SUFFIX = ("mlp_hidden", "attn_out")


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


def _median_lo(vals):
    """下中位数（len<2 时同 _median）。ghost 自身基线专用：基线代表"常规水平
    下沿"——需求跳变持续期不因基线上浮而失明（每个 site 每隔限频间隔保持
    可再生长性），噪声下仍稳定（8-ghost 均值的实测波动 ~5%，门槛 1.5× 远在
    噪声之上）。偶数窗上中位数会把基线吸到跳变后水平（实测：预算守卫挡掉
    一次机会后，下一周期比值即坍回 1.0——信号永久丢失）。"""
    s = sorted(vals)
    return s[(len(s) - 1) // 2] if s else 0.0


def _quantile(vals, q):
    """最近邻秩分位数（空集 → None）。设定点带上浮（R1）与成熟度判据（R2）
    的公共原语：门槛随系统自身历史分布走——值自成，不写死绝对量。"""
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


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
    """连续结构恒温器 + 双半球营养账本 + 形态记录 + 状态机。

    由 Dolphin 持有（d.life_ctl）；save/load 经 to_state/from_state 持久化
    （锚、状态机、形态、账本、恒温器内部状态——规格 C：锚=历史最优 probe，
    持久化进 save/load）。
    """

    def __init__(self):
        self.state = STATE_NORMAL
        self.plateau = 0            # 连续平台期周期数（margin < EPS_PLATEAU；
        #                              恒温器不再以此判决——它是成熟度刹车原料）
        self.anchor_best = None     # 锚：历史最优 probe NLL（持久化）
        self.plan = None            # 待执行手术计划（下一周期 pre_train 执行）
        self.campaign = None        # 战役 dict（grow/wither 进行时）
        self.lr_window = 0          # 手术 LR 重启窗口剩余周期（之后交还 Plasticity）
        self.anneal_pending = False  # 退役阶梯的 LR 退火（一次性；活性路径不置位）
        self.anneal_band = 0        # 已退火到的阶梯号（退役件状态）
        self.consec_rollback = 0
        self.grow_attempts = 0
        self.cooldown = 0           # 手术冷却剩余周期（每周期在恒温器入口递减）
        self.morphology = {}        # 半球名 → morphology dict（持久化）
        self.vitals = {}            # 半球名 → Vitals（持久化）
        # —— 恒温器内部状态（全部随档）——
        self.lambda_g = LAMBDA_G            # 乘性生长率（增益自适应的被调量）
        self.deadband_scale = 1.0           # 死区自校准旋钮（振荡放宽/静默收窄）
        self.maturity = 0.0                 # 成熟度 m∈[0,1]：改善率跌破自身滚动
                                            # 分布低分位的 EMA（R2：世界信号非日历）
        self.margin_hist = []               # gate margin 滚动窗（list[float]，
                                            # 成熟度分布原料；R2）
        self.thermo_hist = {}               # 账本位 → [[cycle, gap, hot], …]
        self.ghost_hist = {}                # "L{li}.{site}" → [[cycle, gain], …]
        self.gapbar_hist = []               # [[cycle, gap̄], …]（全局过剩指数）
        self.dorm_hist = []                 # [[cycle, 休眠占比], …]（趋势原料）
        self.site_last_action = {}          # 账本位 → [cycle, "grow"|"shrink"]
        self.site_fuse = {}                 # 账本位 → 熔断截止 cycle
        self.osc_events = []                # [[cycle, 账本位], …]（振荡账）
        self.supply_hist = []               # [[cycle, fresh_bytes], …]（守卫③）
        self.seen_hashes = []               # 新鲜度指纹集（list 持久化，int）
        self.fresh_since_surgery = 0.0      # 上次结构手术以来新鲜字节（守卫④）
        self.grown_since_surgery = 0        # 上次结构手术以来生长通道数（守卫④）
        self.last_shrink_cycle = None       # 上次收缩计划周期（限频③）
        self.last_born_again = None         # 上次 born-again 计划周期（连环缩阻尼）
        self.no_action_streak = 0           # 连续"持有"周期数（死区自校准）
        self.pending_capture = None         # 捕获对账单 {due_cycle, keys, hname}
        self._seen = set()                  # 运行时指纹集（自 seen_hashes 重建）

    # ---------- 持久化 ----------

    def to_state(self):
        return {"state": self.state, "plateau": self.plateau,
                "anchor_best": self.anchor_best, "plan": self.plan,
                "campaign": self.campaign, "lr_window": self.lr_window,
                "anneal_band": self.anneal_band,
                "consec_rollback": self.consec_rollback,
                "grow_attempts": self.grow_attempts, "cooldown": self.cooldown,
                "morphology": self.morphology,
                "vitals": {k: v.to_state() for k, v in self.vitals.items()},
                # 恒温器状态（全部 JSON 化安全：list/dict/标量）
                "lambda_g": self.lambda_g, "deadband_scale": self.deadband_scale,
                "maturity": self.maturity, "margin_hist": self.margin_hist,
                "thermo_hist": self.thermo_hist, "ghost_hist": self.ghost_hist,
                "gapbar_hist": self.gapbar_hist, "dorm_hist": self.dorm_hist,
                "site_last_action": self.site_last_action, "site_fuse": self.site_fuse,
                "osc_events": self.osc_events, "supply_hist": self.supply_hist,
                "seen_hashes": self.seen_hashes,
                "fresh_since_surgery": self.fresh_since_surgery,
                "grown_since_surgery": self.grown_since_surgery,
                "last_shrink_cycle": self.last_shrink_cycle,
                "last_born_again": self.last_born_again,
                "no_action_streak": self.no_action_streak,
                "pending_capture": self.pending_capture}

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
        # 恒温器状态（旧档缺键 → 初值；from_state 容忍升级）
        self.lambda_g = float(st.get("lambda_g", LAMBDA_G))
        self.deadband_scale = float(st.get("deadband_scale", 1.0))
        self.maturity = float(st.get("maturity", 0.0))
        self.margin_hist = [float(x) for x in (st.get("margin_hist") or [])]
        self.thermo_hist = st.get("thermo_hist") or {}
        self.ghost_hist = st.get("ghost_hist") or {}
        self.gapbar_hist = st.get("gapbar_hist") or []
        self.dorm_hist = st.get("dorm_hist") or []
        self.site_last_action = st.get("site_last_action") or {}
        self.site_fuse = st.get("site_fuse") or {}
        self.osc_events = st.get("osc_events") or []
        self.supply_hist = st.get("supply_hist") or []
        self.seen_hashes = st.get("seen_hashes") or []
        self.fresh_since_surgery = float(st.get("fresh_since_surgery", 0.0))
        self.grown_since_surgery = int(st.get("grown_since_surgery", 0))
        self.last_shrink_cycle = st.get("last_shrink_cycle")
        self.last_born_again = st.get("last_born_again")
        self.no_action_streak = int(st.get("no_action_streak", 0))
        self.pending_capture = st.get("pending_capture")
        self._seen = set(self.seen_hashes)
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

    # ---------- 恒温器：阻尼与守卫的原语 ----------

    def lambda_g_eff(self):
        """成熟刹车下的有效生长率：λ_g·(1−0.5m) ≥ λ_g·0.5 > 0（永不归零）。"""
        return self.lambda_g * (1.0 - 0.5 * self.maturity)

    def h_eff(self):
        """有效死区宽度：噪声定标 × 自校准旋钮 × 成熟刹车放大。"""
        return H_SHRINK * self.deadband_scale * (1.0 + 0.5 * self.maturity)

    def _mature_input(self):
        """成熟度原料（R2 修复：可流动的世界信号）：近期改善率中位数是否跌破
        自身滚动分布的低分位（MATURITY_Q）。

        平稳期：median(近窗) ≥ q25(滚动窗) → 0（刹车释放——平稳系统对自身
        分布不构成下台阶）；改善率下台阶（世界停止投喂新需求/内容饱和）→ 1。
        margin 历史不足 MARGIN_HIST_CAP → 0（冷启动不刹车——样本不足时
        "分布"无从谈起）。

        选它而非"距上次结构动作的周期数"的理由：研究报告 §1.6 明令"刹车信号
        的来源必须是世界，不是日历（adapt.py 前世死因）……禁止用周期计数器"，
        周期数恰是计数器；margin 分位是 probe 判决面的可测统计，且门槛（q25）
        随系统自身历史走——值自成，无写死的绝对量。它对旧口径的解药性：呼吸
        实验实测真实系统 margin 恒 +0.025~0.055 ≫ EPS_PLATEAU=0.005，绝对 ε
        平台期计数恒 0 → 旧原料下 m≡0、刹车与死区收窄两路结构性死；新原料
        读的是"改善率相对自身历史的下台阶"，真实变化即流动。
        """
        if len(self.margin_hist) < MARGIN_HIST_CAP:
            return 0.0
        rec = sorted(self.margin_hist[-MATURITY_RECENT:])
        med = rec[len(rec) // 2]
        q = _quantile(self.margin_hist, MATURITY_Q)
        return 1.0 if med < q else 0.0

    def _lagged_base(self, vals):
        """带上浮的公共滞后窗：p90 只读确认视野（XI_LAG）之前的历史。

        滞后必须 ≥ BORN_PERSIST（带慢过最慢判决——否则持续越带被带当场吸收，
        born_sustained 变成新的数学不可达，重蹈阶梯覆辙）；冷启动历史不足时
        滞后收短到 CONFIRM_M，让上浮尽早生效——真实部署早期是带最盲的窗口
        （监督审计 R1："真实部署早期重锤几乎必触发"）。"""
        lag = max(CONFIRM_M, min(XI_LAG, len(vals) - XI_HIST_MIN))
        return vals[:-lag] if lag else vals

    def _xi_hi_eff(self):
        """过剩设定点带（值自成；监督审计 R1 修复）：max(XI_HI, gap̄ 滞后滚动 p90)。

        为什么带要浮：XI_HI=0.20 下限取自"健康周转余量 5–20%"类比，而 57M
        真实身体实测稳态 gap̄≈0.45（定标/gapbar_定标结果.json：两半球
        0.448–0.468，16 个可动刀位全部 > 0.20）——带低于稳态且不可自校准
        ⇒ 稳态即越带 ⇒ born_sustained 12 周期重锤在健康系统上必触发。带改
        从系统自己的 gap̄ 历史取高分位（"值自成"的应有之义：设定点随系统
        自身历史走，不写死），XI_HI 只兜冷启动。滞后窗见 _lagged_base。"""
        base = self._lagged_base([g for _, g in self.gapbar_hist])
        q = _quantile(base, XI_P) if len(base) >= XI_HIST_MIN else None
        return XI_HI if q is None else max(XI_HI, q)

    def _site_xi_eff(self, key):
        """逐位过剩设定点带（R1 同款上浮，读该位 thermo_hist 的 gap 列）：
        site 级越带收缩同样不得在稳态误触发。"""
        base = self._lagged_base([e[1] for e in self.thermo_hist.get(key, [])])
        q = _quantile(base, XI_P) if len(base) >= XI_HIST_MIN else None
        return XI_HI if q is None else max(XI_HI, q)

    def supply_open(self):
        """守卫③数据供给门：近 SUPPLY_WINDOW 周期有新鲜内容摄入才允许动刀。

        历史为空（冷启动/旧档升级）→ 默认放行（初始化期不因缺观测冻结）。
        新鲜度=经验内容指纹（见 _supply_account）：反刍复训旧记忆不算新供给。
        """
        if not self.supply_hist:
            return True
        recent = self.supply_hist[-SUPPLY_WINDOW:]
        return sum(f for _, f in recent) >= SUPPLY_MIN_BYTES

    def supply_account(self, d, sel):
        """守卫③④的记账步：本周期入选经验的指纹账。

        新鲜度 = sha256 前 8 字节指纹首见。反刍（做梦注入/复活/重复喂同一批）
        的内容指纹已在册 → 不计入 fresh——结构投资必须有新证据流支撑
        （研究报告 §3.5；Draganski"停练回缩"的守卫面）。记忆库首次被消化的
        滞留条目算新鲜：对训练而言它是真正的新经验。
        """
        if sel is None:
            return None  # 无入选名单（直调场景）：账目保持原状
        fresh = 0
        for _, e in sel:
            h = int.from_bytes(hashlib.sha256(e.data).digest()[:8], "big")
            if h not in self._seen:
                self._seen.add(h)
                self.seen_hashes.append(h)
                fresh += len(e.data)
        if len(self.seen_hashes) > SEEN_CAP:  # 有界指纹集：裁掉最旧的
            cut = len(self.seen_hashes) - SEEN_CAP
            dropped = self.seen_hashes[:cut]
            del self.seen_hashes[:cut]
            self._seen.difference_update(dropped)
        self.supply_hist.append([d.cycle, fresh])
        del self.supply_hist[:-(SUPPLY_WINDOW + 1)]
        self.fresh_since_surgery += fresh
        return fresh

    def _confirmed(self, hist, pred, m=CONFIRM_M):
        """Schmitt 确认窗：最近 m 个观察全部满足 pred 才算越界（进入）；
        任一窗口回带内即不满足（退出只需 1 窗——阻尼②）。"""
        win = hist[-m:]
        return len(win) >= m and all(pred(e) for e in win)

    def _site_ready(self, key, cycle):
        """限频③：同 site 间隔 ≥MIN_GAP_SAME_SITE 且未熔断。"""
        if self.site_fuse.get(key, 0) > cycle:
            return False
        last = self.site_last_action.get(key)
        return not (last and cycle - last[0] < MIN_GAP_SAME_SITE)

    def _record_action(self, keys, kind, cycle, report):
        """动作记账 + 振荡检测（阻尼⑤熔断）：同 site 反向操作间隔
        <MIN_GAP_SAME_SITE 记一次振荡；OSC_WINDOW 内 ≥2 次 → 熔断该 site
        OSC_FUSE_CYCLES 周期 + 全局死区放宽 ×1.5。"""
        for k in keys:
            last = self.site_last_action.get(k)
            if last and last[1] != kind and cycle - last[0] < MIN_GAP_SAME_SITE:
                self.osc_events.append([cycle, k])
                recent = [e for e in self.osc_events if cycle - e[0] < OSC_WINDOW]
                if len(recent) >= 2:
                    self.site_fuse[k] = cycle + OSC_FUSE_CYCLES
                    self.deadband_scale = min(3.0, self.deadband_scale * 1.5)
                    report["m4_oscillation"] = {"site": k,
                                                "fuse_until": self.site_fuse[k],
                                                "deadband_scale": self.deadband_scale}
            self.site_last_action[k] = [cycle, kind]
        del self.osc_events[:-OSC_WINDOW * 4]

    def _body_frac(self, d, hname):
        """当前体型相对出生体型的当量占比（d_model 与 n_layers 的乘积口径；
        shrink 记录是绝对值、已含平铺史——born-again 换装后 d.cfg 仍是出生基座）。"""
        shrink = self.morph_for(d, hname).get("shrink") or {}
        d0 = int(shrink.get("d_model") or d.cfg.d_model)
        n0 = int(shrink.get("n_layers") or d.cfg.n_layers)
        return (d0 / d.cfg.d_model) * (n0 / d.cfg.n_layers)

    # ---------- 恒温器：每周期入口 ----------

    def thermo_cycle(self, d, report, v, stream, sel=None):
        """连续结构恒温器每周期入口（rem_phase 调用）：观察 → 判决 → body 报告。

        战役中（非 NORMAL）只做观察与记账，不做新判决。每周期至多一次结构
        动作（判决在 NORMAL 且无 pending 计划时发生）。
        """
        self.supply_account(d, sel)
        # 成熟度（R2 修复）：原料从"平台期计数（margin<绝对 ε）"改为 margin
        # 滚动分布分位（_mature_input）——绝对 ε 口径在真实系统结构性不可达
        # （margin 恒 +0.025~0.055 ≫ 0.005），m 恒 0 → 成熟刹车与死区收窄
        # 两路皆死（与被退役阶梯同型病在守卫内复发）。
        self.maturity += (1.0 / MATURITY_TAU) * (self._mature_input() - self.maturity)
        if self.cooldown > 0:  # 冷却递减唯一入口（退役阶梯不再代管）
            self.cooldown -= 1
        self._capture_check(d, report, v)
        gaps = self._observe(d, v)
        gap_bar = self._gap_bar(gaps)
        self.gapbar_hist.append([d.cycle, round(gap_bar, 6)])
        del self.gapbar_hist[:-XI_HIST_CAP]
        _, frac = v.dormant_report()
        self.dorm_hist.append([d.cycle, round(frac, 6)])
        del self.dorm_hist[:-8]

        action = {"kind": "hold", "reason": "战役进行中（非 NORMAL，不做新判决）"}
        if self.state == STATE_NORMAL and self.plan is None:
            action = self._decide(d, report, v, stream, gaps, gap_bar)
        if action.get("kind") == "hold":
            self.no_action_streak += 1
            if (self.no_action_streak > 0
                    and self.no_action_streak % DEADBAND_CAL_EVERY == 0
                    and self.maturity >= MATURITY_NARROW):
                # 死区自校准（慢环）：持续"改善率下台阶"（成熟度门，R2——
                # 原 plateau≥1 门不可达）且零结构操作 → 死区收窄 ×0.9
                self.deadband_scale = max(DEADBAND_FLOOR, self.deadband_scale * 0.9)
                action["deadband_narrowed"] = self.deadband_scale
        else:
            self.no_action_streak = 0
        self._report_body(d, report, gap_bar, action)

    def _observe(self, d, v):
        """测量阻尼①：每账本位记录 [cycle, gap, hot]（stable 口径；禁 tag）。"""
        gaps = {}
        for key, led in v.sites.items():
            if ".new." in key:
                continue  # 旁路新账本量纲不同（输出侧代理），不进恒温器读数
            gap, hot = led.utilization_indices(p=HOT_TOP_P)
            hist = self.thermo_hist.setdefault(key, [])
            hist.append([int(d.cycle), round(gap, 6), round(hot, 6)])
            del hist[:-XI_HIST_CAP]
            gaps[key] = gap
        return gaps

    def _gap_bar(self, gaps):
        """全局过剩指数 gap̄：可动刀主位（mlp_hidden/attn_out）的均值。
        ln1_out/ln2_out 不进 gap̄——残差流位不可动刀（对质红线同源），
        它们的失衡读数不代表可回收的单元容量。"""
        vals = [g for k, g in gaps.items() if k.endswith(ACTIONABLE_SUFFIX)]
        return sum(vals) / len(vals) if vals else 0.0

    def _capture_check(self, d, report, v):
        """捕获对账（研究报告 §3.7：标签-捕获的固化侧）。probation 期满时，
        新单元 stable_act ≥ 老单元中位一半才算"捕获"；未捕获计入生长尝试的
        失败账 → λ_g 下调。移植体属于执行时睡脑——换班后归醒脑名下，故按
        hname 取账本；该半球尚未再训练（账本全零）时顺延，**至多 1 次**
        （deferred 置位后不再顺延——2026-10-05 P3 对齐：原注释"最多 3 次"
        与代码不符，实况是至多 1 次）。"""
        pc = self.pending_capture
        if not pc or d.cycle < pc.get("due_cycle", 0):
            return
        hv = self.vitals.get(pc.get("hname") or "")
        if hv is None:
            self.pending_capture = None
            return
        if not pc.get("deferred"):
            led0 = hv.sites.get((pc.get("keys") or [""])[0])
            if led0 is not None and all(a == 0.0 for a in led0.stable_act):
                pc["deferred"] = 1  # 移植体还没再睡过：账本无观测，顺延
                pc["due_cycle"] = d.cycle + 1
                return
        per, earned_n, total = {}, 0, 0
        for key in pc.get("keys") or []:
            led = hv.sites.get(key)
            parent = hv.sites.get(key.rsplit(".new.", 1)[0])
            if led is None or parent is None:
                continue  # 回滚/形态变化：无从对账，如实跳过
            med_old = _median(parent.stable_act)
            thr = 0.5 * max(med_old, 1e-12)
            e = sum(1 for a in led.stable_act if a >= thr)
            per[key] = [e, led.C]
            earned_n += e
            total += led.C
        if total == 0:
            self.pending_capture = None  # 无可对账对象：清除，不动增益
            return
        rate = earned_n / total
        report["m4_capture"] = {"rate": round(rate, 3), "per": per}
        if rate >= CAPTURE_RATE_MIN:
            self.lambda_g = min(self.lambda_g * 1.1, LAMBDA_MAX)  # 增益自适应⑤
        else:
            self.lambda_g = max(self.lambda_g * 0.9, LAMBDA_MIN)
        report["m4_lambda_g"] = round(self.lambda_g, 5)
        self.pending_capture = None

    def _report_body(self, d, report, gap_bar, action):
        """可观测义务（L11 措辞防线）：每周期 body 字段——体型/参数总量/
        设定点/当前利用率缺口/本次调整方向与幅度。所有"自适应"措辞都能指认到
        这里的具体误差量/设定点/增益。"""
        m = d.sleeping().model
        xi_eff = self._xi_hi_eff()
        report["body"] = {
            "d_model": m.cfg.d_model, "n_layers": m.cfg.n_layers,
            "params": sum(p.numel() for p in m.parameters()),
            "setpoint": {"xi_hi": round(xi_eff, 4), "xi_hi_floor": XI_HI,
                         "xi_lo_hot": XI_LO_HOT,
                         "band_hi": round(xi_eff + self.h_eff(), 4)},
            "util_gap": round(gap_bar, 4),
            "action": action,
            "lambda_g": round(self.lambda_g_eff(), 5),
            "lambda_g_base": round(self.lambda_g, 5),
            "deadband_scale": round(self.deadband_scale, 3),
            "maturity": round(self.maturity, 3),
            "supply": "open" if self.supply_open() else "closed",
        }

    # ---------- 恒温器：判决 ----------

    def _hold(self, cls, reason, **extra):
        return {"kind": "hold", "class": cls, "reason": reason, **extra}

    def _decide(self, d, report, v, stream, gaps, gap_bar):
        """每周期判决：外环五分类（审计）+ 双信号驱动（研究报告 §3.0/§3.3）。

        收缩两路：绝对带（gap 越 XI_HI+h_eff）与自身趋势（休眠占比上斜）；
        生长一路：ghost 相对自身基线 × headroom 过热（双信号 AND，不对称④）。
        重锤 born-again 由持续过剩升级触发（BORN_PERSIST）。
        """
        h = self.h_eff()
        # 浮动设定点带（R1 修复）：xi_eff 随系统自身 gap̄ 稳态分布上浮，
        # born 阈随带抬升（max(GAP_BORN, xi_eff)）——稳态即越带的病根消除。
        xi_eff = self._xi_hi_eff()
        # —— 外环分类（判决不以此为门——平台期门数学不可达，2026-10-05 诊断）——
        if not self.supply_open():
            return self._hold("数据耗尽（供给门）",
                              f"近 {SUPPLY_WINDOW} 周期新鲜摄入 "
                              f"{sum(f for _, f in self.supply_hist[-SUPPLY_WINDOW:])}B "
                              f"< {SUPPLY_MIN_BYTES}B——反刍期只消化不动刀（守卫③）")
        if self.lr_window > 0:
            return self._hold("限频", f"手术 LR 重启窗还剩 {self.lr_window} 周期（阻尼③）")
        if self.consec_rollback >= ROLLBACK_GUARD:
            return self._hold("限频", f"连续回滚 {self.consec_rollback} 次，先养脑不动刀")
        ok, why = self.surgery_allowed(d)
        if not ok:
            return self._hold("限频/守卫", why)

        # —— 收缩判决（先于生长；site 冷却中的收缩让位给生长——细胞自主）——
        born_now = self._confirmed(self.gapbar_hist,
                                   lambda e: e[1] > max(GAP_BORN, xi_eff) + h)
        born_sustained = (len(self.gapbar_hist) >= BORN_PERSIST
                          and all(e[1] > xi_eff + h
                                  for e in self.gapbar_hist[-BORN_PERSIST:]))
        shrink_sites = []
        for k in gaps:
            if not k.endswith(ACTIONABLE_SUFFIX):
                continue
            xi_s = self._site_xi_eff(k)
            if self._confirmed(self.thermo_hist.get(k, []),
                               lambda e, x=xi_s: e[1] > x + h):
                shrink_sites.append(k)
        dorm, frac = v.dormant_report()
        trend = self._dorm_trend()
        if born_now or born_sustained:
            out = self._plan_born_again(d, report, gap_bar,
                                        born_sustained and not born_now)
            if out is not None:
                return out  # 容量下限封顶时返回 hold（不落到生长——重锤被刹车）
        if dorm and self.last_shrink_ok(d.cycle) and (
                any(self._site_ready(k, d.cycle) for k in shrink_sites) or trend):
            return self._plan_decay(d, report, dorm, shrink_sites, trend)
        # —— 生长判决（双信号 AND：ghost 需求 × headroom 过热——不对称④）——
        return self._plan_grow(d, report, v, stream)

    def last_shrink_ok(self, cycle):
        return self.last_shrink_cycle is None \
            or cycle - self.last_shrink_cycle >= MIN_GAP_SAME_SITE

    def _dorm_trend(self):
        """收缩趋势路：休眠占比相对自身滚动基线上斜（呼吸实验教训：
        bottom-5%×2 确认的占比上限 ~5%，绝对门槛 10%/25% 构造性不可达）。"""
        if len(self.dorm_hist) < 5:
            return False
        base = _median([f for _, f in self.dorm_hist[-5:-1]])
        cur = self.dorm_hist[-1][1]
        return cur > base * 1.5 and (cur - base) > 0.01

    def _plan_born_again(self, d, report, gap_bar, escalated):
        """born-again 计划：目标 T 从 gap̄ 导出（T=1−SHRINK_COEF·gap̄，废除写死
        0.7）；连环缩阻尼（3 周期内再缩 → 系数减半）；成熟刹车容量下限非零
        （累计收缩占比 < MIN_BODY_FRAC → 刹车或拒绝——防无限萎缩，守卫⑥）。"""
        coef = SHRINK_COEF * (0.5 if self.last_born_again is not None
                              and d.cycle - self.last_born_again < 3 else 1.0)
        T = max(0.5, min(0.85, 1.0 - coef * gap_bar))
        body_frac = self._body_frac(d, d.sleeping().name)
        if body_frac * T < MIN_BODY_FRAC:
            T_need = MIN_BODY_FRAC / body_frac
            if T_need > 0.85:
                return self._hold("容量过剩（结构级）",
                                  f"成熟刹车容量下限已到：当前体型占比 "
                                  f"{body_frac:.3f}，再缩将破 {MIN_BODY_FRAC} 下限"
                                  f"（守卫⑥：防无限萎缩）", escalated=escalated)
            T = T_need  # 刹车：只缩到下限允许的幅度
        self.plan = {"kind": "wither", "mode": "born_again",
                     "target": round(T, 4), "source": "thermostat",
                     "escalated": escalated}
        self.state = STATE_WITHER_PLAN
        self.last_shrink_cycle = d.cycle
        self.last_born_again = d.cycle
        report["m4_plan"] = ["wither", "born_again", round(T, 3)]
        return {"kind": "born_again", "target": round(T, 4), "class": "容量过剩（结构级）",
                "reason": f"gap̄={gap_bar:.3f}"
                          f"{'（持续过剩升级）' if escalated else ' 超结构错配阈'}",
                "escalated": escalated}

    def _plan_decay(self, d, report, dorm, shrink_sites, trend):
        """软衰减分代计划（轻中度过剩）：目标=确认休眠名单（2 周期确认，
        ReDo 之后仍在册者）。"""
        self.plan = {"kind": "wither", "mode": "decay", "targets": dorm,
                     "source": "thermostat"}
        self.state = STATE_WITHER_PLAN
        self.last_shrink_cycle = d.cycle
        report["m4_plan"] = ["wither", "decay", len(dorm)]
        self._record_action(sorted(dorm), "shrink", d.cycle, report)
        why = f"越带位 {len(shrink_sites)} 个" if shrink_sites else "休眠占比上斜"
        return {"kind": "decay", "targets": len(dorm), "class": "容量过剩（site 级）",
                "reason": why + ("（趋势路）" if trend else "（绝对带）"),
                "trend": trend}

    def _plan_grow(self, d, report, v, stream):
        """生长判决（研究报告 §3.3 生长行）：headroom 过热前置门（省扫描）→
        ghost_scan（只读真实回放流——L8 红线：选址禁用探测集）→ 逐位相对
        自身基线的需求确认 → argmax 选址 → 乘性生长律 delta=λ_g·W·e →
        预算/供给比（守卫④）与显存预算 → 计划。"""
        h = d.sleeping()
        if not stream or len(stream) < d.cfg.block_size + 2:
            return self._hold("观测不足", "回放流过短，无法幽灵扫描")
        # 前置门：轴位的利用率账本过热（轴②读 attn_out 位——v 通道即 proj 输入）
        cands, blocked = [], []
        for li in range(len(h.model.blocks)):
            for gsite, axis in (("mlp_hidden", "mlp"), ("attn_v", "attn_v")):
                led_key = f"b{li}.{_AXIS_LEDGER[axis]}"
                if not self._confirmed(self.thermo_hist.get(led_key, []),
                                       lambda e: e[2] > XI_LO_HOT):
                    continue
                site_id = f"b{li}.{_AXIS_LEDGER[axis]}"
                if not self._site_ready(site_id, d.cycle):
                    blocked.append(site_id)
                    continue
                cands.append((li, gsite, axis, site_id))
        if not cands:
            if blocked:  # 有过热轴位但在冷却/熔断（限频③/振荡熔断⑤）
                return self._hold("限频", f"轴位冷却/熔断中：{sorted(set(blocked))}")
            return self._hold("容量受限候选未确认",
                              f"无过热轴位（headroom ≤ {XI_LO_HOT}，确认窗未满）")
        # 幽灵扫描：方向货币（GradMax/cascade-correlation 同款；L5 合规——
        # 数据是选拔出的真实回放流，与探测集零接触）
        try:
            gains = ghost_scan(h.model, stream[:4096], seed=d.cycle)
        except (ValueError, RuntimeError):
            gains = {}
        if not gains:
            return self._hold("观测不足", "幽灵扫描无结果")
        # 逐位记账（细胞自主）：判决用**滞后基线**——确认窗（最近 CONFIRM_M 次
        # 扫描）对基线窗（其前 GHOST_BASE_WINDOW 次扫描的中位数）求比。基线不含
        # 确认窗自身，需求跳变才不会被基线当场吸收（自参照带的小窗教训）。
        for (li, gsite), g in sorted(gains.items()):
            gkey = f"{li}.{gsite}"
            gh = self.ghost_hist.setdefault(gkey, [])
            gh.append([d.cycle, g])
            del gh[:-(GHOST_BASE_WINDOW + CONFIRM_M)]
        thr = 1.0 + GHOST_K * GHOST_SE_REL
        confirmed = []
        for gk, gh in self.ghost_hist.items():
            if len(gh) < CONFIRM_M + 1:
                continue
            prior = [g for _, g in gh[:-CONFIRM_M][-GHOST_BASE_WINDOW:]]
            base = _median_lo(prior)
            if base <= 1e-15:
                continue
            win = [g for _, g in gh[-CONFIRM_M:]]
            if all(g > base * thr for g in win):
                confirmed.append((win[-1] / base, gk))
        if not confirmed:
            best_obs = max((g for gh in self.ghost_hist.values() for _, g in gh[-1:]),
                           default=0.0)
            return self._hold("容量受限候选未确认",
                              f"ghost 需求未过自身基线门槛 {thr:.2f}"
                              f"（确认窗未满或未越线）")
        ratio, gk = max(confirmed)
        li, gsite = gk.split(".")
        li, gsite = int(li), gsite
        axis = {"mlp_hidden": "mlp", "attn_v": "attn_v"}[gsite]
        site_id = f"b{li}.{_AXIS_LEDGER[axis]}"
        if not self._site_ready(site_id, d.cycle):
            return self._hold("限频", f"{site_id} 动作间隔未满 {MIN_GAP_SAME_SITE} 周期")
        # 乘性生长律（Butz ν 移植）：delta = λ_g·W·e，e=clip(需求比−1, 0, 2)
        W = self._true_width(h, li, axis)
        e = max(0.0, min(2.0, ratio - 1.0))
        delta = max(8, round(self.lambda_g_eff() * W * e))
        delta = min(delta, max(8, round(GROW_DELTA_CAP_FRAC * W)))  # 单次上限
        # 守卫④预算/供给比：生长通道 ≤ β×(新鲜字节/通道当量)
        budget_units = int(BETA_SUPPLY * self.fresh_since_surgery
                           / SUPPLY_BYTES_PER_UNIT) - self.grown_since_surgery
        if budget_units < 8:
            return self._hold("守卫④预算/供给比",
                              f"新鲜证据 {self.fresh_since_surgery:.0f}B 只够 "
                              f"{budget_units} 通道（< 最小有意义步 8）——投资等供给")
        delta = min(delta, budget_units)
        if axis == "attn_v":  # 轴②整除保持：不足一个头的取整到头宽
            nh = h.model.blocks[li].attn.n_heads
            delta = max(nh, (delta // nh) * nh)
        okm, why = self._mem_ok(d, axis, delta)
        if not okm:
            return self._hold("显存预算", f"显存预算不过：{why}")
        self.plan = {"kind": "grow", "axis": axis, "layer": li, "delta": delta,
                     "seed": d.cycle, "source": "thermostat", "site": gk}
        self.state = STATE_GROW_PLAN
        report["m4_plan"] = ["grow", axis, li, delta]
        self._record_action([site_id], "grow", d.cycle, report)
        return {"kind": "grow", "axis": axis, "site": site_id, "delta": delta,
                "delta_ratio": round(ratio, 3), "class": "容量受限",
                "reason": f"ghost 需求 {ratio:.2f}× 自身基线（门槛 {thr:.2f}）"
                          f" × headroom 过热"}

    def _true_width(self, h, li, axis):
        """P0-2 承袭：宽度按睡脑**真实**脑形计（含移植体），不用基座 cfg。"""
        blk = h.model.blocks[li]
        if axis == "mlp":
            return blk.mlp.fc.out_features + getattr(blk.mlp, "bypass_width", lambda: 0)()
        return (blk.attn.qkv.out_features - 2 * blk.attn.qkv.in_features
                + getattr(blk.attn, "bypass_width", lambda: 0)())

    # ---------- 守卫 ----------

    def sleep_debt(self, d):
        """睡眠债（Bellesi 2017 守卫）：距上次睡眠的交互数 / 期望间隔。"""
        return d._since_sleep / max(1, d.target_interval)

    def surgery_allowed(self, d):
        """结构手术守卫：冷却 / 睡眠债高 / 连续回滚 / 数据供给门（守卫③）。

        供给门在此复查使"排程后供给枯竭"的执行自动走推迟语义（计划保留）。"""
        if self.cooldown > 0:
            return False, f"手术冷却中（还剩 {self.cooldown} 周期）"
        if self.sleep_debt(d) > SLEEP_DEBT_GUARD:
            return False, (f"睡眠债 {self.sleep_debt(d):.1f} > {SLEEP_DEBT_GUARD}"
                           f"（Bellesi 2017 守卫：剥夺期禁突触发生）")
        if self.consec_rollback >= ROLLBACK_GUARD:
            return False, f"连续回滚 {self.consec_rollback} 次，先养脑不动刀"
        if not self.supply_open():
            return False, (f"数据供给门：近 {SUPPLY_WINDOW} 周期新鲜摄入不足"
                           f"（反刍期只消化不动刀——守卫③）")
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
        else:
            # 旁路新单元账本预置（规格 A"初始化即 probation"——下一周期 attach
            # 因 C 匹配而复用本账本，新生通道豁免休眠判决）。捕获对账单同步建立
            # （研究报告 §3.7：标签-捕获的固化侧——ghost 预测 → 利用率实现）。
            blk = h.model.blocks[plan["layer"]]
            if rec["axis"] == "mlp":
                led_key = f"b{plan['layer']}.mlp_hidden.new.{len(blk.mlp.bypass_fcs) - 1}"
                C = blk.mlp.bypass_fcs[-1].out_features
            else:
                led_key = f"b{plan['layer']}.attn_v.new.{len(blk.attn.bypass_qkvs) - 1}"
                C = blk.attn.bypass_qkvs[-1].out_features
            v.sites[led_key] = SiteLedger(
                led_key, C, {"is_new": [True] * C,
                             "probation": [PROBATION_CYCLES] * C})
            self.pending_capture = {"due_cycle": d.cycle + PROBATION_CYCLES,
                                    "keys": [led_key], "hname": h.name}
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
        self.campaign = {"kind": "grow", "gen": 1, "snap": snap,
                         "delta": int(plan["delta"]),
                         "source": plan.get("source", "manual")}
        # 守卫④账目：本次投资记账 + 供给账重新累计（新结构须由其后证据流供养）
        self.grown_since_surgery += int(plan["delta"])
        self.fresh_since_surgery = 0.0
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
            # 目标体型：恒温器计划由 gap̄ 导出（plan["target"]）；手工/退役路径
            # 无 target → 落到 WITHER_TARGET（退役件身份常量）。
            target = float(plan.get("target", WITHER_TARGET))
            student, opt, info = born_again_student(
                h.model.cfg, v, target, d.device, seed=d.cycle)
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
            rec["target"] = target
            self.last_born_again = d.cycle
            # 守卫④账目：整体重置（新体型从零记账）
            self.grown_since_surgery = 0
            self.fresh_since_surgery = 0.0
        else:
            rec["targets"] = {k: len(c) for k, c in (plan.get("targets") or {}).items()}
        self.campaign = {"kind": "wither", "mode": mode, "gen": 1,
                         "max_gen": WITHER_MAX_GEN, "snap": snap,
                         "targets": plan.get("targets") or {},
                         "source": plan.get("source", "manual")}
        self.state = STATE_WITHERING
        self.plan = None
        report["m4_wither_start"] = rec

    # ---------- M4 钩子②：REM 样子相位（训练后、体检前） ----------

    def rem_phase(self, d, report, stream, sel=None):
        h = d.sleeping()
        v = self.vitals_for(d, h.name)
        # ① ReDo 回收（常规代谢，每周期；不占恒温器的"每周期至多一次结构动作"——
        #    回收是代谢不是手术）
        report["m4_redo"] = self._redo(d, h, v)
        # ② 凋零战役的软衰减分代（decay 模式的固定动作；born-again 模式无需衰减）
        if self.state == STATE_WITHERING and self.campaign.get("mode") == "decay":
            report["m4_decay"] = self._decay_generation(d, h, v)
        # ③ 连续结构恒温器（活性路径唯一判决者；2026-10-05 取代平台期阶梯——
        #    阶梯触发链数学不可达，见模块 docstring 诊断节）
        self.thermo_cycle(d, report, v, stream, sel)
        # （平台期阶梯 `_schedule` 已退役：方法体仅为 tests 钉住的历史件保留，
        #   活性路径零调用——退役身份，不许再挂回。）

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
            # 2026-10-04 零点实验发现并修复：ReDo 首次在 GPU 上真实触发（有休眠
            # 单元可回收）即抛 RuntimeError("Expected a 'cuda' device type for
            # generator but found 'cpu'")——torch.Generator() 默认 CPU，而权重在
            # CUDA。generator 必须与被重置权重同设备。CPU 路径语义不变（tests 全绿）。
            # （交接 §6.6b 已挂账 ReDo 种子的进程间非确定问题；本修复只动设备侧，
            # 种子口径 hash((key, cycle)) 原样保留，待 M4 遗留账统一处理。）
            g = torch.Generator(device=d.device).manual_seed(hash((key, d.cycle)) & 0x7FFFFFFF)
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

    # —— 退役件（2026-10-05 触发链诊断：平台期阶梯数学不可达，活性路径已由
    #    连续结构恒温器取代。以下两个方法体仅为 tests/test_conformance.py 钉住的
    #    历史行为保留（铁律：既有断言只增不降），生产路径零调用——退役身份，
    #    不许再挂回活性路径。） ——

    def _schedule(self, d, report, v, stream):
        """【退役件】平台期阶梯：band=plateau//3 → ≥2 阶 LR 退火、≥3 阶排程手术。
        退役原因：呼吸实验实测 margin 恒 +0.025~0.055（系统被设计成永不停学），
        9 连击期望 ~1300 周期——数学不可达。保留仅为既有测试钉住。"""
        if self.cooldown > 0:  # 递减已移至 thermo_cycle（每周期唯一入口）
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
        """【退役件】固定步长幅度（GROW_DELTA_FRAC=5%）。恒温器的乘性律
        （λ_g·W·e，见 _plan_grow）已取代它；本方法仅为 P0-2 断言钉住：
        幅度必须按睡脑**真实**脑形（d.sleeping().model.cfg）计，不能用基座
        d.cfg——born-again 学生脑上名义 5% 会实际排成 7%+。"""
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

    # ---------- M4 钩子③：体检后（锚 / 平台期 / 战役验收 / 二阶阻尼） ----------

    def post_exam(self, d, report, passed):
        """返回换班指令："swap" | "rollback" | "keep"。

        NORMAL：与 M0 逐字等价（passed→swap / fail→rollback）。
        GROWN：fail → ROLLBACK（还原形态+装回醒脑权重，规格 C 状态机）+
        增益自适应⑤（λ_g×0.5，可塑性不归零）。
        WITHERING：fail → keep（学生保留标志，规格 C）；分代耗尽 → 放弃还原。
        """
        probe_new = report.get("probe_new")
        margin = report.get("gate_margin")
        # margin 滚动账（R2）：成熟度原料 _mature_input 的分布窗
        if margin is not None:
            self.margin_hist.append(round(margin, 6))
            del self.margin_hist[:-MARGIN_HIST_CAP]
        # 平台期计数（退役阶梯的原料：margin<绝对 ε；活性路径的成熟度已改用
        # margin 滚动分位——绝对 ε=0.005 在真实系统结构性不可达，见 _mature_input）
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
            # R3（监督审计 2026-10-05）：delta 必须在清 campaign **之前**取——
            # 原实现先 self.campaign = None 再读它扣账，回滚扣的恒是 0，
            # grown_since_surgery 永不回冲（守卫④的预算账目成死账，动态验证
            # 16→16）。
            snap = (self.campaign or {}).get("snap")
            delta = int((self.campaign or {}).get("delta") or 0)
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
            # 增益自适应⑤（研究报告 §3.4.5）：生长验收失败 → λ_g×0.5（下限
            # LAMBDA_MIN>0——可塑性永不归零）+ 供给账回冲 + 捕获对账作废
            self.lambda_g = max(self.lambda_g * 0.5, LAMBDA_MIN)
            self.grown_since_surgery = max(0, self.grown_since_surgery - delta)
            self.fresh_since_surgery = 0.0
            self.pending_capture = None
            report["m4_lambda_g"] = round(self.lambda_g, 5)
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

    # —— M4 钩子②：REM 样子相位（ReDo 回收 / 软衰减 / 连续结构恒温器）——
    if m4:
        ctl.rem_phase(dolphin, report, stream, sel)

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

    # 值自成：学习率反馈（Plasticity 唯一控制器）+ 手术 LR 窗口 / 退役阶梯退火（不再触发）
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
        elif ctl.anneal_pending:  # 退役阶梯遗留：活性路径不再置位，兼容旧档
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
        body = report.get("body") or {}
        act = body.get("action", {})
        print(f"[睡眠 {report['cycle']}] 选拔 {report['selected']} / 滞留 {report['residue']}"
              f"  体检 {report.get('probe_old')} → {report.get('probe_new')}"
              f"  lr={new_lr:.2e}"
              f"  体 {body.get('d_model', '?')}/{'%.2f' % body.get('util_gap', 0)}"
              f"  {'✓ 换班' if report.get('swapped') else '✗ 作废回滚'}"
              f"{'  刀:' + str(act.get('kind')) if act.get('kind', 'hold') != 'hold' else ''}")
    return report
