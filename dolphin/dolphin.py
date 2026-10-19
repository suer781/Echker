"""双半球管理与对外接口（律 L2 / L3）。

两个 hemisphere 初始化时权重同源；一次只有一个在训练（睡脑），
另一个权重冻结负责服务（醒脑）。交换只发生在体检通过后的换班瞬间。

惊讶值链路：mean_nll → goldilocks_gate → band_w → score。
"""
import glob
import hashlib
import os
import secrets
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime

import torch

from .adapt import Plasticity
from .amygdala import Amygdala
from .autotune import Autotune
from .debt import DebtLedger
from .experience import Experience, ExperienceBuffer
from .goldilocks import PRIOR_ABILITY, gate as goldilocks_gate
from .life import LifeController
from .memory import MemoryEntry, MemoryStore
from .param_adaptive import ParamAdaptive
from .model import ByteTransformer, Config
from .probe import evaluate as probe_eval
from .probe import gate_decision, load_chunks

# 律定（G9）：睡眠压力阈值域。域本身是律（封堵 ×1.10/×0.90 无钳位的两个失效方向：
# 高惊讶语料下冲上天文数字→压力触发永久失效；低语料下趋零→周期连发），
# 端点值自成（暂定：下限=单条经验惊讶度量级，上限≈先验惊讶度 ln256 的十倍量级）。
THRESHOLD_LO = 1.0
THRESHOLD_HI = 50.0


def clamp_threshold(v):
    """G9：睡眠压力阈值域钳位——全部阈值修改点统一走此函数（maybe_sleep 与
    life._threshold_feedback（喂食路径阈值反馈，现居 dolphin/life.py），不许出现
    第三份乘法实现）。"""
    return max(THRESHOLD_LO, min(THRESHOLD_HI, float(v)))


def _migrate_mlp_state_dict(sd):
    """旧版 MLP（fc/proj）存档 → DualMLP（rule/body/out）权重迁移（2026-10-19）。

    旧 FFN: y = proj(gelu(fc(x)))。
    新 FFN: y = out(gelu(rule(x)) * tanh(body(x)))。

    迁移策略（近似等价启动）：
      rule ← fc（树突通道继承原第一层）；
      body ← 权重 0、偏置 1（tanh(1)≈0.7616 常数调制，rule ⊙ body ≈ 0.76·rule）；
      out  ← proj × (1/tanh(1))（补偿常数缩放，使初始输出接近旧模型）。
    旧 WidenedMLP 只有 bypass_fcs 旁路，迁移时同步补齐 bypass_bodies
    （权重 0、偏置 1，与主 body 同策略）。已是 DualMLP 键名且旁路齐备则
    内容不变。
    """
    out = dict(sd)
    if any(".mlp.fc.weight" in k for k in sd):
        import math
        scale = 1.0 / math.tanh(1.0)
        # fc → rule（原样）
        for k in list(out):
            if ".mlp.fc." in k:
                out[k.replace(".mlp.fc.", ".mlp.rule.")] = out.pop(k)
        # rule → body（权重 0，偏置 1）
        for k in list(out):
            if ".mlp.rule.weight" in k:
                bias_key = k.replace(".mlp.rule.weight", ".mlp.rule.bias")
                out[k.replace(".mlp.rule.weight", ".mlp.body.weight")] = torch.zeros_like(out[k])
                out[k.replace(".mlp.rule.weight", ".mlp.body.bias")] = torch.ones_like(out[bias_key])
        # proj → out（权重 ×scale 补偿，偏置原样）
        for k in list(out):
            if ".mlp.proj." in k:
                nk = k.replace(".mlp.proj.", ".mlp.out.")
                if nk.endswith(".weight"):
                    out[nk] = out.pop(k) * scale
                else:
                    out[nk] = out.pop(k)
    # 旁路 body 通道补齐（旧 WidenedMLP 只有 bypass_fcs）：权重 0、偏置 1，
    # 与主 body 同策略（tanh(1) 常数调制，宽化瞬间旁路零输出保持不变）。
    for k in list(out):
        if ".mlp.bypass_fcs." in k and k.endswith(".weight"):
            bb_w = k.replace(".bypass_fcs.", ".bypass_bodies.")
            if bb_w not in out:
                bias_key = k.replace(".weight", ".bias")
                out[bb_w] = torch.zeros_like(out[k])
                out[bb_w.replace(".weight", ".bias")] = torch.ones_like(out[bias_key])
    return out


class GateDisabled(RuntimeError):
    """G8：体检探测集缺失/为空时拒绝开睡的 fail-fast 信号。

    旧版静默失效链：probe 缺失 → dolphin 容忍空集 → probe.evaluate 空集返回 inf
    → passed = (inf < inf) = False 恒假 → 每个周期正常训练后被无条件作废回滚，
    无任何报警的静默永久回滚。新行为：开睡前检查，缺失即抛——拒绝无效睡眠周期。
    probe 文件恢复后由 ensure_gate_ready 自动重新加载并解禁（自动恢复）。"""


class Hemisphere:
    def __init__(self, name, model, lr):
        self.name = name
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


class Dolphin:
    def __init__(self, cfg=None, device=None, lr=5e-5, probe_path=None):
        # lr：睡眠学习率。睡眠是温和巩固，不是重训——过猛会扰动基础权重，
        # 体检门控会连续否决（已在实验中证实）。此值自成，体检失败时自动退火。
        self.cfg = cfg or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m1 = ByteTransformer(self.cfg).to(self.device)
        m2 = ByteTransformer(self.cfg).to(self.device)
        m2.load_state_dict(m1.state_dict())  # 初始化同源
        self.h = [Hemisphere("A", m1, lr), Hemisphere("B", m2, lr)]
        self.awake_idx = 0
        self.buffer = ExperienceBuffer()
        self.memory = MemoryStore()
        self.debt_ledger = DebtLedger()          # 双半球债务账本（胼胝体通道）
        self.hemi_pressure = [0.0, 0.0]          # 每半球睡眠压力（值自成，随档持久化）
        # S5 蒸馏方向交替标志（2026-10-13 修复）：feeding 生产路径连续周期
        # 互为师生（防止双半球分裂）。False=swap-first 相位（训练对象=旧醒脑，
        # 教师=当前醒脑）；True=跳过 swap-first、直接以当前睡脑为训练对象。
        # 每周期结束翻转；随档持久化，旧档缺省 False（兼容）。
        self._teacher_flip = False
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.probe_path = probe_path or os.path.join(root, "probes", "probe.txt")
        self.probe_chunks = []
        self.gate_disabled = False  # G8：体检门控可用性标志（探测集缺失时拒绝开睡）
        if os.path.exists(self.probe_path):
            self.probe_chunks = load_chunks(self.probe_path, self.cfg.block_size)
        if not self.probe_chunks:
            # G8：探测集缺失/为空 → 门控语义失效。醒目报警（stderr）+ fail-fast 标志：
            # sleep/feed 的门控段将抛 GateDisabled 拒绝开睡，取代旧版"体检 inf<inf
            # 恒假 → 每周期无条件作废回滚"的静默永久回滚。恢复探测集后自动解禁。
            self.gate_disabled = True
            print(f"[Dolphin][警报] 体检探测集缺失或为空：{self.probe_path}\n"
                  f"[Dolphin][警报] 门控语义失效（律 L8）——睡眠周期将被拒绝开睡"
                  f"（GateDisabled），恢复探测集文件后自动解禁。", file=sys.stderr)
        self.budget = 0.45        # 值：自成（律 L9：部分有损传输）
        self.note_k = 5
        self.dream_k = 5          # 值：律定（暂）——每周期做梦注入上限，防旧知识挤占新经验选拔预算（审计 G11）；M4 接入反馈回路
        self.target_interval = 15  # 值：自成（期望睡眠间隔，交互数）
        # M2 修复（2026-10-06）：kd_alpha/kd_T/sleep_steps 三个量此前只写进
        # autotune.tunables、训练路径从不读取（死控制）。现初始化为与
        # Autotune.tunables 一致的初值，并在 Autotune.adjust() 同步回这些属性，
        # 训练路径（life.run_cycle 值自成覆盖）从此读取 autotune 调好的值。
        self.kd_alpha = 0.5
        self.kd_T = 2.0
        self.sleep_steps = 120
        # 三档睡眠模式（2026-10-11）：light 浅睡/rem REM 各自独立的自适应步数域，
        # 默认值落在各自物理域内（light [8,40] 默认 16，rem [20,80] 默认 40）。
        # autotune.adjust() 会按与 sleep_steps 相同的调节律同步回这些属性。
        self.sleep_steps_light = 16
        self.sleep_steps_rem = 40
        self.sleep_steps_nrem = 80
        # 判决带传导系数（2026-10-11）：K_MARGIN 真正接入 autotune 慢速反馈——
        # 默认 2.0 与 probe.K_MARGIN 一致，autotune 按体检通过率/边缘信号在
        # [1.0,3.0] 域内慢速自调，Dolphin.gate 传此值给 gate_decision。
        self.k_margin = 2.0
        # 值自成自动调参（2026-10-05）：note_k/dream_k/promote_hits/kd_alpha/kd_T/
        # target_interval/sleep_steps 的世界信号反馈控制器——这些量不再写死，
        # 由 Autotune 慢速积分自调（律固定，值自成）。随档持久化。
        self.autotune = Autotune()
        # 杏仁核温度自动托管（2026-10-08）：用系统自身的世界信号（体检 margin/
        # 回滚率/KD 蒸馏比/记忆命中率）自动调节 serve 的生成温度。威胁高→低温
        # （保守稳定），威胁低→高温（创造丰富）。慢环（AMYGDALA_EVERY=20 周期
        # 一次）确保短测试永不触发。随档持久化。
        self.amygdala = Amygdala()
        # 参数自适应控制器（2026-10-06 用户定案）：独立于恒温器判决主路径的
        # 排计划控制器。默认开启（param_adaptive_enabled=True，生产默认启用"完全
        # 自学习"）；测试用 make_dolphin 工厂显式设 False 做隔离。
        self.param_adaptive = ParamAdaptive()   # 参数自适应控制器
        self.param_adaptive_enabled = True       # 默认开启（生产默认启用）
        # 非加密用途（重放变异抽样），仍统一使用密码学安全随机源
        self.rng = secrets.SystemRandom()
        self.plasticity = Plasticity()  # 成年灵敏度：睡眠学习率的唯一控制器
        self.cycle = 0
        # 训练债平衡（2026-10-11）：两半球各自累计实际训练步数，用于判定
        # 睡眠模式时识别"欠训侧"，优先安排 light 浅睡快速训练并换班以缩小差距。
        # 换班/回滚后不变——只累加实际训练过的那个半球（sleeping_idx）。
        self.hemi_trained_steps = [0, 0]
        self._since_sleep = 0
        # 2026-10-06 修复：上次睡眠的墙钟时间戳（None=冷启动/测试）。
        # 用于 sleep_debt 的真实时间债务度量——批量喂食下睡眠周期频繁触发，
        # 时间债务低，不阻塞恒温器手术窗口（修复结构性封锁）。
        self._last_sleep_wall = None
        # 部署期双线程（feed.py）：锁只护共享状态交换，不护计算。
        self.feed_lock = threading.Lock()  # learn 的缓冲交换 与 训练线程快照/换班 互斥
        self._busy_lock = threading.Lock()  # learn_busy 的专用锁（审计②-3：原子计数）
        self.learn_busy = 0                # 在途读取（learn/serve）计数：训练线程动睡脑前等待清零，保证读到的永远是静态权重（L3）
        # G3 自续骨：部署侧经验流入总量（learn/serve 都计）。feed.py 守护模式
        # 用它判断"是否有新经验流入"以自动恢复正式喂食；喂食游标的语义与推进
        # 由 feed.py 全权维护，本类只做透传存取（职责分离）。
        self.learn_total = 0
        self.feed_cursor = None
        # 模型自评估日志（2026-10-14）：替代用户手动打分，后台自动评估。
        # 每次交互后记录 {eid, reward, dim}，reward 与原手动打分格式兼容，
        # dim 含三维度判断依据，便于后续校验与优化。随档持久化，旧档兼容空列表。
        self.self_eval_log = []
        # 用户负面反馈事件日志（2026-10-15）：训完上线后的"有错就改"回路。
        # 每次捕获到异常事件（如模型误删用户内容/服务错误/答非所问/空响应等）
        # 记录 {kind, eid, detail, ts}，并同步对交互打负分。随档持久化，
        # 旧档兼容空列表。驱动 autotune 的 negative_feedback_rate 信号。
        self.anomaly_log = []
        # M4 生命节律（律 L11）：自生长/自凋零控制器 + 人工总开关安全阀。
        # life_enabled=False 时 life.run_cycle 的全部 M4 钩子失效（纯 M0 语义）；
        # 人工只保留这个总开关，禁止干预手术的时机/位置/幅度（律 L11）。
        self.life_ctl = LifeController()
        self.life_enabled = True
        # 学习停滞检测原料（2026-10-12）：最近 N 周期体检 margin 滚动窗。
        # 由 run_cycle 在体检报告生成后追加；随档持久化，旧档兼容空列表。
        self.last_margins = []
        # 金发姑娘价值门控（律 L4）：能力基线 ability = 系统当前的"惊讶度先验"，
        # 初始 = 随机模型的先验惊讶度 ln256（≈5.545）。随睡眠周期末用入选经验的
        # mean surprise 慢速 EMA 更新（见 life.run_cycle）。随档持久化。
        self.ability = PRIOR_ABILITY
        # 门控观测计数：太熟跳过 / 太怪挂起的累计**窗口数**（2026-10-17 改
        # 片段级门控后按窗口计），随档持久化，写进日志观测。
        self.skipped_too_easy = 0
        self.suspended_too_hard = 0
        # 片段级噪声屏蔽观测：累计被屏蔽的噪声字节数（2026-10-17 新增）。
        # learn() 按窗口判决，噪声窗口不入缓冲，此计数反映实际被屏蔽的数据量。
        self.noise_masked_bytes = 0
        # 金发姑娘带通阈值（2026-10-13）：由 autotune 慢速自调（域内值自成），
        # 初值 0.05/0.20 与模块常量一致；learn() 从 autotune 读取当前值。
        self.goldilocks_lo = 0.05
        self.goldilocks_hi = 0.20
        # 语义饱和机制（L7 强化，2026-10-17 用户定案：长期+短期双尺度）。
        # 短期窗口：短时间内大量重复 → 强抑制（衰减 30%/次，下限 10%）；
        # 长期窗口：历史较长范围内的偶尔重复 → 弱抑制（衰减 5%/次，下限 60%），
        # 避免"最近 200 条重复"式单窗口导致某些字被排斥、语义识别错误。
        self.satiation_window_short = 50
        self.satiation_decay_short = 0.30
        self.satiation_min_short = 0.10
        self.satiation_window_long = 2000
        self.satiation_decay_long = 0.05
        self.satiation_min_long = 0.60
        self.saturation_threshold = 0.50     # 单批重复占比阈值（>50% 触发饱和防护）
        self.satiation_counts_short = {}   # 短期窗口指纹计数
        self.satiation_order_short = deque()
        self.satiation_counts_long = {}       # 长期窗口指纹计数
        self.satiation_order_long = deque()
        self.saturation_events = 0           # 饱和防护触发计数（随档持久化，观测用）
        self.saturation_last_ratio = 0.0     # 最近一次窗口内重复占比（观测用）
        # 近期惊讶值分布（2026-10-17 用户定案：金发姑娘动态偏差用）
        self._surprise_hist = []
        # 睡醒考试开关（2026-10-17 用户定案：移除硬编码 probe 考试——模型对
        # 固定探测集过度熟悉后会产生"应试技巧"，考试越来越容易但真实能力未必
        # 提升）。默认 True 保持既有行为/测试兼容；生产训练入口（feed.py）默认
        # 关闭（--no-gate-exam），改为学习质量自审计 + 局部回滚。
        self.gate_exam_enabled = True

    def awake(self):
        return self.h[self.awake_idx]

    def sleeping(self):
        return self.h[1 - self.awake_idx]

    def swap(self):
        self.awake_idx = 1 - self.awake_idx

    def memory_scale(self):
        """记忆处理规模系数（对数尺度，值自适应的依据）。
        小模型（≈1e5 参数）→0.1；大模型（≈1e8 参数）→1.0。
        小模型避免记忆处理过载导致上下文溢出，大模型充分利用处理能力。"""
        import math
        n = sum(p.numel() for p in self.awake().model.parameters())
        s = (math.log10(max(float(n), 1e5)) - 5.0) / 3.0
        return max(0.1, min(1.0, s))

    def adaptive_cold_cap(self):
        """冷层自适应上限（2026-10-17 用户定案）：模型参数越多 → 冷层上限越高。

        基础 2 万条，按模型参数量对数尺度放大，封顶 100 万条——防止冷层无限
        膨胀、比模型本身还离谱（曾致 4 天 52 万条/2.1GB、95% 零命中）。
        参数 1e5 → 2 万条；1e8 → 100 万条（对数插值）。
        """
        import math
        n = sum(p.numel() for p in self.awake().model.parameters())
        s = (math.log10(max(float(n), 1e5)) - 5.0) / 3.0  # 0.0（1e5）~1.0（1e8）
        cap = int(20000 * (1.0 + max(0.0, min(1.0, s)) * 49.0))
        return max(20000, min(1_000_000, cap))

    def _busy_inc(self):
        """learn_busy 原子自增（审计②-3）：serve 与 learn 并发时防止丢失更新。

        专用 _busy_lock 只护计数器本身，不包裹部署计算——部署侧读权重
        依旧永不排队等锁（与 feed_lock 的并发语义完全无关）。
        """
        with self._busy_lock:
            self.learn_busy += 1

    def _busy_dec(self):
        """learn_busy 原子自减（审计②-3）：serve 与 learn 并发时防止丢失更新。

        与 _busy_inc 对称，用同一把专用锁保证计数严格配对。
        """
        with self._busy_lock:
            self.learn_busy -= 1

    def probe_loss(self, hem):
        return probe_eval(hem.model, self.probe_chunks, self.device)[0]

    def gate(self, old_hem, new_hem):
        """律 L8 体检判决：全卷逐块评分 + 配对 margin 判决带。

        唯一实现在此——life.run_cycle（sleep.run_cycle 的 M4 继任者）调用本方法，
        判决逻辑（probe.gate_decision）与双模型评分编排零复制，两份
        训练路径的实现自此无从漂移（M4 评审警告消解）。
        返回 (passed, detail)；detail 直接并入睡眠报告。
        """
        per_old = probe_eval(old_hem.model, self.probe_chunks, self.device)[1]
        per_new = probe_eval(new_hem.model, self.probe_chunks, self.device)[1]
        # k_margin 由 autotune 慢速自调（域 [1.0,3.0]，随档持久化）；默认 2.0。
        return gate_decision(per_old, per_new, k_margin=self.k_margin)

    # ---------- 体检门控就绪检查（G8 护航骨） ----------

    def ensure_gate_ready(self):
        """G8 fail-fast：开睡前必须通过的门控就绪检查（life.run_cycle 的唯一
        共用入口，无从漂移）。

        - 探测集就绪 → 放行（若此前曾缺失则自动重新加载并解除警报，自恢复）；
        - 缺失/为空 → 显式告警（stderr）、gate_disabled=True、抛 GateDisabled。
          门控语义完整性优先于可用性（设计决定）：体检无法判决的周期宁可不开跑，
          也不许"训练完 → inf<inf 恒假 → 无条件作废回滚"地静默空转。
        """
        if not self.probe_chunks and os.path.exists(self.probe_path):
            try:  # 修复后自动恢复：probe 文件回来了就重新装卷
                self.probe_chunks = load_chunks(self.probe_path, self.cfg.block_size)
            except OSError:
                self.probe_chunks = []
        if self.probe_chunks:
            if self.gate_disabled:
                self.gate_disabled = False
                print("[Dolphin] 体检探测集已恢复，门控重新启用（G8 自动恢复）。",
                      file=sys.stderr)
            return
        self.gate_disabled = True
        print(f"[Dolphin][警报] 开睡被拒绝：体检探测集缺失或为空（{self.probe_path}）。"
              f"恢复探测集文件后自动解禁。", file=sys.stderr)
        raise GateDisabled(
            f"体检探测集缺失或为空（{self.probe_path}）：门控语义失效（律 L8），"
            f"拒绝开睡——G8 fail-fast 取代静默回滚")

    def probe_sha256(self):
        """体检卷面指纹（G7）：入档用于 load 时校验卷面未被替换/损坏。"""
        try:
            with open(self.probe_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return None  # 存档时就没有探测集 → 存档里记 None，load 时无从校验

    # ---------- 服务（律 L3：醒脑权重冻结） ----------

    @torch.no_grad()
    def serve(self, user_text: str, max_new=48, temperature=None, top_k=None, top_p=None):
        """部署服务。与 feed 并发时纳入读屏障（审计②-5：serve 原先不在保护体系内）。

        temperature=None（默认）：由杏仁核（Amygdala）自动托管——用系统自身的
        世界信号（威胁等级）决定生成温度，无需人工设温度。
        top_k=None / top_p=None（默认）：同样由杏仁核自动托管（威胁高收紧、
        威胁低放松）。显式传入 temperature/top_k/top_p 会覆盖杏仁核
        （ward/演示等固定参数场景不变）。
        """
        self._busy_inc()  # 读屏障：换班栅栏把 serve 与 learn 同等对待（律 L3）
        try:
            ub = user_text.encode("utf-8")
            # 梦境相关热提升（2026-10-17 用户定案）：用户提及与冷层梦境记忆
            # 相关的内容时，相关记忆临时复制到热层（serve 的 retrieve 快通道
            # 立即能引用）。冷层检索锁外做，失败静默（不阻塞服务）。
            try:
                self.memory.promote_cold_related(ub, k=2)
            except Exception:
                pass
            if temperature is None:
                # 杏仁核三层调节（2026-10-17 用户定案）：先量输入内容的陌生
                # 程度（L1 实时），再叠加上下文窗口统计（L2）与用户情绪感同
                # 身受（L3：生气→恐惧降温/开心→升温）。
                surprise = self.awake().model.mean_nll(ub, self.device)
                temperature = self.amygdala.temperature_for_user(
                    user_text, surprise,
                    model=self.awake().model, device=self.device)
            else:
                surprise = None  # 显式传参：不预计算，生成后按原路径算
            if top_k is None:
                top_k = self.amygdala.top_k
            if top_p is None:
                top_p = self.amygdala.top_p
            notes = self.memory.retrieve(ub, k=1)  # 律 L10 预支快通道：引用记忆，不碰权重
            prefix = ("【记忆】" + notes[0].text[-120:] + "\n") if notes else ""
            prompt = (prefix + user_text).encode("utf-8")[-self.cfg.block_size:]
            idx = torch.tensor(list(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
            out = self.awake().model.generate(
                idx, max_new, temperature, top_k=top_k, top_p=top_p)
            resp = bytes(out[0, idx.shape[1]:].tolist()).decode("utf-8", errors="replace")
            if surprise is None:
                surprise = self.awake().model.mean_nll(ub, self.device)  # 对用户输入的困惑度
            with self.feed_lock:
                # 语义饱和：重复的用户输入同样获得衰减因子（原始惊讶值保留）
                sat_factor = self._apply_semantic_saturation(ub, surprise)
                eid = self.buffer.add(ub, surprise, sat_factor=sat_factor)
                self._since_sleep += 1
                self.learn_total += 1  # G3：部署侧经验流入总量（守护模式的恢复信号）
            return resp, eid
        finally:
            self._busy_dec()

    def feedback(self, eid, reward):
        return self.buffer.feedback(eid, reward)

    def capture_anomaly(self, kind, eid=None, detail=None):
        """训完上线后的"有错就改"：实时捕获用户负面反馈事件并对交互打负分。

        与前置拦截门（体检）职责分离——本方法不拦截任何输入/更新，只在异常
        已经发生（如模型误删除用户内容、产生服务错误、答非所问、空响应等）
        时，事后记录事件并对该交互打负分（-1.0），同时把负面反馈率信号送入
        autotune（值自成：评估阈值随负面事件率自适应收紧/放松）。

        返回 (ok, eid)。eid 为 None（纯事件、无对应交互）时只记录事件与
        更新信号，不写缓冲。
        """
        import time as _time
        ok = False
        if eid is not None:
            ok = self.feedback(eid, -1.0)
        self.anomaly_log.append({
            "kind": kind,
            "eid": eid,
            "detail": detail or "",
            "ts": _time.time(),
        })
        if len(self.anomaly_log) > 2000:  # 防日志无限膨胀
            del self.anomaly_log[: len(self.anomaly_log) - 2000]
        # 负面反馈信号：EMA 累加（0.1/次），update_signals 中每周期 *0.9 衰减。
        au = getattr(self, "autotune", None)
        if au is not None:
            s = au.signals
            s["negative_feedback_rate"] = min(
                1.0, s.get("negative_feedback_rate", 0.0) + 0.1)
        return ok, eid

    def negative_feedback_rate(self):
        """当前用户负面反馈率（EMA [0,1]，供外部观测）。"""
        au = getattr(self, "autotune", None)
        if au is None:
            return 0.0
        return float(au.signals.get("negative_feedback_rate", 0.0))

    def self_evaluate(self, text, resp, eid, max_new=64):
        """模型自评估打分（2026-10-14，替代用户手动打分，后台无感执行）。

        基于三个维度自主判断本次交互是否达成用户预期目标：
          - relevance  需求匹配度：用户诉求（输入）与最终输出字节 4-gram 重叠度
                        （输出越紧扣输入，匹配度越高）。
          - completeness 问题解决完整性：输出非空、长度相对输入合理、未过早截断。
          - accuracy   响应准确性：模型对输出自身的置信度（mean NLL 相对能力基线，
                        NLL 越低于基线越准）。

        合成总分映射到与原用户手动打分完全兼容的三档 reward（-1.0 / 0.0 / +1.0），
        并通过原有 feedback(eid, reward) 写入经验缓冲（数据结构不变，不影响后续
        统计与分析流程）。判断依据记录到 self.self_eval_log（随档持久化）。
        整个过程后台执行，无感知替换用户手动打分环节。
        """
        text_b = text.encode("utf-8", errors="replace") if isinstance(text, str) else bytes(text)
        resp_b = resp if isinstance(resp, bytes) else resp.encode("utf-8", errors="replace")
        if not resp_b:
            dim = {"relevance": 0.0, "completeness": 0.0, "accuracy": 0.0,
                   "total": 0.0, "reason": "空响应"}
            self.feedback(eid, -1.0)
            self.self_eval_log.append({"eid": eid, "reward": -1.0, "dim": dim})
            return -1.0

        # —— 自适应评估阈值（2026-10-15，值自成）：从 autotune 读取当前值，
        # 由用户实时负面反馈率驱动慢速自调（域内），不再写死。缺省回落初值。
        au = getattr(self, "autotune", None)
        pass_thresh = float(au.get("eval_pass_thresh", 0.55)) if au else 0.55
        fail_thresh = float(au.get("eval_fail_thresh", 0.30)) if au else 0.30
        garbage_repl = float(au.get("eval_garbage_repl", 0.30)) if au else 0.30
        garbage_print = float(au.get("eval_garbage_print", 0.50)) if au else 0.50

        # —— F-04 修复：乱码/不可读输出直接差评（防自评估正向强化噪声）——
        # 模型处于未学会说话阶段时，输出可能是任意字节（非法 UTF-8 被替换为
        # U+FFFD，或大量控制字符）。这类输出没有信息价值，绝不能给正奖励。
        decoded_resp = resp_b.decode("utf-8", errors="replace")
        repl_ratio = decoded_resp.count("\ufffd") / max(1, len(decoded_resp))
        printable_ratio = sum(1 for ch in decoded_resp
                               if ch.isprintable() and ch != "\ufffd") / max(1, len(decoded_resp))
        if repl_ratio > garbage_repl or printable_ratio < garbage_print:
            dim = {"relevance": 0.0, "completeness": 0.0, "accuracy": 0.0,
                   "total": 0.0, "reason": "输出为乱码/不可读"}
            self.feedback(eid, -1.0)
            self.self_eval_log.append({"eid": eid, "reward": -1.0, "dim": dim})
            return -1.0

        # —— 需求匹配度：输入/输出 4-gram 重叠 ——
        def _grams(s, n=4):
            return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}
        try:
            gin = _grams(text_b.decode("utf-8", "replace"))
            gout = _grams(resp_b.decode("utf-8", "replace"))
            overlap = len(gin & gout) / max(1, len(gin))
        except Exception:
            overlap = 0.0
        relevance = max(0.0, min(1.0, overlap * 4.0))  # 20% 重叠视为满配

        # —— 响应准确性：模型对输出自身的 mean NLL（越低越准）——
        try:
            out_nll = self.awake().model.mean_nll(resp_b, self.device)
        except Exception:
            out_nll = float(getattr(self, "ability", 0.0) or 5.545)
        ability = float(getattr(self, "ability", 0.0)) or 5.545
        # NLL 低于/接近基线 → 高分；高出基线越多 → 越低分
        accuracy = max(0.0, min(1.0, 1.0 - max(0.0, (out_nll - ability)) / ability * 1.5))

        # —— 问题解决完整性：非空 + 长度相对输入合理 + 未过早截断 ——
        truncated = len(resp_b) >= max_new * 0.95
        len_min = max(4, min(16, len(text_b) // 4))
        len_ok = len(resp_b) >= len_min
        completeness = 0.0
        if len_ok:
            completeness = 0.7 + 0.3 * (0.0 if truncated else 1.0)

        total = 0.4 * relevance + 0.3 * completeness + 0.3 * accuracy
        # —— F-04 修复：零重叠（答非所问）时总分封顶到中性档，禁止正奖励 ——
        # 输入有实质内容（非空）但输出与输入没有任何 4-gram 共享时，即使输出
        # 可读，也不能判定为"达成用户预期"。封顶到 pass_thresh-0.01（低于
        # 自适应正奖励线）。输入短于 4 字符时 4-gram 天然为空（overlap=0），
        # 同样封顶，避免短输入被 completeness+accuracy 推成正奖励。
        if text_b and overlap <= 0.0:
            total = min(total, pass_thresh - 0.01)
        # 自适应三档 reward（值自成，不写死）：阈值由 autotune 按用户实时
        # 负面反馈率慢速自调（负面反馈高 → 正奖励线抬升、差评线下移）。
        if total >= pass_thresh:
            reward = 1.0
        elif total < fail_thresh:
            reward = -1.0
        else:
            reward = 0.0
        dim = {
            "relevance": round(relevance, 4),
            "completeness": round(completeness, 4),
            "accuracy": round(accuracy, 4),
            "total": round(total, 4),
            "overlap": round(overlap, 4),
            "out_nll": round(out_nll, 4),
            "truncated": truncated,
            "resp_len": len(resp_b),
        }
        self.feedback(eid, reward)
        self.self_eval_log.append({"eid": eid, "reward": reward, "dim": dim})
        if len(self.self_eval_log) > 2000:  # 防日志无限膨胀
            del self.self_eval_log[: len(self.self_eval_log) - 2000]
        return reward

    def _apply_semantic_saturation(self, data: bytes, surprise: float) -> float:
        """语义饱和：长期+短期双尺度衰减因子（2026-10-17 用户定案）。

        对同一数据（sha256 指纹）分别在**短期窗口**与**长期窗口**内计数重复，
        返回两个线性衰减因子中的**较大者**：
        - 短期（50 条内大量重复，如刷屏）：衰减 30%/次，下限 10% → 强抑制；
        - 长期（2000 条内偶尔重复）：衰减 5%/次，下限 60% → 弱抑制。

        取较大者：短期大量重复会被明显降权，但长期内偶尔重复不会把该内容
        排斥到无法学习（避免"最近 200 条重复"式单窗口导致某些字被排斥、
        语义识别错误）。原始惊讶值保持不变（带通中心/门控判决依赖真实值）。

        单批重复占比 = 短期窗口内重复占比；超过阈值时短期衰减率增强（上限
        0.5）并累计 saturation_events。
        """
        h = hashlib.sha256(bytes(data)).hexdigest()[:16]
        # 短期计数（FIFO 滑动窗口）
        n_short = self.satiation_counts_short.get(h, 0)
        self.satiation_counts_short[h] = n_short + 1
        self.satiation_order_short.append(h)
        while len(self.satiation_order_short) > self.satiation_window_short:
            old = self.satiation_order_short.popleft()
            c = self.satiation_counts_short.get(old, 1)
            if c <= 1:
                del self.satiation_counts_short[old]
            else:
                self.satiation_counts_short[old] = c - 1
        # 长期计数（FIFO 滑动窗口）
        n_long = self.satiation_counts_long.get(h, 0)
        self.satiation_counts_long[h] = n_long + 1
        self.satiation_order_long.append(h)
        while len(self.satiation_order_long) > self.satiation_window_long:
            old = self.satiation_order_long.popleft()
            c = self.satiation_counts_long.get(old, 1)
            if c <= 1:
                del self.satiation_counts_long[old]
            else:
                self.satiation_counts_long[old] = c - 1
        # 批内重复占比（短期窗口观测 + 防护触发）
        if self.satiation_order_short:
            self.saturation_last_ratio = (
                1.0 - len(set(self.satiation_order_short)) / len(self.satiation_order_short))
        decay_short = self.satiation_decay_short
        if self.saturation_last_ratio > self.saturation_threshold:
            decay_short = min(0.5, decay_short * 1.5)
            self.saturation_events += 1
        factor_short = max(self.satiation_min_short, 1.0 - decay_short * n_short)
        factor_long = max(self.satiation_min_long,
                         1.0 - self.satiation_decay_long * n_long)
        # 取较大者：短期强抑制、长期弱抑制，取宽松侧避免排斥字
        return max(factor_short, factor_long)

    def learn(self, text: str, source="dataset"):
        """部署期喂养：片段级金发姑娘门控（2026-10-17 改）——**屏蔽噪声片段，
        保留有效内容**，而不是整条略过。

        原始律 L4 的"太熟跳过/太怪挂起"在**整条数据**粒度上判决，会把"大部分
        正常、小部分噪声"的数据整条扔掉——不是真正的噪声屏蔽。本实现改为：

          1. 将输入按 block_size 无重叠切分为窗口；
          2. 批量计算每窗口的惊讶度（mean_nll_batch）；
          3. 对每窗口用金发姑娘带通三态判决：
               delta < -LO  → 太熟窗口（已会）→ 屏蔽（计数 skipped_too_easy）
               delta > +HI  → 太怪窗口（噪声/学不会）→ 屏蔽（计数 suspended_too_hard）
               其余          → 有效窗口 → 保留
          4. 所有保留窗口**拼接**后进入经验缓冲（有效内容正常学习）；
             被屏蔽窗口字节数计入 noise_masked_bytes（观测）。

        只有整条全部窗口都被屏蔽时才不进缓冲（等效旧"整条丢弃"）。窗口粒度
        让模型学会"屏蔽噪声、保留正文"——这正是自学习应有的噪声免疫。

        双线程约束不变：mean_nll 是部署计算，锁外做；只有缓冲交换持 feed_lock。
        """
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        self._busy_inc()
        try:
            # 片段级惊讶度：无重叠切窗（尾部不足 2 字节的碎片丢弃，无法计算）
            windows = self._split_windows(data, self.cfg.block_size)
            if not windows:
                with self.feed_lock:
                    self.learn_total += 1  # G3：部署侧经验流入总量（守护恢复信号）
                    return None
            surprises = self.awake().model.mean_nll_batch(windows, self.device)
        finally:
            self._busy_dec()
        with self.feed_lock:
            # 金发姑娘带通：基准阈值来自 autotune（律定域内值自成），再按
            # 近期惊讶值分布的变异系数**动态制定**（2026-10-17 用户定案）。
            base_lo = self.goldilocks_lo
            base_hi = self.goldilocks_hi
            au = getattr(self, "autotune", None)
            if au is not None:
                base_lo = au.get("goldilocks_lo", base_lo)
                base_hi = au.get("goldilocks_hi", base_hi)
            self._surprise_hist.extend(float(s) for s in surprises)
            del self._surprise_hist[:-256]
            lo, hi = self._dynamic_band(base_lo, base_hi)
            kept, kept_surp = [], []
            masked = 0
            skip_n = 0
            suspend_n = 0
            for w, s in zip(windows, surprises):
                if len(w) < 2:
                    # 无法计算有意义的惊讶值：直接保留（用户定案：不丢弃），
                    # 惊讶值用能力线代替（中间带语义）。
                    kept.append(w)
                    kept_surp.append(s if s > 0 else self.ability)
                    continue
                state, _ = goldilocks_gate(s, self.ability, lo=lo, hi=hi)
                if state == "skip":
                    # 太熟窗口（已会）：跳过丢弃，不进缓冲也不进冷层
                    skip_n += 1
                    masked += len(w)
                    continue
                if state == "suspend":
                    # 太怪窗口（噪声/学不会）：屏蔽出训练流，但进冷层留待能力
                    # 增长后再碰（保留原"挂起"语义；噪声片段不再污染有效学习）。
                    suspend_n += 1
                    masked += len(w)
                    self.memory.add_cold(w, s, self.cycle, kind="suspended")
                    continue
                kept.append(w)
                kept_surp.append(s)
            self.skipped_too_easy += skip_n
            self.suspended_too_hard += suspend_n
            self.noise_masked_bytes += masked
            if not kept:
                # 整条全部窗口都被屏蔽：仍计经验流入，但不进缓冲
                self.learn_total += 1
                return None
            kept_data = b"".join(kept)
            # 保留窗口的加权平均惊讶度（按字节数加权）作为该条经验入缓冲的
            # surprise——带通中心/选拔分数对应的是实际入缓冲的内容，而非原始整条。
            total_w = sum(len(w) for w in kept)
            surprise = sum(len(w) * s for w, s in zip(kept, kept_surp)) / total_w
            # 语义饱和：对实际入缓冲的保留数据计重复衰减（原始惊讶值保留口径不变）。
            sat_factor = self._apply_semantic_saturation(kept_data, surprise)
            eid = self.buffer.add(kept_data, surprise, sat_factor=sat_factor)
            self._since_sleep += 1
            self.learn_total += 1  # G3：部署侧经验流入总量（守护模式的恢复信号）
            return eid

    def _dynamic_band(self, base_lo, base_hi):
        """金发姑娘带通**动态制定**（2026-10-17 用户定案）。

        基于近期惊讶值分布的变异系数（CV）动态放宽/收窄带通：
        - 数据分布越分散（CV 大）→ 带越宽（lo 更小、hi 更大），容纳正常波动；
        - 数据越集中（CV 小）→ 带越窄（筛选更精确）。
        结果被钳位在合理域：lo ∈ [0.01, 0.15]，hi ∈ [0.05, 0.60]。
        """
        hist = self._surprise_hist
        if len(hist) < 8:
            return base_lo, base_hi
        mean = sum(hist) / len(hist)
        if mean <= 1e-9:
            return base_lo, base_hi
        var = sum((x - mean) ** 2 for x in hist) / len(hist)
        cv = max(0.0, min(1.0, (var ** 0.5) / mean))
        lo = max(0.01, min(0.15, base_lo * (1.0 - cv * 0.5)))
        hi = max(0.05, min(0.60, base_hi * (1.0 + cv)))
        return lo, hi

    @staticmethod
    def _split_windows(data, chunk_size):
        """自适应切分字节流（2026-10-17 用户定案：**即使无法切块也不丢弃**）。

        - 空输入返回 []；
        - 整条长度 ≤ chunk_size → 作为一块返回（哪怕只有 1 字节也不丢弃）；
        - 否则按 chunk_size 无重叠切分，**尾部不足一块时也独立保留**（不再
          并入前一块，避免块超过 chunk_size；1 字节窗口由 learn() 兜底保留），
          保证没有任何数据被切分逻辑丢弃。
        """
        if not data:
            return []
        if len(data) <= chunk_size:
            return [data]
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # ---------- 睡眠触发（值自成：压力阈值 + 间隔反馈） ----------

    def update_hemi_pressure(self):
        """多因素睡眠压力 + 对方训练进度换算（下丘脑稳态调节映射）。
        权重/系数为【初值】，经 autotune 慢调（律定域内值自成）。
        输出压力钳位到律定域 [0, 50]（与 sleep_threshold 上限 THRESHOLD_HI 对齐），
        防止 _since_sleep 极大时压力爆炸导致 maybe_sleep 恒真（验收 #4 修复）。"""
        au = getattr(self, "autotune", None)
        w1 = au.get("pressure_w1", 0.5) if au else 0.5
        w2 = au.get("pressure_w2", 0.3) if au else 0.3
        w3 = au.get("pressure_w3", 0.2) if au else 0.2
        k = au.get("pressure_k", 0.3) if au else 0.3
        buf_p = self.buffer.pressure()
        debt = self.debt_ledger.debts
        trained = self.hemi_trained_steps
        total_t = max(1, trained[0] + trained[1])
        for i in range(2):
            opp_progress = trained[1 - i] / total_t
            raw = buf_p * w1 + debt[i] * w2 + float(self._since_sleep) * w3
            self.hemi_pressure[i] = raw * (1.0 - k * opp_progress)
            # 律定域钳位 [0, 50]：与 sleep_threshold 上限对齐，压力信号有界。
            self.hemi_pressure[i] = max(0.0, min(50.0, self.hemi_pressure[i]))

    def maybe_sleep(self, force=False, **kw):
        """单线程睡眠路径（M0 兼容）。M4 起派发到 life.run_cycle（唯一睡眠周期
        实现；sleep.py 冻结为历史件，M0 等价性由 tests 固定）。

        注意：与其他线程的 learn/serve 并发使用不安全（本方法不走 feed_lock
        快照流程）——双线程部署请走 feed.py 的 SleepTrainer。
        """
        p = self.buffer.pressure()
        self.update_hemi_pressure()
        overflow = self.buffer.bytes > self.buffer.cap * 0.9
        if not (force or p >= self.buffer.sleep_threshold
                or max(self.hemi_pressure) >= self.buffer.sleep_threshold
                or overflow):
            return None
        from .life import run_cycle
        report = run_cycle(self, **kw)
        if self._since_sleep < self.target_interval // 2:
            # 值自成：睡眠频率反馈（G9：乘法结果必须过 clamp_threshold 域钳位）
            self.buffer.sleep_threshold = clamp_threshold(self.buffer.sleep_threshold * 1.10)
        elif self._since_sleep > self.target_interval * 2:
            self.buffer.sleep_threshold = clamp_threshold(self.buffer.sleep_threshold * 0.90)
        self._since_sleep = 0
        return report

    # ---------- 持久化（全部走 weights_only 安全反序列化；G7 护航骨） ----------

    BACKUP_KEEP = 3  # 律定（G7）：滚动备份份数——断电损坏唯一存档时的兜底深度
    TRASH_WARN_COUNT = 100  # trash/ 归档上限（软阈值）：超出只告警不删除，留人工清理

    def _rotate_backup(self, path):
        """滚动备份（G7）：每次成功存档后把新档复制进同目录 archive/（带时间戳），
        超出 K 份把最旧移入 archive/trash/ 归档（人工可清理）。存档瞬间断电最多
        损失本次，K 份历史快照兜底——对策"唯一存档损坏全归零"。备份失败只报警，
        不拦存档主体。

        注意（2026-10-12 根治）：旧实现用 os.remove 批量删除最旧备份，在受沙箱
        安全钩子保护的环境（WorkBuddy）里会触发 [safe-delete] 批量删除确认并阻塞
        等待人工确认 → 无人值守训练线程死锁。现改为 shutil.move 移入 trash/，
        永不执行批量 os.remove；trash 累计过多只告警，由人工清理。"""
        d = os.path.dirname(os.path.abspath(path))
        archive = os.path.join(d, "archive")
        trash_dir = os.path.join(archive, "trash")
        try:
            os.makedirs(archive, exist_ok=True)
            base = os.path.basename(path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 微秒精度：连续存档不撞名
            dst = os.path.join(archive, f"{base}.{stamp}.bak")
            while os.path.exists(dst):  # 同微秒撞名（理论可能）：补随机尾巴
                dst = os.path.join(archive, f"{base}.{stamp}_{secrets.token_hex(2)}.bak")
            shutil.copyfile(path, dst)
            baks = sorted(glob.glob(os.path.join(archive, f"{base}.*.bak")))
            for old in baks[:-self.BACKUP_KEEP]:  # 文件名字典序=时间序
                try:  # 并发存档时另一线程可能已移动同一旧档，忽略
                    os.makedirs(trash_dir, exist_ok=True)
                    shutil.move(old, trash_dir)
                except FileNotFoundError:
                    pass
                except OSError as e:  # 移动失败只告警，不拦存档主体
                    print(f"[Dolphin] 旧备份归档失败（不拦存档）：{e!r}",
                          file=sys.stderr)
            # trash/ 软上限：只告警不删除——无人值守代码绝不批量 os.remove
            trash_count = len(glob.glob(os.path.join(trash_dir, f"{base}.*.bak")))
            if trash_count > self.TRASH_WARN_COUNT:
                print(f"[Dolphin] 备份 trash/ 已积累 {trash_count} 份（超过 "
                      f"{self.TRASH_WARN_COUNT}），请人工清理 {trash_dir}。",
                      file=sys.stderr)
        except OSError as e:
            print(f"[Dolphin] 滚动备份失败（不拦存档）：{e!r}", file=sys.stderr)

    def save(self, path):
        # 2026-10-04 修复：旧实现构造 payload 时不持 feed_lock，而主线程 learn()
        # 正并发 buffer.add —— 快照是撕裂的（eid/next_id 与 buffer.items 可能不对应，
        # 导致重启后恢复出不一致的缓冲）。现在把"读共享状态"的整段放在 feed_lock
        # 内（锁内只有名单级操作：快照构造毫秒级）；torch.save 落盘在锁外，
        # 不长时间持锁、不与训练线程的长计算互斥（与"锁只护共享状态交换"原则一致）。
        # 注：flush_cold 取 _cold_lock，锁序 feed_lock→_cold_lock 是既有单向顺序，无死锁。
        with self.feed_lock:
            try:
                self.memory.flush_cold()  # 冷层命中进度随档落盘
            except OSError:
                pass
            payload = {
                "version": 3,
                "cfg": asdict(self.cfg),
                "states": [h.model.state_dict() for h in self.h],
                "lrs": [h.opt.param_groups[0]["lr"] for h in self.h],
                "opt_states": [h.opt.state_dict() for h in self.h],  # 动量入档：--resume 非热重启
                # G7 身份入档：v3 起 (eid, data, surprise, reward) 四元组——eid 随档
                # 恢复，逐出+重启后 feedback 不会错挂（旧版 add 重建会平移 id）。
                # 2026-10-15 起增补 sat_factor 为第五元（语义饱和衰减因子随档透传）。
                "buffer": [(e.id, e.data, e.surprise, e.reward, getattr(e, "sat_factor", 1.0))
                           for e in self.buffer.items],
                # 经验身份计数器透传（experience.py 本轮禁改，包内管理层透传 _next_id；
                # M4 可将其正式化）。缺它则重启后新经验与幸存旧经验撞 id。
                "next_id": self.buffer._next_id,
                "awake_idx": self.awake_idx,
                "budget": self.budget,
                "threshold": self.buffer.sleep_threshold,
                "cycle": self.cycle,
                "since_sleep": self._since_sleep,
                "note_k": self.note_k,
                "dream_k": self.dream_k,
                "target_interval": self.target_interval,
                "promote_hits": self.memory.promote_hits,
                # M2 修复（2026-10-06）：kd_alpha/kd_T/sleep_steps 随档透传，
                # 与 autotune.tunables 一致（load 后属性与 tunables 同步恢复）。
                "kd_alpha": self.kd_alpha,
                "kd_T": self.kd_T,
                "sleep_steps": self.sleep_steps,
                # 三档睡眠（2026-10-11）：light/rem 独立自适应步数域随档透传。
                "sleep_steps_light": self.sleep_steps_light,
                "sleep_steps_rem": self.sleep_steps_rem,
                # S6 四档睡眠（2026-10-13）：nrem 步数域随档透传（与 light/rem 同律）。
                "sleep_steps_nrem": self.sleep_steps_nrem,
                # 判决带传导系数（2026-10-11）：随档透传，autotune 慢速自调。
                "k_margin": self.k_margin,
                # 训练债平衡（2026-10-11）：两半球累计训练步数随档透传。
                "hemi_trained_steps": list(self.hemi_trained_steps),
                # S3/S5（2026-10-13）：双半球睡眠压力 + 债务账本（胼胝体通道）随档透传。
                "hemi_pressure": list(self.hemi_pressure),
                "debt": self.debt_ledger.to_state(),
                "self_eval_log": list(self.self_eval_log),
                "anomaly_log": list(self.anomaly_log),
                # S5 蒸馏方向交替标志（2026-10-13）：feeding 生产路径连续周期
                # 互为师生；随档透传，旧档缺省 False（兼容）。
                "_teacher_flip": bool(self._teacher_flip),
                # G7：created 入档——记忆逐出优先级 (score, created) 跨重启不漂移
                "memory": [(e.text, e.score, e.hits, e.cycle, e.kind, e.created)
                           for e in self.memory.entries],
                "probe_sha256": self.probe_sha256(),  # G7：体检卷面指纹，load 时校验
                # G3 自续骨：喂食游标透传（语义与推进全权归 feed.py，本类只存取）
                "feed_cursor": self.feed_cursor,
                # M4（律 L11）：生命节律状态随档——历史最优 probe 锚（规格 C）、
                # 状态机、双半球形态（手术后模型可跨重启复原）、营养因子账本。
                # 战役进行中时 campaign 里含权重快照（体积翻倍仅限战役期）。
                "life": self.life_ctl.to_state(),
                # 值自成自动调参（2026-10-05）：律定量反馈控制器状态随档。
                "autotune": self.autotune.to_state(),
                # 杏仁核温度自动托管（2026-10-08）：温度/信号/调整周期随档，
                # 重启后恢复上次的自动托管状态。
                "amygdala": self.amygdala.to_state(),
                # 学习停滞检测原料（2026-10-12）：margin 滚动窗随档透传。
                "last_margins": list(self.last_margins),
                # 参数自适应控制律（2026-10-06）：三段式下跌状态机随档。
                "param_adaptive": self.param_adaptive.to_state(),
                # 金发姑娘价值门控（律 L4）：能力基线 + 门控观测计数随档透传。
                "ability": self.ability,
                "skipped_too_easy": self.skipped_too_easy,
                "suspended_too_hard": self.suspended_too_hard,
                # 片段级噪声屏蔽观测（2026-10-17）：累计被屏蔽噪声字节数随档透传。
                "noise_masked_bytes": self.noise_masked_bytes,
                # 语义饱和防护触发计数（2026-10-15）：观测用，随档透传。
                "saturation_events": self.saturation_events,
            }
        # G7 原子写：先写同目录临时文件再 os.replace——任何瞬间断电，目标文件
        # 要么是旧版要么是新版，绝不出现写了一半的存档。
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".saving"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)  # 同目录原子替换
        finally:
            if os.path.exists(tmp):  # torch.save 中途崩掉时清理残尸，旧档完好
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self._rotate_backup(path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        # G7 卷面完整性校验先行（门控语义完整性优先于可用性——设计决定）：
        # probe.txt 被替换/损坏时，体检判决的语义已静默改变，带着旧档的
        # 换班史继续跑等于用错试卷对答案，宁可拒绝加载。旧档（v2-）无此字段则跳过。
        want = ck.get("probe_sha256")
        if want is not None and want != self.probe_sha256():
            raise RuntimeError(
                f"体检探测集校验失败（G7）：存档 sha256={want}，"
                f"当前={self.probe_sha256()}（{self.probe_path}）。\n"
                f"probe.txt 已被替换或损坏——门控语义完整性优先于可用性，拒绝加载。")
        self.cfg = Config(**ck["cfg"])
        # M4：先恢复生命节律状态（形态记录是重建半球模型的前提——手术后模型
        # 结构与 base cfg 不同，须按形态重放移植体/平铺，再装入权重）。
        if ck.get("life"):
            self.life_ctl.from_state(ck["life"])
        if ck.get("autotune"):
            self.autotune.from_state(ck["autotune"])
            # 2026-10-13：把 autotune 管理的被调量同步回 Dolphin 属性，确保
            # 重启后 learn()/life.run_cycle 读到的是随档持久化的当前值，而非初值。
            # 旧档没有这些量时 autotune 保持初值，同步后仍与模块常量一致。
            self.goldilocks_lo = float(self.autotune.get("goldilocks_lo", self.goldilocks_lo))
            self.goldilocks_hi = float(self.autotune.get("goldilocks_hi", self.goldilocks_hi))
        if ck.get("amygdala"):  # 2026-10-08：杏仁核状态随档恢复（旧档无此字段则保持初值）
            self.amygdala.from_state(ck["amygdala"])
        if ck.get("param_adaptive"):
            self.param_adaptive.from_state(ck["param_adaptive"])
        from .surgery import apply_morphology
        lrs = ck.get("lrs", [h.opt.param_groups[0]["lr"] for h in self.h])
        for i, h in enumerate(self.h):
            morph = self.life_ctl.morphology.get(h.name) or {}
            k = int(morph.get("d_model_k", 1) or 1)
            # 旧 MLP（fc/proj）存档 → DualMLP（rule/body/out）权重迁移（2026-10-19）。
            sd = _migrate_mlp_state_dict(ck["states"][i])
            # shrink（born-again 学生体型）也走形态重建——否则学生脑 load 按基座
            # cfg 重建 → state_dict 形状不匹配崩溃。layers（深度生长史）同理。
            if (k > 1 or morph.get("mlp") or morph.get("attn_v")
                    or morph.get("shrink") or morph.get("layers")):
                m, _ = apply_morphology(self.cfg, morph, self.device)
                m.load_state_dict(sd)
            else:
                m = ByteTransformer(self.cfg).to(self.device)
                m.load_state_dict(sd)
            h.model = m
            h.opt = torch.optim.AdamW(m.parameters(), lr=lrs[i], weight_decay=0.01)
            if "opt_states" in ck:  # v2 存档：动量一并恢复，续喂不再是热重启
                # DualMLP 架构升级（2026-10-19）：旧 MLP 的 Adam 参数组
                # （fc/proj）与迁移后模型（rule/body/out，新增 body 通道）
                # 不匹配，动量无法跨架构迁移——检测到旧格式存档则重置
                # 优化器状态（权重已由 _migrate_mlp_state_dict 转换）。
                legacy_mlp = any(".mlp.fc." in k for k in ck["states"][i])
                if legacy_mlp:
                    print("[load] 旧版 MLP 存档：优化器动量不跨架构迁移"
                          "（新增 body 通道），Adam 状态重置。")
                else:
                    h.opt.load_state_dict(ck["opt_states"][i])
        self.awake_idx = ck["awake_idx"]
        self.budget = ck["budget"]
        self.buffer.sleep_threshold = ck["threshold"]
        self.cycle = ck["cycle"]
        if "since_sleep" in ck:
            self._since_sleep = ck["since_sleep"]
        if "note_k" in ck:
            self.note_k = ck["note_k"]
        if "dream_k" in ck:
            self.dream_k = ck["dream_k"]
        if "target_interval" in ck:
            self.target_interval = ck["target_interval"]
        if "promote_hits" in ck:
            self.memory.promote_hits = ck["promote_hits"]
        # M2 修复（2026-10-06）：恢复 kd_alpha/kd_T/sleep_steps 属性（缺省=初值，
        # 兼容旧档）。load 后这些属性与 autotune.tunables 一起恢复，训练路径
        # 值自成覆盖会读取它们。
        self.kd_alpha = float(ck.get("kd_alpha", self.kd_alpha))
        self.kd_T = float(ck.get("kd_T", self.kd_T))
        self.sleep_steps = int(ck.get("sleep_steps", self.sleep_steps))
        # 三档睡眠（2026-10-11）：light/rem 独立步数域，缺省=初值兼容旧档。
        self.sleep_steps_light = int(ck.get("sleep_steps_light", self.sleep_steps_light))
        self.sleep_steps_rem = int(ck.get("sleep_steps_rem", self.sleep_steps_rem))
        # S6 四档睡眠（2026-10-13）：nrem 步数域随档恢复，缺省=初值 80 兼容旧档。
        self.sleep_steps_nrem = int(ck.get("sleep_steps_nrem", 80))
        # 判决带传导系数（2026-10-11）：缺省=初值 2.0，兼容旧档。
        self.k_margin = float(ck.get("k_margin", self.k_margin))
        # 训练债平衡（2026-10-11）：两半球累计训练步数，缺省=[0,0] 兼容旧档。
        self.hemi_trained_steps = [int(x) for x in ck.get("hemi_trained_steps", [0, 0])]
        # S3/S5（2026-10-13）：双半球睡眠压力 + 债务账本随档恢复（旧档无此字段
        # → 保持初值/空账本，缺省兼容）。
        self.hemi_pressure = [float(x) for x in ck.get("hemi_pressure", [0.0, 0.0])]
        self.debt_ledger.from_state(ck.get("debt"))
        self.self_eval_log = list(ck.get("self_eval_log", []))
        # 用户负面反馈事件日志（2026-10-15）：旧档无此字段 → 空列表（兼容升级）。
        self.anomaly_log = list(ck.get("anomaly_log", []))
        # S5 蒸馏方向交替标志（2026-10-13）：随档恢复；旧档无此字段 → False。
        self._teacher_flip = bool(ck.get("_teacher_flip", False))
        # 学习停滞检测原料（2026-10-12）：旧档无此字段 → 空列表（兼容升级）。
        self.last_margins = [float(x) for x in ck.get("last_margins", [])]
        # 金发姑娘价值门控（律 L4）：能力基线 + 观测计数随档恢复。
        # 旧档无这些字段 → 用初值（ability=ln256、计数=0）兼容升级。
        self.ability = float(ck.get("ability", PRIOR_ABILITY))
        self.skipped_too_easy = int(ck.get("skipped_too_easy", 0))
        self.suspended_too_hard = int(ck.get("suspended_too_hard", 0))
        self.noise_masked_bytes = int(ck.get("noise_masked_bytes", 0))
        # 语义饱和防护触发计数（2026-10-15）：旧档无此字段 → 0（兼容升级）。
        self.saturation_events = int(ck.get("saturation_events", 0))
        if "feed_cursor" in ck:  # G3：喂食游标透传恢复（语义归 feed.py 解释）
            self.feed_cursor = ck["feed_cursor"]
        if "buffer" in ck:  # v2 起经验缓冲随档恢复，换班季中断零丢失
            self.buffer.items = []
            self.buffer.bytes = 0
            for tup in ck["buffer"]:
                if len(tup) == 5:  # v3.1（2026-10-15）：含语义饱和衰减因子
                    eid, data, sup, rw, satf = tup
                elif len(tup) == 4:  # v3：身份随档恢复（G7——eid 不平移，feedback 不错挂）
                    eid, data, sup, rw = tup
                    satf = 1.0
                else:              # v2 兼容：id 现场重排
                    data, sup, rw = tup
                    eid = self.buffer._next_id
                    self.buffer._next_id += 1
                    satf = 1.0
                e = Experience(eid, data, sup, sat_factor=float(satf))
                e.reward = float(rw)
                self.buffer.items.append(e)
                self.buffer.bytes += len(e.data)
            if "next_id" in ck:  # G7：身份计数器随档恢复，新经验不与幸存旧经验撞 id
                self.buffer._next_id = max(int(ck["next_id"]), self.buffer._next_id)
            elif self.buffer.items:  # 旧档兜底：至少保证不撞幸存条目
                self.buffer._next_id = max(e.id for e in self.buffer.items) + 1
        self.memory.entries = []
        for tup in ck["memory"]:
            if len(tup) == 6:  # v3：created 随档恢复（G7——逐出优先级跨重启不漂移）
                t, s, hits, c, k, created = tup
                en = MemoryEntry(t, s, c, k)
                en.created = float(created)
            else:              # v2 兼容
                t, s, hits, c, k = tup
                en = MemoryEntry(t, s, c, k)
            en.hits = int(hits)
            self.memory.entries.append(en)

    @classmethod
    def from_birth(cls, path, **kw):
        """用出生基座权重（birth.pt）初始化：两个半球从同一个基座复制。"""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = Config(**ck["cfg"])
        d = cls(cfg=cfg, **kw)
        for h in d.h:
            h.model.load_state_dict(ck["sd"])
        return d
