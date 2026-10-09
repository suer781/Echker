"""双半球管理与对外接口（律 L2 / L3）。

两个 hemisphere 初始化时权重同源；一次只有一个在训练（睡脑），
另一个权重冻结负责服务（醒脑）。交换只发生在体检通过后的换班瞬间。
"""
import glob
import hashlib
import os
import secrets
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime

import torch

from .adapt import Plasticity
from .amygdala import Amygdala
from .autotune import Autotune
from .experience import Experience, ExperienceBuffer
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
        # M4 生命节律（律 L11）：自生长/自凋零控制器 + 人工总开关安全阀。
        # life_enabled=False 时 life.run_cycle 的全部 M4 钩子失效（纯 M0 语义）；
        # 人工只保留这个总开关，禁止干预手术的时机/位置/幅度（律 L11）。
        self.life_ctl = LifeController()
        self.life_enabled = True

    def awake(self):
        return self.h[self.awake_idx]

    def sleeping(self):
        return self.h[1 - self.awake_idx]

    def swap(self):
        self.awake_idx = 1 - self.awake_idx

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
        return gate_decision(per_old, per_new)

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
            if temperature is None:
                temperature = self.amygdala.temperature  # 杏仁核自动托管
            if top_k is None:
                top_k = self.amygdala.top_k
            if top_p is None:
                top_p = self.amygdala.top_p
            ub = user_text.encode("utf-8")
            notes = self.memory.retrieve(ub, k=1)  # 律 L10 预支快通道：引用记忆，不碰权重
            prefix = ("【记忆】" + notes[0].text[-120:] + "\n") if notes else ""
            prompt = (prefix + user_text).encode("utf-8")[-self.cfg.block_size:]
            idx = torch.tensor(list(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
            out = self.awake().model.generate(
                idx, max_new, temperature, top_k=top_k, top_p=top_p)
            resp = bytes(out[0, idx.shape[1]:].tolist()).decode("utf-8", errors="replace")
            surprise = self.awake().model.mean_nll(ub, self.device)  # 对用户输入的困惑度
            with self.feed_lock:
                eid = self.buffer.add(ub, surprise)
                self._since_sleep += 1
                self.learn_total += 1  # G3：部署侧经验流入总量（守护模式的恢复信号）
            return resp, eid
        finally:
            self._busy_dec()

    def feedback(self, eid, reward):
        return self.buffer.feedback(eid, reward)

    def learn(self, text: str, source="dataset"):
        """部署期喂养：算惊讶度入缓冲，不生成回复（与 serve 相对）。

        双线程约束：mean_nll 是部署计算，锁外做（部署永不排队等锁）；
        只有缓冲交换持 feed_lock，与训练线程的快照/换班互斥。
        learn_busy 供训练线程在动睡脑前等待在途读取清零——
        刚换下的醒脑可能还有 learn 在读，梯度训练前必须等它读完。
        """
        data = text.encode("utf-8")
        self._busy_inc()
        try:
            surprise = self.awake().model.mean_nll(data, self.device)
        finally:
            self._busy_dec()
        with self.feed_lock:
            eid = self.buffer.add(data, surprise)
            self._since_sleep += 1
            self.learn_total += 1  # G3：部署侧经验流入总量（守护模式的恢复信号）
        return eid

    # ---------- 睡眠触发（值自成：压力阈值 + 间隔反馈） ----------

    def maybe_sleep(self, force=False, **kw):
        """单线程睡眠路径（M0 兼容）。M4 起派发到 life.run_cycle（唯一睡眠周期
        实现；sleep.py 冻结为历史件，M0 等价性由 tests 固定）。

        注意：与其他线程的 learn/serve 并发使用不安全（本方法不走 feed_lock
        快照流程）——双线程部署请走 feed.py 的 SleepTrainer。
        """
        p = self.buffer.pressure()
        overflow = self.buffer.bytes > self.buffer.cap * 0.9
        if not (force or p >= self.buffer.sleep_threshold or overflow):
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

    def _rotate_backup(self, path):
        """滚动备份（G7）：每次成功存档后把新档复制进同目录 archive/（带时间戳），
        超出 K 份删最旧。存档瞬间断电最多损失本次，K 份历史快照兜底——
        对策"唯一存档损坏全归零"。备份失败只报警，不拦存档主体。"""
        d = os.path.dirname(os.path.abspath(path))
        archive = os.path.join(d, "archive")
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
                try:  # 并发存档时另一线程可能已删除同一旧档，忽略
                    os.remove(old)
                except FileNotFoundError:
                    pass
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
                # 恢复，逐出+重启后 feedback 不会错挂（旧版 add 重建会平移 id）
                "buffer": [(e.id, e.data, e.surprise, e.reward) for e in self.buffer.items],
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
                # 参数自适应控制律（2026-10-06）：三段式下跌状态机随档。
                "param_adaptive": self.param_adaptive.to_state(),
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
        if ck.get("amygdala"):  # 2026-10-08：杏仁核状态随档恢复（旧档无此字段则保持初值）
            self.amygdala.from_state(ck["amygdala"])
        if ck.get("param_adaptive"):
            self.param_adaptive.from_state(ck["param_adaptive"])
        from .surgery import apply_morphology
        lrs = ck.get("lrs", [h.opt.param_groups[0]["lr"] for h in self.h])
        for i, h in enumerate(self.h):
            morph = self.life_ctl.morphology.get(h.name) or {}
            k = int(morph.get("d_model_k", 1) or 1)
            # shrink（born-again 学生体型）也走形态重建——否则学生脑 load 按基座
            # cfg 重建 → state_dict 形状不匹配崩溃。
            if k > 1 or morph.get("mlp") or morph.get("attn_v") or morph.get("shrink"):
                m, _ = apply_morphology(self.cfg, morph, self.device)
                m.load_state_dict(ck["states"][i])
            else:
                m = ByteTransformer(self.cfg).to(self.device)
                m.load_state_dict(ck["states"][i])
            h.model = m
            h.opt = torch.optim.AdamW(m.parameters(), lr=lrs[i], weight_decay=0.01)
            if "opt_states" in ck:  # v2 存档：动量一并恢复，续喂不再是热重启
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
        if "feed_cursor" in ck:  # G3：喂食游标透传恢复（语义归 feed.py 解释）
            self.feed_cursor = ck["feed_cursor"]
        if "buffer" in ck:  # v2 起经验缓冲随档恢复，换班季中断零丢失
            self.buffer.items = []
            self.buffer.bytes = 0
            for tup in ck["buffer"]:
                if len(tup) == 4:  # v3：身份随档恢复（G7——eid 不平移，feedback 不错挂）
                    eid, data, sup, rw = tup
                else:              # v2 兼容：id 现场重排
                    data, sup, rw = tup
                    eid = self.buffer._next_id
                    self.buffer._next_id += 1
                e = Experience(eid, data, sup)
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
