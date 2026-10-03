"""B 路线部署期喂食驱动器——部署-训练双线程，永不停机（G3 自续骨）。

主线程 = 部署循环：持续 learn() 喂数据源记录，进度打印不断流，从不等待训练
（唯一的例外是最终退出时 join 收尾）。
工作线程 = 睡眠训练器：缓冲压力够（或主线程通知）就在睡脑上训练，
体检通过热切换（翻 awake_idx）；换班后下一轮自动拾取新睡脑（现有机制）。

守护模式（G3，默认开启，--no-guard 关闭）：数据源迭代耗尽后不退场，转入
"反刍"——周期照常触发，选拔来源变为记忆库做梦采样 + 复活条目 + 缓冲残留
（反刍=消化已吃的，L7 明文允许跨周期重复）；部署侧有任何新经验流入
（learn/serve，learn_total 口径）则自动恢复正式喂食。退出条件只剩
Ctrl+C 与 --max-idle（连续 N 个空选拔周期后允许收工，防真死转——N 值自成）。

喂食游标（G3）：主线程维护 {"sources", "source_idx", "record_idx"}，随档
透传存取（dolphin.save/load 只存取不解释语义，职责分离）；--resume 从游标
续喂，不再从第 0 条重放。

奖励接线（G3 最小闭环）：喂食时给每条经验一个自动第二信号——同文重复出现
次数的负值（结构信号，可复核；非内容评价）。如实标注：这是接线占位，真正
的内容奖励（用户点赞/求解器验证）留给部署时代（M4 归属贡献过于复杂）。

线程安全原则：锁只护共享状态交换，不护计算。
- 共享状态：experience buffer（主线程 add / 训练线程 select+clear）、awake_idx。
- dolphin.feed_lock：主线程的 buffer.add 与训练线程的"选拔+摘除"快照互斥；
  锁内只有名单级操作（毫秒级），变异重放、梯度训练、体检全部在锁外。
- dolphin.learn_busy：在途 learn 计数。训练线程动手训练睡脑前等它清零——
  刚换班退休的醒脑可能还有 learn 在读，读静态权重是部署侧的底线（律 L3）。
- sleep.py 禁改，故 run_cycle 的线程安全版在 feed.py 内复刻（trainer_cycle）。
  与原版的已声明差异：a) 缓冲交接为摘除式快照（主线程喂入不丢）；b) 带通在
  快照时取定（原版在周期末重取）；c) 早退/异常路径同样执行阈值反馈与睡脑回滚。

优雅退出：Ctrl+C（或喂到 --limit / --max-idle 收工）→ 通知训练线程 → 当前
睡眠周期完整收尾（不腰斩梯度更新）→ join → d.save() → 查房总结。

用法：
  python feed.py --sources logiqa,cmath,distil --limit 5000 \
      --sleep-every 100 --sleep-steps 120 --model 57
  python feed.py --resume   # 从 dolphin/fed_state.pt 按喂食游标续喂
"""
import argparse
import hashlib
import itertools
import os
import signal
import sys
import threading
import time

import torch
import torch.nn.functional as F

from dolphin.datasets import DEFAULT_ORDER, SOURCES, iter_source, serialize
from dolphin.dolphin import Dolphin, GateDisabled, clamp_threshold
from dolphin.model import Config
from dolphin.sleep import varied_replay  # 纯函数，线程安全，复用不复制

FED_STATE = os.path.join("dolphin", "fed_state.pt")
PROGRESS_EVERY = 20  # 部署侧进度打印节奏（也用于产出不停机的时间戳证据）
MAX_IDLE_DEFAULT = 8  # 值：自成——守护模式连续 8 个空选拔周期视为消化完毕，允许收工

_PRINT_LOCK = threading.Lock()
_T0 = time.monotonic()


def log(tag, msg):
    """统一带相对时间戳的行打印（单次持打印锁，防两线程行内交错）。"""
    with _PRINT_LOCK:
        print(f"[+{time.monotonic() - _T0:8.3f}s][{tag}] {msg}", flush=True)


# ---------------- 睡眠训练（run_cycle 的线程安全复刻，sleep.py 禁改） ----------------

def _threshold_feedback(d):
    """值自成：睡眠频率反馈（与 maybe_sleep 语义一致；全部早退/异常路径同样计入）。

    模块级唯一实现：trainer_cycle 内部与 SleepTrainer.run 的异常兜底共用——
    后者原先调用的是 trainer_cycle 作用域里的闭包，异常路径真到时会 NameError。
    G9：乘法结果必须过 clamp_threshold 域钳位（与 maybe_sleep 同一公共函数）。
    """
    with d.feed_lock:
        if d._since_sleep < d.target_interval // 2:
            d.buffer.sleep_threshold = clamp_threshold(d.buffer.sleep_threshold * 1.10)
        elif d._since_sleep > d.target_interval * 2:
            d.buffer.sleep_threshold = clamp_threshold(d.buffer.sleep_threshold * 0.90)
        d._since_sleep = 0


def trainer_cycle(d, steps, kd_alpha=0.5, kd_T=2.0):
    """在睡脑上跑一轮完整睡眠周期。锁内只做快照与换班，计算全在锁外。"""
    # G8 fail-fast：探测集缺失 → GateDisabled 拒绝开睡（与 sleep.run_cycle 共用
    # Dolphin.ensure_gate_ready 同一入口，无从漂移）。probe 恢复后自动解禁。
    d.ensure_gate_ready()
    lock = d.feed_lock
    report = {"cycle": d.cycle}

    # ① 锁内快照（毫秒级）：复活 + 做梦 + 选拔 + 摘除。主线程 add 最多排队这一下。
    with lock:
        report["candidates"] = len(d.buffer.items)
        # 带通在摘除前取定（对完整缓冲计算，与 run_cycle 语义一致），复活注入也用它
        center, width = d.buffer.band()
        for rb in d.memory.resurrect():
            d.buffer.add(rb, surprise=center)
        # 做梦（G1）：按检索分数从记忆库抽样旧事注入缓冲——喂食模式的复活通道
        # 不再依赖人工对话提供 hits。注入量 ≤ dream_k，防旧知识挤占新经验选拔
        # 预算（审计 G11）。休眠路径（run_cycle）不做梦：交互模式由 serve 的
        # 检索天然供血，此不对称是设计决定，非实现漂移。
        queries = [e.data for e in d.buffer.items[-8:]]
        dreamed = d.memory.dream(queries, k=d.dream_k)
        for en in dreamed:
            d.buffer.add(en.text.encode("utf-8", errors="replace"), surprise=center)
        report["dreamed"] = len(dreamed)
        sel, rest = d.buffer.select(d.budget)
        report["selected"], report["residue"] = len(sel), len(rest)
        d.buffer.clear()  # 摘除：名单已在本轮处置，缓冲即刻腾空，主线程继续喂数不丢

    # ② L9 周期层：滞留一律降级进记忆库——包括空选拔周期（审计③-1 反例路径）。
    # serve 经 learn_busy 屏障与训练线程互斥，memory 此处无并发访问。
    for s, e in rest:
        d.memory.add(e.data, s, d.cycle, "residue")

    if not sel:
        d.cycle += 1
        _threshold_feedback(d)  # 早退路径同样计入睡眠频率统计（审计②-4）
        report["note"] = "无可训经验（滞留已全部入记忆库）"
        return report

    # ③ 锁外（计算）：变异重放（L6，纯函数）拼成一条字节流
    donors = [e.data for _, e in sel]
    parts = []
    for i, (_, e) in enumerate(sel):
        donor = donors[(i + 1) % len(donors)] if len(donors) > 1 else None
        parts.append(varied_replay(e.data, d.rng, donor))
    stream = b"\n".join(parts)
    blk = d.cfg.block_size
    bt = torch.tensor(list(stream), dtype=torch.long, device=d.device)
    if bt.numel() < blk + 2:
        for s, e in sel:  # L9：训不了的选拔经验同样降级记忆库（审计③-1 同族路径）
            d.memory.add(e.data, s, d.cycle, "residue")
        d.cycle += 1
        _threshold_feedback(d)
        report["note"] = "样本过短"
        return report

    # ④ 动睡脑前等在途 learn 清零：刚退休的醒脑此刻变睡脑，不能边训边被读（L3）
    sleeping, awake = d.sleeping(), d.awake()
    while d.learn_busy > 0:
        time.sleep(0.001)

    # ⑤ 锁外（计算）：睡脑训练（L5：硬标签 + 蒸馏软标签锚定同一份真实数据）
    sleeping.model.train()
    awake.model.eval()
    opt = sleeping.opt
    ce_hist, kd_hist = [], []
    N = bt.numel()
    for _ in range(steps):
        s0 = d.rng.randrange(0, N - blk - 1)
        x = bt[s0:s0 + blk].unsqueeze(0)
        y = bt[s0 + 1:s0 + blk + 1].unsqueeze(0)
        with torch.no_grad():
            t_logits, _ = awake.model(x)
        s_logits, _ = sleeping.model(x)
        ce = F.cross_entropy(s_logits.reshape(-1, d.cfg.vocab), y.reshape(-1))
        p_t = F.softmax(t_logits / kd_T, dim=-1)
        kd = (kd_T ** 2) * F.kl_div(
            F.log_softmax(s_logits / kd_T, dim=-1), p_t, reduction="batchmean")
        loss = (1 - kd_alpha) * ce + kd_alpha * kd
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sleeping.model.parameters(), 1.0)
        opt.step()
        ce_hist.append(ce.item())
        kd_hist.append(kd.item())

    # ⑥ 体检（L8）：全卷逐块评分 + margin 判决带。判决唯一实现在 Dolphin.gate
    # （→ probe.gate_decision），sleep.run_cycle 调用同一方法——零复制，无从漂移
    passed, gate = d.gate(awake, sleeping)
    report.update(gate)
    report["passed"] = passed
    report["ce_last"] = round(ce_hist[-1], 4)
    report["kd_last"] = round(kd_hist[-1], 4)

    # ⑦ 换班/回滚：awake_idx 只在锁内翻（一次赋值），锁绝不覆盖训练计算
    if passed:
        with lock:
            d.swap()
        report["swapped"] = True
        # L10 快通道：新醒脑最自信的片段入记忆库（预支）。memory 训练线程独占
        # 必须显式给 key：并列 NLL 会让元组比较去比 Experience，而它没有 __lt__（律：不可比语义）
        # 次级键用稳定的 Experience.id，保证并列时顺序确定、可复现（禁止随机）
        ranked = sorted(
            ((d.awake().model.mean_nll(e.data, d.device), e) for _, e in sel),
            key=lambda t: (t[0], t[1].id),
        )
        for nll, e in ranked[: d.note_k]:
            d.memory.add(e.data, nll, d.cycle, "note")
        d.budget = min(0.60, d.budget * 1.05)  # 值自成：体检连续通过 → 预算放宽
    else:
        report["swapped"] = False
        sleeping.model.load_state_dict(awake.model.state_dict())  # 作废回滚
        d.budget = max(0.25, d.budget * 0.90)

    # ⑧ 值自成：学习率反馈（Plasticity，带通已在快照时取定）+ 睡眠阈值反馈
    _threshold_feedback(d)
    mean_surp = sum(e.surprise for _, e in sel) / len(sel)
    cur_lr = sleeping.opt.param_groups[0]["lr"]
    new_lr = d.plasticity.next_lr(cur_lr, mean_surp, center, width, passed)
    for h in d.h:
        h.opt.param_groups[0]["lr"] = new_lr
    report["lr"] = new_lr

    d.cycle += 1
    return report


class SleepTrainer(threading.Thread):
    """睡眠训练器：后台循环，触发即训，体检通过热切换。

    主线程对它只有两个非阻塞动作：request_sleep()（通知）和退出时 join()。
    """

    def __init__(self, d, steps, save_every=5):
        super().__init__(name="sleep-trainer", daemon=False)
        self.d = d
        self.steps = steps
        self.save_every = save_every
        self.wake = threading.Event()       # 主线程通知：该睡了
        self.stop_flag = threading.Event()  # 退出通知：当前周期收尾后就下线
        self.cycles = 0
        self.swaps = 0
        self.rollbacks = 0
        self.busy = 0.0                     # 睡眠周期内墙钟（算占墙钟比）
        self.idle_cycles = 0                # 连续空选拔周期数（守护模式 --max-idle 的计数依据，G3）

    def request_sleep(self):
        self.wake.set()  # 非阻塞：主线程从不等待训练

    def shutdown(self):
        self.stop_flag.set()
        self.wake.set()  # 把它从 wait 里叫醒去看停止旗

    def run(self):
        log("训练线程", f"上线待命（steps={self.steps}）")
        while not self.stop_flag.is_set():
            # 触发三选一：主线程通知 / 压力达阈值 / 缓冲将满（自主部署时不通知也会睡）
            triggered = self.wake.wait(timeout=0.2)
            self.wake.clear()
            if self.stop_flag.is_set():
                break
            pressure = self.d.buffer.pressure()
            overflow = self.d.buffer.bytes > self.d.buffer.cap * 0.9
            if not (triggered or pressure >= self.d.buffer.sleep_threshold or overflow):
                continue
            log("训练线程", f"开始睡眠 cycle={self.d.cycle} "
                f"压力 {pressure:.2f}/{self.d.buffer.sleep_threshold:.2f} "
                f"缓冲 {self.d.buffer.bytes}B  触发={'通知' if triggered else '压力'}")
            t0 = time.monotonic()
            try:
                report = trainer_cycle(self.d, self.steps)
            except GateDisabled as e:  # G8：探测集缺失，拒绝开睡（fail-fast，非静默回滚）
                self.busy += time.monotonic() - t0
                log("训练线程", f"门控不可用，拒绝开睡（部署喂入继续，等 probe 恢复自动解禁）：{e}")
                continue
            except Exception as e:  # 单轮失败不带塌部署：记录后继续喂数与训练
                self.busy += time.monotonic() - t0
                try:  # 异常路径同样回滚：半训练权重不得成为下一轮基座（审计②-4）
                    self.d.sleeping().model.load_state_dict(self.d.awake().model.state_dict())
                except Exception as e2:
                    log("训练线程", f"回滚也失败（保留现场排查）：{e2!r}")
                _threshold_feedback(self.d)  # 第三条早退路径同样计入（审计③-6；原闭包调用在 run 作用域会 NameError）
                self.d.cycle += 1
                log("训练线程", f"周期异常终止（部署不停机）：{e!r}")
                continue
            self.busy += time.monotonic() - t0
            self.cycles += 1
            if report.get("selected"):
                self.idle_cycles = 0
            else:
                self.idle_cycles += 1  # 空选拔周期：反刍"消化完毕"的计数证据（G3）
            self.swaps += 1 if report.get("swapped") else 0
            self.rollbacks += 0 if report.get("swapped") else 1
            log("训练线程", f"睡眠报告 {report}")
            if self.cycles % self.save_every == 0:
                self.d.save(FED_STATE)  # 周期边界存档：两半球状态一致
                log("训练线程", f"每 {self.save_every} 睡一存 → {FED_STATE}")
        log("训练线程", "下线")


# ---------------- 喂食管线与守护模式（G3 自续骨） ----------------

def structural_reward(d, eid, text, seen):
    """自动第二信号（G3 奖励接线，最小闭环）：记录新鲜度衰减。

    r = 同文重复出现次数的负值——数据集内重复条目自动获得负奖励，在窗口去重
    （L7，只杀时间窗）之外给选拔层第二证据：跨窗口的重复同样压分。
    如实标注：这是结构信号（可复核：同文 sha256 计数），不是内容评价——
    真正的内容奖励（用户点赞/求解器验证）留给部署时代（M4：归属贡献过于复杂）。
    负奖励的压分边界 [-0.9, 9.0] 由 experience.select 唯一持有（分数恒正）。
    """
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    n = seen.get(h, 0)
    seen[h] = n + 1
    if n == 0:
        return 0.0  # 首现零信号，不画蛇添足
    with d.feed_lock:  # feedback 与训练线程的选拔/摘除互斥（锁内只有名单级操作）
        d.feedback(eid, float(-n))
    return float(-n)


def ruminate(d, k=None):
    """反刍一步（G3 守护模式）：从记忆库做梦采样旧事，经正常部署管线注入缓冲。

    反刍=消化已吃的：旧事全部是真实经验数据（L5 安全——记忆库只收真实经验的
    滞留与笔记，绝无模型自生成内容），L7 明文允许跨周期重复（间隔重复=巩固燃料）。
    注入经 learn 算惊讶度入缓冲 → 压力积累 → 周期照常触发 → 做梦+复活+选拔照旧，
    选拔来源由此成为"记忆库做梦采样 + 复活条目 + 缓冲残留"。
    返回注入条数（0=记忆库暂时无可反刍，如全员 hits 达标——那是 resurrect 的业务）。
    """
    ents = list(d.memory.entries) + list(d.memory.cold_entries)
    if not ents:
        return 0
    queries = [en.text.encode("utf-8", errors="replace")
               for en in d.rng.sample(ents, min(8, len(ents)))]  # 联想线索从记忆库自取
    batch = d.memory.dream(queries, k=k or d.dream_k)
    for en in batch:
        d.learn(en.text, source="ruminate")  # 走与正式喂食同一条部署管线
    return len(batch)


def feed_from(d, names, cursor, seen, trainer=None, sleep_every=500, quota=None,
              stats=None, fed_base=0, progress_every=PROGRESS_EVERY):
    """正式喂食：从喂食游标（G3）起迭代数据源，喂入 + 自动第二信号接线。

    游标语义：record_idx 计数据源内已消费的记录数（含碎片跳过）。cursor 与
    d.feed_cursor 是同一引用：喂食推进即"入档就绪"（周期边界/收工存档自动携带）。
    quota=本次最多喂入条数（None=不限，喂到数据尽）。返回 (fed, limit_hit, skipped)。
    """
    fed = skipped = 0
    limit_hit = False
    si = cursor["source_idx"]
    while si < len(names) and not limit_hit:
        name = names[si]
        start = cursor["record_idx"] if si == cursor["source_idx"] else 0
        for ri, rec in enumerate(itertools.islice(iter_source(name), start, None), start):
            if quota is not None and fed >= quota:
                limit_hit = True
                break
            text = serialize(rec)
            cursor["source_idx"], cursor["record_idx"] = si, ri + 1
            if not text:
                skipped += 1  # 碎片过滤：序列化后短于 30 字符
                continue
            eid = d.learn(text, source=name)  # 部署主线程：锁外算惊讶度，从不等待训练
            structural_reward(d, eid, text, seen)  # G3 奖励接线：自动第二信号
            fed += 1
            if stats is not None:
                stats[name] = stats.get(name, 0) + 1
            fed_total = fed_base + fed
            if trainer is not None and sleep_every and fed_total % sleep_every == 0:
                trainer.request_sleep()  # 非阻塞通知，喂入继续
                log("部署", f"已喂 {fed_total} 条 → 通知训练线程可以睡了")
            elif fed_total % progress_every == 0:
                log("部署", f"已喂 {fed_total} 条（当前源 {name}）"
                    f"  缓冲 {d.buffer.bytes}B  压力 {d.buffer.pressure():.2f}")
        if not limit_hit:  # 本源耗尽：游标推进到下一源
            si += 1
            cursor["source_idx"], cursor["record_idx"] = si, 0
    return fed, limit_hit, skipped


def guard_loop(d, trainer, max_idle, log_every=50):
    """守护模式主循环（G3 自续骨）：数据尽不退场。

    状态机（文字版）：
      反刍：ruminate() 从记忆库做梦采样旧事经 learn 注入缓冲 → request_sleep()
      让周期照常触发（触发三通道：通知/压力/溢出，均不变）→ 周期内的选拔来源
      =记忆库做梦采样 + 复活条目 + 缓冲残留；带通选出空集 → 继续反刍。
      转移：部署侧新经验流入（learn/serve，learn_total 口径）→ 恢复正式喂食；
      连续 max_idle 个空选拔周期（消化完毕）→ 收工；Ctrl+C → 收工（调用方捕获）。
    进入时 idle 计数清零：喂食末尾的空周期不算反刍的无所事事。
    返回 (resume, ruminate_fed)。
    """
    log("守护", f"数据源已尽 → 进入反刍（周期照常触发，选拔来源=记忆库做梦+复活+缓冲残留，"
        f"L7 跨周期重复）。退出：Ctrl+C 或连续 {max_idle} 个空选拔周期（--max-idle）；"
        f"部署侧有新经验流入（learn/serve）将自动恢复正式喂食")
    trainer.idle_cycles = 0
    baseline = d.learn_total  # 流入基线：反刍自己的 learn 同步计入 ruminate_fed，不入 inflow
    ruminate_fed = 0
    while True:
        inflow = d.learn_total - baseline - ruminate_fed
        if inflow > 0:
            log("守护", f"检测到部署侧 {inflow} 条新经验流入 → 恢复正式喂食")
            trainer.request_sleep()  # 让周期优先消化缓冲里的外部新经验
            return True, ruminate_fed
        if trainer.idle_cycles >= max_idle:
            log("守护", f"连续 {max_idle} 个空选拔周期，消化完毕 → 收工（防真死转）")
            return False, ruminate_fed
        n = ruminate(d)
        ruminate_fed += n
        trainer.request_sleep()  # 周期照常触发：有料消化料；无料空选拔也计入 idle（真死转防护）
        if n:
            if ruminate_fed % log_every == 0:
                log("反刍", f"已反刍 {ruminate_fed} 条  缓冲 {d.buffer.bytes}B  "
                    f"记忆库 {len(d.memory.entries)} 条")
        else:
            time.sleep(0.05)  # 无料小睡节流（request_sleep 非阻塞，空选拔周期照常跑）


# ---------------- CLI 与主流程 ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="B 路线部署期喂食（部署-训练双线程，守护模式默认开启）")
    ap.add_argument("--sources", default=",".join(DEFAULT_ORDER),
                    help="逗号分隔的数据源（默认按质量排序，logiconbench 需显式指定）")
    ap.add_argument("--limit", type=int, default=None, help="本次最多喂入条数（默认喂到数据尽）")
    ap.add_argument("--sleep-every", type=int, default=500, help="每喂多少条通知训练线程睡眠")
    ap.add_argument("--sleep-steps", type=int, default=120, help="每次睡眠的训练步数")
    ap.add_argument("--model", default="57", choices=["57", "small"],
                    help="57=默认 57M 身体；small=冒烟用小身体")
    ap.add_argument("--resume", action="store_true", help="从 dolphin/fed_state.pt 按喂食游标续喂")
    ap.add_argument("--no-guard", action="store_true",
                    help="关闭守护模式（数据尽即收工；默认开启：数据尽后转入反刍等待新经验）")
    ap.add_argument("--max-idle", type=int, default=MAX_IDLE_DEFAULT,
                    help=f"守护模式：连续 N 个空选拔周期后允许收工（值：自成，默认 {MAX_IDLE_DEFAULT}）")
    return ap.parse_args()


def make_cfg(which):
    if which == "small":
        return Config(d_model=128, n_layers=2, n_heads=2, block_size=256)
    return Config(d_model=768, n_layers=8, n_heads=8, block_size=256)  # ≈57M 参数


def _sigbreak_to_interrupt(sig, frame):
    """Windows 自动化验证用：CTRL_BREAK 也走优雅退出（与 Ctrl+C 同路径）。"""
    raise KeyboardInterrupt


def main():
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigbreak_to_interrupt)

    args = parse_args()
    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    for n in names:
        if n not in SOURCES:
            sys.exit(f"未知数据源 {n}，可选：{SOURCES}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Dolphin(cfg=make_cfg(args.model), device=dev)
    resumed = False
    if args.resume and os.path.exists(FED_STATE):
        d.load(FED_STATE)  # load 会用存档里的 cfg 覆盖 --model 选项
        log("续喂", f"从 {FED_STATE} 读档：cycle={d.cycle} 醒脑={d.awake().name}")
        resumed = True
    elif args.resume:
        log("续喂", f"{FED_STATE} 不存在，按随机初始化新出生")

    # —— 喂食游标（G3）：存档带着上次喂食位置，--resume 从游标续喂，不再第 0 条重放 ——
    cursor = {"sources": list(names), "source_idx": 0, "record_idx": 0}
    if resumed and isinstance(d.feed_cursor, dict) and d.feed_cursor.get("sources") == list(names):
        c = d.feed_cursor
        si, ri = int(c.get("source_idx", 0)), int(c.get("record_idx", 0))
        if 0 <= si <= len(names):
            cursor["source_idx"], cursor["record_idx"] = si, ri
            if si >= len(names):  # 游标已越过全部数据源：无余粮，直接进守护判断
                log("续喂", "喂食游标显示数据源已吃尽 → 不再从第 0 条重放，直接进入守护判断")
            else:
                log("续喂", f"喂食游标生效：从源 {si + 1}/{len(names)} 第 {ri} 条续喂"
                    f"（不再从第 0 条重放）")
    elif resumed and d.feed_cursor:
        log("续喂", "存档游标与本次 --sources 不一致 → 从第 0 条重喂")
    d.feed_cursor = cursor  # 游标与存档直连：周期边界/收工存档自动携带最新位置

    if resumed:
        log("部署", "海豚已带着记忆上岗，继续喂食")
    else:
        log("出生", f"随机初始化（B 路线不用胎教基座）  身体 {args.model}  设备 {dev}"
            f"  探测集 {len(d.probe_chunks)} 块")
    if d.gate_disabled:
        log("部署", "警告：体检探测集缺失，训练线程将拒绝开睡（G8 fail-fast），喂入照常")

    trainer = SleepTrainer(d, steps=args.sleep_steps)
    trainer.start()
    t_train_span0 = time.monotonic()

    stats = {n: 0 for n in names}
    skipped = 0
    total = 0
    ruminate_total = 0
    seen = {}  # 结构信号的重复计数表（G3 奖励接线）
    guard = not args.no_guard
    stop_reason = "数据尽/limit"
    try:
        while True:  # feed ↔ ruminate 状态机（G3 自续骨：触发、执行、验收全内部完成）
            quota = None if args.limit is None else max(0, args.limit - total)
            fed, limit_hit, skip = feed_from(
                d, names, cursor, seen, trainer=trainer, sleep_every=args.sleep_every,
                quota=quota, stats=stats, fed_base=total)
            total += fed
            skipped += skip
            if limit_hit:
                stop_reason = "limit"
                break
            if not guard:
                stop_reason = "数据尽"
                break
            resume, r = guard_loop(d, trainer, args.max_idle)
            ruminate_total += r
            if not resume:
                stop_reason = f"反刍收工（max-idle={args.max_idle}）"
                break
            # resume=True：部署侧有新经验流入 → 恢复正式喂食（数据源已尽则空转一圈
            # 回反刍，周期已收到通知优先消化缓冲里的外部经验）
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
        log("部署", "收到中断信号，优雅退出：不再喂入，等训练线程收尾")
    finally:
        trainer.shutdown()
        trainer.join()  # 唯一的等待：当前睡眠周期完整收尾，不腰斩梯度更新
        train_span = time.monotonic() - t_train_span0
        d.save(FED_STATE)
        log("存档", f"收工存档 → {FED_STATE}")

        # —— 查房总结 ——
        awake = d.awake()
        print("\n== 查房总结 ==", flush=True)
        print(f"退出原因：{stop_reason}")
        print(f"总喂入 {total} 条（碎片跳过 {skipped} 条）")
        for n in names:
            print(f"  {n:<13} {stats[n]} 条")
        print(f"反刍 {ruminate_total} 条  连续空选拔周期 {trainer.idle_cycles} 个")
        print(f"睡眠 {trainer.cycles} 次：换班 {trainer.swaps} 次 / 作废回滚 {trainer.rollbacks} 次"
              f"  周期 {d.cycle}")
        print(f"训练线程占墙钟比：{trainer.busy:.3f}s / {train_span:.3f}s"
              f" = {trainer.busy / train_span * 100:.1f}%（部署线程其余时间全程在线）")
        print(f"最终探测集损失（醒脑 {awake.name}）：{d.probe_loss(awake):.4f}")
        print(f"记忆库 {len(d.memory.entries)} 条  缓冲滞留 {len(d.buffer.items)} 条"
              f"  传输预算 {d.budget:.3f}")
        print(f"喂食游标：source_idx={cursor['source_idx']}/{len(names)}"
              f" record_idx={cursor['record_idx']}（随档续喂）")
        print(f"当前醒脑：半球 {awake.name}")


if __name__ == "__main__":
    main()
