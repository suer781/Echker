"""B 路线部署期喂食驱动器——部署-训练双线程，持续运行（G3 自续运行）。

主线程 = 部署循环：持续 learn() 喂数据源记录，进度持续打印，从不等待训练
（唯一的例外是最终退出时 join 收尾）。
工作线程 = 睡眠训练器：缓冲压力够（或主线程通知）就在睡脑上训练，
体检通过热切换（切换 awake_idx）；换班后下一轮自动使用新睡脑（现有机制）。

守护模式（G3，默认开启，--no-guard 关闭）：数据源迭代耗尽后不退场，转入
"反刍"——周期照常触发，选拔来源变为记忆库做梦采样 + 复活条目 + 缓冲残留
（反刍=消化已摄入数据，L7 明文允许跨周期重复）；部署侧有任何新经验流入
（learn/serve，learn_total 口径）则自动恢复正式喂食。退出条件只剩
Ctrl+C 与 --max-idle（连续 N 个空选拔周期后允许收工，防真死转——N 值自成）。

喂食游标（G3）：主线程维护 {"sources", "source_idx", "record_idx", "round"}，随档
透传存取（dolphin.save/load 只存取不解释语义，职责分离）；--resume 从游标
续喂，不再从第 0 条重放。round=轮转轮次（无限循环喂食的轮转标记，2026-10-07；
旧存档可能没有该字段，读取用 get("round", 0) 兼容；非 loop 模式不写）。

奖励接线（G3 最小闭环）：喂食时给每条经验一个自动第二信号——同文重复出现
次数的负值（结构信号，可复核；非内容评价）。说明：这是接线占位，真正
的内容奖励（用户反馈/求解器验证）留给部署时代（M4 归属贡献过于复杂）。

线程安全原则：锁只护共享状态交换，不护计算。
- 共享状态：experience buffer（主线程 add / 训练线程 select+clear）、awake_idx。
- dolphin.feed_lock：主线程的 buffer.add 与训练线程的"选拔+摘除"快照互斥；
  锁内只有名单级操作（毫秒级），变异重放、梯度训练、体检全部在锁外。
- dolphin.learn_busy：在途 learn 计数。训练线程开始训练睡脑前等待其清零——
  刚换班下线的醒脑可能还有 learn 在读，读静态权重是部署侧的底线（律 L3）。
- M4 取代路线：sleep.py 冻结为历史件，trainer_cycle 已删除——线程安全的睡眠
  周期由 dolphin.life.run_cycle(feeding=True) 唯一实现（做梦注入、摘除式快照、
  learn_busy 等待、锁内换班、内部阈值反馈全部在 life.py 内），本文件只剩
  SleepTrainer 触发循环。与 M0 的已声明差异：a) 缓冲交接为摘除式快照（主线程
  喂入不丢）；b) 带通在快照时取定；c) 早退/异常路径同样执行阈值反馈与睡脑回滚。

优雅退出：Ctrl+C（或喂到 --limit / --max-idle 收工）→ 通知训练线程 → 当前
睡眠周期完整收尾（不腰斩梯度更新）→ join → d.save() → 运行总结。

用法：
  python feed.py --sources logiqa,cmath,distil --limit 5000 \
      --sleep-every 100 --sleep-steps 120 --model 57
  python feed.py --resume   # 从 dolphin/fed_state.pt 按喂食游标续喂

存档与恢复：启动时若存在 dolphin/fed_state.pt（--resume 或 --model auto）则读档
恢复全部状态，否则随机初始化（B 路线不用预训练基座）；运行中每 5 个睡眠周期自动
存档（feed.py 的 save_every=5），退出时收工存档。

无模式统一自循环（2026-10-07）：任何输入（用户对话、批量文件、数据流）统一
成为学习信号，走同一条"经验→缓冲→选拔→睡眠训练→体检门控→换班"链路。终端
对话经 stdin 通道进入 serve()，用户输入自动 add 进 buffer（learn_total+1），
守护等待中检测到用户输入即恢复喂食——没有用户模式/训练投喂模式之分，不新增 --mode。

无限循环喂食（--loop-feed 默认开启，2026-10-07）：数据源喂尽自动轮转回第一个
源从头再喂，持续运行——不存在"数据尽"或"序位尽头"，喂食持续进行。游标退化
为"轮转位置标记"（防同轮内重复），不再是"终点"。间隔重复本身
就是巩固机制（L7 明文允许跨周期重复），循环重喂符合律。守护模式/反刍/待机
等待保留为"无数据源时"的兜底（--sources 为空或全部禁用时），默认路径下不再
触发"数据尽"。用 --no-loop-feed 可关闭（恢复旧有限批次语义）。
"""
import argparse
import gzip
import hashlib
import itertools
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import torch

from dolphin.datasets import DEFAULT_ORDER, SOURCES, iter_source, serialize
from dolphin.dolphin import Dolphin, GateDisabled, clamp_threshold
from dolphin.life import _threshold_feedback  # noqa: F401  兼容再导出（tests G9 用）
from dolphin.life import run_cycle as life_run_cycle
from dolphin.model import Config

FED_STATE = os.path.join("dolphin", "fed_state.pt")
PROGRESS_EVERY = 20  # 部署侧进度打印节奏（也用于产出持续运行的时间戳证据）
MAX_IDLE_DEFAULT = 8  # 值：自成——守护模式连续 8 个空选拔周期视为消化完毕，允许收工
# 发育实验（2026-10-05）：结构化生长时间线落盘路径——参数总量时间线是发育模式的
# 核心交付（生长曲线），dict repr 日志不适合机器复绘，故每周期追加一行 JSON。
GROWTH_TIMELINE = os.path.join("实验记录", "发育时间线_20261004.jsonl")

# 日志轮转（2026-10-10，无人值守可靠性）：目录级轮转阈值与回落目标比例。
# 只处理日志文件（.log/.jsonl/.log.gz/.jsonl.gz 等实验记录），绝不触碰生产资产
# （dolphin/fed_state.pt、dolphin/memory_cold.jsonl、探针、存档、归档目录等）。
LOG_DIR = os.path.join("实验记录", "")
LOG_ROTATE_MAX_BYTES = int(os.environ.get("LOG_ROTATE_MAX_MB", "100")) * 1024 * 1024
LOG_ROTATE_TARGET_RATIO = 0.80  # 压缩到总大小低于阈值的 80%
LOG_ROTATE_KEEP_GZ = int(os.environ.get("LOG_ROTATE_KEEP_GZ", "5"))  # 压缩后仍超限时保留的最近 .gz 份数

# 崩溃退避（2026-10-10）：指数退避上限与默认最大连续崩溃次数（保持原行为 5）。
CRASH_BACKOFF_CAP = 300.0  # 指数退避上限（秒）
CRASH_MAX_RESTARTS_DEFAULT = 5  # 默认最大连续崩溃次数（保持原行为）

_PRINT_LOCK = threading.Lock()
_T0 = time.monotonic()
_CURRENT_CYCLE = None  # 最近一次运行中的 Dolphin.cycle（崩溃记录用，模块级线程安全近似）

# 日志文件扩展名集合（目录级轮转只处理这些"可压缩日志"）。
_LOG_EXTS = (".log", ".jsonl", ".txt", ".out", ".gz")


def _is_log_filename(name):
    """判断文件名是否属于可轮转的日志文件（.log/.jsonl 等实验记录）。"""
    low = name.lower()
    return any(low.endswith(ext) for ext in _LOG_EXTS)


def _log_candidates(log_dir):
    """列出日志目录下应参与轮转的日志文件（不含子目录、不含生产资产）。

    只扫描 log_dir 顶层文件（不递归）：实验记录/ 下的归档子目录存放的是
    .pt 存档等生产资产，绝不进入轮转逻辑（红线）。
    """
    out = []
    if not os.path.isdir(log_dir):
        return out
    for name in os.listdir(log_dir):
        full = os.path.join(log_dir, name)
        if not os.path.isfile(full):
            continue  # 子目录（归档等）一律跳过
        if _is_log_filename(name):
            out.append(full)
    return out


def _total_size(paths):
    try:
        return sum(os.path.getsize(p) for p in paths)
    except OSError:
        return 0


def rotate_log_dir(log_dir=None, max_bytes=None, target_ratio=None, keep_gz=None):
    """启动时目录级日志轮转：压缩最旧日志直到总大小低于阈值。

    策略（2026-10-10）：
      1. 扫描日志目录（默认 实验记录/）顶层日志文件（.log/.jsonl 等）；
      2. 若总大小超过 max_bytes（默认 100MB，可用环境变量 LOG_ROTATE_MAX_MB 覆盖）：
         按 mtime 从旧到新，将最旧日志压缩为 .gz（gzip，压缩后删除原文件），
         直到总大小低于 max_bytes * target_ratio（默认 80%）；
      3. 若全部压缩后仍超阈值，则删除最旧的 .gz（保留最近 keep_gz 份）。
    红线：只处理日志文件，绝不删除/移动/修改生产资产（dolphin/fed_state.pt、
    dolphin/memory_cold.jsonl、探针、存档、归档目录等）。

    返回动作描述列表（测试用）。"""
    log_dir = log_dir if log_dir is not None else LOG_DIR
    max_bytes = max_bytes if max_bytes is not None else LOG_ROTATE_MAX_BYTES
    target_ratio = target_ratio if target_ratio is not None else LOG_ROTATE_TARGET_RATIO
    keep_gz = keep_gz if keep_gz is not None else LOG_ROTATE_KEEP_GZ
    actions = []
    files = _log_candidates(log_dir)
    if not files:
        return actions
    total = _total_size(files)
    if total <= max_bytes:
        return actions  # 未超阈值，无需轮转

    target = max_bytes * target_ratio
    # 按 mtime 从旧到新排序（旧文件优先压缩）。
    files.sort(key=lambda p: os.path.getmtime(p))
    # 只压缩尚未压缩的日志（.gz 已压缩，压缩无收益，跳过）。
    plain = [p for p in files if not p.lower().endswith(".gz")]
    for path in plain:
        if _total_size(files) <= target:
            break
        try:
            gz_path = path + ".gz"
            with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
            os.remove(path)  # 压缩成功后删除原日志
            files.remove(path)
            files.append(gz_path)
            actions.append(f"compress {os.path.basename(path)} -> {os.path.basename(gz_path)}")
        except OSError as e:
            actions.append(f"compress FAIL {os.path.basename(path)}: {e!r}")

    # 压缩后仍超阈值：删除最旧 .gz（保留最近 keep_gz 份）。
    if _total_size(files) > target:
        gz_files = sorted((p for p in files if p.lower().endswith(".gz")),
                        key=os.path.getmtime)
        while len(gz_files) > keep_gz and _total_size(files) > target:
            oldest = gz_files.pop(0)
            try:
                sz = os.path.getsize(oldest)
                os.remove(oldest)
                files.remove(oldest)
                actions.append(f"delete {os.path.basename(oldest)} ({sz} B)")
            except OSError as e:
                actions.append(f"delete FAIL {os.path.basename(oldest)}: {e!r}")
    return actions


def crash_log_path():
    """崩溃记录文件：实验记录/CRASH_YYYYMMDD.log（按天追加）。"""
    return os.path.join("实验记录", f"CRASH_{datetime.now().strftime('%Y%m%d')}.log")


def write_crash_record(cycle, exc):
    """每次崩溃时追加写崩溃记录到 CRASH_YYYYMMDD.log（失败不影响主流程）。"""
    try:
        os.makedirs("实验记录", exist_ok=True)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with open(crash_log_path(), "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n"
                    f"cycle={cycle}\n{tb}\n")
    except Exception as e:
        log("崩溃记录", f"写入失败（忽略）：{e!r}")


def write_crash_fatal(cycle, exc, crash_count):
    """超过 --max-restarts 最终放弃时写 CRASH_FATAL_<时间戳>.md 告警标记。"""
    try:
        os.makedirs("实验记录", exist_ok=True)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("实验记录", f"CRASH_FATAL_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# CRASH FATAL（请人工介入）\n\n"
                    f"- 时间：{datetime.now().isoformat(timespec='seconds')}\n"
                    f"- 崩溃次数：{crash_count}\n"
                    f"- 当前 cycle：{cycle}\n"
                    f"- 异常类型：{type(exc).__name__}\n"
                    f"- 异常内容：{exc!r}\n\n"
                    f"## 最后 traceback\n\n```\n{tb}\n```\n\n"
                    f"## 建议\n\n系统超过最大连续崩溃次数后放弃自动重启。"
                    f"请人工检查 feed.py 运行环境（数据源/磁盘/模型存档）。"
                    f"存档未被删除/清空（dolphin/fed_state.pt 保留现场）。\n")
        log("崩溃记录", f"最终放弃，已写告警标记 → {path}")
    except Exception as e:
        log("崩溃记录", f"CRASH_FATAL 写入失败（忽略）：{e!r}")


def log(tag, msg):
    """统一带相对时间戳的行打印（单次持打印锁，防两线程行内交错）。"""
    with _PRINT_LOCK:
        print(f"[+{time.monotonic() - _T0:8.3f}s][{tag}] {msg}", flush=True)


class SleepTrainer(threading.Thread):
    """睡眠训练器：后台循环，触发即训练，体检通过热切换。

    主线程对它只有两个非阻塞动作：request_sleep()（通知）和退出时 join()。
    """

    def __init__(self, d, steps, save_every=5, autotune_steps=False):
        super().__init__(name="sleep-trainer", daemon=False)
        self.d = d
        self.steps = steps
        # M2 修复（2026-10-06）：autotune_steps=True 表示每周期步数由 autotune
        # 决定（run 里传 None 给 life.run_cycle，让它从 d.sleep_steps 读取已调好的
        # 值），不再使用人工 --sleep-steps 参数。测试直接构造 SleepTrainer(d,
        # steps=N) 时 autotune_steps 默认 False，仍走显式 steps（行为钉死不变）。
        self.autotune_steps = autotune_steps
        self.save_every = save_every
        self.wake = threading.Event()       # 主线程通知：触发睡眠周期
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
        self.wake.set()  # 唤醒 wait 以检查停止标志

    def _timeline(self, report):
        """发育实验（2026-10-05）：每睡眠周期追加一行结构化生长时间线——
        参数总量/d_model/状态机/门控判决/结构调整记录（生长曲线的核心数据源）。
        只增不改：写盘失败绝不影响喂食（部署持续运行原则）。"""
        import json
        try:
            body = report.get("body") or {}
            rec = {
                "wall": round(time.monotonic() - _T0, 3),
                "cycle": report.get("cycle"),
                "params": body.get("params"),
                "d_model": body.get("d_model"),
                "n_layers": body.get("n_layers"),
                "state": getattr(self.d.life_ctl, "state", None),
                "action_kind": (body.get("action") or {}).get("kind"),
                "action_class": (body.get("action") or {}).get("class"),
                "action_reason": (body.get("action") or {}).get("reason"),
                "m4_plan": report.get("m4_plan"),
                "m4_surgery": {k: v for k, v in (report.get("m4_surgery") or {}).items()
                               if k != "new_model"},
                "m4_defer": report.get("m4_defer"),
                "m4_rollback": report.get("m4_rollback"),
                "m4_lambda_g": report.get("m4_lambda_g"),
                "probe_old": report.get("probe_old"),
                "probe_new": report.get("probe_new"),
                "gate_margin": report.get("gate_margin"),
                "gate_eps": report.get("gate_eps"),
                "passed": report.get("passed"),
                "swapped": report.get("swapped"),
                "lambda_g_eff": body.get("lambda_g"),
                "util_gap": body.get("util_gap"),
                "maturity": body.get("maturity"),
                "supply": body.get("supply"),
                "selected": report.get("selected"),
                "dreamed": report.get("dreamed"),
                "ce_last": report.get("ce_last"),
                "kd_last": report.get("kd_last"),
                "lr": report.get("lr"),
                "awake": self.d.awake().name,
            }
            os.makedirs(os.path.dirname(GROWTH_TIMELINE), exist_ok=True)
            with open(GROWTH_TIMELINE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # 生长曲线打印（发育模式核心观测行：参数总量/状态机/d_model/门控判决）。
            # 空选拔早退周期无 body 字段——如实打一行轻量记录，不做 None 格式化。
            if rec["params"] is None:
                log("生长曲线", f"cycle={rec['cycle']} （空选拔早退周期，无体检/无 body 观测）"
                    f" 选拔={rec['selected']} 状态={rec['state']}")
                return
            surg = rec["m4_surgery"]
            knife = f"  调整:{surg.get('axis')}{surg.get('layer')}(+{surg.get('delta')})" if surg else ""
            log("生长曲线", f"cycle={rec['cycle']} 参数总量={rec['params']:,} "
                f"d_model={rec['d_model']} 状态={rec['state']} "
                f"probe {rec['probe_old']}→{rec['probe_new']} "
                f"margin={rec['gate_margin']} 判决={'过' if rec['passed'] else '否'}"
                f"{' 换班' if rec['swapped'] else ''}{knife}")
        except Exception as e:  # 时间线是证据通道，不是核心功能
            log("生长曲线", f"时间线落盘失败（不影响喂食）：{e!r}")

    def run(self):
        log("训练线程", f"已启动（steps={self.steps}）")
        while not self.stop_flag.is_set():
            # 触发三选一：主线程通知 / 压力达阈值 / 缓冲将满（自主部署时不通知也会触发）
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
                # M4：睡眠周期唯一实现在 dolphin.life.run_cycle（feeding=True =
                # 原 trainer_cycle 的线程安全语义；trainer_cycle 已删除）
                # M2 修复：autotune_steps=True 时传 None，让 run_cycle 从
                # d.sleep_steps 读取 autotune 调好的步数（值自成）；否则走显式 steps。
                run_steps = None if self.autotune_steps else self.steps
                report = life_run_cycle(self.d, run_steps, feeding=True)
            except GateDisabled as e:  # G8：探测集缺失，拒绝开始睡眠（fail-fast，非静默回滚）
                self.busy += time.monotonic() - t0
                log("训练线程", f"门控不可用，拒绝开始睡眠（部署喂入继续，等 probe 恢复自动解禁）：{e}")
                continue
            except Exception as e:  # 单轮失败不带塌部署：记录后继续喂数与训练
                self.busy += time.monotonic() - t0
                try:  # 异常路径同样回滚：半训练权重不得成为下一轮基座（审计②-4）
                    self.d.sleeping().model.load_state_dict(self.d.awake().model.state_dict())
                except Exception as e2:
                    log("训练线程", f"回滚也失败（保留现场排查）：{e2!r}")
                _threshold_feedback(self.d)  # 第三条早退路径同样计入（审计③-6；原闭包调用在 run 作用域会 NameError）
                self.d.cycle += 1
                log("训练线程", f"周期异常终止（部署持续运行）：{e!r}")
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
            self._timeline(report)  # 发育实验：结构化生长时间线 + 生长曲线打印
            if self.cycles % self.save_every == 0:
                self.d.save(FED_STATE)  # 周期边界存档：两半球状态一致
                log("训练线程", f"每 {self.save_every} 个周期存档一次 → {FED_STATE}")
        log("训练线程", "下线")


# ---------------- 喂食管线与守护模式（G3 自续运行） ----------------

def structural_reward(d, eid, text, seen):
    """自动第二信号（G3 奖励接线，最小闭环）：记录新鲜度衰减。

    r = 同文重复出现次数的负值——数据集内重复条目自动获得负奖励，在窗口去重
    （L7，仅限时间窗）之外给选拔层第二证据：跨窗口的重复同样压分。
    如实标注：这是结构信号（可复核：同文 sha256 计数），不是内容评价——
    真正的内容奖励（用户点赞/求解器验证）留给部署时代（M4：归属贡献过于复杂）。
    负奖励的压分边界 [-0.9, 9.0] 由 experience.select 唯一持有（分数恒正）。
    """
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    n = seen.get(h, 0)
    seen[h] = n + 1
    if n == 0:
        return 0.0  # 首现零信号，不加额外信号
    with d.feed_lock:  # feedback 与训练线程的选拔/摘除互斥（锁内只有名单级操作）
        d.feedback(eid, float(-n))
    return float(-n)


def ruminate(d, k=None):
    """反刍一步（G3 守护模式）：从记忆库做梦采样旧数据，经正常部署管线注入缓冲。

    反刍=消化已摄入数据：旧数据全部是真实经验数据（L5 安全——记忆库只收真实经验的
    滞留与笔记，绝无模型自生成内容），L7 明文允许跨周期重复（间隔重复=巩固机制）。
    注入经 learn 算惊讶度入缓冲 → 压力积累 → 周期照常触发 → 做梦+复活+选拔照旧，
    选拔来源由此成为"记忆库做梦采样 + 复活条目 + 缓冲残留"。
    返回注入条数（0=记忆库暂时无可反刍，如全员 hits 达标——该情况由 resurrect 处理）。
    """
    ents = list(d.memory.entries) + list(d.memory.cold_entries)
    if not ents:
        return 0
    queries = [en.text.encode("utf-8", errors="replace")
               for en in d.rng.sample(ents, min(8, len(ents)))]  # 联想线索从记忆库自动获取
    batch = d.memory.dream(queries, k=k or d.dream_k)
    for en in batch:
        d.learn(en.text, source="ruminate")  # 走与正式喂食同一条部署管线
    return len(batch)


def serve_demo(d, fed_total):
    """V2 证据通道【零点实验专用，非生产功能】：喂食进行中用固定演示问题调
    serve() 一次。训练窗口（睡眠周期）与回答窗口的时间戳交错即"边训练边回答"
    的证据。serve 与 learn 同样计 learn_busy + feed_lock（dolphin.py 读屏障），
    主线程内串行调用，不新增线程、不改变既有并发语义。"""
    q = "小明有3个苹果，又买了2个，一共有几个苹果？"
    t0 = time.monotonic()
    try:
        resp, _ = d.serve(q)
    except Exception as e:  # 演示失败不拦喂食（部署持续运行原则）
        log("服务演示", f"第 {fed_total} 条边界 serve 异常：{e!r}")
        return
    dt = time.monotonic() - t0
    log("服务演示", f"（训练窗口进行中，耗时 {dt:.2f}s）问：{q}  "
        f"答：{resp.replace(chr(10), ' ')[:60]!r}")


# ---------------- 无模式 stdin 对话通道（2026-10-07） ----------------
# 任何输入都是学习信号：serve 内部已把用户输入 add 进 buffer（learn_total+1），
# 与批量喂食走同一条"经验→缓冲→选拔→睡眠训练→体检门控→换班"链路。
# 不新增模式概念、不新增 --mode：对话只是"输入到达"的另一种形态。

_CHAT_COUNT_LOCK = threading.Lock()
_CHAT_COUNT = 0


def _bump_chat_count(n=1):
    """线程安全累加对话处理条数（运行总结用，纯计数不涉及模型状态）。"""
    global _CHAT_COUNT
    with _CHAT_COUNT_LOCK:
        _CHAT_COUNT += n
        return _CHAT_COUNT


def chat_count():
    """已处理的 stdin 对话条数（模块级累计，供运行总结打印）。"""
    return _CHAT_COUNT


def chat_once(d, text, source="stdin_chat"):
    """无模式单轮对话：serve 回答 + 用户输入自动成为学习信号 + 可选打分。

    serve() 内部已把用户输入 add 进 buffer（learn_total+1），与批量喂食同一条
    链路——这就是"没有用户模式/训练投喂模式"的体现。打印回复后从 stdin 读
    一行打分：+1/-1/0（空=跳过），调用 d.feedback(eid, reward) 写奖励。
    异常不影响部署（部署持续运行原则）：打印错误后返回 None。
    返回 (eid, resp) 或 None。
    """
    try:
        resp, eid = d.serve(text)
    except Exception as e:  # 对话失败不影响喂食/训练（部署持续运行原则）
        log("对话", f"serve 异常（跳过本轮）：{e!r}")
        return None
    print(f"[对话] 用户：{text}", flush=True)
    print(f"[对话] 蓝蓟智能：{resp}", flush=True)
    _bump_chat_count()
    try:
        print("[对话] 打分？（+1/-1/0 或直接回车跳过）", flush=True)
        line = sys.stdin.readline()
        if line is None:  # 流已关闭：不能再读，跳过打分
            return (eid, resp)
        reward_line = line.strip()
        if reward_line in ("+1", "1", "+", "赞", "好"):
            reward = 1.0
        elif reward_line in ("-1", "0", "-", "踩", "差"):
            reward = -1.0 if reward_line in ("-1", "-", "踩", "差") else 0.0
        elif reward_line == "":
            log("对话", "跳过打分（空输入）")
            return (eid, resp)
        else:
            try:
                reward = float(reward_line)
            except ValueError:
                log("对话", f"无法识别的打分 {reward_line!r}，跳过（可输入 +1/-1/0 或空）")
                return (eid, resp)
        d.feedback(eid, reward)
        log("对话", f"已写入奖励 {reward:+.1f} → 经验 {eid}")
    except Exception as e:  # 打分失败不影响回复与学习信号（部署持续运行）
        log("对话", f"打分写入异常（忽略）：{e!r}")
    return (eid, resp)


def stdin_chat_loop(d, stop_flag, wake_cb=None):
    """无模式 stdin 对话通道（独立线程，daemon=True 由调用方设置）。

    逐行读 sys.stdin（阻塞）。feed.py 主线程不用 stdin，所以独占 stdin 安全。
    每读到一行非空输入 → chat_once(d, line)（打分提示同样从 stdin 读）。
    每处理一条输入可调用 wake_cb()（可选）——通常不需要：serve 已把经验加入
    buffer，SleepTrainer 的压力/通知通道会自然触发睡眠。
    线程退出条件：stop_flag.is_set() 且 stdin 无更多输入（EOF 时退出）。
    """
    log("对话", "stdin 对话通道已启动（无模式：任何输入自动成为学习信号）")
    while not stop_flag.is_set():
        try:
            line = sys.stdin.readline()
        except Exception as e:  # stdin 读取异常：守护线程记录后短暂退避，避免忙转
            log("对话", f"stdin 读取异常（退避 1s 后继续）：{e!r}")
            time.sleep(1.0)
            continue
        if line == "":
            # EOF：stdin 关闭（如管道输入结束）。stop_flag 未置位时继续等——
            # 外部可能重新打开输入流/守护模式仍在运行；置位则退出。
            if stop_flag.is_set():
                break
            time.sleep(0.2)
            continue
        text = line.strip()
        if not text:
            continue  # 空行不是学习信号，跳过
        try:
            chat_once(d, text)
        except Exception as e:  # chat_once 已内部捕获，双保险（部署持续运行）
            log("对话", f"对话处理异常（继续监听）：{e!r}")
        if wake_cb is not None:
            try:
                wake_cb()
            except Exception:
                pass
    log("对话", "stdin 对话通道下线")


def feed_from(d, names, cursor, seen, trainer=None, sleep_every=500, quota=None,
              stats=None, fed_base=0, progress_every=PROGRESS_EVERY, serve_every=0,
              feed_interval=0.0, loop=False):
    """正式喂食：从喂食游标（G3）起迭代数据源，喂入 + 自动第二信号接线。

    游标语义：record_idx 计数据源内已消费的记录数（含碎片跳过）。cursor 与
    d.feed_cursor 是同一引用：喂食推进即"入档就绪"（周期边界/收工存档自动携带）。
    quota=本次最多喂入条数（None=不限，喂到数据尽）。返回 (fed, limit_hit, skipped)。

    loop（无限循环喂食，--loop-feed 默认开启，2026-10-07）：True 时数据源喂尽
    自动轮转回第一个源从头再喂（游标退化为"轮转位置标记"，不再有"数据尽/序位
    尽头"——喂食持续进行）；False 时保持旧语义：所有源耗尽返回 limit_hit=False
    （测试路径与 --no-loop-feed 均走此分支）。quota（--limit）在 loop 模式下仍
    生效：达到人工限次即 break（人工限次不是数据尽头）。轮转时写入 cursor
    ["round"] 轮次数（仅 loop 模式写，非 loop 模式绝不改动 cursor 结构）。

    feed_interval（发育实验 2026-10-05 新增，默认 0=现状不变）：部署侧节流——
    每条 learn 之间 sleep 该秒数，模拟交互式部署的到达节奏（真实用户不会以
    每秒 8 条的速度输入语料）。这只是喂食节奏旋钮：睡眠触发（压力/通知）、
    结构调整判决（恒温器/守卫）全部仍是内部信号，律 L11 人工接口未动。
    动机（如实记录）：批量满速喂食下，睡眠周期中点的 _since_sleep 被本周期内
    的持续喂入抬到 200+，Bellesi 睡眠债守卫（>4 禁调整）结构性封锁结构调整窗口——
    阶段一实测 6/6 周期 hold("睡眠债 15.0 > 4.0")。系统每 ~41s 触发一次睡眠（并无
    剥夺），债务虚高纯因 target_interval=15 按"交互数"计而批量喂食交互极快。
    """
    fed = skipped = 0
    limit_hit = False
    si = cursor["source_idx"]
    # 是否已实际开始过至少一个源（用于区分"游标越界恢复"与"完整一圈零产出"）：
    # 游标越界恢复（source_idx >= len）时尚未跑过任何源，应允许轮转恢复喂食；
    # 已实际跑过源但 fed 未增长（空源/全碎片）则直接退出，不再空转轮转。
    started_any = False
    circle_start_fed = 0  # 当前这一圈开始时的 fed（用于检测空源/全碎片圈）
    while not limit_hit:
        if not loop and si >= len(names):
            break  # 非 loop 模式：所有源耗尽 = 数据尽（旧语义）
        if loop and si >= len(names):
            if not names or (started_any and fed == circle_start_fed):
                # 无数据源可喂，或已实际跑过源但整圈零产出（空源/全碎片）：
                # 这不是"数据尽"（游标无终点概念），而是"无数据源可喂"——
                # 直接退出由调用方走守护兜底（反刍/待机等待），
                # 不轮转、不递增 round、不污染游标。
                break
            # 无限循环喂食：数据源喂尽后轮转回第一个源，喂食持续进行。
            si = 0
            cursor["source_idx"], cursor["record_idx"] = si, 0
            cursor["round"] = int(cursor.get("round", 0)) + 1
            circle_start_fed = fed
            log("部署", f"数据源已轮转一圈，开始第 {cursor['round'] + 1} 轮喂食"
                        f"（loop 模式）")
        name = names[si]
        started_any = True
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
            if feed_interval > 0:
                time.sleep(feed_interval)  # 部署节流（发育实验）：交互节奏模拟
            if stats is not None:
                stats[name] = stats.get(name, 0) + 1
            fed_total = fed_base + fed
            if trainer is not None and sleep_every and fed_total % sleep_every == 0:
                trainer.request_sleep()  # 非阻塞通知，喂入继续
                log("部署", f"已喂 {fed_total} 条 → 通知训练线程触发睡眠周期")
            elif fed_total % progress_every == 0:
                log("部署", f"已喂 {fed_total} 条（当前源 {name}）"
                    f"  缓冲 {d.buffer.bytes}B  压力 {d.buffer.pressure():.2f}")
            if serve_every and fed_total % serve_every == 0:  # 零点实验专用（V2 证据）
                serve_demo(d, fed_total)
            if quota is not None and fed >= quota:
                limit_hit = True  # 人工限次已达：源内精确收工（limit 是人工限次，不是数据尽头）
                break
        if not limit_hit:  # 本源耗尽：游标推进到下一源
            si += 1
            cursor["source_idx"], cursor["record_idx"] = si, 0
            if loop and si >= len(names) and quota is not None and fed >= quota:
                limit_hit = True  # 人工限次已达：直接收工（limit 是人工限次，不是数据尽头）
                break
    return fed, limit_hit, skipped


def _daemon_wait(d, trainer, args, poll_interval=5.0):
    """守护模式待机等待：数据尽+反刍完后，周期性检查是否有新经验流入。

    新经验 = d.learn_total 增长（外部进程喂数据 / serve 交互 / 手动 ruminate）。
    一旦检测到新经验，通知训练线程并返回，调用方恢复正式喂食。
    若收到 KeyboardInterrupt（用户 Ctrl+C）则抛出让外层退出。
    """
    baseline = d.learn_total
    log("守护", f"进入待机等待（基线 learn_total={baseline}，每 {poll_interval:.0f}s 检查）")
    while True:
        time.sleep(poll_interval)
        if d.learn_total > baseline:
            log("守护", f"检测到新经验流入（{d.learn_total - baseline} 条）→ 恢复正式喂食")
            trainer.request_sleep()  # 优先消化缓冲里的外部新经验
            return
        # 状态检查：每 60s 打一行，确认进程运行中（无人值守可观测性）
        if int(time.monotonic()) % 60 < poll_interval:
            log("守护", f"状态检查：learn_total={d.learn_total} 记忆={len(d.memory.entries)} 条"
                f" 周期={d.cycle} 醒脑={d.awake().name}")


def guard_loop(d, trainer, max_idle, log_every=50):
    """守护模式主循环（G3 自续运行）：数据尽不退场。

    状态机（文字版）：
      反刍：ruminate() 从记忆库做梦采样旧数据经 learn 注入缓冲 → request_sleep()
      让周期照常触发（触发三通道：通知/压力/溢出，均不变）→ 周期内的选拔来源
      =记忆库做梦采样 + 复活条目 + 缓冲残留；带通选出空集 → 继续反刍。
      转移：部署侧新经验流入（learn/serve，learn_total 口径）→ 恢复正式喂食；
      连续 max_idle 个空选拔周期（消化完毕）→ 收工；Ctrl+C → 收工（调用方捕获）。
    进入时 idle 计数清零：喂食末尾的空周期不算反刍的空转。
    返回 (resume, ruminate_fed)。
    """
    log("守护", f"数据源已耗尽 → 进入反刍（周期照常触发，选拔来源=记忆库做梦+复活+缓冲残留，"
        f"L7 跨周期重复）。退出条件：Ctrl+C 或连续 {max_idle} 个空选拔周期（--max-idle）；"
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
            log("守护", f"连续 {max_idle} 个空选拔周期，反刍消化完毕 → 收工（防真死转）")
            return False, ruminate_fed
        n = ruminate(d)
        ruminate_fed += n
        trainer.request_sleep()  # 周期照常触发：有数据则消化；无数据空选拔也计入 idle（真死转防护）
        if n:
            if ruminate_fed % log_every == 0:
                log("反刍", f"已反刍 {ruminate_fed} 条  缓冲 {d.buffer.bytes}B  "
                    f"记忆库 {len(d.memory.entries)} 条")
        else:
            time.sleep(0.05)  # 无数据时短等待节流（request_sleep 非阻塞，空选拔周期照常跑）


# ---------------- CLI 与主流程 ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="B 路线部署期喂食（部署-训练双线程，守护模式默认开启）")
    ap.add_argument("--sources", default=",".join(DEFAULT_ORDER),
                    help="逗号分隔的数据源（默认按质量排序，logiconbench 需显式指定）")
    ap.add_argument("--limit", type=int, default=None, help="本次最多喂入条数（默认不限："
                    "--loop-feed 开启时无限循环喂食；"
                    "--no-loop-feed 时喂到数据尽）")
    ap.add_argument("--sleep-every", type=int, default=500, help="每喂多少条通知训练线程睡眠")
    ap.add_argument("--sleep-steps", type=int, default=120, help="每次睡眠的训练步数")
    ap.add_argument("--model", default="auto", choices=["57", "small", "seed", "auto"],
                    help="57=历史默认出生档（≈57M 参数，非容量上限）；small=冒烟用小模型；"
                         "seed=发育模式最小种子 d32×2 层×2 头（41,856 参数随机初始化，"
                         "生长机制可用——模型规模随经验增长）；"
                         "auto=有存档则读存档 cfg（默认），无存档回落历史默认出生档 57")
    ap.add_argument("--resume", action="store_true", help="从 dolphin/fed_state.pt 按喂食游标续喂")
    ap.add_argument("--no-guard", action="store_true",
                    help="关闭守护模式（数据尽即收工；默认开启：数据尽后转入反刍等待新经验）")
    ap.add_argument("--max-idle", type=int, default=MAX_IDLE_DEFAULT,
                    help=f"守护模式：连续 N 个空选拔周期后允许收工（值：自成，默认 {MAX_IDLE_DEFAULT}）")
    ap.add_argument("--feed-interval", type=float, default=0.0,
                    help="部署侧节流：每条 learn 之间 sleep 的秒数（默认 0=满速；"
                         "发育实验用 0.3-0.4 模拟交互式到达节奏——批量满速下睡眠债"
                         "守卫会结构性封锁结构调整窗口，见 feed_from docstring）")
    ap.add_argument("--serve-every", type=int, default=0,
                    help="【零点实验专用】每喂 N 条用固定演示问题调 serve() 一次"
                         "（V2 边训练边回答的证据通道；默认 0=关闭）")
    ap.add_argument("--param-adaptive", action="store_true", default=True,
                    help="启用参数自适应控制律（2026-10-06 用户定案）："
                         "轻限制上限=当前参数×1.571；压力触发三段式下跌"
                         "（30%%→10%%→3%%）；正常态每周期 ±8 单元微调。"
                         "默认开启（参数自适应）；用 --no-param-adaptive 关闭。")
    ap.add_argument("--no-param-adaptive", dest="param_adaptive", action="store_false",
                    help="关闭参数自适应控制律（默认开启，此开关可显式关闭）")
    ap.add_argument("--daemon", action="store_true", default=True,
                    help="常驻守护模式（默认开启）：数据尽+反刍完不退出，进入休眠等待"
                         "新数据/新交互；进程异常自动重启。用 --no-daemon 关闭。")
    ap.add_argument("--no-daemon", dest="daemon", action="store_false",
                    help="关闭常驻守护模式（数据尽+反刍完即收工退出）")
    ap.add_argument("--stdin-chat", action="store_true", default=True,
                    help="统一自循环对话通道：任何输入自动成为学习信号（无模式）。"
                         "终端逐行读入→AI 回答→输入自动入缓冲→可打分→自动睡眠巩固。"
                         "默认开启；用 --no-stdin-chat 关闭。")
    ap.add_argument("--no-stdin-chat", dest="stdin_chat", action="store_false",
                    help="关闭 stdin 对话通道（仅批量数据源消化；默认开启，"
                         "任何输入自动成为学习信号——无模式）")
    ap.add_argument("--loop-feed", action="store_true", default=True,
                    help="无限循环喂食（默认开启）：数据源喂尽后自动轮转回第一个源，"
                         "游标退化为轮转位置标记（防同轮重复）。--limit 仍生效（人工限次）。")
    ap.add_argument("--no-loop-feed", dest="loop_feed", action="store_false",
                    help="关闭无限循环喂食（数据源喂尽后走旧路径：反刍/待机等待，"
                         "即旧守护模式语义）")
    ap.add_argument("--max-restarts", type=int, default=CRASH_MAX_RESTARTS_DEFAULT,
                    help=f"常驻守护：最大连续崩溃次数，超过后放弃自动重启"
                         f"（默认 {CRASH_MAX_RESTARTS_DEFAULT}，保持原行为；"
                         f"退避为指数退避，上限 {CRASH_BACKOFF_CAP:.0f}s）")
    return ap.parse_args()


def make_cfg(which):
    if which == "small":
        return Config(d_model=128, n_layers=2, n_heads=2, block_size=256)
    if which == "seed":
        # 发育模式最小种子（2026-10-05）：d32×2 层×2 头 ≈ 0.042M 参数随机初始化。
        # 物理前提：字面"零参数"不存在（无旋钮则无处存放学习）——最小可行种子即
        # 最接近的诚实实现。固定部分（字节表 256/transformer 配方/block_size）是
        # "基因组"=律，不生长；可生长部分（d_model/MLP 隐层/attn-v）交给恒温器。
        return Config(d_model=32, n_layers=2, n_heads=2, block_size=256)
    if which == "auto":
        # 无人值守默认档：有存档则用存档模型（load 会覆盖），无存档回落历史默认出生档
        # （≈57M，非容量上限；生产起点已改为最小种子档）。
        # 实际模型由 Dolphin.load() 从存档 cfg 决定，此处只提供占位 Config。
        if os.path.exists(FED_STATE):
            try:
                ck = torch.load(FED_STATE, map_location="cpu", weights_only=True)
                return Config(**ck["cfg"])
            except Exception:
                pass
        return Config(d_model=768, n_layers=8, n_heads=8, block_size=256)  # 历史默认出生档 ≈57M 参数（非容量上限）
    return Config(d_model=768, n_layers=8, n_heads=8, block_size=256)  # 历史默认出生档 ≈57M 参数（非容量上限）


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

    # —— 日志轮转（2026-10-10）：启动时压缩/清理超限日志，保护磁盘 ——
    try:
        actions = rotate_log_dir()
        if actions:
            for a in actions:
                log("日志轮转", a)
            log("日志轮转", f"共 {len(actions)} 个动作（阈值 "
                f"{LOG_ROTATE_MAX_BYTES // (1024 * 1024)}MB）")
        else:
            log("日志轮转", f"日志未超阈值（{LOG_ROTATE_MAX_BYTES // (1024 * 1024)}MB），跳过")
    except Exception as e:
        log("日志轮转", f"轮转失败（不影响启动）：{e!r}")

    # —— 常驻守护（无人值守自循环）：进程级异常自动重启，持续运行 ——
    # 只有 KeyboardInterrupt（用户主动 Ctrl+C）才真正退出；其余异常记录后
    # 指数退避重试（最多 --max-restarts 次连续失败后放弃，避免无限崩溃循环）。
    max_restarts = max(1, args.max_restarts)
    consecutive_crashes = 0
    while True:
        try:
            _run_once(args, names)
            # _run_once 正常返回 = 数据尽+反刍完且非守护模式（或守护模式被
            # 外部停止）→ 按用户意图收工。
            if not args.daemon:
                return
            # 守护模式：正常收工（理论上不达，除非 --max-idle 生效）→ 等待后重跑
            log("守护", "一轮喂食+反刍完成，进入常驻等待（10s 后重新检查数据源/新经验）")
            time.sleep(10)
            consecutive_crashes = 0  # 正常轮次重置崩溃计数
        except KeyboardInterrupt:
            log("守护", "收到用户中断，退出常驻循环")
            return
        except Exception as e:
            consecutive_crashes += 1
            write_crash_record(_CURRENT_CYCLE, e)
            log("守护", f"进程异常（第 {consecutive_crashes} 次）：{e!r}——自动重启")
            if consecutive_crashes >= max_restarts:
                write_crash_fatal(_CURRENT_CYCLE, e, consecutive_crashes)
                log("守护", f"连续 {max_restarts} 次异常，放弃自动重启（请人工检查）")
                raise
            # 指数退避：5/10/20/40/80/160/300s…（上限 300s）
            backoff = min(CRASH_BACKOFF_CAP, 5 * (2 ** (consecutive_crashes - 1)))
            log("守护", f"退避 {backoff:.0f}s 后重启（指数退避）")
            time.sleep(backoff)


def _run_once(args, names):
    """单轮喂食+反刍（可被守护循环反复调用）。返回退出原因字符串。"""
    global _CURRENT_CYCLE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 无人值守默认 auto：有存档自动续喂（load 用存档 cfg 覆盖 --model），无存档随机初始化
    d = Dolphin(cfg=make_cfg(args.model), device=dev)
    resumed = False
    if (args.resume or args.model == "auto") and os.path.exists(FED_STATE):
        d.load(FED_STATE)  # load 会用存档里的 cfg 覆盖 --model 选项
        log("续喂", f"从 {FED_STATE} 读档：cycle={d.cycle} 醒脑={d.awake().name}")
        resumed = True
        _CURRENT_CYCLE = d.cycle
    elif args.resume:
        log("续喂", f"{FED_STATE} 不存在，按随机初始化创建")

    # —— 喂食游标（G3）：存档携带上次喂食位置，--resume 从游标续喂，不再第 0 条重放 ——
    # round = 轮次数（无限循环喂食的轮转标记，2026-10-07）：旧存档可能没有该字段，
    # 读取时用 get("round", 0) 兼容；非 loop 模式（--no-loop-feed / 测试直调）不写它。
    cursor = {"sources": list(names), "source_idx": 0, "record_idx": 0, "round": 0}
    if resumed and isinstance(d.feed_cursor, dict) and d.feed_cursor.get("sources") == list(names):
        c = d.feed_cursor
        si, ri = int(c.get("source_idx", 0)), int(c.get("record_idx", 0))
        if 0 <= si <= len(names):
            cursor["source_idx"], cursor["record_idx"] = si, ri
            cursor["round"] = int(c.get("round", 0))  # 旧存档无 round → 0（兼容）
            if si >= len(names):  # 游标已越过全部数据源
                if args.loop_feed:
                    log("续喂", "喂食游标已越过全部数据源（无限循环：轮转回第一个源继续喂）")
                else:
                    log("续喂", "喂食游标显示数据源已耗尽 → 不再从第 0 条重放，直接进入守护判断")
            else:
                log("续喂", f"喂食游标生效：从源 {si + 1}/{len(names)} 第 {ri} 条续喂"
                    f"（不再从第 0 条重放）")
    elif resumed and d.feed_cursor:
        log("续喂", "存档游标与本次 --sources 不一致 → 从第 0 条重喂")
    d.feed_cursor = cursor  # 游标与存档直连：周期边界/收工存档自动携带最新位置

    if resumed:
        log("部署", "存档已加载，继续喂食")
    else:
        log("初始化", f"随机初始化（B 路线不用预训练基座）  模型 {args.model}  设备 {dev}"
            f"  探测集 {len(d.probe_chunks)} 块")
    if d.gate_disabled:
        log("部署", "警告：体检探测集缺失，训练线程将拒绝开始睡眠（G8 fail-fast），喂入照常")
    # 参数自适应控制律总开关（2026-10-06 用户定案）：默认 True（生产默认
    # 启用"完全自学习"）；--no-param-adaptive 显式关闭。
    d.param_adaptive_enabled = args.param_adaptive
    if args.param_adaptive:
        log("部署", "参数自适应控制律已启用：压力三段式下跌 + 每周期 ±8 单元微调")
    else:
        log("部署", "参数自适应控制律已关闭（--no-param-adaptive）")

    # M2 修复（2026-10-06）：autotune_steps=True —— 每周期步数由 autotune 决定
    # （run 里传 None 给 life.run_cycle，读 d.sleep_steps），不再使用人工
    # --sleep-steps 参数（值自成，完全自学习）。
    trainer = SleepTrainer(d, steps=args.sleep_steps, autotune_steps=True)
    trainer.start()
    # 无模式 stdin 对话通道（2026-10-07）：trainer 启动后立刻开监听。daemon=True
    # 保证守护等待中检测到用户输入即恢复喂食（learn_total 增长 → _daemon_wait 恢复喂食）。
    chat_thread = None
    if args.stdin_chat:
        chat_thread = threading.Thread(
            target=stdin_chat_loop, args=(d, trainer.stop_flag), daemon=True,
            name="stdin-chat")
        chat_thread.start()
    t_train_span0 = time.monotonic()

    stats = {n: 0 for n in names}
    skipped = 0
    total = 0
    ruminate_total = 0
    seen = {}  # 结构信号的重复计数表（G3 奖励接线）
    guard = not args.no_guard
    stop_reason = "数据尽/limit"
    try:
        while True:  # feed ↔ ruminate 状态机（G3 自续运行：触发、执行、验收全内部完成）
            _CURRENT_CYCLE = d.cycle  # 崩溃记录用：保持最后已知周期
            quota = None if args.limit is None else max(0, args.limit - total)
            fed, limit_hit, skip = feed_from(
                d, names, cursor, seen, trainer=trainer, sleep_every=args.sleep_every,
                quota=quota, stats=stats, fed_base=total, serve_every=args.serve_every,
                feed_interval=args.feed_interval, loop=args.loop_feed)
            total += fed
            skipped += skip
            if limit_hit:
                stop_reason = "limit"
                break
            if args.loop_feed:
                # 无限循环喂食：feed_from 有数据时内部无限轮转（喂食持续进行），
                # 永不返回；只有"空源/全碎片零产出"才返回（无数据源可喂）。
                # 此时不是数据尽，而是无数据源——走下方守护兜底（反刍/待机等待）。
                if not guard:
                    stop_reason = "无数据源（空源零产出）"
                    break
            else:
                if not guard:
                    stop_reason = "数据尽"
                    break
            resume, r = guard_loop(d, trainer, args.max_idle)
            ruminate_total += r
            if not resume:
                if not args.daemon:
                    stop_reason = f"反刍收工（max-idle={args.max_idle}）"
                    break
                # 守护模式（无人值守自循环）：数据尽+反刍完不退场，进入"待机等待"
                # —— 周期性检查是否有新经验流入（learn_total 变化），有则自动恢复
                # 正式喂食；外部进程/用户可通过新增数据或 serve 交互触发恢复。
                stop_reason = "守护等待"
                log("守护", f"数据源已耗尽且反刍消化完毕（连续 {args.max_idle} 个空周期），"
                            f"进入待机等待新经验（learn_total={d.learn_total}）……")
                _daemon_wait(d, trainer, args)
                # 检测到新经验 → 继续 while True 恢复正式喂食
                resume = True
            # resume=True：部署侧有新经验流入 → 恢复正式喂食（数据源已耗尽则空转一圈
            # 回反刍，周期已收到通知优先消化缓冲里的外部经验）
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
        log("部署", "收到中断信号，优雅退出：停止喂入，等待训练线程完成收尾")
    finally:
        trainer.shutdown()
        # 无模式对话通道：stop_flag 已置位 → 通知 stdin 线程退出。阻塞在
        # readline 上时短超时 join，避免因终端无输入而阻塞收工。
        if chat_thread is not None:
            chat_thread.join(timeout=2.0)
        trainer.join()  # 唯一的等待：当前睡眠周期完整收尾，不腰斩梯度更新
        train_span = time.monotonic() - t_train_span0
        d.save(FED_STATE)
        log("存档", f"收工存档 → {FED_STATE}")

        # —— 运行总结 ——
        awake = d.awake()
        print("\n== 运行总结 ==", flush=True)
        print(f"退出原因：{stop_reason}")
        print(f"总喂入 {total} 条（碎片跳过 {skipped} 条）")
        for n in names:
            print(f"  {n:<13} {stats[n]} 条")
        print(f"反刍 {ruminate_total} 条  连续空选拔周期 {trainer.idle_cycles} 个")
        print(f"stdin 对话 {chat_count()} 条（无模式：任何输入自动成为学习信号）")
        print(f"睡眠 {trainer.cycles} 次：换班 {trainer.swaps} 次 / 作废回滚 {trainer.rollbacks} 次"
              f"  周期 {d.cycle}")
        print(f"训练线程占墙钟比：{trainer.busy:.3f}s / {train_span:.3f}s"
              f" = {trainer.busy / train_span * 100:.1f}%（部署线程其余时间全程运行）")
        print(f"最终探测集损失（醒脑 {awake.name}）：{d.probe_loss(awake):.4f}")
        print(f"记忆库 {len(d.memory.entries)} 条  缓冲滞留 {len(d.buffer.items)} 条"
              f"  传输预算 {d.budget:.3f}")
        print(f"喂食游标：source_idx={cursor['source_idx']}/{len(names)}"
              f" record_idx={cursor['record_idx']}"
              f" round={cursor.get('round', 0)}（随档续喂；无限循环下为轮转位置标记）")
        print(f"当前醒脑：半球 {awake.name}")


if __name__ == "__main__":
    main()
