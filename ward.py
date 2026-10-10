"""查房（ward round）：查看蓝蓟智能的当前状态、身体结构与生成能力。

2026-10-04 升级（查房读真实状态）：
- 优先加载 B 路线部署存档 dolphin/fed_state.pt（双半球 + 记忆库 + 喂食游标，
  身体 cfg 以存档为准，不依赖胎教基座 birth.pt）；
- 存档不存在 / 损坏 / G7 卷面校验失败 → 回退旧逻辑（birth.pt [+ 旧 state.pt]），
  并在报告首行明示"未部署状态"——查房必须如实标注所读取的是哪个状态，
  严禁把胎教基座冒充部署态（措辞漂移禁令之状态漂移）；
- 查房新增字段：状态来源与存档时间戳、喂食游标 feed_cursor、总喂入
  learn_total、记忆库热/冷层条数、当前周期与醒脑（原有格式风格保留）。

2026-10-10 新增（独立看门狗 --watch）：
- 不破坏原查房功能；--watch 进入常驻模式，周期性检查 feed.py 是否存活、
  dolphin/fed_state.pt 是否陈旧；若 feed.py 不在运行且存档陈旧则自动拉起。
- 防循环：内存记录最近 1 小时拉起时间戳，超过 --max-restarts-per-hour 进入
  3600s 冷却并写告警。日志写到 实验记录/watchdog_YYYYMMDD.log。
- --dry-run：只打印检查结果，不拉起进程（验证用）。

注：learn_total 的累计口径当前不随档落盘（feed.py 的存档 payload 未含），
存档里没有时查房如实报"本次查房会话计数"并注明口径，绝不谎报为累计值；
存档未来带上 learn_total 字段后此处自动切换为存档口径。
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

from dolphin.dolphin import Dolphin
from dolphin.memory import MemoryStore
from dolphin.model import Config

ROOT = os.path.dirname(os.path.abspath(__file__))
FED_STATE = os.path.join(ROOT, "dolphin", "fed_state.pt")
BIRTH = os.path.join(ROOT, "dolphin", "birth.pt")
OLD_STATE = os.path.join(ROOT, "dolphin", "state.pt")

WATCH_LOG_DIR = os.path.join(ROOT, "实验记录")


def param_breakdown(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    return total, emb


def _fmt_time(ts):
    # Windows 的 strftime 不支持 %f（交接文档 §8），此处只用秒级精度
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 独立看门狗（--watch，2026-10-10） ----------------

def _watch_log(msg):
    """看门狗日志：追加写 实验记录/watchdog_YYYYMMDD.log，同时打印到 stdout。"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(WATCH_LOG_DIR, exist_ok=True)
        path = os.path.join(WATCH_LOG_DIR, f"watchdog_{datetime.now().strftime('%Y%m%d')}.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[看门狗] 日志写入失败（忽略）：{e!r}", flush=True)


def _feed_pids():
    """用 PowerShell 检测 feed.py 进程，返回 PID 列表（可能为空）。

    Windows 中文路径编码：subprocess 以 UTF-8 解码输出（errors=replace 兜底）。
    排除 powershell 自身（$PID）与只含命令字符串的误匹配——只匹配 python 进程。
    """
    ps_cmd = ("Get-CimInstance Win32_Process | "
              "Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'feed\\.py' "
              "-and $_.ProcessId -ne $PID } | "
              "Select-Object -ExpandProperty ProcessId")
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, timeout=30)
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        if not out:
            return []
        return [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    except Exception as e:
        _watch_log(f"进程检测失败：{e!r}")
        return []


def _state_age():
    """返回 fed_state.pt 距今秒数；不存在返回 None。"""
    if not os.path.exists(FED_STATE):
        return None
    return time.time() - os.path.getmtime(FED_STATE)


def _relaunch_feed():
    """拉起 python feed.py --resume（独立新进程，不阻塞看门狗）。

    日志重定向到 实验记录/喂养_watchdog_<时间戳>.log（.log 后缀可被日志轮转覆盖）。
    """
    os.makedirs(WATCH_LOG_DIR, exist_ok=True)
    log_path = os.path.join(WATCH_LOG_DIR,
                            f"喂养_watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    try:
        logf = open(log_path, "a", encoding="utf-8")
    except OSError as e:
        _watch_log(f"拉起日志文件打开失败：{e!r}")
        logf = None
    flags = 0
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "feed.py", "--resume"],
            cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=flags,
            stdin=subprocess.DEVNULL)
        _watch_log(f"已拉起 feed.py（PID {proc.pid}），日志 → {os.path.basename(log_path)}")
        return proc.pid
    except Exception as e:
        _watch_log(f"拉起 feed.py 失败：{e!r}")
        return None


def watch_loop(args):
    """常驻看门狗主循环：检查存档新鲜度 + feed.py 进程存活，必要时自动拉起。"""
    _watch_log("看门狗启动（--watch）："
               f"interval={args.interval}s stale_after={args.stale_after}s "
               f"max_restarts_per_hour={args.max_restarts_per_hour} "
               f"dry_run={bool(getattr(args, 'dry_run', False))}")
    launch_times = []          # 最近 1 小时拉起时间戳（防循环）
    cooldown_until = 0.0       # 冷却截止时间戳（超过上限后 3600s 不拉起）
    bootstrap_done = False     # 全新环境（无存档）首次拉起标记

    while True:
        now = time.time()
        age = _state_age()
        pids = _feed_pids()
        alive = bool(pids)

        if age is None:
            # 存档不存在（全新环境）：首次拉起一次，之后以存档新鲜度为准。
            if not bootstrap_done and not getattr(args, "dry_run", False):
                _watch_log("存档不存在（全新环境）→ 首次拉起 feed.py 初始化并创建存档")
                _relaunch_feed()
                launch_times.append(now)
                bootstrap_done = True
            elif not bootstrap_done:
                _watch_log("存档不存在（全新环境）→ dry-run：本应首次拉起（已跳过）")
                bootstrap_done = True
            else:
                _watch_log(f"存档仍不存在（距上次拉起已等待），不再反复拉起")
            time.sleep(args.interval)
            continue

        age_s = int(age)
        stale = age_s > args.stale_after
        if alive:
            _watch_log(f"检查：feed.py 运行中（PID {pids}），存档年龄 {age_s}s"
                       f"{'（陈旧）' if stale else '（新鲜）'}")
        else:
            _watch_log(f"检查：feed.py 未运行，存档年龄 {age_s}s"
                       f"{'（陈旧）' if stale else '（新鲜，无需拉起）'}")

        if not alive and stale:
            # 过滤超过 1 小时的时间戳（防循环窗口）
            launch_times[:] = [t for t in launch_times if now - t < 3600]
            if now < cooldown_until:
                remain = int(cooldown_until - now)
                _watch_log(f"冷却中（还剩 {remain}s），跳过拉起")
            elif len(launch_times) >= args.max_restarts_per_hour:
                cooldown_until = now + 3600
                _watch_log(f"告警：1 小时内已拉起 {len(launch_times)} 次，"
                           f"超过上限 {args.max_restarts_per_hour}，进入 3600s 冷却")
            else:
                if getattr(args, "dry_run", False):
                    _watch_log("dry-run：本应拉起 feed.py（已跳过）")
                else:
                    _relaunch_feed()
                    launch_times.append(time.time())
        time.sleep(args.interval)


def load_dolphin(dev=None, fed_state_path=None, birth_path=None,
                 old_state_path=None, cold_path=None, probe_path=None):
    """加载蓝蓟智能：优先 B 路线部署存档 fed_state.pt，回退旧逻辑并明示未部署状态。

    返回 (d, info)。info.deployed=True 表示读到的是部署存档（B 路线真实状态）；
    False 时 info.note 必须给出回退原因，报告据此显式标注"未部署状态"。
    cold_path / probe_path 供测试注入隔离路径；生产默认 None（用真实资产）。
    """
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    fed = fed_state_path or FED_STATE
    info = {"deployed": False, "path": None, "version": None, "saved_at": None,
            "learn_total_arch": None, "note": ""}
    if os.path.exists(fed):
        try:
            ck = torch.load(fed, map_location="cpu", weights_only=True)
            cfg = Config(**ck["cfg"])  # 身体以存档为准（B 路线无胎教基座可依赖）
            d = Dolphin(cfg=cfg, device=dev, probe_path=probe_path)
            if cold_path is not None:  # 测试隔离口：替换后 load 才开始恢复记忆
                d.memory = MemoryStore(cold_path=cold_path)
            d.load(fed)  # G7 卷面校验在 load 内先行；不匹配会抛 RuntimeError
            saved_at = _fmt_time(os.path.getmtime(fed))
            info.update(deployed=True, path=fed, version=ck.get("version"),
                        learn_total_arch=ck.get("learn_total"), saved_at=saved_at,
                        note=f"B 路线部署存档（v{ck.get('version')}，"
                             f"存档于 {saved_at}）")
            return d, info
        except Exception as e:
            info["note"] = (f"[警报] 部署存档 {fed} 加载失败"
                            f"（{type(e).__name__}: {e}）→ 回退旧逻辑。"
                            f"以下不是 B 路线真实状态！")
            print(f"[查房]{info['note']}", file=sys.stderr)
    else:
        info["note"] = f"未部署状态：{fed} 不存在——以下为胎教基座/旧存档，非 B 路线真实状态"

    # —— 旧逻辑回退（升级前行为）：birth.pt 出生 + 旧 state.pt ——
    birth = birth_path or BIRTH
    old = old_state_path or OLD_STATE
    if os.path.exists(birth):
        d = Dolphin.from_birth(birth, device=dev, probe_path=probe_path)
    else:
        d = Dolphin(device=dev, probe_path=probe_path)
        info["note"] += "（birth.pt 亦不在场，按随机初始化出生）"
    if cold_path is not None:
        d.memory = MemoryStore(cold_path=cold_path)
    if os.path.exists(old):
        try:
            d.load(old)
            info["note"] += f"（已叠加旧存档 {old}）"
        except Exception as e:
            info["note"] += f"（旧存档 {old} 加载失败：{type(e).__name__}: {e}）"
    return d, info


def report(d, title, info=None):
    info = info or {}
    print(f"\n===== {title} =====")
    print(f"状态来源：{info.get('note') or '未标注（load_dolphin 未提供 info）'}")
    cfg = d.cfg
    total, emb = param_breakdown(d.awake().model)
    print(f"身体：{cfg.n_layers} 层 × {cfg.n_heads} 头 × {cfg.d_model} 维，"
          f"上下文 {cfg.block_size} 字节（约 {cfg.block_size // 3} 个汉字）")
    print(f"参数 {total / 1e6:.2f}M（字库嵌入 {emb / 1e6:.2f}M + 变换器 {(total - emb) / 1e6:.2f}M）")
    print(f"状态：第 {d.cycle} 周期  醒脑={d.awake().name}  传输预算={d.budget:.3f}"
          f"  睡眠阈值={d.buffer.sleep_threshold:.2f}  学习率={d.awake().opt.param_groups[0]['lr']:.2e}")
    for h in d.h:
        role = "醒脑（服务中）" if h is d.awake() else "睡脑（待训）"
        print(f"  半球{h.name} [{role}] 探测集损失 {d.probe_loss(h):.4f}")
    kinds = {}
    for e in d.memory.entries:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    try:
        d.memory._ensure_cold()  # 冷层影子索引惰性加载（与 resurrect/retrieve 同一入口）
        cold_n = len(d.memory.cold_entries)
    except OSError as e:
        cold_n = -1
        print(f"  [警告] 冷层不可读：{e!r}")
    print(f"记忆库：热层 {len(d.memory.entries)} 条 {kinds}  冷层 {cold_n} 条")
    cur = d.feed_cursor
    if isinstance(cur, dict):
        srcs = cur.get("sources") or []
        print(f"喂食游标：源 {cur.get('source_idx')}/{len(srcs)}"
              f"  record_idx={cur.get('record_idx')}"
              f"  sources={list(srcs)}（随档续喂）")
    else:
        print("喂食游标：无（存档未含游标——v2 旧档或尚未正式喂食）")
    lt = info.get("learn_total_arch")
    if lt is not None:
        print(f"总喂入 learn_total：{lt}（存档口径）")
    else:
        print(f"总喂入 learn_total：{d.learn_total}"
              f"（本次查房会话计数；存档未含累计口径，feed 存档 payload 暂无此字段）")


def samples(d, prompts, n=60):
    for q in prompts:
        resp, _ = d.serve(q, max_new=n, temperature=0.7)
        print(f"  问：{q}\n  答：{resp.replace(chr(10), ' ')[:60]!r}")


def main_cli():
    ap = argparse.ArgumentParser(description="查房 / 独立看门狗（ward）")
    ap.add_argument("--watch", action="store_true",
                    help="常驻看门狗模式：周期性检查 feed.py 进程与存档新鲜度，"
                         "必要时自动拉起 feed.py（不破坏原查房功能）")
    ap.add_argument("--interval", type=int, default=60,
                    help="看门狗检查间隔秒数（默认 60）")
    ap.add_argument("--stale-after", type=int, default=600,
                    help="存档 mtime 超过该秒数视为陈旧（默认 600）")
    ap.add_argument("--max-restarts-per-hour", type=int, default=5,
                    help="1 小时内最大拉起次数，超过进入 3600s 冷却（默认 5）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印检查结果，不实际拉起 feed.py（验证用）")
    args = ap.parse_args()

    if args.watch:
        watch_loop(args)
        return

    # —— 原查房功能（零参数运行，保持不变）——
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, info = load_dolphin(dev)
    report(d, "查房 · 蓝蓟智能当前状态", info)
    print("\n----- 生成输出示例（字节级模型，观察结构）-----")
    samples(d, ["心脏的功能", "感冒了怎么办", "烫伤后第一步", "人体最大的器官是"])


if __name__ == "__main__":
    main_cli()
