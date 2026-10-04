"""候选题流接入——试卷池的真实候选供给侧（2026-10-04 堵"换卷必须人工"缺口）。

问题：rolling.rotate() 机制已就绪而 probes/probe_pool/ 为空——真实候选流
从未接入，换卷在自学习循环里依旧等于不存在（交接文档待办 #2/#5）。

候选来源（全部是"只许进探测体系、永不进训练粮"的评测 split，律 L8 前置；
与 probes/build_logic_probe.py 同源规则——同一解析器、同一防撞预过滤）：
  1. LLMEval-Logic base+hard（Z3 验证中文逻辑题）
  2. LogiQA-1.0 zh_test.txt / zh_eval.txt（与 LogiQA2 train 撞题剔除）
  3. CMATH cmath_test.jsonl（与 cmath_dev 撞题剔除）

入池管线：collect（复用 build_logic_probe.collect_candidates）→ 入池质检
（probes/qc.overlap_check：每条候选按 100 字符滑窗全量切片，与当前卷
probe.txt、退休档 retired/*.txt 做 ≥50 字符共享子串比对；--qc-grain 追加
与全部训练粮的比对，~9 亿字符单遍扫描约数分钟）→ 按条拒绝撞车候选 →
按 rolling 既有指针协议写入 probes/probe_pool/。

铁律与边界：
- probe.txt 本体在此绝不触碰（律 L8）；转正/退休只由 rolling.rotate 执行；
- 本模块对池只做一件事：追加新文件 cand_<时间戳>_<来源>.txt（字典序=入池
  序，天然排在既有指针之后，不干扰消费进度）；消费进度永远归 rolling 的
  pool_pointer.json，本模块绝不读改指针；
- 台账 probes/candidate_ledger.json 记录历次入池题干指纹：重复填充不重复
  入池；质检拒绝的条目不入台账——卷面滚动退休后它们可在下次填充重新受检；
- 候选也要与退休档比对：退休段可回流训练粮，若候选与退休段撞车而退休段
  回流了粮仓，等于从后门把卷面内容喂进训练（L8 违例），必须在入池前拦下；
- 入池质检与当前卷比对的道理：转正会把池内容接上卷面，若候选与卷内既有
  内容撞车，同一道题将在卷面出现两次，体检卷被自我抄袭污染。

用法：
  python probes/candidates.py --fill [--limit N] [--qc-grain]
  python probes/candidates.py --status
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # 包内导入（项目根在 path）与同目录脚本执行两种姿势都要能跑
    from probes.build_logic_probe import collect_candidates, TRAIN_SOURCES, grain_stream
    from probes.qc import overlap_check
    from probes import rolling
except ImportError:  # python probes/candidates.py 直跑时 probes/ 自身在 path
    from build_logic_probe import collect_candidates, TRAIN_SOURCES, grain_stream
    from qc import overlap_check
    import rolling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_DIR = rolling.POOL_DIR
PROBE_PATH = rolling.PROBE_PATH
RETIRED_DIR = rolling.RETIRED_DIR
LEDGER_PATH = os.path.join(ROOT, "probes", "candidate_ledger.json")

LEDGER_VERSION = 1


def item_text(q, a):
    """卷面条目格式——与 build_logic_probe.build_probe 完全一致（同源规则）：
    转正段接上卷面后与既有卷面浑然一体，体检评分无格式断层。"""
    return "问：" + q + "\n答：" + a + "\n\n"


def _stem_sha(q, a):
    """题干指纹（去重键）：q+answer 分隔哈希，防相邻拼接巧合。"""
    return hashlib.sha256((q + "\x00" + a).encode("utf-8")).hexdigest()


# ---------------- 台账（跨次填充去重 + 对账凭据） ----------------

def load_ledger(path=None):
    p = path or LEDGER_PATH
    if os.path.exists(p):
        try:
            led = json.load(open(p, encoding="utf-8"))
            if isinstance(led, dict) and isinstance(led.get("stems"), list):
                led.setdefault("version", LEDGER_VERSION)
                led.setdefault("runs", [])
                return led
        except (json.JSONDecodeError, OSError):
            pass  # 台账损坏按空账重建：宁可重复受检不可静默跳过质检
    return {"version": LEDGER_VERSION, "runs": [], "stems": []}


def save_ledger(ledger, path=None):
    p = path or LEDGER_PATH
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = p + ".writing"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ledger, f, ensure_ascii=False)
    os.replace(tmp, p)  # 原子写：崩溃不留半截账本


def ledger_report(path=None):
    """台账对账凭据：历次填充的入池/拒绝计数与累计题数。"""
    led = load_ledger(path)
    return {"fills": len(led["runs"]), "stems_total": len(led["stems"]),
            "runs": led["runs"]}


# ---------------- 候选流（同源规则复用） ----------------

def iter_candidate_items():
    """真实候选流：返回 (items, dropped)。

    items = [(来源名, 题面, 答案), ...]——与 build_logic_probe.collect_candidates
    同一解析与防撞预过滤（LogiQA2 train / cmath_dev 撞题剔除），保证候选从
    出厂起就不是训练粮。
    """
    return collect_candidates()


# ---------------- 入池 ----------------

def fill_pool(pool_dir=None, probe_path=None, retired_dir=None,
              ledger_path=None, limit=None, items=None, qc_grain=False,
              log=print):
    """解析候选 → 入池质检 → 按条拒绝撞车候选 → 追加进 probe_pool/。

    items=None 时走真实候选流；测试可注入合成 items（各自 (来源, 题, 答)）。
    返回报告 dict（parsed/qc_rejected/dup_run/dup_ledger/admitted/files/
    qc 摘要），供对账与测试断言。
    """
    pool_dir = pool_dir or POOL_DIR
    probe_path = probe_path or PROBE_PATH
    retired_dir = retired_dir or RETIRED_DIR
    ledger_path = ledger_path or LEDGER_PATH

    # ① 候选流
    if items is None:
        items, dropped = iter_candidate_items()
        if log:
            for k, v in dropped.items():
                if v:
                    log(f"[候选] 防撞预过滤剔除（{k}）：{v} 条")
    else:
        items = list(items)
        dropped = {}
    parsed = len(items)

    # ② 去重：台账（跨次）+ 本次（防同 run 重复进场）
    led = load_ledger(ledger_path)
    stems = set(led["stems"])
    seen_run = set()
    kept, dup_ledger, dup_run = [], 0, 0
    for src, q, a in items:
        h = _stem_sha(q, a)
        if h in stems:
            dup_ledger += 1
            continue
        if h in seen_run:
            dup_run += 1
            continue
        seen_run.add(h)
        kept.append((src, q, a, h))

    # ③ 入池质检（qc.overlap_check：滑窗全量切片，覆盖率 100%）
    #    比对来源 = 当前卷 + 退休档（转正撞车面）+（可选）全部训练粮（L8 加固）。
    #    当前卷/退休档是 KB 级小额文本：既进 gram 流式通道（抓 ≥50 字符局部
    #    撞车），也进全文逐字通道（抓短于 50 字符的整条候选原文撞车——CMATH
    #    短题在 gram 机制下天然失明，2026-10-04 生产首填实证补盲必要）。
    sources = []
    verbatim = []
    if os.path.exists(probe_path):
        probe_text = open(probe_path, encoding="utf-8").read()
        if probe_text:
            sources.append(("当前卷 probe.txt", probe_text))
            verbatim.append(("当前卷 probe.txt", probe_text))
    else:
        if log:
            log(f"[质检][警告] 当前卷 {probe_path} 不在场——本次无卷面可比，"
                f"转正前请确认卷面已就位")
    if os.path.isdir(retired_dir):
        for name in sorted(os.listdir(retired_dir)):
            fp = os.path.join(retired_dir, name)
            if os.path.isfile(fp):
                t = open(fp, encoding="utf-8", errors="replace").read()
                if t:
                    sources.append((f"退休档 {name}", t))
                    verbatim.append((f"退休档 {name}", t))
    if qc_grain:
        for name in TRAIN_SOURCES:  # 同 build_logic_probe.overlap_check 的来源表
            sources.append((f"训练粮:{name}", grain_stream(name)))

    texts = [item_text(q, a) for _, q, a, _ in kept]
    qc_rep = overlap_check(texts, sources, verbatim_against=verbatim, log=log) \
        if kept else {
        "passed": True, "coverage": 1.0, "windows_total": 0,
        "windows_checked": 0, "hit_items": []}
    rejected = set(qc_rep.get("hit_items") or [])

    # ④ 按条拒绝撞车候选，余者入池（limit 计 admitted 口径）
    stamp = time.strftime("%Y%m%d_%H%M%S")
    admitted, qc_rejected = [], 0
    by_src = {}
    for i, (src, q, a, h) in enumerate(kept):
        if i in rejected:
            qc_rejected += 1
            by_src.setdefault(src, {"admitted": 0, "qc_rejected": 0})["qc_rejected"] += 1
            continue
        if limit is not None and len(admitted) >= limit:
            break
        admitted.append((src, q, a, h))
        by_src.setdefault(src, {"admitted": 0, "qc_rejected": 0})["admitted"] += 1

    # ⑤ 写池：每来源一个文件 cand_<时间戳>_<来源>.txt（字典序=入池序）
    files = []
    total_bytes = 0
    os.makedirs(pool_dir, exist_ok=True)
    for src in by_src:  # dict 保序 = 候选流首现序
        rows = [it for it in admitted if it[0] == src]
        if not rows:
            continue
        fname = f"cand_{stamp}_{src}.txt"
        body = "".join(item_text(q, a) for _, q, a, _ in rows)
        with open(os.path.join(pool_dir, fname), "w", encoding="utf-8",
                  newline="\n") as f:
            f.write(body)
        files.append({"file": fname, "source": src, "items": len(rows),
                      "bytes": len(body.encode("utf-8"))})
        total_bytes += files[-1]["bytes"]

    # ⑥ 台账：只记已入池题干（被拒者下次填充重新受检）
    for src, q, a, h in admitted:
        stems.add(h)
    run_rec = {"stamp": stamp, "parsed": parsed, "dup_ledger": dup_ledger,
               "dup_run": dup_run, "qc_rejected": qc_rejected,
               "admitted": len(admitted), "qc_grain": qc_grain,
               "qc_coverage": qc_rep.get("coverage"),
               "sources": by_src}
    led["runs"].append(run_rec)
    led["stems"] = sorted(stems)
    if admitted:
        save_ledger(led, ledger_path)

    rep = {"parsed": parsed, "dup_ledger": dup_ledger, "dup_run": dup_run,
           "qc_rejected": qc_rejected, "admitted": len(admitted),
           "files": files, "bytes": total_bytes, "pool_dir": pool_dir,
           "qc_passed": qc_rep.get("passed"), "qc_coverage": qc_rep.get("coverage"),
           "qc_windows": qc_rep.get("windows_total"),
           "qc_verbatim_hits": qc_rep.get("verbatim_hits", 0),
           "dropped": dropped, "stamp": stamp}
    if log:
        log(f"[入池] 解析 {parsed} 条 → 台账重复 {dup_ledger} / 本轮重复 {dup_run}"
            f" / 质检拒绝 {qc_rejected} → 入池 {len(admitted)} 条"
            f"（{total_bytes / 1024:.1f} KB，{len(files)} 个文件）")
        for fi in files:
            log(f"  + {fi['file']}  {fi['items']} 条 / {fi['bytes']:,} B")
    return rep


# ---------------- 对账 ----------------

def status(live=True, ledger_path=None, pool_dir=None, probe_path=None,
           retired_dir=None, log=print):
    """候选源对账：实时解析各源条数 + 台账累计 + 池内现状（rolling.status）。"""
    pool_dir = pool_dir or POOL_DIR
    probe_path = probe_path or PROBE_PATH
    retired_dir = retired_dir or RETIRED_DIR

    rep = {"ledger": ledger_report(ledger_path)}
    if live:
        try:
            items, dropped = iter_candidate_items()
            per = {}
            for src, _, _ in items:
                per[src] = per.get(src, 0) + 1
            rep["live"] = {"total": len(items), "per_source": per,
                           "prefilter_dropped": dropped}
        except Exception as e:  # 数据盘不在场不拖垮对账
            rep["live"] = {"error": repr(e)}
    rep["rolling"] = rolling.status(probe_path=probe_path, pool_dir=pool_dir,
                                    retired_dir=retired_dir)
    if log:
        led = rep["ledger"]
        log(f"[候选源] 台账：{led['fills']} 次填充，累计入池 {led['stems_total']} 题指纹")
        lv = rep.get("live") or {}
        if "per_source" in lv:
            log("[候选源] 实时解析：" +
                "  ".join(f"{k}={v}" for k, v in lv["per_source"].items()) +
                f"  合计 {lv['total']} 题"
                + (f"（预过滤剔除 {lv['prefilter_dropped']}）"
                   if any(lv["prefilter_dropped"].values()) else ""))
        elif "error" in lv:
            log(f"[候选源] 实时解析不可用：{lv['error']}")
        r = rep["rolling"]
        log(f"[候选源] 池内：{r['pool_files']} 文件 / {r['pool_bytes_left']} B 待转正；"
            f"卷 {r['probe_bytes']} B（{r['probe_sections']} 段）；"
            f"退休 {r['retired_files']} 文件 / {r['retired_bytes']} B")
    return rep


def main():
    ap = argparse.ArgumentParser(description="候选题流接入试卷池（rolling 的供给侧）")
    ap.add_argument("--fill", action="store_true", help="解析候选→质检→入池")
    ap.add_argument("--status", action="store_true", help="候选源对账（默认）")
    ap.add_argument("--limit", type=int, default=None, help="本次最多入池条数")
    ap.add_argument("--qc-grain", action="store_true",
                    help="追加与全部训练粮的重叠比对（单遍扫描约数分钟，一次性填充建议开）")
    args = ap.parse_args()
    if args.fill:
        fill_pool(limit=args.limit, qc_grain=args.qc_grain)
    status()


if __name__ == "__main__":
    main()
