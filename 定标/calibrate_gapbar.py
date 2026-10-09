"""gap̄ 稳态分布定标（监督审计 R1-b，2026-10-05）。

义务来源：监督审计 R1"设定点带'稳态即越带' + '实测'引用无工件"——life.py 的
XI_HI/GAP_BORN 注释与架构设计引用"真实身体稳态 gap̄"，但分布工件从未归档，
"实测"引用不可复核。本脚本在 57M 真实身体（dolphin/fed_state.pt，B 路线真实
部署档）+ 真实数据流上测 gap̄ 的稳态分布，工件归档为 定标/gapbar_定标结果.json。

两种取样：
  A. 存档挖掘（CPU，零风险，默认包含）：fed_state.pt + dolphin/archive/ 的
     滚动备份（真实部署在不同周期落的档）→ 各周期两半球的 gap̄ 与全部
     可动刀位的逐位 gap。vitals 的 stable_act 是半衰期 240 步的 EMA，每份
     存档代表"落档时刻前约 2 个周期的稳态水平"。
  B. 现跑周期（GPU，--live N）：从 fed_state.pt 载入真实身体，重复喂同一批
     已喂过的真实记录（cmath 呼吸切片——稳态协议：无新需求、无结构手术
     干扰），跑 N 个短周期（steps=60，测量受限选择；stable EMA 半衰期 240
     步，账本主体仍是 67 个真实周期的历史），逐周期从恒温器账本读 gap̄。
     全程**不存档**——生产资产零写入（收尾 sha256 断言，任一变化即失败）。

口径（与 life.py/_gap_bar、vitals.SiteLedger.utilization_indices 一致）：
  gap(site) = (med − bottom-5% 均值) / med（stable 激活，probation 豁免池）
  gap̄       = 可动刀位（mlp_hidden / attn_out）的均值

复跑：
  PY="python"
  $PY 定标/calibrate_gapbar.py                 # 模式 A（CPU，秒级）
  $PY 定标/calibrate_gapbar.py --live 2        # 模式 A+B（GPU，约 1.5 分钟）
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- 冷层隔离（防线①，问题记录后加固）----
# 首轮现跑实测发现：live 周期里 memory._demote 对**新文本**是 append-only 直写
# 生产 memory_cold.jsonl（1206→1208 行），不存档也落盘。本定标的修复
# 同 smoke_test.py P2-1：MemoryStore 默认解析指向生产冷层时一律改写进沙箱，
# 现跑的降级/复活全部落在沙箱文件里，生产冷层零触碰（sha256 哨兵仍在收尾兜底）。
from dolphin.memory import MemoryStore  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_gapbar_sandbox_")

_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = os.path.join(_SANDBOX, "memory_cold.jsonl")
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init

ROOT = _ROOT
FED_STATE = os.path.join(ROOT, "dolphin", "fed_state.pt")
OUT_JSON = os.path.join(ROOT, "定标", "gapbar_定标结果.json")

# 生产资产红线清单：定标全程只读，收尾逐一断言 sha256 不变
GUARDED = [os.path.join(ROOT, "probes", "probe.txt"),
           os.path.join(ROOT, "dolphin", "memory_cold.jsonl"),
           os.path.join(ROOT, "dolphin", "state.pt"),
           os.path.join(ROOT, "dolphin", "fed_state.pt")]
# 现跑协议喂的呼吸切片源（真实数据流；与呼吸实验阶段 1 同源）
LIVE_SOURCE = "cmath"
LIVE_SLICE = 24          # 喂入条数/周期（呼吸实验阶段 1 同量级）
LIVE_STEPS = 60          # 训练步数/周期（120 的一半——测量受限，见模块 docstring）


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gap_of_site(led):
    """SiteLedger 状态 dict → utilization_gap（vitals.SiteLedger.utilization_indices
    同公式；probation 豁免池、stable 口径）。返回 None 表示无信号（冷账本）。"""
    C = led["C"]
    prob = led.get("probation") or [0] * C
    pool = [c for c in range(C) if prob[c] == 0]
    if not pool:
        return None
    vals = sorted(led["stable_act"][c] for c in pool)
    n = len(vals)
    med = vals[n // 2]
    if med <= 1e-12:
        return None
    k = max(1, int(round(0.05 * n)))
    tail = sum(vals[:k]) / k
    return (med - tail) / med


def snapshot_gaps(life_state):
    """life 状态 dict → ({hname: gap̄}, {hname: {site: gap}})（只读可动刀位）。"""
    gapbars, sites = {}, {}
    for hn, vs in sorted((life_state.get("vitals") or {}).items()):
        per = {}
        for key, led in sorted((vs.get("sites") or {}).items()):
            if ".new." in key or not key.endswith(("mlp_hidden", "attn_out")):
                continue
            g = gap_of_site(led)
            if g is not None:
                per[key] = round(g, 4)
        if per:
            gapbars[hn] = round(sum(per.values()) / len(per), 4)
            sites[hn] = per
    return gapbars, sites


def quantile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def stats_of(vals):
    if not vals:
        return None
    return {"n": len(vals), "min": round(min(vals), 4),
            "median": round(quantile(vals, 0.50), 4),
            "p90": round(quantile(vals, 0.90), 4),
            "max": round(max(vals), 4)}


def mine_archives():
    """模式 A：fed_state.pt + archive 滚动备份 → 各周期 gap̄ 样本。"""
    import torch
    paths = [FED_STATE] + sorted(glob.glob(os.path.join(ROOT, "dolphin", "archive",
                                                        "fed_state*.bak")))
    samples = []
    for p in paths:
        st = torch.load(p, map_location="cpu", weights_only=False)
        life = st.get("life") or {}
        gapbars, sites = snapshot_gaps(life)
        if not gapbars:
            continue  # 旧格式/空账本档：如实跳过
        rel = os.path.relpath(p, ROOT)
        samples.append({"source": rel, "cycle": st.get("cycle"),
                        "gapbar_per_hemisphere": gapbars,
                        "per_site": sites})
        print(f"  [挖掘] {rel}  cycle={st.get('cycle')}  gap̄={gapbars}")
    return samples


def live_cycles(n_cycles):
    """模式 B：真实身体 + 真实数据流（重复呼吸切片——稳态协议）现跑 N 周期。"""
    import itertools

    import torch
    from dolphin.datasets import iter_source, serialize
    from dolphin.dolphin import Dolphin
    from dolphin.model import Config

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [现跑] 设备 {dev}：载入 {FED_STATE} …")
    t0 = time.monotonic()
    d = Dolphin(cfg=Config(), device=dev)
    d.load(FED_STATE)
    print(f"  [现跑] 载入完成（{time.monotonic() - t0:.1f}s）cycle={d.cycle} "
          f"params={sum(p.numel() for p in d.h[0].model.parameters()):,}/半球")
    # 呼吸切片（cmath 前段真实记录；重复喂——稳态协议，无新需求不排手术）
    texts = []
    for rec in itertools.islice(iter_source(LIVE_SOURCE), 0, LIVE_SLICE):
        t = serialize(rec)
        if t:
            texts.append(t)
    assert len(texts) >= 8, "呼吸切片过少"
    samples = []
    for i in range(1, n_cycles + 1):
        tc0 = time.monotonic()
        for t in texts:
            d.learn(t, source="gapbar_live_slice")
        rep = __import__("dolphin.life", fromlist=["run_cycle"]).run_cycle(
            d, steps=LIVE_STEPS, feeding=True, verbose=False)
        ctl = d.life_ctl
        gapbars, sites = snapshot_gaps(ctl.to_state())
        entry = {"source": f"live#{i}", "cycle": rep.get("cycle"),
                 "steps": LIVE_STEPS, "seconds": round(time.monotonic() - tc0, 1),
                 "gapbar_per_hemisphere": gapbars, "per_site": sites,
                 "passed": rep.get("passed"),
                 "margin": rep.get("gate_margin"),
                 "action": (rep.get("body") or {}).get("action", {}).get("kind"),
                 "xi_hi_eff": (rep.get("body") or {}).get("setpoint", {}).get("xi_hi")}
        samples.append(entry)
        print(f"  [现跑#{i}] {entry['seconds']}s  gap̄={gapbars}  "
              f"体检={rep.get('probe_new')}  margin={rep.get('gate_margin')}  "
              f"动作={entry['action']}  xi_hi_eff={entry['xi_hi_eff']}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0, metavar="N",
                    help="现跑 N 个真实周期（GPU；0=只做存档挖掘）")
    args = ap.parse_args()

    before = {p: sha256(p) for p in GUARDED if os.path.exists(p)}
    print("== gap̄ 稳态分布定标 ==")
    print("[模式 A] 存档挖掘（fed_state.pt + archive 滚动备份）：")
    mined = mine_archives()

    live = []
    if args.live > 0:
        print(f"[模式 B] 现跑 {args.live} 周期（真实身体 + 重复呼吸切片，不存档）：")
        live = live_cycles(args.live)

    # —— 汇总统计 ——
    hemi_vals = []   # 每半球每快照一个 gap̄ 样本（稳态分布的主体）
    live_vals = []
    site_vals = []   # 逐位 gap（跨位分布）
    for s in mined:
        hemi_vals += list(s["gapbar_per_hemisphere"].values())
        for hn, per in s["per_site"].items():
            site_vals += list(per.values())
    for s in live:
        live_vals += list(s["gapbar_per_hemisphere"].values())
        for hn, per in s["per_site"].items():
            site_vals += list(per.values())

    out = {"date": "2026-10-05",
           "purpose": "监督审计 R1-b：XI_HI/GAP_BORN 设定点带的稳态分布定标工件",
           "metric": "gap=(med−bottom5%均值)/med（stable 口径）；gap̄=可动刀位均值",
           "body": "57M 真实部署身体（fed_state.pt，B 路线，cycle 67）",
           "archive_samples": mined,
           "live_samples": live,
           "stats": {"gapbar_稳态分布_存档挖掘": stats_of(hemi_vals),
                     "gapbar_稳态分布_现跑": stats_of(live_vals) if live else None,
                     "gapbar_合并": stats_of(hemi_vals + live_vals),
                     "逐位gap_合并": stats_of(site_vals)},
           "notes": [
               "XI_HI=0.20 的律定下限取自'健康周转余量 5-20%'类比；本工件证明真实"
               "身体稳态 gap̄ 远高于它——带必须随系统自身稳态分布上浮（life.py "
               "_xi_hi_eff：max(XI_HI, gap̄ 滞后滚动 p90)，XI_P=0.90）。",
               "样本量如实声明：存档挖掘样本来自 2 个真实周期（62/67）× 2 半球；"
               "p90 由小样本取的是分布上沿的保守估计，_online_ 的浮动带会在运行中"
               "以同样口径（滚动 p90）持续自校准，不依赖本工件的绝对数。",
               "现跑协议：重复喂同一批真实记录（cmath 呼吸切片，稳态语义——无新"
               "需求），冷层走沙箱（_demote 对新文本 append-only 直写，不隔离则"
               "碰生产冷层——2026-10-05 首轮现跑实测发现，已修复并加固）。",
           ]}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[工件] → {OUT_JSON}")
    print(f"[统计] gap̄（存档+现跑合并）：{out['stats']['gapbar_合并']}")
    print(f"[统计] 逐位 gap（合并）：{out['stats']['逐位gap_合并']}")

    # —— 红线断言：生产资产零触碰 ——
    changed = [p for p, h in before.items() if sha256(p) != h]
    if changed:
        raise SystemExit(f"红线违约：定标过程改动了生产资产 {changed}")
    print(f"[红线] 生产资产 sha256 开工/收工一致（{len(before)} 项）：零触碰 ✓")


if __name__ == "__main__":
    main()
