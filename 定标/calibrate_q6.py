# -*- coding: utf-8 -*-
"""Q6 定标脚本——测量两个科学未知数（M4 动工前的最后测量，只测量、只定标）。

本脚本不创建 vitals.py / surgery.py / life.py，不实现任何 M4 架构代码；
不改 dolphin/ 与 tests/ 的任何既有文件；探测集只读不训（律 L8）。

实验 1（exp1 / exp1-analysis）：账本估计方差
  57M 真身体（Config(d_model=768, n_layers=8, n_heads=8, block_size=256)，随机
  初始化）+ logiqa 前 200 条 serialize 拼成的真实字节流，按生产睡眠配方
  （trainer_cycle：AdamW lr 5e-5 wd 0.01 + grad-clip 1.0，CE+KD(kd_alpha=0.5,
  kd_T=2.0) 锚定真实数据，batch 1×256 随机窗口）连续训练 240 步（=两个 120 步
  周期）。期间用 hook 逐步采集：
    - 每层每通道激活 L1 范数（ReDo 式利用率的原料）；
    - 每层每通道一阶 Taylor 项（梯度×激活；signed 与 abs 两个口径）；
    - 每层残差流有效秩（RankMe 式奇异值熵）：每步 svdvals + 两个 120 步 Gram
      （堆叠矩阵奇异值的精确累计，每步去均值）。
  通道口径 = 各 Linear 的输入激活维度（手术对象是"单元"）：
    b{i}.ln1_out(768) / b{i}.attn_out(768) / b{i}.ln2_out(768) / b{i}.mlp_hidden(3072)
  每块 5376 通道，8 块共 43008 通道。
  离线分析（exp1-analysis）：每周期内 5 次随机 60/60 对半切分（共 10 份对照）+
  120/120 跨周期切分，给出通道间相对波动中位数/90 分位、半侧秩相关（Spearman）、
  240 步趋势漂移、滞后一阶自相关、休眠占比稳定性（层内 1% 分位法 + ReDo 式
  1%/3%×max 法）、有效秩波动，以及 EMA 半衰期扫描。

实验 2（exp2-pre / exp2-slice / exp2-summary）：探测集噪声底
  生产探测集 probes/probe.txt（61889 字节 → 242 块全卷，只读）逐块 NLL：
    - 同权重连续 evaluate 两次（确定性验证，应严格 0 差异）；
    - bootstrap 1000 重采样 → 均值 NLL 的 SE 与 95% CI；
    - 3 组数据切片（logiqa 记录 0-65 / 66-131 / 132-199）× 3 个随机种子（窗口
      采样种子 11/22/33）：每个 run 从同一随机初始化出生，按生产损失训练 120 步，
      用生产 gate_decision（配对判决带 K_MARGIN=2.0）判决，并测：
      配对 SE=std(d)/sqrt(n) 的实测数量级、bootstrap 判决翻转率、配对 vs 非配对
      SE（共同波动消除了多少）、跨种子 post-post 配对差（纯种子噪声对照）。

用法（GPU 单命令 ≤ 2 分钟，可分多次跑）：
  python 定标/calibrate_q6.py exp1
  python 定标/calibrate_q6.py exp1-analysis
  python 定标/calibrate_q6.py exp2-pre
  python 定标/calibrate_q6.py exp2-slice --k 0   # 再跑 --k 1 / --k 2
  python 定标/calibrate_q6.py exp2-summary
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time

# 3GB 卡 + WDDM：可扩展段分配器减少碎片（必须在 import torch 前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from dolphin.model import ByteTransformer, Config  # noqa: E402
from dolphin.datasets import iter_source, serialize  # noqa: E402
from dolphin import probe as probe_mod  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = os.path.join(WORKDIR, "定标")
PROBE_PATH = os.path.join(WORKDIR, "probes", "probe.txt")

# 生产身体与生产睡眠配方（feed.make_cfg("57") + trainer_cycle），只测量不改动
CFG = Config(d_model=768, n_layers=8, n_heads=8, block_size=256)
DEV = "cuda" if torch.cuda.is_available() else None
LR = 5e-5           # adapt.py 锚点
WD = 0.01           # Hemisphere AdamW weight_decay
KD_ALPHA = 0.5      # trainer_cycle 默认
KD_T = 2.0
CYCLE = 120         # 生产默认 --sleep-steps 120
N_RECORDS = 200     # logiqa 前 200 条
INIT_SEED_EXP2 = 2024   # 实验 2 的出生种子（各 run 共用，配对设计隔离种子效应）
SLICE_BOUNDS = [(0, 66), (66, 132), (132, 200)]
SLICE_SEEDS = (11, 22, 33)
N_BOOT = 1000

SITES_PER_BLOCK = [  # (语义名, 对应 Linear) —— 通道 = 该 Linear 的输入激活维度
    ("ln1_out", "attn.qkv"),     # LayerNorm1 输出 → 注意力 QKV 输入（残差流）
    ("attn_out", "attn.proj"),   # 注意力输出 → attn.proj 输入
    ("ln2_out", "mlp.fc"),       # LayerNorm2 输出 → MLP fc 输入（残差流）
    ("mlp_hidden", "mlp.proj"),  # MLP 隐层（GELU 后）→ mlp.proj 输入
]


def die_if_no_gpu():
    if DEV is None:
        sys.exit("[定标] CUDA 不可用——本测量必须在真实 GPU 身体上做（交接文档 §8）")


def log(msg):
    print(f"[定标 +{time.monotonic():.1f}] {msg}", flush=True)


def load_records():
    """logiqa 前 200 条 serialize（与 feed.py 同一解析与序列化管线）。"""
    recs = []
    for rec in iter_source("logiqa"):
        s = serialize(rec)
        if s:
            recs.append(s)
        if len(recs) >= N_RECORDS:
            break
    return recs


def records_stream(records):
    return ("\n".join(records)).encode("utf-8")


def to_device_stream(stream: bytes):
    return torch.tensor(list(stream), dtype=torch.long, device=DEV)


def birth_pair(init_seed):
    """出生一对同源半球：student（可训）+ teacher（冻结醒脑）。"""
    torch.manual_seed(init_seed)
    student = ByteTransformer(CFG).to(DEV)
    teacher = ByteTransformer(CFG).to(DEV)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
    return student, teacher, opt


def production_step(student, teacher, opt, bt, s0):
    """trainer_cycle ⑤ 的单步复刻（CE+KD 锚定真实数据，batch 1×256，clip 1.0）。"""
    blk = CFG.block_size
    x = bt[s0:s0 + blk].unsqueeze(0)
    y = bt[s0 + 1:s0 + blk + 1].unsqueeze(0)
    with torch.no_grad():
        t_logits, _ = teacher(x)
    s_logits, _ = student(x)
    ce = F.cross_entropy(s_logits.reshape(-1, CFG.vocab), y.reshape(-1))
    p_t = F.softmax(t_logits / KD_T, dim=-1)
    kd = (KD_T ** 2) * F.kl_div(
        F.log_softmax(s_logits / KD_T, dim=-1), p_t, reduction="batchmean")
    loss = (1 - KD_ALPHA) * ce + KD_ALPHA * kd
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    return ce.item()


def meta_common():
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cfg": {"d_model": CFG.d_model, "n_layers": CFG.n_layers,
                "n_heads": CFG.n_heads, "block_size": CFG.block_size},
        "lr": LR, "wd": WD, "kd_alpha": KD_ALPHA, "kd_T": KD_T,
        "cycle_steps": CYCLE,
        "probe_sha256": hashlib.sha256(open(PROBE_PATH, "rb").read()).hexdigest(),
    }


# ============================ 实验 1：账本估计方差 ============================

def install_ledger_hooks(model, steps, svd_stride=1):
    """逐步采集账本原料：每通道 act/taylor（各 Linear 输入侧）+ 每层 ER。

    返回 (store, arrays, er_series, sv_store, grams, handles)。
    arrays[(metric, site)] = np[steps, C]；er_series=np[steps, n_layers]（未采步=NaN）；
    grams[li][half] = 该层残差流（每步去均值）X^T X 的精确累计（half=周期半）。

    实现注记（2026-10-04 首跑教训）：不能用张量钩子 x.register_hook(闭包捕获 x)——
    x→钩子表→闭包→x 构成引用环，环不破则 x 的整段反向子图（数十 MB/步）被滞留，
    120 步后显存 8.27GB 溢出 3GB 显存进 WDDM 共享内存（分页又拖慢 6 倍）。
    正确做法：模块级 register_full_backward_hook（每模块注册一次，无每步注册），
    前向钩子只把输入 x 暂存进 cur_x（此刻反向图本来就要用它），反向钩子用完即
    pop——零滞留、零引用环。
    """
    store = {"idx": 0}
    arrays = {}
    handles = []
    cur_x = {}
    site_mods = []
    for li, blk in enumerate(model.blocks):
        mods = {"attn.qkv": blk.attn.qkv, "attn.proj": blk.attn.proj,
                "mlp.fc": blk.mlp.fc, "mlp.proj": blk.mlp.proj}
        for site, mp in SITES_PER_BLOCK:
            site_mods.append((f"b{li}.{site}", mods[mp]))
    for name, lin in site_mods:
        C = lin.in_features
        for m in ("act", "tsign", "tabs"):
            arrays[(m, name)] = np.zeros((steps, C), dtype=np.float32)

    def make_fwd(name):
        def fwd(mod, inp, out):
            cur_x[name] = inp[0]
        return fwd

    def make_bwd(name):
        def bwd(mod, grad_input, grad_output):
            x = cur_x.pop(name, None)
            g = grad_input[0] if grad_input else None
            if x is None or g is None:
                return
            with torch.no_grad():
                xf = x.detach()
                p = g * xf
                trio = torch.stack([
                    xf.abs().mean(dim=(0, 1)),      # 激活 L1 范数（ReDo 式利用率原料）
                    p.mean(dim=(0, 1)),             # 一阶 Taylor 项（符号）
                    p.abs().mean(dim=(0, 1)),       # 一阶 Taylor 项（幅度）
                ]).cpu().numpy()
            idx = store["idx"]
            arrays[("act", name)][idx] = trio[0]
            arrays[("tsign", name)][idx] = trio[1]
            arrays[("tabs", name)][idx] = trio[2]
        return bwd

    for name, lin in site_mods:
        handles.append(lin.register_forward_hook(make_fwd(name)))
        handles.append(lin.register_full_backward_hook(make_bwd(name)))

    n_layers = CFG.n_layers
    grams = [torch.zeros(2, CFG.d_model, CFG.d_model, dtype=torch.float64, device=DEV)
             for _ in range(n_layers)]
    er_series = np.full((steps, n_layers), np.nan, dtype=np.float64)
    sv_store = np.full((steps, n_layers, CFG.block_size), np.nan, dtype=np.float32)

    def blk_factory(li):
        def fwd(mod, inp, out):
            # 整段 no_grad：Gram/svdvals 是测量，绝不能并进自动微分图
            # （否则 grams 被反向图链住，逐步滞留激活图——首跑 2845MB 峰值的根因）
            with torch.no_grad():
                idx = store["idx"]
                X = out[0]  # (T, d_model) 残差流
                Xc = X - X.mean(dim=0, keepdim=True)  # 每步去均值（去 DC）
                half = 0 if idx < steps // 2 else 1
                grams[li][half] += (Xc.T @ Xc).double()
                if idx % svd_stride == 0:
                    try:
                        sv = torch.linalg.svdvals(Xc)
                        sv_store[idx, li] = sv.cpu().numpy()
                        p = sv / sv.sum()
                        er_series[idx, li] = float(
                            torch.exp(-(p * p.clamp_min(1e-30).log()).sum()).item())
                    except Exception:
                        pass  # 留 NaN
        return fwd
    for li, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(blk_factory(li)))
    return store, arrays, er_series, sv_store, grams, handles


def gram_effective_rank(G):
    """RankMe 式有效秩：堆叠矩阵奇异值的熵 exp(-Σ p ln p)，p=σ/Σσ。"""
    Gs = ((G.detach() + G.detach().T) / 2.0).cpu().numpy()
    w, _ = np.linalg.eigh(Gs)
    s = np.sqrt(np.clip(w, 0.0, None))
    ss = s.sum()
    if ss <= 0:
        return float("nan")
    p = s / ss
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def cmd_exp1(args):
    die_if_no_gpu()
    t0 = time.time()
    records = load_records()
    stream = records_stream(records)
    bt = to_device_stream(stream)
    N = bt.numel()
    log(f"exp1 开始：{len(records)} 条 logiqa 记录 → 字节流 {N} B "
        f"(sha256 {hashlib.sha256(stream).hexdigest()[:16]})，训练 {2*CYCLE} 步")

    student, teacher, opt = birth_pair(1234)
    steps = 2 * CYCLE
    store, arrays, er_series, sv_store, grams, handles = install_ledger_hooks(
        student, steps, svd_stride=args.svd_stride)
    rng = random.Random(777)
    blk = CFG.block_size
    ce_hist = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        store["idx"] = i
        if i % 40 == 0:
            el = time.time() - t0
            proj = el / max(1, i) * steps if i else 0
            log(f"exp1 step {i}/{steps}  已用 {el:.1f}s  预计总 {proj:.0f}s")
        s0 = rng.randrange(0, N - blk - 1)
        ce_hist.append(production_step(student, teacher, opt, bt, s0))
    for h in handles:
        h.remove()
    log(f"exp1 训练完成 {time.time()-t0:.1f}s  ce 首末 {ce_hist[0]:.4f} → {ce_hist[-1]:.4f}"
        f"  显存峰值 {torch.cuda.max_memory_allocated()/2**20:.0f} MB")

    er_gram = {"c1": [gram_effective_rank(grams[li][0]) for li in range(CFG.n_layers)],
               "c2": [gram_effective_rank(grams[li][1]) for li in range(CFG.n_layers)]}
    del grams, student, teacher, opt
    torch.cuda.empty_cache()

    payload = {f"{m}__{name}": arr for (m, name), arr in arrays.items()}
    payload["er_series"] = er_series
    payload["sv_store"] = sv_store
    payload["ce_hist"] = np.array(ce_hist, dtype=np.float64)
    payload["meta_json"] = np.array(json.dumps({
        **meta_common(),
        "records": len(records), "stream_bytes": N,
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "init_seed": 1234, "window_seed": 777,
        "steps": steps, "svd_stride": args.svd_stride,
        "ce_first": ce_hist[0], "ce_last": ce_hist[-1],
        "er_gram_c1": er_gram["c1"], "er_gram_c2": er_gram["c2"],
        "elapsed_s": time.time() - t0,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
    }, ensure_ascii=False))
    path = os.path.join(OUT, "exp1_data.npz")
    np.savez(path, **payload)
    log(f"exp1 数据已存 {path}（{os.path.getsize(path)/2**20:.0f} MB），"
        f"Gram-ER c1={er_gram['c1']}\n          c2={er_gram['c2']}")


# ---------------------------- 实验 1 离线分析 ----------------------------

def ewm_mean(M, half_life):
    """半周期内的指数加权均值（half_life=None → 简单均值）。"""
    n = M.shape[0]
    if half_life is None:
        return M.mean(axis=0)
    w = 0.5 ** ((n - 1 - np.arange(n)) / half_life)
    w = w / w.sum()
    return w @ M


def rel_diff(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    return np.abs(a - b) / np.maximum(denom, 1e-12)


def rank1d(a):
    """平均秩（并列取平均）。"""
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
    avg = (cum[:-1] + cum[1:] + 1.0) / 2.0
    return avg[inv]


def spearman(a, b):
    ra, rb = rank1d(np.asarray(a, dtype=np.float64)), rank1d(np.asarray(b, dtype=np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def col_trend_spearman(M):
    """每列与步序号 [0..n) 的 Spearman 相关（漂移诊断）。"""
    n, C = M.shape
    order = np.argsort(M, axis=0, kind="stable")
    ranks = np.empty((n, C), dtype=np.float64)
    ranks[order, np.broadcast_to(np.arange(C), (n, C))] = \
        np.arange(1, n + 1, dtype=np.float64)[:, None]
    t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    rc = ranks - ranks.mean(axis=0, keepdims=True)
    num = (rc * tc[:, None]).sum(axis=0)
    den = np.sqrt((rc * rc).sum(axis=0) * (tc * tc).sum())
    return num / np.maximum(den, 1e-30)


def col_lag1(M):
    """每列滞后一阶自相关（步间独立还是漂移主导）。"""
    a = M[:-1].astype(np.float64)
    b = M[1:].astype(np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return num / np.maximum(den, 1e-30)


def jaccard(sa, sb):
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def jaccard_random(a, b, C):
    """同大小随机两集合的期望 Jaccard（显著性基线）。"""
    if a + b == 0:
        return 1.0
    inter = a * b / C
    union = a + b - inter
    return inter / union if union > 0 else 1.0


METRICS = {"act": "激活L1范数", "tabs": "Taylor|g×x|", "tsign": "Taylor符号g×x(取|半侧均值|)"}
HALF_LIVES = [None, 60, 30, 15]


def cmd_exp1_analysis(args):
    path = os.path.join(OUT, "exp1_data.npz")
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    steps, cycle = meta["steps"], meta["cycle_steps"]
    er_series = d["er_series"]
    sites = sorted({k.split("__", 1)[1] for k in d.files if "__" in k})
    metrics = sorted({k.split("__", 1)[0] for k in d.files if "__" in k})
    log(f"载入 {path}：steps={steps} cycle={cycle} sites={len(sites)} metrics={metrics}")

    rng = np.random.default_rng(2026)
    results = {"meta": meta, "per_site": {}, "aggregate": {}}

    # ---- 通道级指标：相对波动 / 秩相关 / 漂移 ----
    for metric in metrics:
        agg_rel = {f"split60_h{h}": [] for h in HALF_LIVES}
        per_site = {}
        for name in sites:
            M = d[f"{metric}__{name}"]
            e = {}
            spearmans = []
            for c in (0, 1):
                Mc = M[c * cycle:(c + 1) * cycle]
                for _ in range(5):
                    perm = rng.permutation(cycle)
                    A, B = Mc[perm[:cycle // 2]], Mc[perm[cycle // 2:]]
                    for h in HALF_LIVES:
                        rel = rel_diff(ewm_mean(A, h), ewm_mean(B, h))
                        agg_rel[f"split60_h{h}"].append(rel)
                    spearmans.append(spearman(A.mean(axis=0), B.mean(axis=0)))
            e["spearman_median"] = float(np.median(spearmans))
            ma, mb = M[:cycle].mean(axis=0), M[cycle:].mean(axis=0)
            rel = rel_diff(ma, mb)
            e["cross120"] = {"median": float(np.median(rel)),
                             "p90": float(np.quantile(rel, 0.9)),
                             "spearman": spearman(ma, mb)}
            e["trend_spearman_median"] = float(np.median(col_trend_spearman(M)))
            e["lag1_median"] = float(np.median(col_lag1(M)))
            # 决策相关子集：激活非休眠通道（半侧均值 ≥ 层内 1% 分位）的波动
            A, B = M[:cycle].mean(axis=0), M[cycle:].mean(axis=0)
            alive = A >= np.quantile(A, 0.01)
            rel_alive = rel_diff(ma[alive], mb[alive])
            e["cross120_alive"] = {"median": float(np.median(rel_alive)),
                                   "p90": float(np.quantile(rel_alive, 0.9))}
            per_site[name] = e
        for h in HALF_LIVES:
            allv = np.concatenate(agg_rel[f"split60_h{h}"])
            results["aggregate"].setdefault(metric, {})[f"split60_h{h}"] = {
                "median": float(np.median(allv)), "p90": float(np.quantile(allv, 0.9))}
        results["aggregate"][metric]["spearman_median"] = float(np.median(
            [per_site[n]["spearman_median"] for n in sites]))
        results["per_site"][metric] = per_site
        del agg_rel

    # ---- 休眠占比稳定性（act 口径） ----
    rng2 = np.random.default_rng(2026)
    dorm = {}
    for name in sites:
        M = d[f"act__{name}"]
        C = M.shape[1]
        e = {"C": int(C)}
        for scheme in ("q1", "redo1", "redo3"):
            js, sizes = [], []
            for c in (0, 1):
                Mc = M[c * cycle:(c + 1) * cycle]
                for _ in range(5):
                    perm = rng2.permutation(cycle)
                    ma = Mc[perm[:cycle // 2]].mean(axis=0)
                    mb = Mc[perm[cycle // 2:]].mean(axis=0)
                    if scheme == "q1":
                        sa = set(np.where(ma <= np.quantile(ma, 0.01))[0].tolist())
                        sb = set(np.where(mb <= np.quantile(mb, 0.01))[0].tolist())
                    else:
                        t = 0.01 if scheme == "redo1" else 0.03
                        sa = set(np.where(ma < t * ma.max())[0].tolist())
                        sb = set(np.where(mb < t * mb.max())[0].tolist())
                    js.append(jaccard(sa, sb))
                    sizes.append((len(sa), len(sb)))
            # 跨周期对照
            ma, mb = M[:cycle].mean(axis=0), M[cycle:].mean(axis=0)
            if scheme == "q1":
                sa = set(np.where(ma <= np.quantile(ma, 0.01))[0].tolist())
                sb = set(np.where(mb <= np.quantile(mb, 0.01))[0].tolist())
            else:
                t = 0.01 if scheme == "redo1" else 0.03
                sa = set(np.where(ma < t * ma.max())[0].tolist())
                sb = set(np.where(mb < t * mb.max())[0].tolist())
            e[scheme] = {
                "jaccard_median": float(np.median(js)),
                "jaccard_min": float(np.min(js)),
                "size_median": float(np.median([s[0] for s in sizes + [(len(sa), len(sb))]])),
                "size_cross": [len(sa), len(sb)],
                "jaccard_cross": jaccard(sa, sb),
                "jaccard_random_baseline": float(np.median(
                    [jaccard_random(a, b, C) for a, b in sizes])),
                "frac_of_channels_cross": [len(sa) / C, len(sb) / C],
            }
        dorm[name] = e
    results["dormancy"] = dorm

    # ---- 补充：ReDo 可行性诊断 + bottom-k% 休眠集稳定性 ----
    rng3 = np.random.default_rng(2027)
    dorm2 = {}
    for name in sites:
        M = d[f"act__{name}"]
        C = M.shape[1]
        # ReDo 比值动态范围：通道半侧均值 / 层内最大通道均值（越窄 → 固定相对阈值越不可行）
        ratios = []
        for c in (0, 1):
            ma = M[c * cycle:(c + 1) * cycle].mean(axis=0)
            ratios.append(ma / ma.max())
        ratios = np.concatenate(ratios)
        bk = {}
        for frac in (0.01, 0.05, 0.10):
            k = max(1, int(round(frac * C)))
            js = []
            for c in (0, 1):
                Mc = M[c * cycle:(c + 1) * cycle]
                for _ in range(5):
                    perm = rng3.permutation(cycle)
                    ma = Mc[perm[:cycle // 2]].mean(axis=0)
                    mb = Mc[perm[cycle // 2:]].mean(axis=0)
                    sa = set(np.argsort(ma)[:k].tolist())
                    sb = set(np.argsort(mb)[:k].tolist())
                    js.append(jaccard(sa, sb))
            ma, mb = M[:cycle].mean(axis=0), M[cycle:].mean(axis=0)
            sa = set(np.argsort(ma)[:k].tolist())
            sb = set(np.argsort(mb)[:k].tolist())
            bk[f"bottom{int(frac*100)}%"] = {
                "k": k, "jaccard_median": float(np.median(js)),
                "jaccard_cross": jaccard(sa, sb),
                "jaccard_random_baseline": jaccard_random(k, k, C),
            }
        dorm2[name] = {
            "ratio_to_max": {"min": float(ratios.min()), "p1": float(np.quantile(ratios, 0.01)),
                             "p5": float(np.quantile(ratios, 0.05)),
                             "p50": float(np.quantile(ratios, 0.50))},
            **bk,
        }
    results["dormancy_extended"] = dorm2

    # ---- 有效秩波动 ----
    er = er_series  # [steps, n_layers]，NaN=未采步
    er_res = {"per_block": {}, "gram_c1": meta["er_gram_c1"], "gram_c2": meta["er_gram_c2"]}
    rels_all = []
    for li in range(CFG.n_layers):
        rels = []
        # 按步索引切分（er_series 与步对齐，NaN=未采步）
        s_full = er[:, li]
        idx_ok = np.where(~np.isnan(s_full))[0]
        vals = s_full[idx_ok]
        for c in (0, 1):
            m_ok = (idx_ok >= c * cycle) & (idx_ok < (c + 1) * cycle)
            sub_idx, sub_val = idx_ok[m_ok], vals[m_ok]
            for _ in range(5):
                perm = rng2.permutation(len(sub_idx))
                n_half = len(sub_idx) // 2
                ea = sub_val[perm[:n_half]].mean()
                eb = sub_val[perm[n_half:2 * n_half]].mean()
                rels.append(abs(ea - eb) / max((abs(ea) + abs(eb)) / 2, 1e-12))
        g1, g2 = meta["er_gram_c1"][li], meta["er_gram_c2"][li]
        er_res["per_block"][f"b{li}"] = {
            "er_series_mean_c1": float(np.nanmean(er[:cycle, li])),
            "er_series_mean_c2": float(np.nanmean(er[cycle:, li])),
            "er_gram_c1": g1, "er_gram_c2": g2,
            "gram_rel_diff": abs(g1 - g2) / max((abs(g1) + abs(g2)) / 2, 1e-12),
            "split60_rel_median": float(np.median(rels)),
        }
        rels_all.append(rels)
    allv = np.concatenate(rels_all)
    er_res["split60_rel_pooled"] = {"median": float(np.median(allv)),
                                    "p90": float(np.quantile(allv, 0.9))}
    results["effective_rank"] = er_res

    out = os.path.join(OUT, "exp1_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log(f"exp1 分析完成 → {out}")

    # ---- 人类可读摘要 ----
    print("\n===== 实验 1 摘要（通道间相对波动，10 份 60/60 切分 + 120/120 跨周期）=====")
    print(f"{'指标':<16}{'h(半衰期)':>10}{'中位数':>10}{'p90':>10}")
    for metric in metrics:
        for h in HALF_LIVES:
            a = results["aggregate"][metric][f"split60_h{h}"]
            print(f"{METRICS[metric]:<16}{str(h):>10}{a['median']*100:>9.1f}%{a['p90']*100:>9.1f}%")
        a = results["aggregate"][metric].get("cross120")
    print("\n跨周期（120/120，含真实漂移）逐层：")
    for metric in ("act", "tabs"):
        print(f"-- {METRICS[metric]} --")
        for name in sites:
            e = results["per_site"][metric][name]
            print(f"  {name:<18} med={e['cross120']['median']*100:6.1f}%  "
                  f"p90={e['cross120']['p90']*100:6.1f}%  "
                  f"Spearman={e['cross120']['spearman']:5.2f}  "
                  f"存活通道 med={e['cross120_alive']['median']*100:5.1f}%")
    print("\n每步趋势（240 步 Spearman 中位数，|ρ|大=漂移主导）与滞后一阶自相关：")
    for metric in ("act", "tabs"):
        tr = [results["per_site"][metric][n]["trend_spearman_median"] for n in sites]
        lg = [results["per_site"][metric][n]["lag1_median"] for n in sites]
        print(f"  {METRICS[metric]:<16} trend med={np.median(tr):5.2f}  lag1 med={np.median(lg):5.2f}")
    print("\n休眠占比稳定性（Jaccard 中位数 / 随机基线；跨周期）：")
    for name in sites:
        e = dorm[name]
        print(f"  {name:<18} q1: {e['q1']['jaccard_median']:.2f}/{e['q1']['jaccard_random_baseline']:.2f}"
              f" (cross {e['q1']['jaccard_cross']:.2f}, n={e['q1']['size_cross']})  "
              f"redo1%: {e['redo1']['jaccard_median']:.2f}"
              f" (cross {e['redo1']['jaccard_cross']:.2f}, n={e['redo1']['size_cross']})  "
              f"redo3%: n={e['redo3']['size_cross']}")
    print("\n有效秩（RankMe ER）：")
    for li in range(CFG.n_layers):
        e = er_res["per_block"][f"b{li}"]
        print(f"  b{li}: Gram-ER c1={e['er_gram_c1']:.2f} c2={e['er_gram_c2']:.2f} "
              f"相对差={e['gram_rel_diff']*100:.1f}%  60/60切分中位={e['split60_rel_median']*100:.1f}%")
    print("\nReDo 可行性诊断（通道均值/层最大值 的分布，两个半周期合并）+ bottom-k% 稳定性：")
    for name in sites:
        e = dorm2[name]
        r = e["ratio_to_max"]
        b1, b5, b10 = e["bottom1%"], e["bottom5%"], e["bottom10%"]
        print(f"  {name:<18} min={r['min']:.3f} p1={r['p1']:.3f} p5={r['p5']:.3f} p50={r['p50']:.3f} | "
              f"Jaccard 1%={b1['jaccard_median']:.2f}(cross {b1['jaccard_cross']:.2f}) "
              f"5%={b5['jaccard_median']:.2f}(cross {b5['jaccard_cross']:.2f}) "
              f"10%={b10['jaccard_median']:.2f}(cross {b10['jaccard_cross']:.2f})")


# ============================ 实验 2：探测集噪声底 ============================

def evaluate_probe(model):
    chunks = probe_mod.load_chunks(PROBE_PATH, CFG.block_size)
    m, per, n = probe_mod.evaluate(model, chunks, DEV)
    return m, per, n, len(chunks)


def bootstrap_mean_se(arr, rng, n_boot=N_BOOT):
    n = len(arr)
    idx = rng.integers(0, n, (n_boot, n))
    boots = arr[idx].mean(axis=1)
    return float(boots.std(ddof=1)), [float(np.quantile(boots, 0.025)),
                                      float(np.quantile(boots, 0.975))]


def cmd_exp2_pre(args):
    die_if_no_gpu()
    torch.manual_seed(INIT_SEED_EXP2)
    model = ByteTransformer(CFG).to(DEV)
    t0 = time.time()
    m1, per1, n1, n_chunks = evaluate_probe(model)
    m2, per2, n2, _ = evaluate_probe(model)
    if per1 == per2:
        det = {"identical": True, "max_abs_diff": 0.0}
    else:
        det = {"identical": False,
               "max_abs_diff": max(abs(a - b) for a, b in zip(per1, per2))}
    log(f"确定性验证：两次逐块评估 {'逐位一致' if det['identical'] else '存在差异'}"
        f"（max|Δ|={det['max_abs_diff']:.2e}）")
    arr = np.array(per1, dtype=np.float64)
    rng = np.random.default_rng(7)
    se_boot, ci = bootstrap_mean_se(arr, rng)
    se_unpaired = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    payload = {**meta_common(),
               "mean_nll": m1, "n_chunks": n_chunks, "n_positions": n1,
               "per_chunk": per1,
               "se_bootstrap": se_boot, "ci95": ci,
               "se_parametric_unpaired": se_unpaired,
               "se_over_mean": se_boot / m1,
               "determinism": det,
               "elapsed_s": time.time() - t0}
    out = os.path.join(OUT, "exp2_pre.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    log(f"exp2-pre 完成 → {out}\n"
        f"  mean NLL={m1:.4f}  SE(bootstrap)={se_boot:.6f} ({se_boot/m1*100:.3f}% of mean)"
        f"  95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")


def cmd_exp2_slice(args):
    die_if_no_gpu()
    k = args.k
    pre_path = os.path.join(OUT, "exp2_pre.json")
    with open(pre_path, encoding="utf-8") as f:
        pre = json.load(f)
    pre_per = pre["per_chunk"]
    pre_arr = np.array(pre_per, dtype=np.float64)

    records = load_records()
    lo, hi = SLICE_BOUNDS[k]
    stream = records_stream(records[lo:hi])
    bt = to_device_stream(stream)
    N = bt.numel()
    blk = CFG.block_size
    log(f"exp2-slice {k}：logiqa 记录 {lo}:{hi} → 字节流 {N} B"
        f" (sha256 {hashlib.sha256(stream).hexdigest()[:16]})")

    t0 = time.time()
    runs = []
    posts = {}
    torch.cuda.reset_peak_memory_stats()
    for seed in SLICE_SEEDS:
        student, teacher, opt = birth_pair(INIT_SEED_EXP2)  # 与 pre 同一出生权重
        rng = random.Random(seed)  # 窗口采样种子 = 生产里唯一的有效随机源
        for i in range(CYCLE):
            production_step(student, teacher, opt, bt, rng.randrange(0, N - blk - 1))
        m_new, per_new, _, n_chunks = evaluate_probe(student)
        passed, detail = probe_mod.gate_decision(pre_per, per_new)  # 生产判决唯一实现
        post = np.array(per_new, dtype=np.float64)
        dd = pre_arr - post
        rngb = np.random.default_rng(1000 + seed)
        se_boot, _ = bootstrap_mean_se(dd, rngb)
        idxb = rngb.integers(0, len(dd), (N_BOOT, len(dd)))
        dboot = dd[idxb].mean(axis=1)
        eps = detail["gate_eps"]
        runs.append({
            "seed": seed,
            "mean_post": m_new,
            "mean_d": detail["gate_margin"], "eps": eps, "se_paired": detail["gate_se"],
            "passed": bool(passed),
            "se_bootstrap_paired": se_boot,
            "p_boot_flip_le0": float((dboot <= 0).mean()),
            "p_boot_flip_le_eps": float((dboot <= eps).mean()),
            "se_unpaired": float(np.sqrt(pre_arr.var(ddof=1) / len(pre_arr)
                                         + post.var(ddof=1) / len(post))),
            "corr_pre_post": float(np.corrcoef(pre_arr, post)[0, 1]),
            "mean_d_pct_of_mean": detail["gate_margin"] / pre["mean_nll"] * 100,
            "se_pct_of_mean": detail["gate_se"] / pre["mean_nll"] * 100,
            "ce_first_last": None,
        })
        posts[seed] = post
        log(f"  seed={seed}: probe {pre['mean_nll']:.4f} → {m_new:.4f}"
            f"  margin={detail['gate_margin']:.5f}  SE={detail['gate_se']:.5f}"
            f"  eps={eps:.5f}  {'通过' if passed else '拒绝'}"
            f"  P(boot≤0)={runs[-1]['p_boot_flip_le0']:.3f}")
        del student, teacher, opt
        torch.cuda.empty_cache()

    cross = []
    for a, b in itertools.combinations(SLICE_SEEDS, 2):
        dc = posts[a] - posts[b]
        cross.append({
            "seeds": [a, b], "mean_diff": float(dc.mean()),
            "se_paired": float(dc.std(ddof=1) / np.sqrt(len(dc))),
            "max_abs_chunk_diff": float(np.abs(dc).max()),
        })
    payload = {**meta_common(), "slice": k, "records": [lo, hi],
               "stream_bytes": N, "init_seed": INIT_SEED_EXP2,
               "seeds": list(SLICE_SEEDS), "steps": CYCLE,
               "mean_pre": pre["mean_nll"], "runs": runs, "cross_seed": cross,
               "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
               "elapsed_s": time.time() - t0}
    out = os.path.join(OUT, f"exp2_slice{k}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    log(f"exp2-slice {k} 完成 → {out}（{time.time()-t0:.1f}s，"
        f"显存峰值 {payload['peak_mem_mb']:.0f} MB）")


def cmd_exp2_summary(args):
    with open(os.path.join(OUT, "exp2_pre.json"), encoding="utf-8") as f:
        pre = json.load(f)
    slices = []
    for k in range(len(SLICE_BOUNDS)):
        p = os.path.join(OUT, f"exp2_slice{k}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                slices.append(json.load(f))
    out = {"pre": {kk: pre[kk] for kk in
                   ("mean_nll", "se_bootstrap", "ci95", "se_parametric_unpaired",
                    "se_over_mean", "n_chunks", "determinism")},
           "slices": [{kk: s[kk] for kk in ("slice", "mean_pre", "runs", "cross_seed")}
                      for s in slices]}
    all_runs = [r for s in slices for r in s["runs"]]
    ses = [r["se_paired"] for r in all_runs]
    margins = [r["mean_d"] for r in all_runs]
    summ = {
        "n_runs": len(all_runs),
        "se_paired_min_med_max": [float(min(ses)), float(np.median(ses)), float(max(ses))] if ses else None,
        "se_pct_of_mean_min_med_max": ([float(min(r['se_pct_of_mean'] for r in all_runs)),
                                        float(np.median([r['se_pct_of_mean'] for r in all_runs])),
                                        float(max(r['se_pct_of_mean'] for r in all_runs))] if all_runs else None),
        "margin_min_med_max": [float(min(margins)), float(np.median(margins)), float(max(margins))] if margins else None,
        "passed": sum(1 for r in all_runs if r["passed"]),
        "corr_pre_post_min": float(min(r["corr_pre_post"] for r in all_runs)) if all_runs else None,
        "se_unpaired_over_paired_med": (float(np.median([r["se_unpaired"] / max(r["se_paired"], 1e-12)
                                                          for r in all_runs])) if all_runs else None),
    }
    out["aggregate"] = summ
    p = os.path.join(OUT, "exp2_summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"exp2-summary → {p}")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description="Q6 定标：账本估计方差 + 探测集噪声底（只测量）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("exp1", help="240 步带 hook 训练，采集账本原料")
    s.add_argument("--svd-stride", type=int, default=1, help="每几步做一次 svdvals（Gram 不受影响）")
    s.set_defaults(fn=cmd_exp1)
    s = sub.add_parser("exp1-analysis", help="离线切分分析（纯 CPU）")
    s.set_defaults(fn=cmd_exp1_analysis)
    s = sub.add_parser("exp2-pre", help="探测集评估×2 + bootstrap 噪声底")
    s.set_defaults(fn=cmd_exp2_pre)
    s = sub.add_parser("exp2-slice", help="一组数据切片 × 3 种子训练 + 生产门控判决")
    s.add_argument("--k", type=int, required=True, choices=[0, 1, 2])
    s.set_defaults(fn=cmd_exp2_slice)
    s = sub.add_parser("exp2-summary", help="汇总实验 2")
    s.set_defaults(fn=cmd_exp2_summary)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
