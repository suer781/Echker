"""Q2 难度对齐实验：CMATH grade 元数据档位 vs 模型 mean_nll 惊讶度。

只读实验：
- 不训练、不保存模型、不写任何生产文件。
- 模型：实验记录/发育实验归档_20261004/state.pt（v3 存档，直接 torch.load + ByteTransformer 加载权重）。
- 数据：O:/数据集/03_数学推理/CMATH/cmath_dev.jsonl（grade 1-6，各 100 条）。
- 方法：按 grade 分层随机抽样每组 50 条，序列化为 "问：...\n答：..." 后调用 model.mean_nll()。
- 统计：每组 mean/std/median；Spearman 秩相关（grade 1-6 vs nll）；Kruskal-Wallis 六组。
- 输出：打印汇总；逐样本结果写入系统临时目录 JSON（可复现）。

用法：python 调查/Q2_难度对齐实验.py [--seed 20261005] [--per-grade 50] [--device cuda|cpu]
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dolphin.model import ByteTransformer, Config  # noqa: E402

CMATH_PATH = "O:/数据集/03_数学推理/CMATH/cmath_dev.jsonl"
ARCHIVE_PATH = "实验记录/发育实验归档_20261004/state.pt"


def load_seed_model(path, device):
    """只读加载 v3 存档：直接取 cfg 与 awake 半球权重，不实例化生产 Dolphin。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**ck["cfg"])
    model = ByteTransformer(cfg).to(device)
    awake_idx = ck.get("awake_idx", 0)
    model.load_state_dict(ck["states"][awake_idx])
    return model, cfg, awake_idx


def load_cmath(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = " ".join(str(rec.get("question") or "").split())
            g = " ".join(str(rec.get("golden") or "").split())
            if q and g:
                rows.append({
                    "grade": int(rec["grade"]),
                    "question": q,
                    "golden": g,
                    "reasoning_step": rec.get("reasoning_step"),
                    "num_digits": rec.get("num_digits"),
                })
    return rows


def serialize(rec):
    return "问：" + rec["question"] + "\n答：" + rec["golden"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--per-grade", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model, cfg, awake_idx = load_seed_model(ARCHIVE_PATH, device)
    model.eval()
    t_load = time.time() - t0

    rows = load_cmath(CMATH_PATH)
    by_grade = {}
    for r in rows:
        by_grade.setdefault(r["grade"], []).append(r)

    rng = random.Random(args.seed)
    sample = []
    for g in sorted(by_grade):
        pool = by_grade[g]
        picked = rng.sample(pool, min(args.per_grade, len(pool)))
        for r in picked:
            text = serialize(r)
            data = text.encode("utf-8")
            nll = model.mean_nll(data, device)
            sample.append({
                "grade": g,
                "nll": nll,
                "text_len": len(data),
                "reasoning_step": r["reasoning_step"],
                "num_digits": r["num_digits"],
                "question": r["question"],
                "golden": r["golden"],
            })
    t_score = time.time() - t0 - t_load

    # ---------- 统计 ----------
    try:
        from scipy import stats
    except ImportError:
        print("scipy 不可用，仅输出描述统计", file=sys.stderr)
        stats = None

    print(f"模型: {ARCHIVE_PATH}")
    print(f"  cfg: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"block_size={cfg.block_size} vocab={cfg.vocab}")
    print(f"  awake_idx={awake_idx}  设备={device}  加载耗时={t_load:.2f}s  打分耗时={t_score:.2f}s")
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数（单半球）: {trainable}")

    print("\n=== 每组描述统计（grade 1-6）===")
    groups = {}
    for g in sorted(set(s["grade"] for s in sample)):
        vals = [s["nll"] for s in sample if s["grade"] == g]
        groups[g] = vals
        import statistics as st
        print(f"grade {g}: N={len(vals)} mean={sum(vals)/len(vals):.4f} "
              f"std={st.stdev(vals):.4f} median={st.median(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f}")

    # 三级聚合（低1-2 / 中3-4 / 高5-6）
    def tier(g):
        return 0 if g <= 2 else (1 if g <= 4 else 2)

    print("\n=== 三级聚合（低年级1-2 / 中年级3-4 / 高年级5-6）===")
    import statistics as st
    tier_groups = {0: [], 1: [], 2: []}
    for s in sample:
        tier_groups[tier(s["grade"])].append(s["nll"])
    for t, vals in sorted(tier_groups.items()):
        print(f"tier {t}: N={len(vals)} mean={sum(vals)/len(vals):.4f} "
              f"std={st.stdev(vals):.4f} median={st.median(vals):.4f}")

    if stats is not None:
        # Spearman：grade 序数 vs nll（逐样本，n=300）
        gs = [s["grade"] for s in sample]
        ns = [s["nll"] for s in sample]
        rho, p_rho = stats.spearmanr(gs, ns)
        print(f"\n=== Spearman（grade 1-6 序数 vs nll，n={len(sample)}）===")
        print(f"rho = {rho:.4f}, p = {p_rho:.6g}")

        # 按 grade 均值再算一次 Spearman（聚合，n=6）
        gmeans = sorted((g, sum(groups[g]) / len(groups[g])) for g in groups)
        rho_m, p_rho_m = stats.spearmanr([x[0] for x in gmeans], [x[1] for x in gmeans])
        print(f"Spearman（每组均值聚合，n=6）: rho = {rho_m:.4f}, p = {p_rho_m:.6g}")

        # 三级聚合 Spearman
        ts = [tier(s["grade"]) for s in sample]
        rho_t, p_rho_t = stats.spearmanr(ts, ns)
        print(f"Spearman（三级 tier 0/1/2 vs nll）: rho = {rho_t:.4f}, p = {p_rho_t:.6g}")

        # Kruskal-Wallis：六组
        h6, p6 = stats.kruskal(*[groups[g] for g in sorted(groups)])
        print(f"\n=== Kruskal-Wallis（grade 1-6 六组）===")
        print(f"H = {h6:.4f}, p = {p6:.6g}")

        # Kruskal-Wallis：三级
        h3, p3 = stats.kruskal(tier_groups[0], tier_groups[1], tier_groups[2])
        print(f"=== Kruskal-Wallis（三级聚合）===")
        print(f"H = {h3:.4f}, p = {p3:.6g}")

    # ---------- 结果落盘（系统临时目录） ----------
    out = os.path.join(tempfile.gettempdir(), "q2_cmath_nll_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "per_grade": args.per_grade,
            "model_archive": ARCHIVE_PATH,
            "cfg": {k: v for k, v in cfg.__dict__.items()},
            "awake_idx": awake_idx,
            "device": device,
            "samples": sample,
        }, f, ensure_ascii=False, indent=1)
    print(f"\n逐样本结果已写入（临时目录）: {out}")


if __name__ == "__main__":
    main()