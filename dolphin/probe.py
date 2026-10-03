"""固定探测集（律 L8）：体检专用。

任何 hemisphere 永不许在探测集内容上训练——它是系统不会越学越傻的保险丝。

2026-10-03 G2 施工：
- 全卷逐块独立评分（旧版 max_chunks=48 + 50% 重叠在 61.9KB 卷面上只批 10.1%）。
- gate_decision：配对 margin 判决带——亚噪声改善一律拒绝（旧版零 margin
  掷硬币，实测 40 次放行 19 次）。判决逻辑唯一实现于此，sleep.run_cycle 与
  feed.trainer_cycle 经 Dolphin.gate 共用，两份复刻实现不再漂移（M4 评审警告）。
"""
import torch

EPS_FLOOR = 1e-6   # 律定：浮点守卫（完全同源的两模型逐块差恒为 0，必须判否），非判决带
K_MARGIN = 2.0     # 律定（暂）：判决带传导系数——真实噪声尺度全部由数据估计（SE），
                   # k 只定"几倍 SE 才算真改善"；单侧 ~97.5% 置信。M4 接入反馈回路再议。


def load_chunks(path, block_size):
    """全卷不重叠切分：每个字节恰好进一块，尾块不足 block_size 也保留（≥2 字节可评分）。

    覆盖率恒等式 sum(len(c)) == 文件总字节——10.1% 缺陷（截断+重叠）的结构性回归点。
    """
    data = open(path, "rb").read()
    bs = max(2, block_size)
    return [data[s:s + bs] for s in range(0, len(data) - 1, bs)]


@torch.no_grad()
def evaluate(model, chunks, device):
    """全卷逐块独立评分。返回 (mean_loss, per_chunk 列表, n)。

    逐块分数必须保留：门控的判决分辨率由配对逐块差的离散度在线估计，
    只报均值就又回到掷硬币。
    """
    model.eval()
    if not chunks:
        return float("inf"), [], 0
    per = []
    for c in chunks:
        b = torch.tensor(list(c), dtype=torch.long, device=device)
        _, loss = model(b[:-1].unsqueeze(0), b[1:].unsqueeze(0))
        per.append(loss.item())
    return sum(per) / len(per), per, len(per)


def gate_decision(per_old, per_new, k=K_MARGIN):
    """律 L8 判决带（唯一实现，两条训练路径共用）。

    配对逐块差 d_i = old_i − new_i（正=改善）。passed ⟺ mean_new < mean_old − ε，
    ε = max(EPS_FLOOR, k·SE)，SE = std(d)/√n 全部由本轮数据在线估计（值：自成）。
    返回 (passed, detail)；detail 直接并入睡眠报告。
    """
    n = min(len(per_old), len(per_new))
    if n == 0:
        return False, {"probe_old": None, "probe_new": None, "gate_eps": None,
                       "gate_se": None, "chunks": 0, "gate_margin": None}
    d = [o - w for o, w in zip(per_old[:n], per_new[:n])]
    mean_d = sum(d) / n
    var = sum((x - mean_d) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = (var ** 0.5) / (n ** 0.5)
    eps = max(EPS_FLOOR, k * se)
    passed = mean_d > eps
    detail = {
        "probe_old": round(sum(per_old[:n]) / n, 4),
        "probe_new": round(sum(per_new[:n]) / n, 4),
        "gate_eps": round(eps, 6),
        "gate_se": round(se, 6),
        "gate_margin": round(mean_d, 6),  # 正=改善幅度；与 eps 同尺度，直读"真改善还是噪声"
        "chunks": n,
    }
    return passed, detail
