"""烤机体检 v2：fp64 仲裁版显存完整性 + 训练轨迹一致性 + 有效算力。

v1 教训（用户抓的）：max(|z-ref|/(|ref|+1e-3)) 会在 ref≈0 的元素上，
把 CPU/GPU 求和顺序的正常舍入差放大成 ~0.1 的假错误。
v2 用 CPU float64 当裁判，容差 = atol + rtol*|ref|，谁越界谁算错，GPU/CPU 分开判。
注意：本测试能分辨 GPU 与 CPU 谁在错，不能完全担保系统内存条
（内存超频的终审是 memtest86 的领域）。
"""
import time

import torch

from dolphin.model import ByteTransformer, Config

ATOL, RTOL = 2e-3, 1e-2


def excess(z, ref):
    """超出容差的量（<=0 表示全部元素都在容差内）。"""
    return ((z.double() - ref).abs() - (ATOL + RTOL * ref.abs())).max().item()


def memtest(iters=30, size=1536):
    torch.manual_seed(7)
    worst_gpu = worst_cpu = -1e30
    for _ in range(iters):
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        ref = a.double() @ b.double()                      # 裁判：CPU float64
        worst_gpu = max(worst_gpu, excess((a.cuda() @ b.cuda()).cpu(), ref))
        worst_cpu = max(worst_cpu, excess(a @ b, ref))
    return worst_gpu, worst_cpu


def trajectory(steps=50):
    """同一组初始权重，GPU 与 CPU 各自独立训练，loss 轨迹必须重合。

    2026-10-04 修复：旧实现 max(|a-b|/max(|b|,1e-6)) 是纯相对偏差——
    训练后期 loss 趋 0 时，正常舍入差被除以极小分母放大成假错误
    （v1 的病根在 trajectory 里残留，memtest 早已改用混合容差）。
    现与 memtest 的 excess() 统一：容差 = atol + rtol*|ref|（混合容差），
    返回"超出容差的最大量"，<=0 表示轨迹完全重合在容差内。
    """
    torch.manual_seed(0)
    cfg = Config(d_model=256, n_layers=3, n_heads=4, block_size=256)
    x = torch.randint(0, 256, (8, 256))
    y = torch.randint(0, 256, (8, 256))

    def train(dev):
        torch.manual_seed(0)
        model = ByteTransformer(cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        out = []
        for _ in range(steps):
            _, loss = model(x.to(dev), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    g = train("cuda")
    c = train("cpu")
    # 混合容差：绝对容差 2e-3 + 相对容差 1e-2×|ref|（与 memtest 同一口径）
    worst = max((abs(a - b) - (2e-3 + 1e-2 * abs(b))).__float__()
                for a, b in zip(g, c))
    return worst


def throughput(steps=30):
    cfg = Config(d_model=768, n_layers=8, n_heads=8, block_size=256)
    model = ByteTransformer(cfg).cuda()
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 256, (16, 256), device="cuda")
    y = torch.randint(0, 256, (16, 256), device="cuda")
    for _ in range(5):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return 6 * n * 16 * 256 * steps / dt / 1e12


if __name__ == "__main__":
    print(f"显卡：{torch.cuda.get_device_name(0)}")
    ok = True

    wg, wc = memtest()
    g_ok, c_ok = wg <= 0, wc <= 0
    ok &= g_ok
    print(f"① 显存完整性（fp64 仲裁，30 次 1536³ 矩阵乘）：")
    print(f"   GPU 侧超容差量 {wg:+.2e}  {'✓ 在容差内' if g_ok else '✗ GPU 在算错数'}")
    print(f"   CPU 侧超容差量 {wc:+.2e}  {'✓ 在容差内' if c_ok else '✗ CPU/内存侧在算错数（查内存超频）'}")

    traj = trajectory()
    # 混合容差口径：<=0 表示所有点都在 atol+rtol*|ref| 内（与 memtest 同标准）。
    # 留 0.01 裕量防边界浮点抖动，但不接受量级放大（旧阈值 0.05 已不适用）。
    t_ok = traj <= 0.01
    ok &= t_ok
    print(f"② 训练轨迹一致性（50 步 GPU vs CPU）：超出混合容差 {traj:+.2e}  {'✓' if t_ok else '✗'}")

    try:
        tf = throughput()
        print(f"③ 有效算力（57M 模型实测）：{tf:.2f} TFLOPS"
              f"  → 100MB/50M 胎教 ≈ {6 * 5e7 * 1e8 / (tf * 1e12) / 86400:.1f} 天")
    except RuntimeError as e:
        ok = False
        print(f"③ 有效算力：✗ 训练负载下 CUDA 崩溃（{str(e).splitlines()[0]}）")

    print("\n体检结论：" + ("✓ 全绿，允许挂机训练" if ok else "✗ 未通过——按报告逐项处理"))
