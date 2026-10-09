"""预训练（M1 阶段主任务）：从干净语料从零预训练基座模型。

语料纪律：corpus/ 里只放干净来源的文本——预训练输入应为预过滤的高信噪比文本。

长时间训练的三重保障：NaN 守卫（超频导致静默计算错误时立即显式失败）、
定期存档（.ckpt，每 200 步覆盖写）、断点续训（--resume）。

用法：python pretrain.py --steps 300
"""
import argparse
import math
import os

import torch

from dolphin.model import ByteTransformer, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="dolphin/birth.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true", help="从 .ckpt 断点续训")
    a = ap.parse_args()

    data = b""
    for fn in sorted(os.listdir(a.corpus)):
        if fn.endswith(".txt"):
            data += open(os.path.join(a.corpus, fn), "rb").read() + b"\n"
    assert len(data) > a.block + 2, "语料太少，先往 corpus/ 放干净来源的 txt"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, block_size=a.block)
    model = ByteTransformer(cfg).to(dev)
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  设备 {dev}"
          f"  语料 {len(data) / 1e6:.2f}MB")

    # 字节流预转张量（整块常驻显存，避免逐字节 Python 转换）
    bt = torch.tensor(list(data), dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    start_step = 0
    ckpt_path = a.out + ".ckpt"
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ck["sd"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"断点续训：从 step {start_step} 恢复")

    def dump(path, step):
        torch.save({
            "cfg": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                    "block_size": cfg.block_size, "dropout": cfg.dropout, "vocab": cfg.vocab},
            "sd": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
        }, path)

    warm = max(10, a.steps // 10)
    for st in range(start_step, a.steps):
        starts = torch.randint(0, bt.numel() - a.block - 1, (a.batch,))
        x = torch.stack([bt[s:s + a.block] for s in starts])
        y = torch.stack([bt[s + 1:s + a.block + 1] for s in starts])
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            dump(ckpt_path, st)
            raise RuntimeError(
                f"step {st} loss={loss.item()}——数值爆炸。"
                f"先查超频稳定性（烤机 30 分钟），再用 --resume 续训，进度已存 {ckpt_path}")
        lr = a.lr * min(1.0, (st + 1) / warm) * (0.5 * (1 + math.cos(math.pi * st / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (st + 1) % 200 == 0:
            dump(ckpt_path, st + 1)
        if st % 20 == 0 or st == a.steps - 1:
            print(f"step {st:5d}  loss {loss.item():.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dump(a.out, a.steps)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("预训练完成 →", a.out)


if __name__ == "__main__":
    main()
