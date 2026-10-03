"""查房（ward round）：查看海豚的当前状态、身体结构与生成能力。"""
import os

import torch

from dolphin.dolphin import Dolphin


def param_breakdown(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    return total, emb


def report(d, title):
    print(f"\n===== {title} =====")
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
    print(f"记忆库：{len(d.memory.entries)} 条 {kinds}")


def samples(d, prompts, n=60):
    for q in prompts:
        resp, _ = d.serve(q, max_new=n, temperature=0.7)
        print(f"  问：{q}\n  答：{resp.replace(chr(10), ' ')[:60]!r}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Dolphin.from_birth("dolphin/birth.pt", device=dev)
    if os.path.exists("dolphin/state.pt"):
        d.load("dolphin/state.pt")
    report(d, "查房 · 海豚当前状态")
    print("\n----- 现在嘴里能说出什么（字节级玩具模型，看结构不求通顺）-----")
    samples(d, ["心脏的功能", "感冒了怎么办", "烫伤后第一步", "人体最大的器官是"])
