"""M0 端到端冒烟测试。

出生同源 → 模拟交互（记录惊讶度/奖励）→ 睡眠周期（选拔→变异重放→
蒸馏训练→体检）→ 换班/回滚 → 记忆库生长 → 存档读档。
"""
import os

import torch

from dolphin.dolphin import Dolphin
from dolphin.model import Config


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)

    if os.path.exists("dolphin/birth.pt"):
        d = Dolphin.from_birth("dolphin/birth.pt", device=dev)
        print("① 从胎教基座出生 ✓")
    else:
        d = Dolphin(cfg=Config(d_model=64, n_layers=2, n_heads=2, block_size=192), device=dev)
        print("① 无胎教基座，随机初始化出生")

    p1 = list(d.h[0].model.parameters())
    p2 = list(d.h[1].model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(p1, p2)), "出生时应同卵双生"
    print("   双半球出生同源 ✓")

    lines = []
    for fn in sorted(os.listdir("corpus")):
        if fn.endswith(".txt"):
            t = open(os.path.join("corpus", fn), encoding="utf-8").read()
            lines += [s.strip() for s in t.replace("\n", "。").split("。") if len(s.strip()) >= 8]
    assert len(lines) >= 10, "语料句子太少"
    print(f"② 语料句 {len(lines)} 条，开始模拟交互")

    for i in range(24):
        q = lines[(i * 7 + 3) % len(lines)]
        resp, eid = d.serve(q, max_new=16)
        if i % 3 == 0:
            d.feedback(eid, 1.0)
    print(f"③ 交互 24 次  缓冲 {d.buffer.bytes} 字节  压力 {d.buffer.pressure():.3f}")

    report = d.maybe_sleep(force=True, steps=150)
    print("④ 睡眠报告:", report)
    assert report is not None and "probe_new" in report, "睡眠周期未完成体检"
    assert 0.25 <= d.budget <= 0.60, "预算超出律定边界"
    assert len(d.memory.entries) > 0, "记忆库应有滞留/笔记条目"
    print(f"⑤ 记忆库 {len(d.memory.entries)} 条  传输预算 {d.budget:.3f}"
          f"  睡眠阈值 {d.buffer.sleep_threshold:.3f}")

    resp, _ = d.serve(lines[0], max_new=16)
    print("⑥ 换班后示例回复:", repr(resp[:40]))

    d.save("dolphin/state.pt")
    d2 = Dolphin(cfg=Config(), device=dev)
    d2.load("dolphin/state.pt")
    assert d2.awake_idx == d.awake_idx and abs(d2.budget - d.budget) < 1e-9
    print("⑦ 存档/读档一致 ✓")
    print("PASS：M0 全循环跑通。")


if __name__ == "__main__":
    main()
