"""mean_nll batch 化改造的等价性/顺序/排序/空列表测试（2026-10-09）。

对应生产改动：dolphin/model.py 新增 mean_nll_batch 批量方法（原 mean_nll
逐窗口串行 → 完整窗口堆叠一次 forward），dolphin/life.py 换班排序调用点
改走批量接口。本测试验证：

1. 等价性：mean_nll(data) 与 mean_nll_batch([data])[0] 绝对差 < 1e-4。
2. 批量顺序：mean_nll_batch([d1,d2,d3]) 与逐个 mean_nll 一一对应。
3. 排序一致性：多条数据 batch 算出的 nll 排序与逐个算出的排序一致。
4. 空列表：mean_nll_batch([], device) 返回 []。

纯 CPU 运行，不依赖 CUDA；不触碰生产资产。
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.model import ByteTransformer, Config

PASS = []


def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


def _pat(n):
    """构造 n 字节的确定性模式字节串（0..255 循环）。"""
    return bytes(i % 256 for i in range(n))


def _make_model():
    cfg = Config(d_model=64, n_layers=2, n_heads=2, block_size=256)
    m = ByteTransformer(cfg)
    m.eval()
    return m


def t_mean_nll_batch():
    print("== mean_nll batch 化改造测试 ==")
    m = _make_model()

    # —— 1. 等价性：各种长度/内容的数据，单条 batch 与逐窗口串行一致 ——
    datas = [
        b"",                                    # 空字节串
        b"a",                                   # 1 字节
        b"short text",                          # 短文本
        bytes(range(256)),                      # 正好 block_size
        _pat(5000),                            # 长文本 5000 字节
        "中文UTF-8混合测试文本，用于验证批量计算的正确性。".encode("utf-8"),
        _pat(1000),                              # 中长文本
        _pat(300),                               # 跨块中长文本
    ]
    max_diff = 0.0
    for i, d in enumerate(datas):
        single = m.mean_nll(d, "cpu")
        batch = m.mean_nll_batch([d], "cpu")[0]
        diff = abs(single - batch)
        max_diff = max(max_diff, diff)
        check(f"等价性 data[{i}] len={len(d)} diff={diff:.2e} < 1e-4",
              diff < 1e-4, f"single={single:.6f} batch={batch:.6f}")
    check("等价性最大绝对差 < 1e-4", max_diff < 1e-4,
           f"max_diff={max_diff:.2e}")

    # —— 2. 批量顺序：batch 结果与逐个 mean_nll 一一对应 ——
    d1, d2, d3 = _pat(2000), "测试批量顺序".encode(), _pat(1800)
    singles = [m.mean_nll(d, "cpu") for d in [d1, d2, d3]]
    batched = m.mean_nll_batch([d1, d2, d3], "cpu")
    check("批量返回长度与输入等长", len(batched) == 3, f"len={len(batched)}")
    order_ok = True
    for j, (s, b) in enumerate(zip(singles, batched)):
        if abs(s - b) >= 1e-4:
            order_ok = False
        print(f"    order[{j}] single={s:.6f} batch={b:.6f} diff={abs(s-b):.2e}")
    check("批量顺序与逐个 mean_nll 一一对应（差 < 1e-4）", order_ok)

    # —— 3. 排序一致性：30 条随机长度数据的 batch 排序 == 逐个排序 ——
    random.seed(42)
    many = [bytes(random.randbytes(random.randint(100, 3000))) for _ in range(30)]
    srt_single = sorted((m.mean_nll(d, "cpu"), i) for i, d in enumerate(many))
    srt_batch = sorted((v, i) for i, v in enumerate(m.mean_nll_batch(many, "cpu")))
    sort_ok = True
    for a, b in zip(srt_single, srt_batch):
        if abs(a[0] - b[0]) >= 1e-4 or a[1] != b[1]:
            sort_ok = False
            break
    check("排序一致性：30 条数据 batch 排序与逐个排序一致", sort_ok)

    # —— 4. 空列表：mean_nll_batch([], device) 返回 [] ——
    empty = m.mean_nll_batch([], "cpu")
    check("空列表返回 []", empty == [] and isinstance(empty, list), f"{empty}")

    # —— 5. 空数据/1字节数据返回 0.0 ——
    zeros = m.mean_nll_batch([b"", b"x", b""], "cpu")
    check("空/1字节数据返回 0.0", zeros == [0.0, 0.0, 0.0], f"{zeros}")

    n_ok = sum(PASS)
    print(f"\n== 结论：{n_ok}/{len(PASS)} 绿 ==")
    return n_ok == len(PASS)


if __name__ == "__main__":
    ok = t_mean_nll_batch()
    sys.exit(0 if ok else 1)