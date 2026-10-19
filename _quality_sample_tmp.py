# -*- coding: utf-8 -*-
"""临时质量采样脚本：只读加载 fed_state.pt，生成几个固定问题回答，不保存。
不启动训练线程、不写存档，避免干扰正在运行的 feed.py。"""
import os
import sys

sys.path.insert(0, r"c:\Users\13682\Documents\自学习AI")
os.chdir(r"c:\Users\13682\Documents\自学习AI")
sys.stdout.reconfigure(encoding="utf-8")

import torch

from dolphin.dolphin import Dolphin
from feed import FED_STATE, make_cfg

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 先读存档字段
ck = torch.load(FED_STATE, map_location="cpu", weights_only=True)
print("== 存档字段 ==")
for k in ("cycle", "ability", "skipped_too_easy", "suspended_too_hard", "round"):
    if k in ck:
        print(f"  {k} = {ck[k]}")
print("  cfg =", ck.get("cfg"))

d = Dolphin(cfg=make_cfg("auto"), device=dev)
d.load(FED_STATE)
n_params = sum(p.numel() for p in d.awake().model.parameters())
print(f"== 模型 ==")
print(f"  cycle={d.cycle} 醒脑={d.awake().name} 参数={n_params:,}")
print(f"  probe_loss={d.probe_loss(d.awake()):.4f}" if hasattr(d, "probe_loss") else "")

print("== 生成采样（固定参数 temp=0.8 top_k=50 top_p=0.95 max_new=48）==")
questions = [
    "你好，你叫什么名字？",
    "小明有3个苹果，又买了2个，一共有几个苹果？",
    "什么是健康的生活方式？",
]
for q in questions:
    try:
        resp, eid = d.serve(q, temperature=0.8, top_k=50, top_p=0.95, max_new=48)
        print(f"\nQ: {q}\nA: {resp!r}")
    except Exception as e:
        print(f"  serve 异常: {e!r}")

print("\nDONE")