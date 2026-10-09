"""用户聊天入口（2026-10-08）：直接运行即可，输入即学习。

设计哲学（与 feed.py 完全一致）：
- 无模式：没有"用户模式/训练模式"之分。本入口提供直接文本输入，
  与后台喂食共享同一份 Dolphin 存档、同一条"经验→选拔→睡眠→换班"链路。
- 使用即学习：用户输入的每句话经 serve() 自动写入经验缓冲（learn_total+1），
  SleepTrainer 后台线程压力触发睡眠训练，自动巩固。
- 杏仁核自动托管：生成温度/top_k/top_p 由 Amygdala 根据系统健康度自动调节，
  无需人工设置。
- 打分反馈：每次回答后输入 +1/-1/0（或回车跳过），奖励进入记忆库影响选拔。

用法：
  python chat.py            # 前台聊天（后台睡眠训练线程保持自循环）
退出：输入 exit 或 Ctrl+C，自动存档（dolphin/fed_state.pt）。

存档与恢复：启动时若存在 dolphin/fed_state.pt 则读档恢复，否则随机初始化；
运行中每 5 个睡眠周期自动存档，退出时收工存档。
"""
import os
import sys
import threading
import time

from dolphin.dolphin import Dolphin
from dolphin.model import Config
from feed import SleepTrainer, FED_STATE, make_cfg

# 聊天时也保持后台睡眠训练（自循环不中断）
SAVE_EVERY = 5


def load_dolphin():
    """加载生产存档（与 feed.py 同一份机体），无存档则新出生。"""
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    d = Dolphin(cfg=make_cfg("auto"), device=dev)
    if os.path.exists(FED_STATE):
        d.load(FED_STATE)
        print(f"[续喂] 从 {FED_STATE} 读档：cycle={d.cycle} 醒脑={d.awake().name}")
    else:
        print("[初始化] 无存档，随机初始化模型")
    return d


def chat_once(d, text):
    """单轮对话：serve 回答 + 输入自动成为学习信号 + 可选打分。"""
    try:
        resp, eid = d.serve(text)
    except Exception as e:
        print(f"[对话] serve 异常：{e!r}")
        return
    print(f"\n你：{text}")
    print(f"蓝蓟智能：{resp}")
    # 打分反馈（输入即学习的内容级奖励）
    try:
        reward_line = input("[打分] +1/-1/0 或回车跳过：").strip()
        if reward_line in ("+1", "1", "+", "赞", "好"):
            d.feedback(eid, 1.0)
            print("  → 已记录 好评 +1")
        elif reward_line in ("-1", "-", "踩", "差"):
            d.feedback(eid, -1.0)
            print("  → 已记录 差评 -1")
        elif reward_line == "0":
            d.feedback(eid, 0.0)
            print("  → 已记录 中性 0")
    except Exception:
        pass


def main():
    d = load_dolphin()
    # 后台睡眠训练线程：保持自循环（压力触发睡眠/换班）
    trainer = SleepTrainer(d, steps=d.sleep_steps, autotune_steps=True, save_every=SAVE_EVERY)
    trainer.start()

    print("\n" + "=" * 56)
    print("  蓝蓟智能（Echker）聊天入口（输入自动成为学习信号，无模式）")
    print("  输入内容直接回车发送；输入 exit 退出")
    print("  每次回答后可以打分：+1 / -1 / 0（回车跳过）")
    print("=" * 56 + "\n")

    try:
        while True:
            try:
                text = input("你：").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in ("exit", "quit", "退出"):
                break
            chat_once(d, text)
    except KeyboardInterrupt:
        print("\n[中断] 正在收尾…")
    finally:
        trainer.shutdown()
        trainer.join(timeout=10)
        d.save(FED_STATE)
        print(f"[存档] 已保存 → {FED_STATE}（cycle={d.cycle}，本轮对话已入经验）")


if __name__ == "__main__":
    main()