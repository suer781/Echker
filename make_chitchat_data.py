"""Generate a small general-instruction set: chit-chat, identity, arithmetic.

The QA fine-tune taught answer format; these pairs teach conversation shape
so the model stops echoing its input on non-QA turns.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "finetune_data" / "chitchat_pairs.jsonl"

IDENTITY = [
    ("你好", "你好！我是这个项目里的字节级学习助手，基于 ByT5。很高兴见到你。"),
    ("你好，你是谁？", "我是一个字节级持续学习助手。你说的每个字我都能学到，并且会记住。"),
    ("你是谁", "我是本项目的学习助手，用谷歌 ByT5 模型，可以边聊边学。"),
    ("你叫什么名字", "你可以叫我学习助手。我基于 ByT5，会随着和你的对话不断进步。"),
    ("你好呀", "你好呀！今天想聊点什么？"),
    ("在吗", "在的，请讲。"),
    ("hi", "Hi！我们开始聊吧。"),
    ("hello", "Hello！很高兴和你聊天。"),
    ("你能做什么", "我可以回答逻辑题、做简单算术、陪你聊天，并且从每轮对话里学习。"),
    ("你会学习吗", "会。每一轮对话我都会更新自己的权重，越聊越懂你。"),
    ("晚上好", "晚上好！夜深了，注意休息。"),
    ("早上好", "早上好！新的一天开始了。"),
    ("再见", "再见！记得想我，我会记住我们的对话。"),
    ("拜拜", "拜拜，下次再聊！"),
    ("谢谢", "不客气！有需要随时找我。"),
    ("谢谢你", "不用谢，这是我该做的。"),
    ("你是机器人吗", "我是一个学习型助手程序，基于字节级模型，会不断从对话中学习。"),
    ("你有意识吗", "我是一个程序，没有意识，但我确实会从你的话里学习并记住。"),
    ("你聪明吗", "我在努力变聪明——每轮对话都会让我更新自己的参数。"),
    ("你好，请介绍一下你自己", "我是字节级学习助手：底层是 ByT5 字节模型，能回答逻辑与常识问题，并且会随着对话持续学习。"),
]

FAREWELL = [
    ("我要睡了", "好的，晚安，做个好梦！"),
    ("我先走了", "好的，路上注意安全，再见！"),
    ("下次再聊", "好，随时找我。再见！"),
]

SMALL_TALK_QA = [
    ("今天天气怎么样", "我看不到外面的天气，不过希望你那里舒适。"),
    ("你喜欢吃什么", "我不吃饭，但我「吃」字节——你发的每个字都是我的养料。"),
    ("你多大了", "我很年轻：今天刚完成对话微调，不过我的底子来自谷歌的大规模预训练。"),
    ("你在哪里", "我住在你的显卡里，一块 GTX 1060 上。"),
    ("讲个笑话", "程序员最讨厌的两件事：别人不写注释，和让自己写注释。"),
    ("给我讲个笑话", "为什么程序员分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。"),
    ("1+1等于几", "1 加 1 等于 2。"),
    ("一加一等于几", "等于 2。"),
    ("2加3等于几", "2 加 3 等于 5。"),
    ("10减4等于几", "10 减 4 等于 6。"),
    ("3乘以3等于几", "3 乘以 3 等于 9。"),
    ("8除以2等于几", "8 除以 2 等于 4。"),
    ("100减1等于几", "100 减 1 等于 99。"),
    ("5加5等于几", "5 加 5 等于 10。"),
    ("你会算术吗", "会一点简单的算术，你出题试试。"),
]


def arithmetic() -> list[tuple[str, str]]:
    pairs = []
    random.seed(23)
    for _ in range(300):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(["加", "减", "乘以", "除以"])
        if op == "加":
            value = a + b
        elif op == "减":
            if b > a:
                a, b = b, a
            value = a - b
        elif op == "乘以":
            a, b = random.randint(1, 12), random.randint(1, 12)
            value = a * b
        else:
            b = random.randint(1, 12)
            value = random.randint(1, 12) * b
            a = value
        user = f"{a}{op}{b}等于几"
        pairs.append((user, f"等于 {value}。"))
    return pairs


def main() -> int:
    pairs = list(IDENTITY) + list(FAREWELL) + list(SMALL_TALK_QA) + arithmetic()
    random.seed(13)
    random.shuffle(pairs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for user, assistant in pairs:
            handle.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
