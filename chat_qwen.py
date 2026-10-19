# -*- coding: utf-8 -*-
"""与外部中文小模型 Qwen2.5-0.5B-Instruct 对话（2026-10-17 接入）。

当前项目 Dolphin 是字节级自学习模型；此脚本直接加载外部预训练中文小模型，
用于体验"装一个现成中文模型"的效果。纯 CPU 可跑（0.5B，约 1GB 内存）。

用法：
    python chat_qwen.py

输入中文对话，输入 exit / quit / 退出 结束。
"""
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "external_models", "Qwen2.5-0.5B-Instruct")
SYSTEM = "你是一个中文 AI 助手。请用简洁、准确、自然的中文回答用户问题。"


def main():
    if not os.path.isdir(MODEL_DIR):
        sys.exit(f"模型目录不存在：{MODEL_DIR}\n请先运行下载命令。")
    print(f"加载 {MODEL_DIR} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    print("就绪！输入内容开始对话（exit/退出 结束）。\n", flush=True)

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "退出"):
            break
        history.append({"role": "user", "content": user})
        try:
            prompt = tok.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt")
            out = model.generate(
                **ids,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            resp = tok.decode(
                out[0][ids["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            print(f"[生成出错] {e!r}")
            history.pop()
            continue
        print(f"Qwen> {resp}", flush=True)
        history.append({"role": "assistant", "content": resp})
        # 控制上下文长度（保留 system + 最近 10 条）
        if len(history) > 12:
            history = [history[0]] + history[-10:]


if __name__ == "__main__":
    main()