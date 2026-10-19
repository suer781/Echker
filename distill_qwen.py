# -*- coding: utf-8 -*-
"""Qwen 引导教材生成器（2026-10-17）。

角色定位：Qwen 只做**引导作用**——从现有蒸馏语料抽少量种子，用 Qwen2.5-0.5B
现场润色/扩写/问答，产出通顺中文文本，作为 Dolphin 的起步教材之一。

不依赖 Qwen：
- 生成量小（默认 200 条），只是众多数据源之一；
- Qwen 发癫产生的坏句子**不做人工过滤**——Dolphin 学习时金发姑娘门控会
  自动把"太怪"的片段屏蔽进冷层（suspended），不会污染训练。

用法：
    python distill_qwen.py [条数] [起始序号]
例：
    python distill_qwen.py 200
断点续跑：已生成的种子自动跳过，追加写入。
"""
import json
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "external_models", "Qwen2.5-0.5B-Instruct")
SEED_SOURCES = [
    r"O:\数据集\12_模型蒸馏\Alpaca-Chinese-5K-Distill-Qwen\Alpaca-Chinese-5K-Distill-Qwen.json",
    r"O:\数据集\12_模型蒸馏\alpaca-zh-R1-Distill-20k\alpaca_data_zh_20k_distill_qwen_32b.json",
]
OUT_DIR = r"O:\数据集\12_模型蒸馏\qwen_live_zh"
OUT_FILE = os.path.join(OUT_DIR, "qwen_live_zh.jsonl")

KINDS = ("polish", "expand", "qa")


def load_seeds(limit):
    """从现有蒸馏语料抽取去重后的中文种子文本。"""
    seeds = []
    for p in SEED_SOURCES:
        if not os.path.exists(p):
            print(f"[跳过] 种子源不存在：{p}")
            continue
        data = None
        if p.endswith(".json"):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data") or data.get("instances") or []
        if not isinstance(data, list):
            continue
        for item in data:
            if isinstance(item, str):
                t = item
            elif isinstance(item, dict):
                t = item.get("text") or item.get("instruction") or ""
            else:
                continue
            t = str(t).strip()
            if t:
                seeds.append(t)
        if len(seeds) >= limit:
            break
    # 去重保序
    seen, out = set(), []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def build_prompt(kind, seed):
    if kind == "polish":
        return f"请把下面这段话改写得更通顺、更完整、更自然，只输出改写后的内容：\n{seed}"
    if kind == "expand":
        return f"请根据下面这段话续写两三句，让内容更丰富、表达更自然，只输出续写后的完整段落：\n{seed}"
    # qa
    return (f"请根据下面这段内容，提出一个相关的问题并给出简洁的答案，"
            f"严格按这个格式输出：\n问：...\n答：...\n\n内容：\n{seed}")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = load_seeds(limit)
    print(f"种子语料 {len(seeds)} 条（目标 {limit}）")

    # 断点续跑：读取已生成的种子
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("seed", ""))
                except Exception:
                    pass
    todo = [s for s in seeds if s not in done]
    print(f"待生成 {len(todo)} 条（已完成 {len(done)}）")
    if not todo:
        print("没有需要生成的条目，退出。")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    count = 0
    with open(OUT_FILE, "a", encoding="utf-8", newline="\n") as f:
        for seed in todo:
            kind = KINDS[count % len(KINDS)]
            prompt = build_prompt(kind, seed)
            try:
                msgs = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt")
                out = model.generate(
                    **ids,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                resp = tok.decode(
                    out[0][ids["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                if len(resp) < 10:
                    print(f"[跳过] 生成过短（{len(resp)}）")
                    continue
                rec = {"seed": seed, "kind": kind, "text": resp}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                count += 1
                print(f"[{count}/{len(todo)}] {kind} OK len={len(resp)}", flush=True)
            except Exception as e:
                print(f"[错误] {e!r}", flush=True)
    print(f"本次生成 {count} 条 → {OUT_FILE}")


if __name__ == "__main__":
    main()