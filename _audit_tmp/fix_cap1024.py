# finetune_byt5 自适应上下文上限：1024 起步，OOM 自动降回 512
from pathlib import Path

p = Path("finetune_byt5.py")
src = p.read_text(encoding="utf-8")

# 1) 数据加载改为存原始文本（编码延迟到训练时按当前 cap 进行）
old = '''    examples = []
    for data_file in sorted(args.data_dir.glob("*.jsonl")):
        rows = 0
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                examples.append(encode_pair(tok, row["user"], row["assistant"]))
                rows += 1
        print(json.dumps({"data": data_file.name, "pairs": rows}, ensure_ascii=False))
    print(json.dumps({"examples": len(examples)}, ensure_ascii=False))'''
new = '''    examples_raw = []
    for data_file in sorted(args.data_dir.glob("*.jsonl")):
        rows = 0
        with data_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                examples_raw.append((row["user"], row["assistant"]))
                rows += 1
        print(json.dumps({"data": data_file.name, "pairs": rows}, ensure_ascii=False))
    print(json.dumps({"examples": len(examples_raw)}, ensure_ascii=False))'''
assert old in src, "数据加载锚点"
src = src.replace(old, new)

# 2) 训练循环：按当前 cap 编码 + OOM 自适应降档
old = '''    model.train()
    random.seed(11)
    ema = None
    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for enc, dec in examples:
            inputs = torch.tensor([enc], device="cuda")
            labels = torch.tensor([dec], device="cuda")
            loss = model(input_ids=inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.item())
            ema = value if ema is None else 0.98 * ema + 0.02 * value
            if step % 200 == 0:
                rate = step / max(1e-9, time.time() - started)
                print(json.dumps({"step": step, "ema_loss": round(ema, 4), "steps_per_s": round(rate, 2)}, ensure_ascii=False), flush=True)'''
new = '''    model.train()
    random.seed(11)
    ema = None
    step = 0
    skipped = 0
    started = time.time()
    enc_cap = 1024   # 自适应上限：OOM 降回 512（长指令不再截断）
    for epoch in range(args.epochs):
        random.shuffle(examples_raw)
        for user_text, assistant_text in examples_raw:
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            dec = tok(assistant_text, add_special_tokens=False).input_ids[-DEC_CAP:] + [tok.eos_token_id]
            inputs = torch.tensor([enc], device="cuda")
            labels = torch.tensor([dec], device="cuda")
            try:
                loss = model(input_ids=inputs, labels=labels).loss
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if enc_cap > 512:
                    enc_cap = 512
                    print(json.dumps({"adaptive": "OOM → 上下文上限降至 512"}, ensure_ascii=False), flush=True)
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = float(loss.item())
            ema = value if ema is None else 0.98 * ema + 0.02 * value
            if step % 200 == 0:
                rate = step / max(1e-9, time.time() - started)
                print(json.dumps({"step": step, "ema_loss": round(ema, 4), "steps_per_s": round(rate, 2),
                                  "enc_cap": enc_cap}, ensure_ascii=False), flush=True)'''
assert old in src, "训练循环锚点"
src = src.replace(old, new)

# 3) 汇总行带 skipped
old = '''    print(json.dumps({"done": True, "steps": step, "ema_loss": round(ema, 4), "saved": str(OUTPUT)}, ensure_ascii=False))'''
new = '''    print(json.dumps({"done": True, "steps": step, "ema_loss": round(ema, 4), "skipped_oom": skipped,
                      "enc_cap_final": enc_cap, "saved": str(OUTPUT)}, ensure_ascii=False))'''
assert old in src, "汇总锚点"
src = src.replace(old, new)

# 4) skipped 递增（二次 OOM 或继续 OOM 的批跳过计数）
old = '''                if enc_cap > 512:
                    enc_cap = 512
                    print(json.dumps({"adaptive": "OOM → 上下文上限降至 512"}, ensure_ascii=False), flush=True)
                continue'''
new = '''                skipped += 1
                if enc_cap > 512:
                    enc_cap = 512
                    print(json.dumps({"adaptive": "OOM → 上下文上限降至 512"}, ensure_ascii=False), flush=True)
                continue'''
assert old in src, "skipped 锚点"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("finetune_byt5 自适应 cap 完成")
