# 修复：多轮上下文 + 温度采样（chat_dual + chat_cli）
from pathlib import Path

# ============ chat_dual.py ============
p = Path("chat_dual.py")
src = p.read_text(encoding="utf-8")

old = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=10)'''
new = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（0=贪心；0.6-0.9 更活泼）")
    parser.add_argument("--history-turns", type=int, default=3,
                        help="编码器携带的最近对话轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")'''
assert old in src, "chat_dual args"
src = src.replace(old, new)

old = '''    turns_since_save = 0

    while True:'''
new = '''    turns_since_save = 0
    history: list = []   # 多轮上下文（用户/助手）

    while True:'''
assert old in src, "chat_dual history 声明"
src = src.replace(old, new)

old = '''        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]'''
new = '''        # 多轮上下文：最近 N 轮拼接进编码器（在线学习让模型认识这个格式）
        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += f"用户：{h_user}\\n助手：{h_resp}\\n"
        full_text = context_text + f"用户：{user_text}\\n助手："
        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(full_text, add_special_tokens=True).input_ids[-enc_cap:]'''
assert old in src, "chat_dual context 生成"
src = src.replace(old, new)

old = '''                output = active_h.gpu_model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                )'''
new = '''                sampling = args.temperature > 0
                output = active_h.gpu_model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=sampling,
                    temperature=max(args.temperature, 1e-5) if sampling else None,
                    top_k=40 if sampling else None,
                    num_beams=1,
                    repetition_penalty=1.3 if sampling else 1.5,
                )'''
assert old in src, "chat_dual generate"
src = src.replace(old, new)

old = '''                try:
                    result = active_h.learn(user_bytes, clean_bytes, intensity)'''
new = '''                # 学习输入与生成输入一致（含上下文）——上下文格式经在线学习
                # 变成模型认识的分布
                learn_bytes = list(full_text.encode("utf-8"))[-enc_cap:]
                if len(learn_bytes) < 2:
                    learn_bytes = (learn_bytes + [32, 32])[:2]
                try:
                    result = active_h.learn(learn_bytes, clean_bytes, intensity)'''
assert old in src, "chat_dual learn 一致性"
src = src.replace(old, new)

old = '''                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                    # P0-1：只有真正通过质量闸门的学习样本才入债务/回放——
                    # 被拒的垃圾不得经巩固路径烧进被动侧权重
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=exchange_ids, labels=exchange_labels,
                        weight=1.0, source=f"stream:{brain.active}",
                    ))'''
new = '''                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                    # P0-1：只有通过质量闸门的学习样本入债务/回放
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=list(user_text.encode("utf-8"))[-256:],
                        labels=clean_bytes[-192:],
                        weight=1.0, source=f"stream:{brain.active}",
                    ))'''
assert old in src, "chat_dual admit 回位"
src = src.replace(old, new)

old = '''                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)

        # 被动侧巩固'''
new = '''                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)
                # 多轮历史更新（拒绝轮也记录上下文，但退化为只记用户侧语义）
                history.append((user_text, response))
                del history[:-args.history_turns]

        # 被动侧巩固'''
assert old in src, "chat_dual history 更新"
src = src.replace(old, new)

# 清理失效变量
src = src.replace('''                exchange_ids = user_bytes[-256:]
                exchange_labels = clean_bytes[-192:]
                if len(exchange_ids) < 2:
                    exchange_ids = (exchange_ids + [32, 32])[:2]
''', "")
src = src.replace('''                # 学习输入与生成输入一致（含上下文）——上下文格式经在线学习
                # 变成模型认识的分布
                learn_bytes = list(full_text.encode("utf-8"))[-enc_cap:]
                if len(learn_bytes) < 2:
                    learn_bytes = (learn_bytes + [32, 32])[:2]
''', '''                # 学习输入与生成输入一致（含上下文）——上下文格式经在线学习
                # 变成模型认识的分布
''')
src = src.replace("result = active_h.learn(learn_bytes, clean_bytes, intensity)",
                  "result = active_h.learn(list(full_text.encode('utf-8'))[-enc_cap:], clean_bytes, intensity)")
p.write_text(src, encoding="utf-8")
print("chat_dual.py 完成")

# ============ chat_cli.py ============
p = Path("chat_cli.py")
src = p.read_text(encoding="utf-8")

old = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")'''
new = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度（0=贪心）")
    parser.add_argument("--history-turns", type=int, default=3, help="多轮上下文轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")'''
assert old in src, "chat_cli args"
src = src.replace(old, new)

old = '''    turns_since_save = 0
    while True:'''
new = '''    turns_since_save = 0
    history: list = []
    while True:'''
assert old in src, "chat_cli history 声明"
src = src.replace(old, new)

old = '''        enc = tok(user_text, add_special_tokens=True).input_ids[-512:]
        inputs = torch.tensor([enc], device="cuda")
        start = time.perf_counter()
        model.eval()
        model.config.use_cache = True
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=gen_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
            )'''
new = '''        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += f"用户：{h_user}\\n助手：{h_resp}\\n"
        full_text = context_text + f"用户：{user_text}\\n助手："
        enc = tok(full_text, add_special_tokens=True).input_ids[-512:]
        inputs = torch.tensor([enc], device="cuda")
        start = time.perf_counter()
        model.eval()
        model.config.use_cache = True
        sampling = args.temperature > 0
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=gen_tokens,
                do_sample=sampling,
                temperature=max(args.temperature, 1e-5) if sampling else None,
                top_k=40 if sampling else None,
                num_beams=1,
                repetition_penalty=1.3 if sampling else 1.5,
            )'''
assert old in src, "chat_cli generate"
src = src.replace(old, new)

old = '''        if args.learn and response.strip():'''
new = '''        if args.learn and response.strip():'''
# （chat_cli 的学习输入也切到 full_text，与生成一致）
old2 = '''                    enc_learn = tok(full_text, add_special_tokens=True).input_ids[-512:]
                    inputs = torch.tensor([enc_learn], device="cuda")'''
# 该块此前未应用，检查 chat_cli 学习输入是否还是 user_text
src = src.replace('''        if args.learn and response.strip():''', '''        if args.learn and response.strip():''')

old3 = '''                    try:
                        loss.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)'''
# chat_cli 学习用的 labels 是 response——学习输入需要重算为 full_text
old4 = '''                        loss.backward()'''
if "enc_learn" not in src:
    # 在 labels 构造后、backward 前插入上下文输入重算
    anchor = '''                    try:
                        model.train()
                        loss = model(input_ids=inputs, labels=label_tensor).loss
                        loss_value = float(loss.item())'''
    repl = '''                    try:
                        enc_learn = tok(full_text, add_special_tokens=True).input_ids[-512:]
                        inputs = torch.tensor([enc_learn], device="cuda")
                        model.train()
                        loss = model(input_ids=inputs, labels=label_tensor).loss
                        loss_value = float(loss.item())'''
    assert anchor in src, "chat_cli learn 重算锚点"
    src = src.replace(anchor, repl)

# 历史更新（学习成功后）
old = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
new = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    history.append((user_text, response))
                    del history[:-args.history_turns]
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
assert old in src, "chat_cli history 更新"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("chat_cli.py 完成")
