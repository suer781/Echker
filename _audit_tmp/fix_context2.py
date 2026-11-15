# 多轮上下文 + 温度采样补丁（v2，锚点按实际文件校准）
from pathlib import Path

# ============ chat_dual.py ============
p = Path("chat_dual.py")
src = p.read_text(encoding="utf-8")

old = '''    parser.add_argument("--no-save", action="store_true", help="冒烟模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()'''
new = '''    parser.add_argument("--no-save", action="store_true", help="冒烟模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（0=贪心；0.6-0.9 更活泼）")
    parser.add_argument("--history-turns", type=int, default=3,
                        help="编码器携带的最近对话轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")
    args = parser.parse_args()'''
if old not in src:
    print("chat_dual args：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]'''
new = '''        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += "用户：" + h_user + "\n助手：" + h_resp + "\n"
        full_text = context_text + "用户：" + user_text + "\n助手："
        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(full_text, add_special_tokens=True).input_ids[-enc_cap:]'''
if old not in src:
    print("chat_dual context：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''                output = active_h.gpu_model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.5,
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
if old not in src:
    print("chat_dual generate：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''                try:
                    result = active_h.learn(user_bytes, clean_bytes, intensity)'''
new = '''                # 学习输入与生成输入一致（含多轮上下文）——上下文格式经在线
                # 学习变成模型认识的分布
                result = active_h.learn(
                    list(full_text.encode("utf-8"))[-enc_cap:], clean_bytes, intensity)'''
if old not in src:
    print("chat_dual learn 一致性：未命中（跳过或已应用）")
else:
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
if old not in src:
    print("chat_dual admit 回位：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

# history 更新：插在学习块后、巩固前（两种可能位置都尝试）
old = '''                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)
'''
if old in src:
    src = src.replace(old, old + '''                history.append((user_text, response))
                del history[:-args.history_turns]
''', 1)
    print("chat_dual history 更新（位置1）")
else:
    old2 = '''                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)
'''
    if old2 in src:
        src = src.replace(old2, old2 + '''                history.append((user_text, response))
                del history[:-args.history_turns]
''', 1)
        print("chat_dual history 更新（位置2）")

# 清理失效构造
src = src.replace('''                exchange_ids = user_bytes[-256:]
                exchange_labels = clean_bytes[-192:]
                if len(exchange_ids) < 2:
                    exchange_ids = (exchange_ids + [32, 32])[:2]
''', "")

p.write_text(src, encoding="utf-8")
print("chat_dual.py 完成")

# ============ chat_cli.py ============
p = Path("chat_cli.py")
src = p.read_text(encoding="utf-8")

old = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")
    args = parser.parse_args()'''
new = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度（0=贪心）")
    parser.add_argument("--history-turns", type=int, default=3, help="多轮上下文轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")
    args = parser.parse_args()'''
if old not in src:
    print("chat_cli args：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''    turns_since_save = 0
    while True:'''
new = '''    turns_since_save = 0
    history = []
    while True:'''
if old not in src:
    print("chat_cli history 声明：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''        enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]'''
new = '''        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += "用户：" + h_user + "\n助手：" + h_resp + "\n"
        full_text = context_text + "用户：" + user_text + "\n助手："
        enc = tok(full_text, add_special_tokens=True).input_ids[-512:]'''
if old not in src:
    print("chat_cli context：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                )'''
new = '''                sampling = args.temperature > 0
                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=sampling,
                    temperature=max(args.temperature, 1e-5) if sampling else None,
                    top_k=40 if sampling else None,
                    num_beams=1,
                    repetition_penalty=1.3 if sampling else 1.5,
                )'''
if old not in src:
    print("chat_cli generate：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

# 学习输入切到 full_text（生成用的 enc 变量被复用，直接重算）
old = '''                    try:
                        model.train()
                        loss = model(input_ids=inputs, labels=label_tensor).loss
                        loss_value = float(loss.item())'''
new = '''                    enc_learn = tok(full_text, add_special_tokens=True).input_ids[-512:]
                    inputs = torch.tensor([enc_learn], device="cuda")
                    try:
                        model.train()
                        loss = model(input_ids=inputs, labels=label_tensor).loss
                        loss_value = float(loss.item())'''
if old not in src:
    print("chat_cli learn 一致性：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

old = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
new = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    history.append((user_text, response))
                    del history[:-args.history_turns]
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
if old not in src:
    print("chat_cli history 更新：未命中（跳过或已应用）")
else:
    src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("chat_cli.py 完成")
