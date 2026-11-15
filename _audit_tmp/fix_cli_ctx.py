# chat_cli.py 专属补丁：上下文 + 温度 + TTT + 历史（锚点按实际文本校准）
from pathlib import Path

p = Path("chat_cli.py")
src = p.read_text(encoding="utf-8")

# 1) 参数
old = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")'''
new = '''    parser.add_argument("--no-save", action="store_true", help="冒烟测试模式：不写任何检查点")
    parser.add_argument("--save-every", type=int, default=20, help="autosave every N learned turns")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度（0=贪心）")
    parser.add_argument("--history-turns", type=int, default=3, help="多轮上下文轮数")
    parser.add_argument("--no-context", action="store_true", help="关闭多轮上下文")'''
assert old in src, "chat_cli 参数锚点"
src = src.replace(old, new)

# 2) history 声明（chat_cli 无空行变体）
old = '''    turns_since_save = 0
    while True:'''
new = '''    turns_since_save = 0
    history = []
    while True:'''
assert old in src, "chat_cli history 声明锚点"
src = src.replace(old, new)

# 3) 生成：上下文编码 + 采样
old = '''        enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
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
                repetition_penalty=1.5,
            )'''
new = '''        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += "用户：" + h_user + "\n" + "助手：" + h_resp + "\n"
        full_text = context_text + "用户：" + user_text + "\n" + "助手："
        enc = tok(full_text, add_special_tokens=True).input_ids[-enc_cap:]
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
                repetition_penalty=1.5,
            )'''
assert old in src, "chat_cli 生成锚点"
src = src.replace(old, new)

# 4) 学习：TTT（编码器=上下文+当前输入，解码目标=当前用户输入）
old = '''                    dec_learn = tok("用户：" + user_text, add_special_tokens=True).input_ids[-512:]
                    inputs = torch.tensor([dec_learn], device="cuda")'''
new = '''                    dec_learn = tok(full_text, add_special_tokens=True).input_ids[-512:]
                    inputs = torch.tensor([dec_learn], device="cuda")'''
assert old in src, "chat_cli TTT 学习锚点"
src = src.replace(old, new)

# 5) 历史更新
old = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
new = '''                    optimizer.zero_grad(set_to_none=True)
                    turns_since_save += 1
                    history.append((user_text, response))
                    del history[:-args.history_turns]
                    print(f"[本轮学习 loss={loss_value:.4f}]", flush=True)'''
assert old in src, "chat_cli 历史更新锚点"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("chat_cli.py 上下文+温度 完成")
