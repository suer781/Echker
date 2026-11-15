# TTT + 多轮上下文 + 回忆巩固 + 温度（chat_dual + chat_cli）
# 锚点全部按当前 HEAD (e1712f4) 实测文本校准
from pathlib import Path

# ============ chat_dual.py ============
p = Path("chat_dual.py")
src = p.read_text(encoding="utf-8")

# 1) 参数
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
assert old in src, "chat_dual 参数锚点"
src = src.replace(old, new)

# 2) history 声明
old = '''    turns_since_save = 0

    while True:'''
new = '''    turns_since_save = 0
    history = []   # 多轮上下文（用户/助手）

    while True:'''
assert old in src, "chat_dual history 声明锚点"
src = src.replace(old, new)

# 3) 生成：上下文编码 + 温度采样
old = '''        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]'''
new = '''        context_text = ""
        if not args.no_context:
            for h_user, h_resp in history[-args.history_turns:]:
                context_text += "用户：" + h_user + "\n" + "助手：" + h_resp + "\n"
        full_text = context_text + "用户：" + user_text + "\n" + "助手："
        response_ids = None
        try:
            brain.ensure_active_gpu()
            active_h = brain.active_h()
            enc = tok(full_text, add_special_tokens=True).input_ids[-enc_cap:]'''
assert old in src, "chat_dual 上下文编码锚点"
src = src.replace(old, new)

# 4) generate 采样
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
                    repetition_penalty=1.5,
                )'''
assert old in src, "chat_dual generate 锚点"
src = src.replace(old, new)

# 5) 学习：目标改为当前用户输入（TTT），编码器含上下文；学习成功后入账
old = '''                try:
                    result = active_h.learn(user_bytes, user_bytes[-192:], intensity)
                except torch.cuda.OutOfMemoryError:'''
new = '''                try:
                    # TTT：编码器=多轮上下文+当前输入，解码器=当前用户输入
                    # （回忆当前轮）；回应永不进训练
                    result = active_h.learn(
                        list(full_text.encode("utf-8"))[-enc_cap:],
                        list(user_text.encode("utf-8"))[-192:], intensity)
                except torch.cuda.OutOfMemoryError:'''
assert old in src, "chat_dual TTT 学习锚点"
src = src.replace(old, new)

# 6) 学习成功后入账 + 回忆巩固 + 历史更新
old = '''                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} TTT 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)'''
new = '''                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} TTT 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)
                # 观测数据无条件入账（用户输入是真实语言；回应永不入账）
                brain.admit_exchange_sample(Seq2SeqSample(
                    ids=user_bytes[-256:], labels=user_bytes[-192:],
                    weight=1.0, source=f"stream:{brain.active}",
                ))
                # 上下文回忆巩固：随机抽一轮历史，给定前文回忆其用户输入
                # ——把上下文压进权重，编码器因此可以减负
                if history and intensity > 0:
                    try:
                        ri = random.randrange(len(history))
                        r_user, _r_resp = history[ri]
                        r_prefix = "".join(
                            "用户：" + u + "\n" + "助手：" + r + "\n"
                            for u, r in history[:ri])
                        r_text = r_prefix + "用户：" + r_user + "\n" + "助手："
                        recall = active_h.learn(
                            list(r_text.encode("utf-8"))[-enc_cap:],
                            list(r_user.encode("utf-8"))[-192:],
                            intensity * 0.5)
                        if recall.get("learned"):
                            print(f"[回忆巩固 #{ri}] loss={recall['loss']:.4f}", flush=True)
                    except torch.cuda.OutOfMemoryError:
                        active_h.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                history.append((user_text, response))
                del history[:-args.history_turns]'''
assert old in src, "chat_dual 巩固+历史锚点"
src = src.replace(old, new)

# 7) random 导入（回忆巩固用）
old = '''import argparse
from pathlib import Path
import sys
import time'''
new = '''import argparse
from pathlib import Path
import random
import sys
import time'''
assert old in src, "random 导入锚点"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("chat_dual.py TTT+上下文+回忆巩固 完成")

# ============ chat_cli.py ============
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

# 2) history 声明
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

# 4) 学习：TTT 目标（当前用户输入）
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
