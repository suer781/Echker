# 修复轮 2：chat_dual.py + chat_cli.py + requirements
from pathlib import Path

# ============ chat_dual.py ============
path = Path(__file__).resolve().parent.parent / "chat_dual.py"
src = path.read_text(encoding="utf-8")

# P0-2 + 输入编码错误 + 未定义命令
old = '''    while True:
        try:
            user_text = input("\\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        if not user_text:
            continue
        if user_text == "/exit":'''
new = '''    while True:
        try:
            user_text = input("\\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        except UnicodeDecodeError as exc:
            print(f"[输入编码错误，已忽略该行：{exc}]", flush=True)
            continue
        if not user_text:
            continue
        # P2-9：以 / 开头但未定义的输入一律按命令处理，绝不喂给模型
        # （否则用户的手滑命令变成训练教材）
        if user_text.startswith("/") and user_text.lower() not in ("/exit", "/save", "/status"):
            print("[未知命令。可用命令：/status /save /exit]", flush=True)
            continue
        if user_text == "/exit":'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual 输入锚点：已应用，跳过")

# P0-2：单字节样本保底 + ValueError 兜底（样本构造移入 try）
old = '''        if args.learn and response.strip():
            mask, ratio = utf8_byte_mask(bytes(raw))
            valid_bytes = sum(mask)
            if valid_bytes < 4 or ratio < 0.4:
                print(f"[本轮学习跳过：正确字节占比 {ratio:.0%} 过低，整轮拒学]", flush=True)
            else:
                # 错误字节（mask=False）被剔除：只让正确字词的字节进入学习
                clean_bytes = [raw[i] for i, ok in enumerate(mask) if ok]
                try:
                    result = active_h.learn(list(user_text.encode("utf-8")), clean_bytes, intensity)
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    print("[显存不足，本轮学习跳过]", flush=True)
                    continue
                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                    # 新数据进入活动侧债务（字节为单位）：构成被动侧的切换压力
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=list(user_text.encode("utf-8"))[-256:],
                        labels=clean_bytes[-192:],
                        weight=1.0,
                        source=f"stream:{brain.active}",
                    ))
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)'''
new = '''        if args.learn and response.strip():
            mask, ratio = utf8_byte_mask(bytes(raw))
            valid_bytes = sum(mask)
            if valid_bytes < 4 or ratio < 0.4:
                print(f"[本轮学习跳过：正确字节占比 {ratio:.0%} 过低，整轮拒学]", flush=True)
            else:
                # 错误字节（mask=False）被剔除：只让正确字词的字节进入学习
                clean_bytes = [raw[i] for i, ok in enumerate(mask) if ok]
                # P2-6：学习与生成使用同一段截断后的用户上下文
                user_bytes = list(user_text.encode("utf-8"))[-enc_cap:]
                # P0-2：单字节输入会让账本校验崩溃——保底到 ≥2 字节
                if len(user_bytes) < 2:
                    user_bytes = (user_bytes + [32, 32])[:2]
                exchange_ids = user_bytes[-256:]
                exchange_labels = clean_bytes[-192:]
                if len(exchange_ids) < 2:
                    exchange_ids = (exchange_ids + [32, 32])[:2]
                try:
                    result = active_h.learn(user_bytes, clean_bytes, intensity)
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=exchange_ids, labels=exchange_labels,
                        weight=1.0, source=f"stream:{brain.active}",
                    ))
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    print("[显存不足，本轮学习跳过]", flush=True)
                    continue
                except ValueError as exc:
                    print(f"[本轮学习跳过：样本构造异常 {exc}]", flush=True)
                    continue
                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual P0-2 锚点：已应用，跳过")

# P1-2：学习 OOM 后清理活动侧残留梯度
old = '''                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    print("[显存不足，本轮学习跳过]", flush=True)
                    continue'''
new = '''                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    active_h.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print("[显存不足，本轮学习跳过（梯度已回滚）]", flush=True)
                    continue'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual P1-2 锚点：已应用，跳过")

# P1-1：--no-learn 时不得巩固/切换（"只对话"名副其实）
old = '''        # 被动侧巩固（换入-训练-换回，债务驱动）
        try:
            result = brain.consolidate_passive(intensity)
            if result:
                print(f"[{brain.passive_h().name} 巩固 steps={result['steps']} loss={result.get('loss')}]", flush=True)
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            torch.cuda.empty_cache()
            brain.ensure_active_gpu()
            print("[显存不足，本轮巩固跳过]", flush=True)

        # 曲线自适应切换（紧迫度累加，满 1 触发）
        try:
            if brain.maybe_switch():
                print(f"[半球切换 #{brain.switch_count}] → {brain.active}（紧迫度曲线已达标，债务交换+2%插值完成）", flush=True)
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            torch.cuda.empty_cache()
            brain.ensure_active_gpu()
            print("[切换时显存不足，已回滚驻留，下轮重试]", flush=True)'''
new = '''        # 被动侧巩固（换入-训练-换回，债务驱动）——P1-1：--no-learn 时跳过
        if args.learn:
            try:
                result = brain.consolidate_passive(intensity)
                if result:
                    print(f"[{brain.passive_h().name} 巩固 steps={result['steps']} loss={result.get('loss')}]", flush=True)
            except torch.cuda.OutOfMemoryError:
                governor.bump_pressure()
                torch.cuda.empty_cache()
                brain.ensure_active_gpu()
                print("[显存不足，本轮巩固跳过]", flush=True)

            # 曲线自适应切换（紧迫度累加，满 1 触发）
            try:
                if brain.maybe_switch():
                    print(f"[半球切换 #{brain.switch_count}] → {brain.active}（紧迫度曲线已达标，债务交换+2%插值完成）", flush=True)
            except torch.cuda.OutOfMemoryError:
                governor.bump_pressure()
                torch.cuda.empty_cache()
                brain.ensure_active_gpu()
                print("[切换时显存不足，已回滚驻留，下轮重试]", flush=True)'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual P1-1 锚点：已应用，跳过")

# use_cache 切换：生成走缓存（快），学习走无缓存
old = '''            model.eval()
            with torch.no_grad():
                output = active_h.gpu_model.generate('''
# 注意实际变量名是 brain.active_h 的模型——定位生成调用块
old = '''            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            inputs = torch.tensor([enc], device="cuda")
            start = time.perf_counter()
            active_h.gpu_model.eval()
            with torch.no_grad():
                output = active_h.gpu_model.generate('''
new = '''            active_h = brain.active_h()
            enc = tok(user_text, add_special_tokens=True).input_ids[-enc_cap:]
            inputs = torch.tensor([enc], device="cuda")
            start = time.perf_counter()
            active_h.gpu_model.eval()
            active_h.gpu_model.config.use_cache = True   # 生成走 KV 缓存（快）
            with torch.no_grad():
                output = active_h.gpu_model.generate('''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual use_cache 已应用，跳过")
old = '''            response_ids = output[0].tolist()
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            print("[显存不足，本轮生成失败（压力已上调，系统自动收紧）。输入继续。]", flush=True)
            continue'''
new = '''            response_ids = output[0].tolist()
            active_h.gpu_model.config.use_cache = False
        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            active_h.gpu_model.config.use_cache = False
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（压力已上调，系统自动收紧）。输入继续。]", flush=True)
            continue'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_dual use_cache 恢复已应用，跳过")

path.write_text(src, encoding="utf-8")
print("chat_dual.py 修复完成")

# ============ chat_cli.py ============
path = Path(__file__).resolve().parent.parent / "chat_cli.py"
src = path.read_text(encoding="utf-8")

# P0-2 相关：输入编码错误 + 未定义命令
old = '''    while True:
        try:
            user_text = input("\\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        if not user_text:
            continue
        if user_text == "/exit":'''
new = '''    while True:
        try:
            user_text = input("\\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "/exit"
        except UnicodeDecodeError as exc:
            print(f"[输入编码错误，已忽略该行：{exc}]", flush=True)
            continue
        if not user_text:
            continue
        if user_text.startswith("/") and user_text.lower() not in ("/exit", "/save", "/reset"):
            print("[未知命令。可用命令：/save /exit]", flush=True)
            continue
        if user_text == "/exit":'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_cli 输入锚点：已应用，跳过")

# P2-2：坏序列的悬挂首字节漏标
old = '''        try:
            emitted = decoder.decode(raw[i : i + 1])
        except UnicodeDecodeError:
            decoder.reset()
            mask[i] = False
            i += 1
            continue
        i += 1'''
new = '''        try:
            emitted = decoder.decode(raw[i : i + 1])
        except UnicodeDecodeError:
            # P2-2：坏字节会让缓冲里悬挂的 lead 字节一起变非法——整段标 False
            pending = len(decoder.getstate()[0])
            for j in range(max(0, i - pending), i + 1):
                mask[j] = False
            decoder.reset()
            i += 1
            continue
        i += 1'''
if old in src:
    src = src.replace(old, new)
else:
    print("P2-2 锚点：已应用，跳过")

# P3-3：曲线下端锚点提到跳过线之上
old = "    ANCHOR_LO = 0.08 * 1024**3      # 低于此空闲贴地滑行"
new = "    ANCHOR_LO = 0.12 * 1024**3      # 曲线下端锚点（须高于 SKIP_FREE，避免死区）"
if old in src:
    src = src.replace(old, new)
else:
    print("P3-3 锚点：已应用，跳过")
old = "    SKIP_FREE = int(0.10 * 1024**3)   # 连生成都不安全的物理底线"
new = "    SKIP_FREE = int(0.08 * 1024**3)   # 连生成都不安全的物理底线（低于曲线锚点）"
if old in src:
    src = src.replace(old, new)
else:
    print("SKIP 锚点：已应用，跳过")

# use_cache：生成走缓存（快），学习保持无缓存
old = """            model.eval()
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,"""
new = """            model.eval()
            model.config.use_cache = True
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=gen_tokens,"""
if old in src:
    src = src.replace(old, new)
else:
    print("chat_cli use_cache 已应用，跳过")
old = """                )
            response_ids = output[0].tolist()
        except torch.cuda.OutOfMemoryError:"""
new = """                )
            model.config.use_cache = False
            response_ids = output[0].tolist()
        except torch.cuda.OutOfMemoryError:
            model.config.use_cache = False
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（已清理缓存）。输入继续。]", flush=True)
            continue"""
if old in src:
    src = src.replace(old, new)
else:
    print("chat_cli use_cache 恢复已应用，跳过")

# P2-8（轻量）：/save 与 /exit 前备份生产权重（.bak 只保留最近一份）
old = '''        if user_text == "/save":
            model.config.use_cache = True'''
new = '''        if user_text == "/save":
            import shutil as _shutil
            saf = Path("byt5_model/model.safetensors")
            if saf.exists():
                _shutil.copy2(saf, Path("byt5_model/model.safetensors.bak"))
            model.config.use_cache = True'''
if old in src:
    src = src.replace(old, new)
else:
    print("chat_cli /save 备份锚点：已应用，跳过")

path.write_text(src, encoding="utf-8")
print("chat_cli.py 修复完成")

# ============ requirements ============
path = Path(__file__).resolve().parent.parent / "requirements.txt"
req = path.read_text(encoding="utf-8")
if "transformers" not in req:
    path.write_text(req.rstrip("\n") + "\ntransformers>=5.23\n", encoding="utf-8")
    print("requirements.txt 已锁定 transformers>=5.23")
