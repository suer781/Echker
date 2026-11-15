# 修复轮 3：审计 P0-1 + P1 组 + P2 批次（共 14 项）
from pathlib import Path

# ============ byt5_dual.py ============
path = Path(__file__).resolve().parent.parent / "byt5_dual.py"
src = path.read_text(encoding="utf-8")

# P0-1：consolidate 加质量闸门（垃圾不得经巩固路径进权重）+ P2-7 梯度 NaN 卫兵
#       + P2-6 只对实际训练过的样本记账 + P3-3 decoder 补 eos 对称
old = '''        self.model.train()
        lr = BASE_LR * max(0.25, intensity)
        steps_done = 0
        last_loss = None
        try:
            for step in range(CONSOLIDATION_STEPS):
                sample = samples[step % len(samples)]
                # seq2seq 语义：encoder = 用户字节，decoder = 回应字节（账本以字节为单位）
                enc_tokens = [self.byte_to_id[b] for b in sample.ids[-ENC_MAX:] if b in self.byte_to_id]
                dec_tokens = [self.byte_to_id[b] for b in sample.labels[-DEC_CAP:] if b in self.byte_to_id]
                inputs = torch.tensor([enc_tokens], device="cuda")
                targets = torch.tensor([dec_tokens], device="cuda")
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(input_ids=inputs, labels=targets)
                loss = out.loss
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                steps_done += 1
                last_loss = float(loss.item())
        except Exception:
            # P1-4：巩固失败必须释放 due() 的预留，否则这批债务本会话
            # 永远不会被再次选中（legacy B4 修复的对应移植）。
            self.debt.release_reservation(samples)
            raise
        if steps_done:
            self.debt.commit_repayment(samples, trained_steps=steps_done)
            self.debt.decay_after_training()
            stored_sample = samples[0]
            self.replay.add(Seq2SeqSample(
                ids=stored_sample.ids.copy(), labels=stored_sample.labels.copy(),
                weight=1.0, source=f"consolidate:{self.name}",
            ))
        else:
            self.debt.release_reservation(samples)
        return {"steps": steps_done, "loss": last_loss}'''
new = '''        self.model.train()
        lr = BASE_LR * max(0.25, intensity)
        steps_done = 0
        last_loss = None
        try:
            for step in range(CONSOLIDATION_STEPS):
                sample = samples[step % len(samples)]
                # seq2seq 语义：encoder = 用户字节，decoder = 回应字节（账本以字节为单位）
                enc_tokens = [self.byte_to_id[b] for b in sample.ids[-ENC_MAX:] if b in self.byte_to_id]
                dec_tokens = [self.byte_to_id[b] for b in sample.labels[-DEC_CAP:] if b in self.byte_to_id]
                dec_tokens += [self.tokenizer.eos_token_id]   # P3-3：与 learn 的目标语义对称
                inputs = torch.tensor([enc_tokens], device="cuda")
                targets = torch.tensor([dec_tokens], device="cuda")
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(input_ids=inputs, labels=targets)
                loss = out.loss
                loss_value = float(loss.item())
                # P0-1：巩固侧同款质量闸门——活动侧被拒的垃圾不得经债务
                # →巩固→交换的路径烧进被动侧真权重。
                if not torch.isfinite(loss) or loss_value > 2.0:
                    continue
                loss.backward()
                # P2-7：反向 NaN 卫兵——clip 范数非有限时 step 会把权重整体
                # 变 NaN 并从此静默停学，必须在 step 前拦截。
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(total_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                steps_done += 1
                last_loss = loss_value
        except Exception:
            # P1-4：巩固失败必须释放 due() 的预留，否则这批债务本会话
            # 永远不会被再次选中（legacy B4 修复的对应移植）。
            self.debt.release_reservation(samples)
            raise
        if steps_done:
            # P2-6：只对实际训练过的样本记账/计入容量（此前全批记账虚增容量）
            committed = samples[:steps_done]
            self.debt.commit_repayment(committed, trained_steps=steps_done)
            self.debt.decay_after_training()
            self.replay.add(Seq2SeqSample(
                ids=committed[0].ids.copy(), labels=committed[0].labels.copy(),
                weight=1.0, source=f"consolidate:{self.name}",
            ))
        # 未训练的剩余样本释放预留（P2-6 配套）
        self.debt.release_reservation(samples[steps_done:])
        return {"steps": steps_done, "loss": last_loss}'''
if old not in src:
    print("consolidate 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# learn() 同款梯度 NaN 卫兵（P2-7）
old = '''        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_value, "learned": True, "intensity": intensity}'''
new = '''        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(total_norm):
            # P2-7：反向 NaN 会让 step 把权重整体毒化为 NaN 并静默停学
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": loss_value, "learned": False, "reason": "non-finite-grad"}
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_value, "learned": True, "intensity": intensity}'''
if old not in src:
    print("learn NaN 卫兵锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P1-3 + P2-1：ensure_gpu/to_cpu 失败兜底
old = '''    def ensure_gpu(self) -> None:
        if self.device == "cuda":
            return
        self.model.to("cuda")
        self.device = "cuda"
        self._move_optimizer_state("cuda")

    def to_cpu(self) -> None:
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.device = "cpu"
        self._move_optimizer_state("cpu")
        torch.cuda.empty_cache()'''
new = '''    def ensure_gpu(self) -> None:
        if self.device == "cuda":
            return
        try:
            self.model.to("cuda")
        except torch.cuda.OutOfMemoryError:
            # P2-1：部分迁移后尽力回到 CPU，避免 device 字段与实际位置撕裂
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self._sync_device_field()
            raise
        self._sync_device_field()
        self._move_optimizer_state("cuda")

    def to_cpu(self) -> None:
        if self.device == "cpu":
            return
        try:
            self.model.to("cpu")
        except Exception:
            self._sync_device_field()
            raise
        self._sync_device_field()
        self._move_optimizer_state("cpu")
        torch.cuda.empty_cache()'''
if old not in src:
    print("P2-1 已应用，跳过")
else:
    src = src.replace(old, new)

# P1-5：save 的 rmtree-before-replace 破坏窗口 → bak 方案
old = '''        memory_tmp = tmp / "memory.json.tmp"
        memory_tmp.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
        os.replace(memory_tmp, tmp / "memory.json")
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)'''
new = '''        memory_tmp = tmp / "memory.json.tmp"
        memory_tmp.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
        os.replace(memory_tmp, tmp / "memory.json")
        # P1-5：绝不先删旧档——旧档改名备份，新档就位后再删备份，
        # 消除"旧档已毁、新档未就位"的断电窗口。
        bak = out.with_suffix(".bak")
        if bak.exists():
            shutil.rmtree(bak)
        if out.exists():
            os.replace(out, bak)
        os.replace(tmp, out)
        shutil.rmtree(bak)'''
if old not in src:
    print("P1-5 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P1-2：.corrupt 二次隔离崩溃（Windows 目录 replace 不能覆盖已存在）
old = '''        except Exception as exc:
            print(f"[{self.name}] 检查点损坏，已隔离该半球档案：{exc}", flush=True)
            corrupt = out.with_suffix(".corrupt")
            if out.exists():
                os.replace(out, corrupt)
            return False'''
new = '''        except Exception as exc:
            print(f"[{self.name}] 检查点损坏，已隔离该半球档案：{exc}", flush=True)
            corrupt = out.with_suffix(".corrupt")
            if out.exists():
                if corrupt.exists():
                    shutil.rmtree(corrupt)   # P1-2：Windows 不能 replace 覆盖已存在目录
                os.replace(out, corrupt)
            return False'''
if old not in src:
    print("P1-2 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P2-3：brain.json 键校验 + 损坏时基线回填
old = '''        meta_path = self.state_dir / "brain.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[双半球] brain.json 损坏（{exc}），元数据按默认处理。", flush=True)
                return True
            self.active = meta.get("active", "left")
            self.switch_urgency_acc = float(meta.get("switch_urgency_acc", 0.0))
            self.pressure_baseline = meta.get("pressure_baseline", self.pressure_baseline)
            self.switch_count = int(meta.get("switch_count", 0))
            self.switch_log = meta.get("switch_log", [])
            return True
        return False'''
new = '''        meta_path = self.state_dir / "brain.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"[双半球] brain.json 损坏（{exc}），元数据按默认处理。", flush=True)
                return True
            # P2-3：逐键校验回填，坏键/缺键不下一轮才炸
            active = meta.get("active")
            self.active = active if active in ("left", "right") else "left"
            self.switch_urgency_acc = max(0.0, min(1.0, float(meta.get("switch_urgency_acc", 0.0))))
            baselines = meta.get("pressure_baseline", {})
            self.pressure_baseline = {
                name: max(0.0, float(baselines.get(name, 0.0))) for name in ("left", "right")
            }
            self.switch_count = int(meta.get("switch_count", 0))
            self.switch_log = meta.get("switch_log", []) if isinstance(meta.get("switch_log"), list) else []
            return True
        return False'''
if old not in src:
    print("P2-3 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P2-2：switch 重排——先做可能 OOM 的 ensure_gpu，零副作用失败可重试
old = '''    def switch(self) -> None:
        old = self.active
        new = self.passive_h().name
        old_h, new_h = self.hemispheres[old], self.hemispheres[new]
        # 1) 债务交换：双方最新 replay 克隆入对方债务（已学材料，非新压力）
        for source_h, target_h in ((old_h, new_h), (new_h, old_h)):
            for item in source_h.replay.latest(2):
                target_h.admit_exchange(
                    Seq2SeqSample(
                        ids=item.ids.copy(), labels=item.labels.copy(),
                        weight=1.0, source=f"exchange:{item.source}",
                    )
                )
        # 2) CPU 上 2% 对称插值（旧活动侧已迁回 CPU，权重即最新状态）
        old_h.to_cpu()
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.model.parameters(), new_h.model.parameters()):
                snapshot = left_p.detach().clone()
                left_p.lerp_(right_p, mix)
                right_p.lerp_(snapshot, mix)
        # 3) 换驻留：旧活动侧落回 CPU，新活动侧上 GPU
        new_h.ensure_gpu()
        self.active = new'''
new = '''    def switch(self) -> None:
        old = self.active
        new = self.passive_h().name
        old_h, new_h = self.hemispheres[old], self.hemispheres[new]
        # 1) 先换驻留（唯一可能 OOM 的操作排最前）：失败时交换/插值均未发生，
        #    整次切换零副作用，下轮重试不会累积漂移（P2-2）
        old_h.to_cpu()
        new_h.ensure_gpu()
        # 2) 债务交换：双方最新 replay 克隆入对方债务（已学材料，非新压力）
        for source_h, target_h in ((old_h, new_h), (new_h, old_h)):
            for item in source_h.replay.latest(2):
                target_h.admit_exchange(
                    Seq2SeqSample(
                        ids=item.ids.copy(), labels=item.labels.copy(),
                        weight=1.0, source=f"exchange:{item.source}",
                    )
                )
        # 3) CPU 上 2% 对称插值（新活动侧此时在 GPU，其 CPU 主副本就是最新权重）
        mix = SYNC_MIX / 2
        with torch.no_grad():
            for left_p, right_p in zip(old_h.model.parameters(), new_h.model.parameters()):
                snapshot = left_p.detach().clone()
                left_p.lerp_(right_p, mix)
                right_p.lerp_(snapshot, mix)
        self.active = new'''
if old not in src:
    print("P2-2 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("byt5_dual.py 修复轮 3 完成")

# ============ chat_dual.py ============
path = Path(__file__).resolve().parent.parent / "chat_dual.py"
src = path.read_text(encoding="utf-8")

# P0-1：admit 移回 learned 分支（修复引入的闸门架空）
old = '''                try:
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
new = '''                try:
                    result = active_h.learn(user_bytes, clean_bytes, intensity)
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
                    # P0-1：只有真正通过质量闸门的学习样本才入债务/回放——
                    # 被拒的垃圾不得经巩固路径烧进被动侧权重
                    brain.admit_exchange_sample(Seq2SeqSample(
                        ids=exchange_ids, labels=exchange_labels,
                        weight=1.0, source=f"stream:{brain.active}",
                    ))
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)'''
if old not in src:
    print("P0-1 chat_dual 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P1-1：命令大小写归一（门与分发统一）
old = '''        if user_text.startswith("/") and user_text.lower() not in ("/exit", "/save", "/status"):
            print("[未知命令。可用命令：/status /save /exit]", flush=True)
            continue
        if user_text == "/exit":'''
new = '''        cmd = user_text.lower()   # P1-1：门与分发统一用小写，/Exit 不再穿透
        if user_text.startswith("/") and cmd not in ("/exit", "/save", "/status"):
            print("[未知命令。可用命令：/status /save /exit]", flush=True)
            continue
        if cmd == "/exit":'''
if old not in src:
    print("P1-1 exit 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)
old = '''        if user_text == "/save":
            if args.no_save:
                print("冒烟模式（--no-save）：未保存。")
            else:
                brain.save()
                print("已保存。")
            continue
        if user_text == "/status":'''
new = '''        if cmd == "/save":
            if args.no_save:
                print("冒烟模式（--no-save）：未保存。")
            else:
                brain.save()
                print("已保存。")
            continue
        if cmd == "/status":'''
if old not in src:
    print("P1-1 save/status 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P1-4：生成 OOM 处理器不再引用可能未绑定的 active_h
old = '''        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            active_h.gpu_model.config.use_cache = False
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（压力已上调，系统自动收紧）。输入继续。]", flush=True)
            continue'''
new = '''        except torch.cuda.OutOfMemoryError:
            governor.bump_pressure()
            gpu_model = brain.active_h().gpu_model   # P1-4：实时解析，防未绑定/陈旧引用
            if gpu_model is not None:
                gpu_model.config.use_cache = False
            torch.cuda.empty_cache()
            print("[显存不足，本轮生成失败（压力已上调，系统自动收紧）。输入继续。]", flush=True)
            continue'''
if old not in src:
    print("P1-4 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("chat_dual.py 修复轮 3 完成")

# ============ chat_cli.py ============
path = Path(__file__).resolve().parent.parent / "chat_cli.py"
src = path.read_text(encoding="utf-8")

# P1-1：大小写归一
old = '''        if user_text.startswith("/") and user_text.lower() not in ("/exit", "/save"):
            print("[未知命令。可用命令：/save /exit]", flush=True)
            continue
        if user_text == "/exit":'''
new = '''        cmd = user_text.lower()   # P1-1
        if user_text.startswith("/") and cmd not in ("/exit", "/save"):
            print("[未知命令。可用命令：/save /exit]", flush=True)
            continue
        if cmd == "/exit":'''
if old not in src:
    print("chat_cli P1-1 exit 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)
old = '''        if user_text == "/save":
            if args.no_save:'''
new = '''        if cmd == "/save":
            if args.no_save:'''
if old not in src:
    print("chat_cli P1-1 save 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

# P2-7：chat_cli 学习的梯度 NaN 卫兵
old = '''                    try:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step()
                    except torch.cuda.OutOfMemoryError:
                        governor.bump_pressure()
                        optimizer.zero_grad(set_to_none=True)
                        print("[显存不足，本轮学习跳过（部分梯度已回滚）]", flush=True)
                        continue'''
new = '''                    try:
                        loss.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        # P2-7：反向 NaN 卫兵
                        if not torch.isfinite(total_norm):
                            optimizer.zero_grad(set_to_none=True)
                            print("[本轮学习跳过：梯度非有限]", flush=True)
                            continue
                        optimizer.step()
                    except torch.cuda.OutOfMemoryError:
                        governor.bump_pressure()
                        optimizer.zero_grad(set_to_none=True)
                        print("[显存不足，本轮学习跳过（部分梯度已回滚）]", flush=True)
                        continue'''
if old not in src:
    print("chat_cli P2-7 锚点: 未命中，跳过（可能已应用）")
else:
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("chat_cli.py 修复轮 3 完成")
