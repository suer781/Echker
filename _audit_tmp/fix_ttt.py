# TTT 语义真正落地（第三轮测试员给出的三处反证逐一修正）
from pathlib import Path

# ============ chat_dual.py：学习目标改为用户输入 + 无条件观测入账 ============
p = Path("chat_dual.py")
src = p.read_text(encoding="utf-8")

old = '''                exchange_labels = clean_bytes[-192:]
                if len(exchange_ids) < 2:
                    exchange_ids = (exchange_ids + [32, 32])[:2]
                try:
                    result = active_h.learn(user_bytes, clean_bytes, intensity)
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    active_h.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print("[显存不足，本轮学习跳过（梯度已回滚）]", flush=True)
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
new = '''                # TTT 语义：学习目标 = 用户的观测输入（真实人类语言），
                # 模型自己的回应永远不进训练——斩断自训练塌缩回路。
                # 观测样本无条件入账（拒学轮也入：退化半球借此积累切换压力）。
                brain.admit_exchange_sample(Seq2SeqSample(
                    ids=user_bytes[-256:], labels=user_bytes[-192:],
                    weight=1.0, source=f"stream:{brain.active}",
                ))
                try:
                    result = active_h.learn(user_bytes, user_bytes[-192:], intensity)
                except torch.cuda.OutOfMemoryError:
                    governor.bump_pressure()
                    active_h.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print("[显存不足，本轮学习跳过（梯度已回滚）]", flush=True)
                    continue
                except ValueError as exc:
                    print(f"[本轮学习跳过：样本构造异常 {exc}]", flush=True)
                    continue
                if result["learned"]:
                    turns_since_save += 1
                    print(f"[{brain.active} TTT 学习 loss={result['loss']:.4f} 强度={result['intensity']:.2f}]", flush=True)
                else:
                    print(f"[{brain.active} 学习跳过：{result.get('reason')} loss={result['loss']:.4f}]", flush=True)'''
assert old in src, "chat_dual TTT 锚点"
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("chat_dual.py TTT 落地")

# ============ byt5_dual.py：consolidate 目标改为账本 ids 的自重建（TTT） ============
p = Path("byt5_dual.py")
src = p.read_text(encoding="utf-8")

old = '''                dec_tokens = [self.byte_to_id[b] for b in sample.labels[-DEC_CAP:] if b in self.byte_to_id]
                dec_tokens += [self.tokenizer.eos_token_id]   # P3-3：与 learn 的目标语义对称'''
new = '''                # TTT 语义：巩固目标 = 账本样本的观测输入自身（ids 的后段），
                # 不再使用 labels（旧回应标签语义已废弃）
                dec_source = sample.ids[-DEC_CAP:]
                dec_tokens = [self.byte_to_id[b] for b in dec_source if b in self.byte_to_id]
                dec_tokens += [self.tokenizer.eos_token_id]'''
assert old in src, "byt5_dual consolidate TTT 锚点"
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("byt5_dual.py TTT 落地")
