# -*- coding: utf-8 -*-
"""P0-1 探针二: 正向对照——正常学习样本必须入债务+回放。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chat_cli import utf8_byte_mask
from byt5_dual import Byt5DualBrain, Seq2SeqSample

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "_audit_tmp" / "qa4_gate_probe_state"
brain = Byt5DualBrain(str(ROOT / "byt5_model"), STATE)
brain.load()
brain.ensure_active_gpu()
h = brain.active_h()

def snapshot(h):
    return (len(h.debt), round(h.debt.total_clean_added, 6), len(h.replay))

def chat_dual_learn_and_admit(brain, h, user_text, resp_bytes):
    user_bytes = list(user_text.encode("utf-8"))
    mask, ratio = utf8_byte_mask(bytes(resp_bytes))
    if sum(mask) < 4 or ratio < 0.4:
        return {"stage": "byte-ratio-reject", "ratio": ratio}
    clean_bytes = [raw for raw, ok in zip(resp_bytes, mask) if ok]
    exchange_ids = user_bytes[-256:]
    exchange_labels = clean_bytes[-192:]
    result = h.learn(user_bytes, clean_bytes, 1.0)
    if result["learned"]:
        brain.admit_exchange_sample(Seq2SeqSample(
            ids=exchange_ids, labels=exchange_labels, weight=1.0, source=f"stream:{brain.active}"))
    return result

# 回声式正常样本（run1 里此类输入 loss 0.2-1.5, 可学）
for i, (u, r) in enumerate([
    ("你好", "你好，很高兴见到你。"),
    ("3+4等于几", "等于 7。"),
]):
    before = snapshot(h)
    result = chat_dual_learn_and_admit(brain, h, u, r.encode("utf-8"))
    after = snapshot(h)
    print(f"对照{i+1} '{u}': {result}  debt {before} -> {after}  入账={after != before}")
print("最终 ledger:", snapshot(h))
