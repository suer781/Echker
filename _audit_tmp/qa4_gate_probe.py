# -*- coding: utf-8 -*-
"""P0-1 探针: 完整复刻 chat_dual 学习+准入管线, 验证被拒样本不入债务/回放。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_cli import utf8_byte_mask
from byt5_dual import Byt5DualBrain, Seq2SeqSample

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "_audit_tmp" / "qa4_gate_probe_state"

brain = Byt5DualBrain(str(ROOT / "byt5_model"), STATE)
restored = brain.load()
print("restored:", restored)
brain.ensure_active_gpu()
h = brain.active_h()
print("active hemisphere:", h.name)

def snapshot(h):
    return (len(h.debt), round(h.debt.total_clean_added, 6), len(h.replay))

def chat_dual_learn_and_admit(brain, h, user_text, resp_bytes):
    """逐行复刻 chat_dual.py L160-198 的学习+准入管线。"""
    user_bytes = list(user_text.encode("utf-8"))
    mask, ratio = utf8_byte_mask(bytes(resp_bytes))
    valid_bytes = sum(mask)
    if valid_bytes < 4 or ratio < 0.4:
        return {"stage": "byte-ratio-reject", "ratio": ratio}
    clean_bytes = [raw for raw, ok in zip(resp_bytes, mask) if ok]
    if len(user_bytes) < 2:
        user_bytes = (user_bytes + [32, 32])[:2]
    exchange_ids = user_bytes[-256:]
    exchange_labels = clean_bytes[-192:]
    if len(exchange_ids) < 2:
        exchange_ids = (exchange_ids + [32, 32])[:2]
    result = h.learn(user_bytes, clean_bytes, 1.0)
    if result["learned"]:
        brain.admit_exchange_sample(Seq2SeqSample(
            ids=exchange_ids, labels=exchange_labels, weight=1.0, source=f"stream:{brain.active}"))
    return result

cases = [
    ("A1 乱码回复(合法UTF8,模型无把握)", "今晚想试试那家新开的川菜馆",
     (b"zxq wkjv bwq 0921 lkjhgf qwert mnbv 7845 poiuy zxcv asdf hjkl ") * 2),
    ("A2 乱码回复(英文问,俄文答)", "Hello, how are you today?",
     "ЖЩЪЫЬ ЭЁЮЯ БЛВГД йцукен фывапр".encode("utf-8") * 2),
    ("B  正常对照回复", "晚饭吃面条还是吃米饭好",
     "面条和米饭都可以，看个人口味。".encode("utf-8")),
    ("C  FFFD退化回复(ratio<0.4, 字节闸门拒绝)", "你好",
     bytes([0xEF,0xBF,0xBD,0x20,0x7F,0x01,0xF0,0x9F,0xDE,0xAD,0xBE] * 10)),
]
for name, user_text, resp in cases:
    before = snapshot(h)
    result = chat_dual_learn_and_admit(brain, h, user_text, resp)
    after = snapshot(h)
    admitted = after != before
    print(f"[{name}]")
    print(f"  learn(): {result}")
    print(f"  debt: {before} -> {after}  入账={admitted}")
