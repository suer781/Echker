# -*- coding: utf-8 -*-
"""qa7 验收：解码 hemispheres/*/memory.json 的 debt.items / replay，
对账 ids/labels 字节到底是用户输入还是模型回应。

用法: python qa7_decode_ledger.py <transcript_log> <state_dir>
"""
import json
import re
import sys
from pathlib import Path

def b2text(arr):
    raw = bytes(max(0, min(255, int(b))) for b in arr)
    return raw.decode("utf-8", errors="replace")

def parse_transcript(path):
    """从 chat_dual 日志提取 (turn, user_text, response, hemisphere) 与事件行。"""
    turns = []
    events = []
    user = None
    hemi = None
    turn_no = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.rstrip()
        m = re.match(r"^助手\((left|right)\):\s*(.*)$", line)
        if m:
            hemi, resp = m.group(1), m.group(2)
            turn_no += 1
            turns.append({"turn": turn_no, "user": user, "resp": resp, "hemi": hemi})
            user = None
            continue
        if line.startswith("你: "):
            user = line[3:]
            continue
        if line.startswith("[") or "检查点" in line:
            events.append(line)
    return turns, events

KNOWN_USERS = [
    "你好",
    "今天天气怎么样",
    "zzz kkk qqq www",
    "1加1等于几",
    "给我讲个故事",
    "最后第六轮测试用户输入",
]

def main():
    log_path, state_dir = sys.argv[1], Path(sys.argv[2])
    turns, events = parse_transcript(log_path)
    # 管道输入不回显：用户输入按已知序列注入
    resp_only = [t for t in turns]
    for i, t in enumerate(resp_only):
        t["user"] = KNOWN_USERS[i] if i < len(KNOWN_USERS) else "(unknown)"
    turns = resp_only
    print("=== 对话轮记录 ===")
    for t in turns:
        print(f"T{t['turn']} 半球={t['hemi']} 用户={t['user']!r} 回应={t['resp']!r}")
    print("\n=== 事件行 ===")
    for e in events:
        print(e)

    user_bytes_list = [t["user"].encode("utf-8") for t in turns if t["user"]]
    resp_bytes_list = [t["resp"].encode("utf-8") for t in turns]

    def classify(label_bytes):
        """labels 字节属于哪类内容：返回匹配描述列表。"""
        hits = []
        for i, ub in enumerate(user_bytes_list, 1):
            if label_bytes and (label_bytes in ub or ub in label_bytes):
                hits.append(f"用户输入T{i}")
        for i, rb in enumerate(resp_bytes_list, 1):
            if label_bytes and (label_bytes in rb or rb in label_bytes):
                hits.append(f"模型回应T{i}")
        return hits

    print("\n=== debt.items 对账 ===")
    summary = {"user_target": 0, "resp_target": 0, "other": 0, "total": 0}
    verdict_rows = []
    for hemi in ("left", "right"):
        mem = state_dir / "hemispheres" / hemi / "memory.json"
        if not mem.exists():
            print(f"[{hemi}] memory.json 不存在")
            continue
        data = json.loads(mem.read_text(encoding="utf-8"))
        for kind in ("debt", "replay"):
            block = data.get(kind, {})
            items = block.get("items", []) if kind == "debt" else block.get("samples", [])
            if isinstance(items, dict):
                items = list(items.values())
            if not items:
                continue
            print(f"\n--- [{hemi}] {kind} ({len(items)} 条) ---")
            for item in items:
                ids_t = b2text(item["ids"])
                labels_t = b2text(item.get("labels", []))
                hits = classify(bytes(item.get("labels", [])))
                id_hits = classify(bytes(item["ids"]))
                src = item.get("source", "")
                w = item.get("weight")
                summary["total"] += 1
                is_user = any(h.startswith("用户输入") for h in hits)
                is_resp = any(h.startswith("模型回应") for h in hits)
                if is_resp and not is_user:
                    summary["resp_target"] += 1
                elif is_user:
                    summary["user_target"] += 1
                else:
                    summary["other"] += 1
                verdict_rows.append((hemi, kind, src, hits, id_hits, w))
                print(f"source={src} weight={w}")
                print(f"  ids   = {ids_t!r}  命中={id_hits}")
                print(f"  labels= {labels_t!r}  命中={hits}")
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False))
    print("\n判定规则：labels 命中『用户输入』= TTT 正确；命中『模型回应』= 旧语义残留（违规）")

if __name__ == "__main__":
    main()
