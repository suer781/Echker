# -*- coding: utf-8 -*-
import json, os

ROOT = r"O:\Documents\智能自学习AI（新）"
jp = os.path.join(ROOT, "validation_runs", "audit-adaptive-switch-2026-11-13.json")
mdp = os.path.join(ROOT, "AUDIT_ADAPTIVE_SWITCH.md")

d = json.load(open(jp, encoding="utf-8"))
print("keys:", list(d.keys()))
print("allow_full:", d["verdict"]["allow_full_dataset_training"], "blocking:", d["verdict"]["blocking_count"])
print("blocking_ids:", [b["id"] + "/" + b["from"] for b in d["blocking_items"]])
print("findings:", len(d["findings"]), d["findings_totals"])
print("criteria:", len(d["criteria_audit"]), "measured_facts:", len(d["measured_facts"]))

text = open(mdp, encoding="utf-8").read()
lines = text.splitlines()
pipe = chr(96)
rows = sum(1 for l in lines if l.startswith("| " + pipe))
print("md_lines:", len(lines), "md_bytes:", len(text.encode("utf-8")))
print("appendix_rows:", rows)
for key in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "尚未实现", "不可混称", "阻断"]:
    print(key, "->", text.count(key))

# 确认既有验证产物未被覆盖
runs = os.path.join(ROOT, "validation_runs")
print("validation_runs entries:", sorted(os.listdir(runs)))
print("json_mtime:", os.path.getmtime(jp))
