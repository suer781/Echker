"""QA9 independent ledger decoder: verify TTT semantics in debt.items.

Checks per hemisphere memory.json:
  A. every debt item: labels == ids[-len(labels):]  (labels are the tail of
     the observed user input, i.e. self-reconstruction of ids)
  B. expected tail cap: len(labels) == min(len(ids), 192)
  C. no item's labels equal any model response produced in this session
  D. all 5 user inputs (incl. garbage "zzz kkk qqq") present as some item's ids
Pure stdlib; run with the project venv python, no repo imports.
"""
import json
import sys
from pathlib import Path

STATE = Path(sys.argv[1])
LOG = Path(sys.argv[2])

# 5 user inputs in feed order (turn 3 = garbage)
USER_INPUTS = [
    "你好呀，今天心情怎么样？",
    "给我讲个笑话吧。",
    "zzz kkk qqq",
    "我喜欢在夜里听雨声。",
    "记住：猫是一种哺乳动物。",
]

# parse model responses from the session log
responses = []
for line in LOG.read_text(encoding="utf-8").splitlines():
    if line.startswith("助手(") and "): " in line:
        responses.append(line.split("): ", 1)[1].strip())
print(f"[log] parsed {len(responses)} model responses: {responses}")

violations = []
all_ids_texts = []   # (hemisphere, decoded ids)
found_inputs = {t: [] for t in USER_INPUTS}

for mem_path in sorted(STATE.glob("hemispheres/*/memory.json")):
    hemi = mem_path.parent.name
    mem = json.loads(mem_path.read_text(encoding="utf-8"))
    debt = mem.get("debt", {})
    items = debt.get("items", [])
    print(f"\n=== hemisphere={hemi} debt.items={len(items)} "
          f"total_created={debt.get('total_created')} total_repaid={debt.get('total_repaid')} ===")
    for idx, item in enumerate(items):
        ids = list(item["ids"])
        labels = list(item["labels"])
        ids_b = bytes(ids)
        labels_b = bytes(labels)
        # A: labels must be the exact tail of ids
        tail_ok = (len(labels) <= len(ids)) and ids[len(ids) - len(labels):] == labels
        # B: tail cap 192
        cap_ok = len(labels) == min(len(ids), 192)
        ids_text = ids_b.decode("utf-8", errors="replace")
        labels_text = labels_b.decode("utf-8", errors="replace")
        # C: labels must not equal a model response (byte-level and decoded-level)
        resp_hit = [r for r in responses
                    if labels_b == r.encode("utf-8") or labels_text == r]
        status = "OK" if (tail_ok and cap_ok and not resp_hit) else "VIOLATION"
        print(f"  [{hemi}#{idx}] source={item.get('source')!r} weight={item.get('weight'):.4f} "
              f"len(ids)={len(ids)} len(labels)={len(labels)} tail_ok={tail_ok} cap_ok={cap_ok} "
              f"labels==response={bool(resp_hit)} -> {status}")
        print(f"        ids    = {ids_text!r}")
        print(f"        labels = {labels_text!r}")
        if not tail_ok:
            violations.append((hemi, idx, "labels != ids tail"))
        if not cap_ok:
            violations.append((hemi, idx, f"len(labels)={len(labels)} != min(len(ids),192)"))
        if resp_hit:
            violations.append((hemi, idx, f"labels equal model response {resp_hit!r}"))
        all_ids_texts.append((hemi, ids_text, labels_text))
        for t in USER_INPUTS:
            if ids_b == t.encode("utf-8"):
                found_inputs[t].append(hemi)

print("\n=== D: user-input coverage (union of both hemispheres' debt) ===")
missing = []
for t in USER_INPUTS:
    hit = found_inputs[t]
    tag = "garbage-turn" if t == "zzz kkk qqq" else ""
    print(f"  {'FOUND ' if hit else 'MISSING'} in {hit} {tag}: {t!r}")
    if not hit:
        missing.append(t)

print("\n=== VERDICT ===")
print(f"ledger violations (A/B/C): {len(violations)}")
for v in violations:
    print(f"  VIOLATION: {v}")
print(f"user inputs missing from ledgers: {len(missing)} {missing}")
if violations or missing:
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: PASS — all debt items are user-input self-reconstruction; "
      "0 labels = model response; all 5 user inputs (incl. garbage turn) in ledger")
