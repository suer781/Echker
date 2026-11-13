import json, os, sys, datetime

ROOT = r"O:\Documents\智能自学习AI（新）\_audit_tmp"
PARTS = ["concurrency", "math", "persistence", "readiness", "novelty"]
SEV_ORDER = {"阻断": 0, "高": 1, "中": 2, "低": 3}

all_rows = []
meta = []
for part in PARTS:
    p = os.path.join(ROOT, part, "findings.json")
    if not os.path.exists(p):
        print(f"MISSING {p}")
        continue
    with open(p, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    fs = data.get("findings") or []
    meta.append({"part": part, "reviewer": data.get("reviewer"),
                 "count": len(fs), "verdict": (data.get("verdict") or "")[:200],
                 "experiments": len(data.get("experiments") or [])})
    for f in fs:
        row = {
            "part": part,
            "id": f.get("id", ""),
            "title": f.get("title", ""),
            "severity": f.get("severity", ""),
            "kind": f.get("kind", ""),
            "strength": f.get("evidence_strength", ""),
            "location": f.get("location", []),
            "evidence": f.get("evidence", ""),
            "impact": f.get("impact", ""),
            "recommendation": f.get("recommendation", ""),
            "regression_test": f.get("regression_test", ""),
        }
        all_rows.append(row)

print("=== META ===")
for m in meta:
    print(m["part"], "|", m["reviewer"], "| findings:", m["count"], "| experiments:", m["experiments"], "| verdict:", m["verdict"][:120])

print("\n=== SEVERITY ===")
from collections import Counter
print(Counter(r["severity"] for r in all_rows))
print(Counter(r["kind"] for r in all_rows))
print(Counter(r["strength"] for r in all_rows))

print("\n=== TABLE ===")
all_rows.sort(key=lambda r: (SEV_ORDER.get(r["severity"], 9), r["part"], r["id"]))
for r in all_rows:
    locs = "; ".join(r["location"][:3]) if isinstance(r["location"], list) else str(r["location"])
    print(f'{r["id"]}\t{r["severity"]}\t{r["kind"]}\t{r["strength"]}\t{locs}\t{r["title"][:70]}')

merged = {
    "audit": "adaptive switch after feed/consolidate starvation fix",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "project_root": r"O:\Documents\智能自学习AI（新）",
    "parts": meta,
    "total_findings": len(all_rows),
    "findings": all_rows,
}
out = os.path.join(ROOT, "merged_findings.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print("\nWROTE", out, os.path.getsize(out))
