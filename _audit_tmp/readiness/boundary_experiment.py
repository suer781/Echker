"""Read-only readiness audit experiments.

CPU only. Does not train, does not touch O:\\数据集, does not write under
life_agent/ tests/ validation_runs/ state_dataset_smoke/. All outputs go next
to this script.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from life_agent.byte_codec import ByteFrame, observation_chunks, HEADER
from life_agent.config import AppConfig

LIMIT = 1048576  # config.json runtime.max_frame_bytes

results: dict[str, object] = {"frame_boundary": {}, "coverage": {}, "throughput": {}}

# ---------- 1. ByteFrame size / validation boundary ----------
hdr_size = HEADER.size
results["frame_boundary"]["header_size_bytes"] = hdr_size

# 1a. zero-byte payload
empty = ByteFrame.raw(b"", {"media_type": "application/octet-stream"})
empty_encoded = empty.to_bytes()
results["frame_boundary"]["empty"] = {
    "payload_bytes": 0,
    "encoded_bytes": len(empty_encoded),
    "expected_payload_length": empty.expected_payload_length(),
    "validate_payload_raises": False,
}
empty.validate_payload()

# 1b. typical ingest metadata, measure its encoded length
typical_meta = {
    "relative_path": "12_模型蒸馏/Chinese-R1-Distill-110k-opencc/data/train-00002-of-00003.parquet",
    "media_type": "application/octet-stream",
    "file_size": 245425765,
    "offset": 16724230,
    "chunk_sha256": "a" * 64,
}
meta_frame = ByteFrame.raw(b"", typical_meta)
meta_len = len(meta_frame._metadata_bytes())
results["frame_boundary"]["typical_metadata_bytes"] = meta_len

def encoded_len(payload: bytes, meta: dict) -> int:
    return len(ByteFrame.raw(payload, meta).to_bytes())

# 1c. exactly 1 MiB payload / 1 MiB + 1 payload
for label, payload in (
    ("payload_1MiB", bytes(LIMIT)),
    ("payload_1MiB_plus1", bytes(LIMIT + 1)),
):
    enc = encoded_len(payload, typical_meta)
    exceeds = enc > LIMIT
    results["frame_boundary"][label] = {
        "payload_bytes": len(payload),
        "encoded_bytes": enc,
        "max_frame_bytes": LIMIT,
        "exceeds_limit": exceeds,
        "agent_validate_would_raise": exceeds,
        "expected_payload_length": ByteFrame.raw(payload, typical_meta).expected_payload_length(),
    }

# 1d. largest payload that still fits with typical metadata
fit_payload = LIMIT - hdr_size - meta_len
results["frame_boundary"]["max_fitting_payload_with_typical_meta"] = fit_payload
results["frame_boundary"]["ingest_chunk_bytes_default"] = 960_000
results["frame_boundary"]["ingest_chunk_guard_limit"] = LIMIT - 65536
results["frame_boundary"]["default_chunk_fits"] = 960_000 <= LIMIT - 65536

# 1e. metadata overflow (>65535 bytes)
big_meta = {"pad": "x" * 70_000}
try:
    ByteFrame.raw(b"abc", big_meta)._metadata_bytes()
    meta_overflow = False
except ValueError:
    meta_overflow = True
results["frame_boundary"]["metadata_overflow_raises"] = meta_overflow

# 1f. observation_chunks: reconstruction + independence at boundaries
payload = bytes((i * 37) & 0xFF for i in range(1200))
frame = ByteFrame.raw(payload, {"kind": "boundary-test"})
encoded = frame.to_bytes()
chunks = list(observation_chunks(frame, 512))
width = 512 - 3
reconstructed: list[int] = []
for ids, _labels in chunks:
    reconstructed.extend(ids[2:-1])
boundary_independent = all(ids[0] == 256 and ids[1] == 258 for ids, _ in chunks)
results["frame_boundary"]["chunks"] = {
    "encoded_bytes": len(encoded),
    "chunk_width_tokens": width,
    "chunk_count": len(chunks),
    "reconstruction_matches_encoded": bytes(reconstructed) == encoded,
    "every_chunk_starts_fresh_BOS_USER": boundary_independent,
    "note": "cross-chunk predecessor bytes never appear in the same sequence",
}

# ---------- 2. coverage of the persisted 20 MB real-data run ----------
inv = json.loads((ROOT / "dataset_inventory.json").read_text(encoding="utf-8"))
prog_path = ROOT / "validation_runs" / "continuous-adaptive-full-7" / "dataset_progress.json"
prog = json.loads(prog_path.read_text(encoding="utf-8"))
touched = prog["files"]
touched_bytes = sum(v["size"] for v in touched.values())
touched_complete = sum(1 for v in touched.values() if v["complete"])
by_ext: dict[str, dict[str, int]] = {}
for rel, v in touched.items():
    ext = Path(rel).suffix.lower() or "<none>"
    bucket = by_ext.setdefault(ext, {"files": 0, "bytes": 0})
    bucket["files"] += 1
    bucket["bytes"] += v["size"]
top_dirs = sorted({rel.split("/")[0] for rel in touched})
results["coverage"] = {
    "progress_file": str(prog_path),
    "touched_files": len(touched),
    "touched_file_sizes_sum_bytes": touched_bytes,
    "touched_complete_files": touched_complete,
    "touched_extensions": by_ext,
    "touched_top_directories": top_dirs,
    "inventory_total_files": inv["total_files"],
    "inventory_total_bytes": inv["total_bytes"],
    "touched_file_fraction": len(touched) / inv["total_files"],
    "touched_size_fraction_of_inventory_bytes": touched_bytes / inv["total_bytes"],
    "held_out_files": inv["holdout_files"],
    "held_out_bytes": inv["holdout_bytes"],
    "holdout_fraction_config": inv["holdout_fraction"],
    "held_out_is_path_hash_only": True,
    "note": "progress sizes are source file sizes, not bytes actually read; "
    "actual read payload was 20,000,000 bytes per report.json",
}

# ---------- 3. throughput arithmetic ----------
total_bytes = inv["total_bytes"]
elapsed = 275.7164018000003
real_payload = 20_000_000
blended_bps = real_payload / elapsed
results["throughput"] = {
    "observed_elapsed_seconds_mixed_plus_real": elapsed,
    "observed_real_payload_bytes": real_payload,
    "blended_payload_bytes_per_second": blended_bps,
    "full_dataset_bytes": total_bytes,
    "full_seconds_at_blended_rate": total_bytes / blended_bps,
    "full_hours_at_blended_rate": total_bytes / blended_bps / 3600,
    "readme_claimed_kb_per_s": 79.3,
    "full_hours_at_readme_79_3KB_s": total_bytes / (79.3 * 1000) / 3600,
    "steps_total_left_plus_right": 5558 + 5618,
    "uncertainty": "elapsed mixes 200 synthetic interactions, 5 checkpoint reloads "
    "and the 20 MB real phase; no per-phase timing is persisted, so the real-data "
    "only rate cannot be decomposed from artifacts.",
}

out = Path(__file__).with_name("experiments.json")
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
