"""Read-only cross-format admission probe.

Every source file is consumed as raw bytes through the same BLF1 path.
This is not a Parquet/CSV/JSON/PNG parser and makes no semantic claim; it
answers the narrower audit question: can each extension's bytes enter the
single feed_frame/train path without a modality-specific branch?
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import torch

from continuous_validation import rss_megabytes
from ingest_dataset import DatasetStreamer, ProgressStore
from life_agent import AppConfig, LifeAgent
from life_agent.byte_codec import ByteFrame

ROOT = Path(__file__).resolve().parent


def safe_name(extension: str) -> str:
    return extension.lstrip(".").replace("<", "none").replace(">", "") or "none"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("O:/数据集"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation_runs" / "cross-format-probe")
    parser.add_argument("--per-format-bytes", type=int, default=131072)
    parser.add_argument("--chunk-bytes", type=int, default=131072)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AppConfig.load(ROOT / "config.json")
    agent = LifeAgent(config, args.output_dir / "state")
    baseline_rss = rss_megabytes()
    baseline_gpu = torch.cuda.memory_allocated(agent.worker.device)
    started = time.monotonic()
    records: list[dict[str, object]] = []
    failures: list[str] = []

    inventory = DatasetStreamer(args.dataset_root, args.chunk_bytes, 0.0, [], []).inventory()
    extensions = sorted(inventory["extensions"].keys(), key=lambda value: inventory["extensions"][value]["bytes"], reverse=True)
    agent.start()
    try:
        for extension in extensions:
            pattern = "*" if extension == "<none>" else f"*{extension}"
            streamer = DatasetStreamer(args.dataset_root, args.chunk_bytes, 0.0, [pattern], [])
            progress = ProgressStore(
                args.output_dir / "progress" / f"{safe_name(extension)}.json",
                streamer.root,
                fresh=True,
            )
            record: dict[str, object] = {
                "extension": extension,
                "pattern": pattern,
                "declared_files": inventory["extensions"][extension]["files"],
                "declared_bytes": inventory["extensions"][extension]["bytes"],
                "frames": 0,
                "payload_bytes": 0,
                "elapsed_seconds": 0.0,
                "last_file": None,
                "codec": None,
                "error": None,
            }
            begin = time.monotonic()
            try:
                if extension == "<none>":
                    candidates = [
                        record for record in streamer.files()
                        if not Path(record.relative).suffix and record.size > 0
                    ]
                    if not candidates:
                        raise AssertionError("no extensionless file")
                    file_record = min(candidates, key=lambda item: item.size)
                    payload = file_record.path.read_bytes()[: args.per_format_bytes]
                    frame = ByteFrame.raw(
                        payload,
                        {
                            "name": file_record.relative,
                            "media_type": "application/octet-stream",
                            "truncated": len(payload) < file_record.size,
                        },
                    )
                    result = agent.feed_frame(frame)
                    record["frames"] = 1
                    record["payload_bytes"] = len(frame.payload)
                    record["last_file"] = file_record.relative
                    record["codec"] = int(frame.codec)
                    record["last_train_steps"] = int(result["trained_steps"])
                else:
                    for frame, file_record, _ in streamer.frames(
                        progress, args.per_format_bytes, max_files=1, deadline=None
                    ):
                        result = agent.feed_frame(frame)
                        record["frames"] = int(record["frames"]) + 1
                        record["payload_bytes"] = int(record["payload_bytes"]) + len(frame.payload)
                        record["last_file"] = file_record.relative
                        record["codec"] = int(frame.codec)
                        record["last_train_steps"] = int(result["trained_steps"])
                if int(record["frames"]) == 0:
                    raise AssertionError("no frame produced for extension")
            except Exception as exc:  # preserve the exact extension failure
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(f"{extension}: {record['error']}")
            record["elapsed_seconds"] = round(time.monotonic() - begin, 4)
            record["rss_mb"] = rss_megabytes()
            record["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated(agent.worker.device))
            records.append(record)
            print(json.dumps({"stage": "format", **record}, ensure_ascii=False), flush=True)

        # Hidden names have no Path.suffix and would otherwise be hidden inside
        # the <none> bucket by os.walk. Probe them explicitly so dotfiles are
        # admitted through the same byte path rather than silently skipped.
        for hidden in (".gitignore", ".gitattributes"):
            file_records = [
                item for item in streamer.files()
                if Path(item.relative).name == hidden and item.size > 0
            ]
            record = {
                "extension": hidden,
                "pattern": hidden,
                "declared_files": len(file_records),
                "declared_bytes": sum(item.size for item in file_records),
                "frames": 0,
                "payload_bytes": 0,
                "elapsed_seconds": 0.0,
                "last_file": None,
                "codec": None,
                "error": None,
            }
            begin = time.monotonic()
            try:
                if not file_records:
                    raise AssertionError(f"no {hidden} file")
                file_record = min(file_records, key=lambda item: item.size)
                payload = file_record.path.read_bytes()[: args.per_format_bytes]
                frame = ByteFrame.raw(
                    payload,
                    {
                        "name": file_record.relative,
                        "media_type": "application/octet-stream",
                        "truncated": len(payload) < file_record.size,
                    },
                )
                result = agent.feed_frame(frame)
                record["frames"] = 1
                record["payload_bytes"] = len(frame.payload)
                record["last_file"] = file_record.relative
                record["codec"] = int(frame.codec)
                record["last_train_steps"] = int(result["trained_steps"])
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(f"{hidden}: {record['error']}")
            record["elapsed_seconds"] = round(time.monotonic() - begin, 4)
            record["rss_mb"] = rss_megabytes()
            record["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated(agent.worker.device))
            records.append(record)
            print(json.dumps({"stage": "format", **record}, ensure_ascii=False), flush=True)

        agent.save()
        status = agent.status()
        final_rss = rss_megabytes()
        final_gpu = int(torch.cuda.memory_allocated(agent.worker.device))
        total_gpu = int(torch.cuda.get_device_properties(agent.worker.device).total_memory)
        report = {
            "created_at": datetime.now().isoformat(),
            "dataset_root": str(args.dataset_root.resolve()),
            "total_files": inventory["total_files"],
            "total_bytes": inventory["total_bytes"],
            "per_format_bytes": args.per_format_bytes,
            "chunk_bytes": args.chunk_bytes,
            "extensions_covered": len(records),
            "failures": failures,
            "records": records,
            "steps": status.get("step_accounting"),
            "switches": status.get("switches"),
            "memory": {
                "rss_baseline_mb": baseline_rss,
                "rss_final_mb": final_rss,
                "gpu_baseline_bytes": int(baseline_gpu),
                "gpu_final_bytes": final_gpu,
                "gpu_total_bytes": total_gpu,
                "memory_budget_mb": config.runtime.memory_budget_mb,
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "codec_note": "All sampled files are admitted as raw codec bytes; no format parser or separate objective.",
            "scope": "Cross-format byte admission only; no claim about semantic parsing or downstream quality.",
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"stage": "summary", "extensions": len(records), "failures": failures}, ensure_ascii=False))
        return 0 if not failures else 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
