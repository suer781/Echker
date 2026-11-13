"""Read-only, resumable BLF1 streaming over a large local dataset.

No source file is modified. Every chunk remains raw bytes; no text tokenizer,
Parquet/CSV/JSON parser, or modality-specific training objective is introduced.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

from life_agent import AppConfig, ByteFrame, LifeAgent

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FileRecord:
    path: Path
    relative: str
    size: int
    mtime_ns: int


class ProgressStore:
    def __init__(self, path: Path, root: Path, fresh: bool = False) -> None:
        self.path = path
        self.root = str(root.resolve())
        self.data: dict[str, Any] = {"version": 1, "root": self.root, "files": {}, "frames": 0}
        if not fresh and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("root") != self.root:
                raise ValueError("progress file belongs to a different dataset root")

    def offset(self, record: FileRecord) -> int:
        saved = self.data["files"].get(record.relative)
        if not saved:
            return 0
        if saved.get("size") != record.size or saved.get("mtime_ns") != record.mtime_ns:
            return 0
        next_offset = int(saved.get("next_offset", 0))
        # A size+mtime check cannot detect in-place content replacement with a
        # preserved timestamp. Verify the last committed chunk when its digest
        # was recorded; a mismatch means the file must be consumed from zero.
        last_len = int(saved.get("last_chunk_bytes", 0))
        last_hash = saved.get("last_chunk_sha256")
        if last_len > 0 and last_hash and next_offset >= last_len:
            try:
                with record.path.open("rb") as source:
                    source.seek(next_offset - last_len)
                    chunk = source.read(last_len)
                if len(chunk) != last_len or hashlib.sha256(chunk).hexdigest() != last_hash:
                    return 0
            except OSError:
                return 0
        return min(next_offset, record.size) if record.size > 0 else 0

    def mark(
        self,
        record: FileRecord,
        next_offset: int,
        last_chunk_sha256: str | None = None,
        last_chunk_bytes: int = 0,
    ) -> None:
        entry = {
            "size": record.size,
            "mtime_ns": record.mtime_ns,
            "next_offset": next_offset,
            "complete": next_offset >= record.size,
        }
        if last_chunk_sha256:
            entry["last_chunk_sha256"] = last_chunk_sha256
            entry["last_chunk_bytes"] = int(last_chunk_bytes)
        self.data["files"][record.relative] = entry
        self.data["frames"] = int(self.data.get("frames", 0)) + 1
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


class DatasetStreamer:
    def __init__(
        self,
        root: Path,
        chunk_bytes: int,
        holdout_fraction: float,
        includes: list[str],
        excludes: list[str],
    ) -> None:
        self.root = root.resolve()
        self.chunk_bytes = chunk_bytes
        self.holdout_fraction = holdout_fraction
        self.includes = includes or ["*"]
        self.excludes = excludes
        if not self.root.is_dir():
            raise ValueError(f"dataset directory does not exist: {self.root}")
        if chunk_bytes < 1 or not 0 <= holdout_fraction < 1:
            raise ValueError("invalid chunk size or holdout fraction")

    def _selected(self, relative: str) -> bool:
        included = any(fnmatch.fnmatch(relative, pattern) for pattern in self.includes)
        excluded = any(fnmatch.fnmatch(relative, pattern) for pattern in self.excludes)
        return included and not excluded

    def _holdout(self, relative: str) -> bool:
        digest = hashlib.sha256(relative.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return value < self.holdout_fraction

    def files(self) -> list[FileRecord]:
        records = []
        for directory, _, filenames in os.walk(self.root, followlinks=False):
            for filename in filenames:
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(self.root).as_posix()
                stat = path.stat()
                records.append(FileRecord(path, relative, stat.st_size, stat.st_mtime_ns))
        return sorted(records, key=lambda item: item.relative)

    def inventory(self) -> dict[str, Any]:
        total_files = total_bytes = eligible_files = eligible_bytes = holdout_files = holdout_bytes = 0
        extensions: dict[str, dict[str, int]] = {}
        for record in self.files():
            total_files += 1
            total_bytes += record.size
            extension = Path(record.relative).suffix.lower() or "<none>"
            bucket = extensions.setdefault(extension, {"files": 0, "bytes": 0})
            bucket["files"] += 1
            bucket["bytes"] += record.size
            if not self._selected(record.relative):
                continue
            if self._holdout(record.relative):
                holdout_files += 1
                holdout_bytes += record.size
                continue
            eligible_files += 1
            eligible_bytes += record.size
        return {
            "root": str(self.root),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "training_files": eligible_files,
            "training_bytes": eligible_bytes,
            "holdout_files": holdout_files,
            "holdout_bytes": holdout_bytes,
            "holdout_fraction": self.holdout_fraction,
            "includes": self.includes,
            "excludes": self.excludes,
            "extensions": dict(sorted(extensions.items(), key=lambda item: item[1]["bytes"], reverse=True)),
            "note": "Inventory only; no claim about semantic quality or duplicate content across formats.",
        }

    def frames(
        self,
        progress: ProgressStore,
        max_bytes: int,
        max_files: int,
        deadline: float | None,
    ) -> Iterator[tuple[ByteFrame, FileRecord, int]]:
        produced_bytes = 0
        touched_files = 0
        for record in self.files():
            if max_files and touched_files >= max_files:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not self._selected(record.relative) or self._holdout(record.relative):
                continue
            offset = progress.offset(record)
            if offset >= record.size and record.size > 0:
                continue
            touched_files += 1
            with record.path.open("rb") as source:
                source.seek(offset)
                while offset < record.size or (record.size == 0 and offset == 0):
                    if deadline is not None and time.monotonic() >= deadline:
                        return
                    if max_bytes and produced_bytes >= max_bytes:
                        return
                    amount = self.chunk_bytes
                    if max_bytes:
                        amount = min(amount, max_bytes - produced_bytes)
                    payload = source.read(amount)
                    if not payload:
                        break
                    next_offset = offset + len(payload)
                    metadata = {
                        "relative_path": record.relative,
                        "media_type": mimetypes.guess_type(record.path.name)[0] or "application/octet-stream",
                        "file_size": record.size,
                        "offset": offset,
                        "chunk_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                    yield ByteFrame.raw(payload, metadata), record, next_offset
                    produced_bytes += len(payload)
                    offset = next_offset
                    if record.size == 0:
                        break


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable raw-byte dataset ingestion")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="actually train; omit for inventory only")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state_dataset")
    parser.add_argument("--inventory", type=Path, default=ROOT / "dataset_inventory.json")
    parser.add_argument("--chunk-bytes", type=int, default=960_000)
    parser.add_argument("--max-bytes", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-files", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-seconds", type=float, default=0, help="0 means unlimited")
    parser.add_argument("--holdout", type=float, default=0.01)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--fresh", action="store_true", help="ignore prior model and dataset progress")
    parser.add_argument("--initialize-from", type=Path, help="copy a trusted checkpoint into a new dataset state")
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AppConfig.load(ROOT / "config.json")
    if args.chunk_bytes > config.runtime.max_frame_bytes - 65536:
        raise ValueError("chunk-bytes leaves insufficient room for BLF1 metadata")
    streamer = DatasetStreamer(args.root, args.chunk_bytes, args.holdout, args.include, args.exclude)
    inventory = streamer.inventory()
    write_json(args.inventory, inventory)
    print(json.dumps({"inventory": inventory}, ensure_ascii=False))
    if not args.train:
        return 0

    args.state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.state_dir / "latest.pt"
    if args.initialize_from and not checkpoint.exists() and not args.fresh:
        if not args.initialize_from.is_file():
            raise ValueError(f"initialization checkpoint not found: {args.initialize_from}")
        shutil.copy2(args.initialize_from, checkpoint)
    progress = ProgressStore(args.state_dir / "dataset_progress.json", args.root, fresh=args.fresh)
    agent = LifeAgent(config, args.state_dir)
    if not args.fresh:
        agent.load()
    agent.start()
    started = time.monotonic()
    deadline = started + args.max_seconds if args.max_seconds else None
    totals = {"frames": 0, "payload_bytes": 0, "encoded_bytes": 0, "chunks": 0, "trained_steps": 0}
    pending_progress: list[tuple[FileRecord, int, str, int]] = []

    def commit_progress() -> None:
        for saved_record, saved_offset, chunk_hash, chunk_bytes in pending_progress:
            progress.mark(saved_record, saved_offset, chunk_hash, chunk_bytes)
        pending_progress.clear()

    try:
        for frame, record, next_offset in streamer.frames(
            progress, args.max_bytes, args.max_files, deadline
        ):
            result = agent.feed_frame(frame)
            pending_progress.append(
                (
                    record,
                    next_offset,
                    str(frame.metadata.get("chunk_sha256", "")),
                    len(frame.payload),
                )
            )
            totals["frames"] += 1
            totals["payload_bytes"] += result["payload_bytes"]
            totals["encoded_bytes"] += result["encoded_bytes"]
            totals["chunks"] += result["chunks"]
            totals["trained_steps"] += result["trained_steps"]
            if args.save_every and totals["frames"] % args.save_every == 0:
                # The checkpoint is committed before its dataset offsets. A
                # crash may repeat recent frames, but cannot skip unsaved learning.
                agent.save()
                commit_progress()
            print(json.dumps({"progress": {**totals, "last_file": record.relative, "offset": next_offset}}, ensure_ascii=False))
        if deadline is not None and time.monotonic() >= deadline:
            totals["stopped_for_time_budget"] = True
    finally:
        agent.close()
        commit_progress()
    elapsed = max(1e-9, time.monotonic() - started)
    summary = {
        **totals,
        "elapsed_seconds": elapsed,
        "payload_bytes_per_second": totals["payload_bytes"] / elapsed,
        "trained_steps_per_second": totals["trained_steps"] / elapsed,
        "state_dir": str(args.state_dir),
        "progress": str(args.state_dir / "dataset_progress.json"),
        "source_access": "read-only (rb); no source writes in this program",
    }
    write_json(args.state_dir / "last_run.json", summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
