"""Holdout acceptance instrument (layer-3 verdict): did the corpus actually
land in the weights?

Scores next-byte cross-entropy on the 1% path-hash holdout files that the
training pipeline never consumes, for:
  (a) the live checkpoint (state_dataset/latest.pt, both hemispheres), and
  (b) a freshly initialized model of the same architecture.

The reported delta is the "corpus was learned" evidence for layer 3 of the
landing checklist. It is a measurement, not a claim of semantic understanding.

Read-only with respect to the training state directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from ingest_dataset import DatasetStreamer
from life_agent import AppConfig, LifeAgent

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("O:\\数据集")
CHECKPOINT = ROOT / "state_dataset" / "latest.pt"
BYTES_PER_FILE = 256 * 1024


def collect_holdout_blobs(config: AppConfig) -> list[tuple[str, bytes]]:
    streamer = DatasetStreamer(DATA_ROOT, 1, config_holdout(config), [], [])
    blobs: list[tuple[str, bytes]] = []
    for record in streamer.files():
        if not streamer._holdout(record.relative):
            continue
        with record.path.open("rb") as source:
            blobs.append((record.relative, source.read(BYTES_PER_FILE)))
    return blobs


def config_holdout(config: AppConfig) -> float:
    return 0.01


def sequences(blobs: list[tuple[str, bytes]], max_len: int) -> list[list[int]]:
    ids_all: list[list[int]] = []
    for _, blob in blobs:
        for start in range(0, len(blob) - max_len + 1, max_len):
            ids_all.append(list(blob[start : start + max_len]))
    return ids_all


def score(agent: LifeAgent, name: str, ids_all: list[list[int]], batch: int = 16) -> float:
    state = agent.state(name)
    device = state.model.device if hasattr(state.model, "device") else next(state.model.parameters()).device
    state.model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(ids_all), batch):
            chunk = ids_all[start : start + batch]
            inputs = torch.tensor([ids[:-1] for ids in chunk], device=device)
            targets = torch.tensor([ids[1:] for ids in chunk], device=device)
            logits = state.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, agent.config.model.vocab_size),
                targets.reshape(-1),
                reduction="mean",
            )
            total += float(loss.item()) * len(chunk)
            count += len(chunk)
    return total / max(1, count)


def main() -> int:
    config = AppConfig.load(ROOT / "config.json")
    if not torch.cuda.is_available():
        config.runtime.training_device = "cpu"
    blobs = collect_holdout_blobs(config)
    ids_all = sequences(blobs, config.model.max_seq_len)
    if not ids_all:
        print(json.dumps({"error": "no holdout sequences collected"}))
        return 1

    with tempfile.TemporaryDirectory(dir=ROOT / "_audit_tmp") as fresh_dir:
        fresh = LifeAgent(config, Path(fresh_dir))
        trained = LifeAgent(config, ROOT / "state_dataset")
        if not trained.load():
            print(json.dumps({"error": "checkpoint load failed", "checkpoint": str(CHECKPOINT)}))
            return 1
        result: dict[str, object] = {
            "holdout_files": [name for name, _ in blobs],
            "holdout_bytes_scored": sum(len(blob) for _, blob in blobs),
            "sequences": len(ids_all),
            "seq_len": config.model.max_seq_len,
            "trained_frames": trained.observed_frames,
            "fresh_loss": {side: score(fresh, side, ids_all) for side in ("left", "right")},
            "trained_loss": {side: score(trained, side, ids_all) for side in ("left", "right")},
        }
        fresh.close() if hasattr(fresh, "close") else None
        trained.close() if hasattr(trained, "close") else None
    reductions = {}
    for side in ("left", "right"):
        before = result["fresh_loss"][side]  # type: ignore[index]
        after = result["trained_loss"][side]  # type: ignore[index]
        reductions[side] = round(100.0 * (before - after) / before, 3)
    result["reduction_percent"] = reductions
    result["interpretation"] = (
        "Positive reduction_percent on holdout bytes = corpus knowledge present in "
        "the weights. This measures corpus learning only, not semantic understanding."
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
