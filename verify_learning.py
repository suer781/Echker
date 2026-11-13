"""Reproducible proof of online weight learning on non-UTF-8 raw bytes.

This is intentionally a small optimization test, not a claim of general
intelligence. It uses the production LifeAgent/feed_frame path and no direct
optimizer calls in the experiment loop.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from life_agent import AppConfig, ByteFrame, LifeAgent
from life_agent.byte_codec import observation_sample, trim_training_pair

ROOT = Path(__file__).resolve().parent


def observation_loss(agent: LifeAgent, frame: ByteFrame) -> float:
    state = agent.state()
    ids, labels = observation_sample(frame)
    ids, labels = trim_training_pair(ids, labels, agent.config.model.max_seq_len)
    inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=agent.worker.device)
    targets = torch.tensor([labels[1:]], dtype=torch.long, device=agent.worker.device)
    state.model.eval()
    with torch.no_grad():
        logits = state.model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 262), targets.reshape(-1))
    return float(loss.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = AppConfig.load(ROOT / "config.json")
    config.runtime.adaptive_switching = False
    config.runtime.checkpoint_interval_s = 1e9
    with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
        agent = LifeAgent(config, Path(directory))
        # 0xFF/0xFE/0xFD cannot form this repeated sequence as valid UTF-8.
        # No text decoder or tokenizer is involved anywhere in this payload.
        payload = bytes([0, 255, 1, 254, 2, 253, 3, 252]) * 24
        frame = ByteFrame.raw(payload, {"purpose": "raw-byte-online-learning"})
        try:
            payload.decode("utf-8", errors="strict")
            utf8_valid = True
        except UnicodeDecodeError:
            utf8_valid = False
        agent.start()
        try:
            state = agent.state()
            active_before = state.name
            before_loss = observation_loss(agent, frame)
            before_weight = state.model.blocks[0].wi_0.weight.detach().clone()
            for _ in range(args.steps):
                agent.feed_frame(frame)
            after_loss = observation_loss(agent, frame)
            update_l2 = torch.linalg.vector_norm(
                state.model.blocks[0].wi_0.weight.detach() - before_weight
            ).item()
            result = {
                "interface": "BLF1",
                "payload_bytes": len(payload),
                "payload_prefix_hex": payload[:16].hex(" "),
                "payload_is_valid_utf8": utf8_valid,
                "raw_byte_ids": list(payload[:16]),
                "model_vocab_size": agent.config.model.vocab_size,
                "control_ids": "256..261 only; payload remains 0..255",
                "active_hemisphere_before_after": [active_before, agent.active],
                "gpu_training_steps": state.global_step,
                "loss_before": before_loss,
                "loss_after": after_loss,
                "loss_reduction": before_loss - after_loss,
                "loss_reduction_percent": 100 * (before_loss - after_loss) / before_loss,
                "wi_0_weight_update_l2": update_l2,
                "interpretation": (
                    "Only demonstrates online optimization on observed raw bytes; "
                    "does not demonstrate general intelligence or semantic understanding."
                ),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if after_loss < before_loss and update_l2 > 0 else 1
        finally:
            agent.close()
            logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
