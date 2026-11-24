"""Qwen dual-hemisphere chat with chunked test-time training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qwen_agent import DEFAULT_MODEL, QwenAgent

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local path")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state_qwen_dual")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading two independent copies of {args.model} ...", flush=True)
    agent = QwenAgent(args.model, args.state_dir, args.device)
    print(
        f"Checkpoint: {'restored' if agent.restored else 'new'}; "
        f"active hemisphere: {agent.brain.active}. "
        "Each generated 8-token chunk is learned immediately; all transcript roles "
        "are retained for replay and background consolidation.",
        flush=True,
    )
    print("Use /status, /save, or /exit.", flush=True)
    history: list[dict[str, str]] = []

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/exit"
        if not text:
            continue
        command = text.lower()
        if command == "/exit":
            agent.close()
            print("Saved.")
            return 0
        if command == "/save":
            agent.save()
            print("Saved.")
            continue
        if command == "/status":
            status = agent.status()
            print(
                f"active={status['active']} switches={status['switch_count']} "
                f"background_consolidation={status['background_consolidation']}"
            )
            if status["consolidation_error"]:
                print(f"consolidation_error={status['consolidation_error']}")
            for name, hemisphere in status["hemispheres"].items():
                print(
                    f"{name}: device={hemisphere['device']} steps={hemisphere['steps']} "
                    f"ttt={hemisphere['ttt_steps']} consolidate={hemisphere['consolidation_steps']} "
                    f"replay={hemisphere['replay']} debt={hemisphere['debt']} "
                    f"mlp_widths={hemisphere['mlp_widths']}"
                )
            continue
        if text.startswith("/"):
            print("Unknown command. Use /status, /save, or /exit.")
            continue

        try:
            answer = agent.generate(
                text,
                history[-12:],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("CUDA out of memory; this turn was not completed. Free GPU memory or use --device cpu.")
            continue
        print(f"Assistant: {answer}")
        history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        history = history[-12:]


if __name__ == "__main__":
    raise SystemExit(main())
