"""Chat with Qwen; online weight updates require an explicit /teach correction."""
from __future__ import annotations

import argparse
from pathlib import Path

from qwen_agent import DEFAULT_MODEL, QwenAgent

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local path")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state_qwen")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    agent = QwenAgent(args.model, args.state_dir, args.device)
    print(
        "Ready. Use /teach <correct answer> to train on the last question, "
        "/status, /save, or /exit.",
        flush=True,
    )
    history: list[dict[str, str]] = []
    last_user: str | None = None

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/exit"
        if not text:
            continue
        command = text.lower()
        if command == "/exit":
            agent.save()
            print("Saved.")
            return 0
        if command == "/save":
            agent.save()
            print("Saved.")
            continue
        if command == "/status":
            print(
                f"model={agent.model_name} device={agent.device} "
                f"explicit_corrections={len(agent.examples)}"
            )
            continue
        if command == "/teach" or command.startswith("/teach "):
            correction = text[len("/teach") :].strip()
            if not correction:
                print("Usage: /teach <correct answer>")
                continue
            if last_user is None:
                print("There is no previous user message to correct.")
                continue
            result = agent.learn_from_correction(last_user, correction)
            if result["trained"]:
                print(
                    f"Correction learned: loss={result['loss']:.4f}, "
                    f"saved examples={result['examples']}"
                )
            else:
                print(f"Correction skipped: {result['reason']}")
            continue
        if text.startswith("/"):
            print("Unknown command. Use /teach, /status, /save, or /exit.")
            continue

        last_user = text
        answer = agent.generate(
            text,
            history[-6:],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"Assistant: {answer}")
        history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
        )
        history = history[-6:]


if __name__ == "__main__":
    raise SystemExit(main())
