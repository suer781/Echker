"""Vocabulary-tokenized Qwen chat with continuous dual-hemisphere learning."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

import torch

from life_agent.memory import Sample

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "你是一个有帮助的助手。请清楚作答；无法确认时明确说明，不要编造事实。"
MAX_SEQUENCE_LENGTH = 512
TTT_CHUNK_TOKENS = 8
LOG = logging.getLogger(__name__)


def completion_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Mask a prompt prefix while keeping the generated completion trainable."""
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("training example must contain a prompt and a non-empty answer")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full conversation")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


def transcript_samples(ids: list[int]) -> list[Sample]:
    if len(ids) < 2:
        raise ValueError("tokenized transcript is too short to train")
    samples = []
    start = 0
    while start < len(ids):
        end = min(start + MAX_SEQUENCE_LENGTH, len(ids))
        if len(ids) - end == 1:
            end -= 1
        chunk = ids[start:end]
        if len(chunk) < 2:
            raise ValueError("transcript chunk is too short to train")
        samples.append(
            Sample(
                ids=chunk,
                labels=chunk.copy(),
                weight=1.0,
                source="conversation:all-roles",
            )
        )
        start = end
    return samples


class QwenAgent:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        state_dir: Path = Path("state_qwen_dual"),
        device: str | None = None,
    ) -> None:
        from qwen_dual import QwenDualBrain

        self.model_name = model_name
        self.state_dir = Path(state_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        metadata_path = self.state_dir / "model_name.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model_name") != model_name:
                raise ValueError(
                    f"checkpoint uses {metadata.get('model_name')!r}, not {model_name!r}"
                )

        self.brain = QwenDualBrain(model_name, self.state_dir, self.device)
        self.restored = self.brain.load()
        self.tokenizer = self.brain.active_h().tokenizer
        if len(self.tokenizer) < 1000:
            raise ValueError("selected tokenizer does not appear to have a normal text vocabulary")
        self.brain.ensure_active_gpu()

        self._consolidator = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen-consolidation",
        )
        self._consolidation_future: Future | None = None
        self._consolidation_error: str | None = None
        self._closed = False
        self._lock: threading.RLock = self.brain.training_lock
        self.last_learning_result: dict[str, Any] | None = None

    @staticmethod
    def _ids(encoded: Any) -> list[int]:
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if torch.is_tensor(encoded):
            encoded = encoded.detach().cpu().tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(value) for value in encoded]

    def _template_ids(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return self._ids(encoded)

    def _await_consolidation(self) -> None:
        future = self._consolidation_future
        if future is None:
            return
        self._consolidation_future = None
        future.result()

    def _consolidate_and_switch(self) -> dict[str, Any] | None:
        result = self.brain.consolidate_passive(1.0)
        with self._lock:
            self.brain.maybe_switch()
        return result

    def _record_consolidation_result(self, future: Future) -> None:
        try:
            future.result()
        except Exception as exc:
            self._consolidation_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("Qwen passive consolidation failed")
        else:
            self._consolidation_error = None

    def _schedule_consolidation(self) -> None:
        future = self._consolidation_future
        if future is not None:
            if not future.done():
                return
        self._consolidation_future = self._consolidator.submit(
            self._consolidate_and_switch,
        )
        self._consolidation_future.add_done_callback(self._record_consolidation_result)

    def _messages_for_user(
        self,
        user: str,
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        prior = list(history or [])
        if prior and prior[0].get("role") == "system":
            messages = prior
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *prior]
        messages.append({"role": "user", "content": user})
        while len(messages) > 2:
            prompt_ids = self._template_ids(messages, add_generation_prompt=True)
            if len(prompt_ids) <= MAX_SEQUENCE_LENGTH:
                return messages
            del messages[1]
        if len(self._template_ids(messages, add_generation_prompt=True)) > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"prompt exceeds the {MAX_SEQUENCE_LENGTH}-token context limit")
        return messages

    def generate(
        self,
        user: str,
        history: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        user = user.strip()
        if not user:
            return ""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        messages = self._messages_for_user(user, history)
        prompt_ids = self._template_ids(messages, add_generation_prompt=True)
        generated: list[int] = []

        with self._lock:
            self.brain.ensure_active_gpu()
            hemisphere = self.brain.active_h()
            model = hemisphere.model
            model.eval()
            for start in range(0, max_new_tokens, TTT_CHUNK_TOKENS):
                context = prompt_ids + generated
                context = context[-MAX_SEQUENCE_LENGTH:]
                input_ids = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=next(model.parameters()).device,
                )
                model.config.use_cache = True
                generation_args: dict[str, Any] = {
                    "input_ids": input_ids,
                    "max_new_tokens": min(TTT_CHUNK_TOKENS, max_new_tokens - start),
                    "repetition_penalty": 1.05,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    generation_args.update(
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=40,
                    )
                else:
                    generation_args.update(
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=50,
                    )
                with torch.inference_mode():
                    output = model.generate(**generation_args)
                chunk = [
                    int(value)
                    for value in output[0, input_ids.shape[1] :].detach().cpu().tolist()
                ]
                if not chunk:
                    break
                generated.extend(chunk)
                training_ids = (prompt_ids + generated)[-MAX_SEQUENCE_LENGTH:]
                trainable_start = max(0, len(training_ids) - len(generated))
                labels = [-100] * trainable_start + training_ids[trainable_start:]
                self.last_learning_result = hemisphere.learn(
                    training_ids,
                    labels,
                    intensity=1.0,
                    kind="ttt",
                )
                if not self.last_learning_result["learned"]:
                    LOG.warning(
                        "Qwen TTT step skipped: %s",
                        self.last_learning_result.get("reason"),
                    )
                if self.tokenizer.eos_token_id in chunk:
                    break

            model.config.use_cache = False
            response = self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
            transcript = messages + [{"role": "assistant", "content": response}]
            self.learn_from_transcript(transcript)
            return response

    def learn_from_transcript(self, messages: list[dict[str, Any]]) -> int:
        """Store every role as a causal-LM target, including tool call/results."""
        if not messages:
            raise ValueError("a conversation transcript must contain at least one message")
        transcript = list(messages)
        if transcript[0].get("role") != "system":
            transcript.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        ids = self._template_ids(transcript, add_generation_prompt=False)
        samples = transcript_samples(ids)

        with self._lock:
            for sample in samples:
                self.brain.admit_exchange_sample(sample)
        self._schedule_consolidation()
        return len(samples)

    def status(self) -> dict[str, Any]:
        with self._lock:
            hemispheres = {}
            for name, hemisphere in self.brain.hemispheres.items():
                if hemisphere.model is None:
                    raise RuntimeError(f"Qwen hemisphere {name} has no loaded model")
                hemispheres[name] = {
                    "device": hemisphere.device,
                    "steps": hemisphere.global_step,
                    "ttt_steps": hemisphere.ttt_steps,
                    "consolidation_steps": hemisphere.consolidate_steps,
                    "loss": hemisphere.last_loss,
                    "replay": len(hemisphere.replay),
                    "debt": len(hemisphere.debt),
                    "repayment_capacity": hemisphere.debt.repayment_capacity,
                    "mlp_widths": [
                        layer.mlp.gate_proj.out_features
                        for layer in hemisphere.model.model.layers
                    ],
                }
            return {
                "model": self.model_name,
                "training_device": self.device,
                "active": self.brain.active,
                "passive": self.brain.passive_h().name,
                "switch_count": self.brain.switch_count,
                "background_consolidation": (
                    self._consolidation_future is not None
                    and not self._consolidation_future.done()
                ),
                "consolidation_error": self._consolidation_error,
                "hemispheres": hemispheres,
            }

    def save(self) -> None:
        self._await_consolidation()
        with self._lock:
            self.brain.save()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_dir / "model_name.json.tmp"
        temporary.write_text(
            json.dumps({"model_name": self.model_name}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_dir / "model_name.json")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.save()
        finally:
            self._consolidator.shutdown(wait=True)
            self._closed = True
