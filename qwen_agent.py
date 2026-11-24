"""Vocabulary-tokenized Qwen chat with explicit-feedback LoRA updates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "你是一个有帮助的助手。请清楚作答；无法确认时明确说明，不要编造事实。"
MAX_SEQUENCE_LENGTH = 512
MAX_CORRECTIONS = 128


def completion_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Mask prompt tokens so optimization is only against the supplied answer."""
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("training example must contain a prompt and a non-empty answer")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full conversation")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


class QwenAgent:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        state_dir: Path = Path("state_qwen"),
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA support is missing; install the project requirements including peft."
            ) from exc

        self.model_name = model_name
        self.state_dir = Path(state_dir)
        self.adapter_dir = self.state_dir / "adapter"
        self.feedback_path = self.state_dir / "feedback.jsonl"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.state_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.state_dir / "model.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model_name") != model_name:
                raise ValueError(
                    f"checkpoint uses {metadata.get('model_name')!r}, not {model_name!r}"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if len(self.tokenizer) < 1000:
            raise ValueError("selected tokenizer does not appear to have a normal text vocabulary")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if self.adapter_dir.exists():
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_dir,
                is_trainable=True,
            )
        else:
            config = LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(base_model, config)
        self.model.to(self.device)
        self.model.config.use_cache = True

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA adapter has no trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=2e-5)
        optimizer_path = self.state_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )
        self.examples = self._load_feedback()
        self.model.eval()

    def _load_feedback(self) -> list[dict[str, str]]:
        if not self.feedback_path.exists():
            return []
        examples: list[dict[str, str]] = []
        with self.feedback_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                user = item.get("user")
                assistant = item.get("assistant")
                if not isinstance(user, str) or not isinstance(assistant, str):
                    raise ValueError(f"invalid feedback record at line {line_number}")
                examples.append({"user": user, "assistant": assistant})
        return examples[-MAX_CORRECTIONS:]

    def _tokenized_example(self, user: str, assistant: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        full = prompt + [{"role": "assistant", "content": assistant}]
        prompt_encoding = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_encoding = self.tokenizer.apply_chat_template(
            full,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = list(prompt_encoding["input_ids"])
        full_ids = list(full_encoding["input_ids"])
        labels = completion_labels(prompt_ids, full_ids)
        if len(full_ids) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"training example is {len(full_ids)} tokens; maximum is {MAX_SEQUENCE_LENGTH}"
            )
        return (
            torch.tensor([full_ids], dtype=torch.long, device=self.device),
            torch.tensor([labels], dtype=torch.long, device=self.device),
        )

    @torch.inference_mode()
    def generate(
        self,
        user: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        while input_ids.shape[1] > MAX_SEQUENCE_LENGTH and len(messages) > 2:
            del messages[1:3]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        if input_ids.shape[1] > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"prompt is {input_ids.shape[1]} tokens; maximum is {MAX_SEQUENCE_LENGTH}"
            )
        input_ids = input_ids.to(self.device)
        self.model.eval()
        generation_args = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args.update(temperature=temperature, top_p=0.9)
        output = self.model.generate(**generation_args)
        return self.tokenizer.decode(
            output[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()

    def learn_from_correction(self, user: str, correction: str) -> dict[str, Any]:
        user = user.strip()
        correction = correction.strip()
        if not user or not correction:
            raise ValueError("both the original user message and correction are required")
        if any(item["user"] == user and item["assistant"] == correction for item in self.examples):
            return {"trained": False, "reason": "duplicate"}

        batch = [{"user": user, "assistant": correction}]
        if self.examples:
            batch.append(random.choice(self.examples))

        self.model.train()
        self.model.config.use_cache = False
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        try:
            for item in batch:
                input_ids, labels = self._tokenized_example(item["user"], item["assistant"])
                output = self.model(input_ids=input_ids, labels=labels)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss; adapter weights were not updated")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach().item()))

            trainable = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient; adapter weights were not updated")
            self.optimizer.step()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            self.model.config.use_cache = True
            self.model.eval()

        self.examples.append({"user": user, "assistant": correction})
        self.examples = self.examples[-MAX_CORRECTIONS:]
        self.save()
        return {"trained": True, "loss": sum(losses) / len(losses), "examples": len(self.examples)}

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: QwenAgent._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [QwenAgent._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(QwenAgent._to_cpu(item) for item in value)
        return value

    def save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        metadata_tmp = self.state_dir / "model.json.tmp"
        metadata_tmp.write_text(
            json.dumps({"model_name": self.model_name}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(metadata_tmp, self.state_dir / "model.json")

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.adapter_dir, safe_serialization=True)
        optimizer_tmp = self.state_dir / "optimizer.pt.tmp"
        torch.save(self._to_cpu(self.optimizer.state_dict()), optimizer_tmp)
        os.replace(optimizer_tmp, self.state_dir / "optimizer.pt")

        feedback_tmp = self.state_dir / "feedback.jsonl.tmp"
        with feedback_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for item in self.examples:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(feedback_tmp, self.feedback_path)
