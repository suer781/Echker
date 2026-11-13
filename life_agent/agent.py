"""Persistent dual-hemisphere continual-learning agent over universal byte frames."""
from __future__ import annotations

import logging
from pathlib import Path
import random
import threading
import time
import traceback
from typing import Any

import torch
from torch.nn import functional as F

from .byte_codec import (
    ASSISTANT,
    BOS,
    EOS,
    PAD,
    SEP,
    USER,
    ByteFrame,
    build_frame_prompt,
    frame_dialogue_sample,
    observation_chunks,
    observation_sample,
)
from .config import AppConfig
from .memory import Sample
from .switching import AdaptiveSwitchController
from .worker import HemisphereState, TrainingWorker

LOG = logging.getLogger(__name__)


class LifeAgent:
    def __init__(self, config: AppConfig, state_dir: Path) -> None:
        config.validate()
        self.config = config
        self.state_dir = state_dir
        self.worker = TrainingWorker(config)
        self.active = "left"
        self.history: list[tuple[ByteFrame, ByteFrame]] = []
        self.interaction_count = 0
        self.observed_frames = 0
        self.switch_count = 0
        self.created_at = time.time()
        self.last_switch = time.monotonic()
        self.last_switch_probe = -1
        self.switch_controller = AdaptiveSwitchController(config.runtime.adaptive_switching)
        self.switch_decisions: list[dict[str, Any]] = []
        self.last_checkpoint = time.monotonic()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.random = random.Random(config.runtime.seed)
        self.torch_random = torch.Generator(device="cpu")
        self.torch_random.manual_seed(config.runtime.seed)
        self.background_errors: list[str] = []
        self.pending_consolidation: dict[str, Any] = {}
        self.monitor_thread = threading.Thread(target=self._monitor, name="homeostasis", daemon=True)
        self.started = False

    @property
    def passive(self) -> str:
        return "right" if self.active == "left" else "left"

    def state(self, name: str | None = None) -> HemisphereState:
        return self.worker.states[name or self.active]

    def start(self) -> None:
        if self.started:
            return
        self.worker.start()
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor, name="homeostasis", daemon=True)
        self.monitor_thread.start()
        self.started = True
        LOG.info("agent started with %s hemisphere active", self.active)

    def _next_token(self, logits: torch.Tensor, generated_count: int, required_minimum: int = 4) -> int:
        runtime = self.config.runtime
        values = logits.detach().float().cpu().clone()
        values = torch.nan_to_num(values, nan=-1e4, posinf=1e4, neginf=-1e4)
        for control in (PAD, BOS, USER, ASSISTANT, SEP):
            values[control] = -1e4
        if generated_count < required_minimum:
            values[EOS] = -1e4
        values = values / max(runtime.temperature, 1e-5)
        if runtime.top_k > 0:
            k = min(runtime.top_k, values.numel())
            threshold = values.topk(k).values[-1]
            values[values < threshold] = -1e4
        probabilities = F.softmax(values, dim=-1)
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        cumulative = sorted_probabilities.cumsum(dim=0)
        remove = cumulative - sorted_probabilities > runtime.top_p
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum().clamp_min(1e-12)
        selected = torch.multinomial(sorted_probabilities, 1, generator=self.torch_random).item()
        return int(sorted_indices[selected].item())

    def _generate_chunk(
        self, prompt: list[int], generated: list[int], limit: int, required_minimum: int = 4
    ) -> tuple[list[int], bool]:
        state = self.state()
        produced: list[int] = []
        reached_eos = False
        for _ in range(limit):
            context = prompt + generated + produced
            if len(context) > self.config.model.max_seq_len:
                context = [BOS, *context[-(self.config.model.max_seq_len - 1):]]
            token_ids = torch.tensor([context], dtype=torch.long, device="cpu")
            with state.lock, torch.inference_mode():
                logits = state.cpu_model(token_ids)[0, -1]
            token = self._next_token(logits, len(generated) + len(produced), required_minimum)
            if token == EOS:
                reached_eos = True
                break
            produced.append(token)
        return produced, reached_eos

    def _validate_frame(self, frame: ByteFrame) -> int:
        size = len(frame.to_bytes())
        if size > self.config.runtime.max_frame_bytes:
            raise ValueError(
                f"encoded frame is {size} bytes; configured limit is {self.config.runtime.max_frame_bytes}"
            )
        return size

    def _remember(self, sample: Sample) -> None:
        with self.state().lock:
            self.state().replay.add(sample)
        with self.state(self.passive).lock:
            self.state(self.passive).debt.add(sample, self.config.runtime.debt_retention)

    def interact(self, user: str) -> str:
        """Text convenience wrapper; the internal path is still only ByteFrame."""
        if not user.strip():
            return ""
        response = self.interact_frame(ByteFrame.text(user.strip()), ByteFrame.text(""))
        return response.as_text(errors="replace")

    def interact_frame(
        self, stimulus: ByteFrame, response_spec: ByteFrame | None = None
    ) -> ByteFrame:
        if not self.started:
            raise RuntimeError("agent must be started before interaction")
        specification = response_spec or stimulus.empty()
        if specification.payload:
            raise ValueError("response_spec must describe format metadata with an empty payload")
        self._validate_frame(stimulus)
        with self.lock:
            prompt = build_frame_prompt(
                self.history,
                stimulus,
                specification,
                self.config.model.max_seq_len,
                self.config.runtime.context_turns,
            )
            generated: list[int] = []
            reached_eos = False
            expected_length = specification.expected_payload_length()
            if expected_length is not None:
                if expected_length < 0 or expected_length > self.config.runtime.max_new_tokens:
                    raise ValueError(
                        f"response structure requires {expected_length} bytes; "
                        f"max_new_tokens is {self.config.runtime.max_new_tokens}"
                    )
                target_length = expected_length
                required_minimum = expected_length
            else:
                target_length = self.config.runtime.max_new_tokens
                required_minimum = 4
            while len(generated) < target_length and not reached_eos:
                remaining = target_length - len(generated)
                limit = min(self.config.runtime.ttt_chunk_tokens, remaining)
                chunk, reached_eos = self._generate_chunk(
                    prompt, generated, limit, required_minimum=required_minimum
                )
                generated.extend(chunk)
                if chunk:
                    partial = specification.with_payload(bytes(generated))
                    ids, labels = frame_dialogue_sample(stimulus, partial, specification)
                    sample = Sample(ids=ids, labels=labels, weight=0.5, source=f"ttt:{stimulus.codec}")
                    self.worker.submit(
                        "ttt", self.active, [sample], self.config.runtime.ttt_steps, wait=True
                    )
            response = specification.with_payload(bytes(generated))
            response.validate_payload()
            ids, labels = frame_dialogue_sample(stimulus, response, specification)
            self._remember(Sample(ids=ids, labels=labels, weight=0.8, source=f"dialogue:{stimulus.codec}"))
            input_ids, input_labels = observation_sample(stimulus)
            self._remember(
                Sample(ids=input_ids, labels=input_labels, weight=0.25, source=f"observe:{stimulus.codec}")
            )
            self.history.append((stimulus, response))
            self.history = self.history[-self.config.runtime.context_turns:]
            self.interaction_count += 1
            self._schedule_passive_consolidation()
            # Interactive traffic has the same starvation risk as continuous
            # feed: high-priority TTT can indefinitely precede consolidation.
            self._drain_passive_consolidation()
            self._maybe_switch()
            return response

    def feed_frame(self, frame: ByteFrame, weight: float = 0.5) -> dict[str, Any]:
        """Observe an entire arbitrary frame in streaming chunks, with no reply."""
        if not self.started:
            raise RuntimeError("agent must be started before interaction")
        size = self._validate_frame(frame)
        with self.lock:
            samples = [
                Sample(ids=ids, labels=labels, weight=weight, source=f"feed:{frame.codec}")
                for ids, labels in observation_chunks(frame, self.config.model.max_seq_len)
            ]
            # One GPU job now covers the whole frame. The worker's batches visit
            # every sample once instead of paying queue latency per 509 bytes.
            steps = max(
                1,
                ((len(samples) + self.config.runtime.batch_size - 1) // self.config.runtime.batch_size)
                * self.config.runtime.feed_steps,
            )
            result = self.worker.submit("train", self.active, samples, steps, wait=True)
            for sample in samples:
                self._remember(sample)
            self.observed_frames += 1
            self.interaction_count += 1
            self._schedule_passive_consolidation()
            # A continuously fed stream otherwise keeps priority-3 training jobs
            # ahead of priority-4 consolidation forever. Drain that one
            # background job so the passive hemisphere actually learns and the
            # adaptive switch sees measurable work.
            self._drain_passive_consolidation()
            self._maybe_switch()
            return {
                "codec": int(frame.codec),
                "encoded_bytes": size,
                "payload_bytes": len(frame.payload),
                "chunks": len(samples),
                "trained_steps": int(result["steps"]),
            }

    def feed_file(self, path: Path) -> dict[str, Any]:
        return self.feed_frame(ByteFrame.from_file(path))

    def _debt_training_samples(self, name: str) -> list[Sample]:
        state = self.state(name)
        with state.lock:
            limit = self.config.runtime.batch_size * self.config.runtime.consolidation_steps
            samples = state.debt.due(limit)
            if state.replay.samples and self.random.random() < self.config.runtime.replay_probability:
                samples.extend(state.replay.choose(max(1, limit // 4), self.random))
        return samples

    def _schedule_passive_consolidation(self) -> None:
        name = self.passive
        future = self.pending_consolidation.get(name)
        if future is not None:
            if not future.done():
                return
            self.pending_consolidation.pop(name, None)
            future.result()  # never hide a failed background consolidation
        samples = self._debt_training_samples(name)
        if not samples:
            return
        self.pending_consolidation[name] = self.worker.submit(
            "consolidate", name, samples, self.config.runtime.consolidation_steps, wait=False
        )

    def _drain_passive_consolidation(self) -> None:
        future = self.pending_consolidation.pop(self.passive, None)
        if future is not None:
            future.result()

    def _wait_consolidations(self) -> None:
        first_error = None
        for name, future in list(self.pending_consolidation.items()):
            try:
                future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self.pending_consolidation.pop(name, None)
        if first_error is not None:
            raise first_error

    def _exchange_switch_debt(self, old: str, new: str) -> None:
        for source_name, target_name in ((old, new), (new, old)):
            source = self.state(source_name)
            target = self.state(target_name)
            with source.lock:
                latest = [item.clone() for item in source.replay.latest(2)]
            with target.lock:
                for sample in latest:
                    target.debt.add(sample, self.config.runtime.debt_retention)

    def switch(self) -> None:
        if not self.started:
            raise RuntimeError("agent is not running")
        with self.lock:
            self._switch_locked()

    def _switch_locked(self, reason: str | None = None, decision: Any | None = None) -> None:
        # A pending consolidation belongs to the current role. Resolve it
        # before roles change, otherwise its future is stranded under the old
        # hemisphere key and its failure is only discovered much later.
        self._wait_consolidations()
        old, new = self.active, self.passive
        self._exchange_switch_debt(old, new)
        incoming = self._debt_training_samples(new)
        if incoming:
            self.worker.submit(
                "consolidate", new, incoming, self.config.runtime.consolidation_steps, wait=True
            )
        self.worker.submit("sync", wait=True)
        self.active = new
        self.switch_count += 1
        self.last_switch = time.monotonic()
        self.last_switch_probe = self.interaction_count
        passive_state = self.state(self.passive)
        with passive_state.lock:
            clean_pressure = passive_state.debt.clean_pressure()
        # Baselines are role-indexed. Passing left/right by name here was the
        # B1 defect: when active=right the two numbers silently swapped roles.
        self.switch_controller.note_switch(
            clean_pressure,
            self.state(self.active).online_learning_steps,
            self.state(self.passive).passive_consolidation_steps,
            self.active,
            self.passive,
        )
        record = decision.to_dict() if decision is not None else {
            "should_switch": True,
            "reason": reason or "manual switch",
            "active_name": old,
            "passive_name": new,
            "manual": True,
        }
        record["switched"] = True
        record["switch_index"] = self.switch_count
        record["active_after"] = self.active
        record["passive_after"] = self.passive
        self.switch_decisions.append(record)
        self.switch_decisions = self.switch_decisions[-128:]
        detail = f"; {reason}" if reason else ""
        LOG.info("hemisphere switched: %s -> %s%s", old, new, detail)
        self._schedule_passive_consolidation()

    def _maybe_switch(self) -> None:
        active_name, passive_name = self.active, self.passive
        passive_state = self.state(passive_name)
        with passive_state.lock:
            clean_pressure = passive_state.debt.clean_pressure()
            learned_capacity = passive_state.debt.repayment_capacity
            has_debt = len(passive_state.debt) > 0
        if not self.switch_controller._roles_match(active_name, passive_name):
            # Legacy checkpoints had no role labels. Establish a correct
            # baseline once, at the current data/step boundary, instead of
            # guessing or switching immediately.
            self.switch_controller.rebase(
                clean_pressure,
                self.state(active_name).online_learning_steps,
                self.state(passive_name).passive_consolidation_steps,
                active_name,
                passive_name,
            )
            LOG.info("re-established role-correct switch baseline for %s/%s", active_name, passive_name)
        decision = self.switch_controller.decide(
            clean_pressure=clean_pressure,
            learned_capacity=learned_capacity,
            active_work=self.state(active_name).online_learning_steps,
            passive_work=self.state(passive_name).passive_consolidation_steps,
            has_debt=has_debt,
            active_name=active_name,
            passive_name=passive_name,
        )
        if decision.should_switch:
            self._switch_locked(
                "adaptive: "
                f"clean_growth={decision.pressure_growth:.4f}, "
                f"capacity={decision.learned_capacity:.4f}, "
                f"work_share={decision.work_share:.3f}",
                decision,
            )

    def _memory_signals(self) -> dict[str, dict[str, float]]:
        # Observation only. The cleaner's familiarity signal affects stored
        # sample weights; it is not an input to the pressure/work switch.
        return {
            name: self.state(name).replay.cleaner.signal() for name in ("left", "right")
        }

    def _monitor(self) -> None:
        while not self.stop_event.wait(1.0):
            try:
                with self.lock:
                    self._schedule_passive_consolidation()
                    self._maybe_switch()
                    if time.monotonic() - self.last_checkpoint >= self.config.runtime.checkpoint_interval_s:
                        self.save()
            except Exception:
                self.background_errors.append(traceback.format_exc())
                LOG.exception("autonomous homeostasis cycle failed")

    def save(self) -> Path:
        from .store import StateStore

        with self.lock:
            # A checkpoint is a snapshot at a quiescent boundary. Without
            # this wait, priority-2 snapshot can overtake priority-4
            # consolidation and persist a debt ledger that has been charged
            # but not trained.
            self._wait_consolidations()
            path = StateStore(self.state_dir).save(self)
            self.last_checkpoint = time.monotonic()
            return path

    def load(self) -> bool:
        from .store import StateStore

        if self.started or self.worker.started:
            raise RuntimeError("checkpoint loading is only allowed before the worker starts")
        with self.lock:
            return StateStore(self.state_dir).load(self)

    def status(self) -> dict[str, Any]:
        with self.lock:
            result = self.worker.metrics()
            result.update(
                {
                    "active": self.active,
                    "passive": self.passive,
                    "interface": "BLF1",
                    "interactions": self.interaction_count,
                    "observed_frames": self.observed_frames,
                    "switches": self.switch_count,
                    "adaptive_switch": (
                        self.switch_controller.last_decision.to_dict()
                        if self.switch_controller.last_decision
                        else None
                    ),
                    "switch_decisions": list(self.switch_decisions),
                    "memory_signals": self._memory_signals(),
                    "background_errors": list(self.background_errors),
                    "step_accounting": {
                        name: {
                            "global": self.state(name).global_step,
                            "train": self.state(name).train_steps,
                            "ttt": self.state(name).ttt_steps,
                            "consolidate": self.state(name).consolidate_steps,
                            "online_learning": self.state(name).online_learning_steps,
                            "passive_consolidation": self.state(name).passive_consolidation_steps,
                        }
                        for name in ("left", "right")
                    },
                    "history_turns": len(self.history),
                    "uptime_seconds": int(time.time() - self.created_at),
                }
            )
            return result

    def close(self) -> None:
        if not self.started and not self.worker.started:
            return
        self.stop_event.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        background_error = None
        with self.lock:
            try:
                self._wait_consolidations()
            except Exception as exc:
                background_error = exc
            try:
                self.save()
            finally:
                self.worker.stop()
                self.started = False
        if background_error is not None:
            raise background_error
        LOG.info("agent stopped and state persisted")
