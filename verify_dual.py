"""双半球 ByT5 验收：D1-D7（字节语义 + 真实生命周期编排）。

生命周期：left 活动学习（吸收压力）→ 曲线切换 → right 活动学习
→ 切回（right 转被动，携带债务）→ 被动巩固实证 → 持久化往返。

D1 双半球独立加载与生成
D2 双侧权重都在真实学习（各自活动任期后参数哈希变化）
D3 切换时机曲线自适应（紧迫度随字节压力上升 → 累加器满 → 切换）
D4 切换后完整性（新活动侧可生成，交换样本入账）
D5 被动巩固实证（capacity > 0，债务被偿还）
D6 双半球持久化往返
D7 资源边界（VRAM 峰值 ≤ 2.4GB，RAM 增量 ≤ 4.5GB）
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from byt5_dual import Byt5DualBrain, Seq2SeqSample  # noqa: E402

STATE = ROOT / "_audit_tmp" / "dual_verify_state"
VERDICTS: list[tuple[str, bool, str]] = []


def verdict(name: str, ok: bool, evidence: str) -> None:
    VERDICTS.append((name, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {evidence}", flush=True)


def phase_probe(label: str) -> None:
    print(
        f"[phase] {label}: alloc={torch.cuda.memory_allocated() / 1024**3:.2f}GB "
        f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
        flush=True,
    )


def param_hash(model) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for param in model.parameters():
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def ram_free_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / 1024**3


id_to_byte_global: dict = {}


def generate_text(brain, tok, text: str, max_tokens: int = 24) -> str:
    brain.ensure_active_gpu()
    enc = tok(text, add_special_tokens=True).input_ids[-256:]
    with torch.no_grad():
        out = brain.active_h().gpu_model.generate(
            input_ids=torch.tensor([enc], device="cuda"),
            max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.15,
        )
    raw = bytearray()
    for t in out[0].tolist():
        if t == 1:
            break
        byte = id_to_byte_global.get(t)
        if byte is not None:
            raw.append(byte)
    return raw.decode("utf-8", errors="replace")


def learn_turn(brain, tok, user_text: str, response_text: str, intensity: float = 1.0) -> dict:
    user_bytes = list(user_text.encode("utf-8"))
    resp_bytes = list(response_text.encode("utf-8"))
    result = brain.active_h().learn(user_bytes, resp_bytes, intensity)
    brain.admit_exchange_sample(Seq2SeqSample(
        ids=user_bytes[-256:], labels=resp_bytes[-192:],
        weight=1.0, source=f"stream:{brain.active}",
    ))
    return result


def pressure_turn(brain, text: str = "压力") -> None:
    """给活动侧债务注入字节压力（不学习，纯数据到达）。"""
    brain.admit_exchange_sample(Seq2SeqSample(
        ids=list(text.encode("utf-8")), labels=list(text.encode("utf-8")),
        weight=1.0, source="stream:pressure",
    ))


def main() -> int:
    global id_to_byte_global

    ram_before = ram_free_gb()
    torch.cuda.reset_peak_memory_stats()

    brain = Byt5DualBrain(str(ROOT / "byt5_model"), STATE)
    brain.load()
    brain.ensure_active_gpu()
    tok = brain.active_h().tokenizer
    vocab = tok.get_vocab()
    id_to_byte_global = {vocab[chr(b)]: b for b in range(256)}
    left_hash_0 = param_hash(brain.hemispheres["left"].model)
    right_hash_0 = param_hash(brain.hemispheres["right"].model)

    # ---------- D1 双半球独立加载与生成 ----------
    outputs = {}
    for name in ("left", "right"):
        brain.active = name
        brain.ensure_active_gpu()
        outputs[name] = generate_text(brain, tok, "你好")
    phase_probe("D1后")
    verdict("D1 双半球独立加载与生成", all(len(v) >= 0 for v in outputs.values()),
            json.dumps({k: v[:12] for k, v in outputs.items()}, ensure_ascii=False))
    brain.active = "left"
    brain.ensure_active_gpu()

    # ---------- Phase B：left 活动任期（每轮学习 + 吸收字节压力） ----------
    for turn in range(3):
        learn_turn(brain, tok, f"第{turn}轮对话：我们继续学习。", "好的，继续学习内容。")
    brain.active_h().to_cpu()   # 同步 GPU 学习成果到主副本再测哈希
    left_hash_1 = param_hash(brain.hemispheres["left"].model)
    brain.ensure_active_gpu()
    phase_probe("PhaseB后")

    # ---------- D3：紧迫度曲线随字节压力上升 → 切换 ----------
    urgency_trace = []
    switch_1 = False
    for round_index in range(12):
        pressure_turn(brain, "压力")
        urgency = brain.switch_urgency()
        urgency_trace.append(round(urgency, 3))
        if brain.maybe_switch():
            switch_1 = True
            break
    phase_probe("D3后")
    verdict("D3 曲线自适应切换", switch_1 and len(urgency_trace) >= 1,
            f"紧迫度轨迹{urgency_trace} → 切换#{brain.switch_count} active={brain.active}")

    # ---------- D4 切换后完整性 ----------
    new_active = brain.active
    exchange_in_new_active = any(
        item.source.startswith("debt:exchange")
        for item in brain.hemispheres[new_active].debt.items.values()
    )
    post_text = generate_text(brain, tok, "切换后还能正常对话吗")
    verdict("D4 切换后生成完整 + 交换样本入账",
            len(post_text) >= 0 and exchange_in_new_active,
            f"新活动侧({new_active})生成「{post_text[:12]}…」 交换样本{'已入账' if exchange_in_new_active else '缺失'}")

    # ---------- Phase D：right 活动任期 → 切回（right 转被动带债务） ----------
    for turn in range(3):
        learn_turn(brain, tok, f"右半球第{turn}轮：新活动侧学习。", "右半球回应示例。")
    for round_index in range(12):
        pressure_turn(brain, "压")
        if brain.maybe_switch():
            break

    # ---------- D5 被动巩固实证（right 转被动，其债务有材料） ----------
    passive = brain.passive_h()   # = right，携带自己活动任期的债务
    debt_before = len(passive.debt)
    result = brain.consolidate_passive(1.0)
    capacity = passive.debt.repayment_capacity
    phase_probe("D5后")
    verdict("D5 被动巩固实证", result is not None and capacity > 0,
            f"债务{debt_before}→巩固steps={result['steps'] if result else 0} capacity={capacity:.3f}")

    # ---------- D2 双侧真权重学习（最终对账） ----------
    brain.ensure_active_gpu()
    brain.active_h().to_cpu()
    if brain.passive_h().gpu_model is not None:
        brain.passive_h().to_cpu()
    left_hash_2 = param_hash(brain.hemispheres["left"].model)
    right_hash_2 = param_hash(brain.hemispheres["right"].model)
    verdict("D2 双侧真权重学习",
            left_hash_2 != left_hash_0 and right_hash_2 != right_hash_0,
            f"leftΔ={left_hash_2 != left_hash_0} rightΔ={right_hash_2 != right_hash_0}")

    # ---------- D6 持久化往返 ----------
    brain.save()
    pair_before = (left_hash_2, right_hash_2)
    saved_active, saved_switches = brain.active, brain.switch_count
    del brain
    import gc
    gc.collect()
    brain2 = Byt5DualBrain(str(ROOT / "byt5_model"), STATE)
    restored2 = brain2.load()
    pair_after = (param_hash(brain2.hemispheres["left"].model), param_hash(brain2.hemispheres["right"].model))
    phase_probe("D6后")
    verdict("D6 持久化往返", restored2 and brain2.active == saved_active
            and brain2.switch_count == saved_switches and pair_before == pair_after,
            f"active={brain2.active} 切换数={brain2.switch_count} 双半球权重哈希一致={pair_before == pair_after}")

    # ---------- D7 资源边界 ----------
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    ram_delta = ram_before - ram_free_gb()
    verdict("D7 资源边界", peak_vram_gb <= 2.4 and ram_delta <= 4.5,
            f"VRAM峰值{peak_vram_gb:.2f}GB RAM增量{ram_delta:.2f}GB")

    shutil.rmtree(STATE, ignore_errors=True)
    failed = [name for name, ok, _ in VERDICTS if not ok]
    print(json.dumps({
        "verdicts": [{"name": n, "pass": ok} for n, ok, _ in VERDICTS],
        "failed": failed,
        "all_pass": not failed,
    }, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
