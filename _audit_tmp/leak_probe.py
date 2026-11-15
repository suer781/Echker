"""泄漏探针：10 轮与 chat_dual 完全相同的代码路径，逐轮测量显存/内存。"""
import ctypes
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from byt5_dual import Byt5DualBrain, Seq2SeqSample  # noqa: E402


def process_commit_gb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = ctypes.windll.kernel32.K32GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb)
    return (pmc.PrivateUsage / 1024**3) if ok else -1.0


def main() -> int:
    state = ROOT / "_audit_tmp" / "leak_state"
    brain = Byt5DualBrain(str(ROOT / "byt5_model"), state)
    brain.load()
    brain.ensure_active_gpu()
    from transformers import AutoTokenizer
    tok = brain.active_h().tokenizer
    vocab = tok.get_vocab()
    id2byte = {vocab[chr(b)]: b for b in range(256)}
    byte_to_id = {b: vocab[chr(b)] for b in range(256)}
    governor_free = torch.cuda.mem_get_info()[0]

    print(f"{'轮':>3} {'alloc':>7} {'reserved':>8} {'commit':>7} {'GPUfree':>8}", flush=True)
    prompts = ["你好", "今天天气怎么样", "1加1等于几", "给我讲个故事", "你叫什么名字",
               "重复一遍刚才的话", "讲个长一点的笑话", "你是谁", "再说说", "测试显存第10轮"]
    for turn, text in enumerate(prompts, 1):
        active = brain.active_h()
        active.ensure_gpu()
        enc = tok(text, add_special_tokens=True).input_ids[-256:]
        inputs = torch.tensor([enc], device="cuda")
        active.model.eval()
        with torch.no_grad():
            out = active.model.generate(input_ids=inputs, max_new_tokens=64,
                                        do_sample=False, repetition_penalty=1.3)
        resp_tokens = [t for t in out[0].tolist() if 3 <= t <= 258]
        resp_bytes = bytes(id2byte.get(t, 63) for t in resp_tokens)
        response = resp_bytes.decode("utf-8", errors="replace")
        # 学习路径（同 chat_dual）
        if response.strip():
            clean = list(resp_bytes)
            user_bytes = list(text.encode("utf-8"))
            result = active.learn(user_bytes, clean, 1.0)
            brain.admit_exchange_sample(Seq2SeqSample(
                ids=user_bytes[-256:], labels=clean[-192:], weight=1.0,
                source=f"stream:{brain.active}"))
        # 巩固 + 切换检查（同 chat_dual）
        try:
            brain.consolidate_passive(1.0)
        except torch.cuda.OutOfMemoryError:
            brain.ensure_active_gpu()
        try:
            brain.maybe_switch()
        except torch.cuda.OutOfMemoryError:
            brain.ensure_active_gpu()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        commit = process_commit_gb()
        gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"{turn:>3} {alloc:7.2f} {reserved:8.2f} {commit:7.2f} {gpu_free:8.2f} "
              f"[{brain.active}] learned={result.get('learned')}", flush=True)
    print("判读：reserved/commit 若逐轮线性增长=泄漏；平台化=正常缓存", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
