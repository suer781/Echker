"""双半球大脑打包器：整个在线学习大脑 ⇄ 单一模型文件（.brainpack）。

一个文件包含 chat_dual 的全部持久化状态：
- 左右半球完整 byt5-small 真权重（fp32 state_dict）
- 双 Adafactor 优化器状态（曲线学习强度依赖它续跑）
- 双回放缓冲 + 债务账本（字节语义）
- 切换状态机元数据（活动侧/紧迫度积分/压力基线/切换历史）

导出 → 单文件可搬家/备份/分享；导入 → 原子落成 chat_dual 可加载的
state 目录，学习从断点无缝继续。

用法：
  python brain_pack.py export --state-dir state_dual --out my_brain.brainpack
  python brain_pack.py import --file my_brain.brainpack --state-dir state_dual
  python brain_pack.py verify --file my_brain.brainpack
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from byt5_dual import BASE, Byt5DualBrain

MAGIC = "BYT5_DUAL_BRAIN_V1"
ROOT = Path(__file__).resolve().parent


def collect_hemisphere(hemisphere) -> dict:
    hemisphere.to_cpu()   # 权重回到 CPU 再收集（幂等）
    return {
        "model": hemisphere.model.state_dict(),
        "optimizer": hemisphere.optimizer.state_dict(),
        "memory": {"replay": hemisphere.replay.to_dict(), "debt": hemisphere.debt.to_dict()},
    }


def export(state_dir: Path, out_file: Path) -> None:
    brain = Byt5DualBrain(BASE, state_dir)
    restored = brain.load()
    if not restored:
        print("[警告] 状态目录无完整检查点：导出的是全新大脑（两侧=基座权重）。", flush=True)
    payload = {
        "magic": MAGIC,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base": BASE,
        "left": collect_hemisphere(brain.hemispheres["left"]),
        "right": collect_hemisphere(brain.hemispheres["right"]),
        "brain": {
            "active": brain.active,
            "switch_count": brain.switch_count,
            "switch_urgency_acc": brain.switch_urgency_acc,
            "pressure_baseline": dict(brain.pressure_baseline),
            "switch_log": brain.switch_log[-64:],
        },
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)
    print(json.dumps({
        "exported": str(out_file),
        "size_gb": round(out_file.stat().st_size / 1024**3, 2),
        "switch_count": brain.switch_count,
        "active": brain.active,
        "note": "payload 含 created 时间戳——即使权重无变化，两次导出的哈希也不同；变更检测请比对权重张量",
    }, ensure_ascii=False), flush=True)


def import_brain(in_file: Path, state_dir: Path, force: bool = False) -> None:
    brain_json = state_dir / "brain.json"
    if brain_json.exists() and not force:
        raise SystemExit(f"[拒绝导入] {state_dir} 已存在一个大脑（brain.json）。"
                         f"确认覆盖请加 --force，或换一个 --state-dir。")
    payload = torch.load(in_file, map_location="cpu", weights_only=True)
    if payload.get("magic") != MAGIC:
        raise SystemExit("[错误] 文件不是有效的 .brainpack（magic 不匹配）。")
    brain = Byt5DualBrain(payload.get("base", BASE), state_dir)
    for name in ("left", "right"):
        hemisphere = brain.hemispheres[name]
        part = payload[name]
        hemisphere.model.load_state_dict(part["model"])          # 全量替换权重（CPU）
        hemisphere.optimizer.load_state_dict(part["optimizer"])  # 状态对应同构参数
        hemisphere.replay.load_dict(part["memory"]["replay"])
        hemisphere.debt.load_dict(part["memory"]["debt"])
    b = payload["brain"]
    brain.active = b.get("active") if b.get("active") in ("left", "right") else "left"
    brain.switch_count = int(b.get("switch_count", 0))
    brain.switch_urgency_acc = max(0.0, min(1.0, float(b.get("switch_urgency_acc", 0.0))))
    baselines = b.get("pressure_baseline", {})
    brain.pressure_baseline = {
        name: max(0.0, float(baselines.get(name, 0.0))) for name in ("left", "right")
    }
    log = b.get("switch_log")
    brain.switch_log = log if isinstance(log, list) else []
    brain.save()   # 原子落成 chat_dual 可加载的目录格式
    print(json.dumps({
        "imported": str(in_file),
        "state_dir": str(state_dir),
        "active": brain.active,
        "switch_count": brain.switch_count,
    }, ensure_ascii=False), flush=True)


def verify(in_file: Path) -> None:
    if not in_file.exists():
        raise SystemExit(f"[错误] 文件不存在：{in_file}")
    try:
        payload = torch.load(in_file, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"[错误] 文件无法解析为 .brainpack：{exc}")
    if payload.get("magic") != MAGIC:
        raise SystemExit("[错误] magic 不匹配。")
    report = {
        "magic": payload["magic"],
        "created": payload.get("created"),
        "base": payload.get("base"),
        "hemispheres": {},
        "brain": payload.get("brain"),
    }
    for name in ("left", "right"):
        part = payload[name]
        model_sd = part["model"]
        param_count = sum(t.numel() for t in model_sd.values() if hasattr(t, "numel"))
        report["hemispheres"][name] = {
            "tensors": len(model_sd),
            "params": param_count,
            "optimizer_entries": len(part["optimizer"].get("state", {})),
            "debt_items": len(part["memory"]["debt"].get("items", {})),
            "replay_samples": len(part["memory"]["replay"].get("samples", [])),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def export_hf(in_file: Path, side: str, out_dir: Path) -> None:
    """从 .brainpack 导出标准 HuggingFace 格式单侧模型目录。

    产出 config.json（tie_word_embeddings=False）+ model.safetensors +
    tokenizer——任何标准 HF 加载器（transformers/vLLM/diffusers 生态）
    都能直接 from_pretrained 使用。注意：这是静态权重快照，不含
    优化器/账本/切换状态（在线学习状态只在 .brainpack 里）。
    """
    if side not in ("left", "right"):
        raise SystemExit("[错误] --side 只能是 left 或 right")
    payload = torch.load(in_file, map_location="cpu", weights_only=True)
    if payload.get("magic") != MAGIC:
        raise SystemExit("[错误] magic 不匹配。")
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    config = AutoConfig.from_pretrained(payload.get("base", BASE))
    config.tie_word_embeddings = False   # 双键值本就分离，显式声明避免加载歧义
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.load_state_dict(payload[side]["model"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(payload.get("base", BASE)).save_pretrained(out_dir)
    print(json.dumps({
        "exported_hf": str(out_dir),
        "side": side,
        "files": sorted(f.name for f in out_dir.iterdir()),
        "note": "标准 HF 格式：transformers.from_pretrained 直接可用（静态权重快照，无在线学习状态）",
    }, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="双半球大脑 ⇄ 单一模型文件")
    sub = parser.add_subparsers(dest="command", required=True)
    p_exp = sub.add_parser("export", help="state 目录 → 单一 .brainpack 文件")
    p_exp.add_argument("--state-dir", type=Path, default=ROOT / "state_dual")
    p_exp.add_argument("--out", type=Path, required=True)
    p_imp = sub.add_parser("import", help="单一 .brainpack 文件 → state 目录")
    p_imp.add_argument("--file", type=Path, required=True)
    p_imp.add_argument("--state-dir", type=Path, default=ROOT / "state_dual")
    p_imp.add_argument("--force", action="store_true", help="覆盖已存在的大脑")
    p_ver = sub.add_parser("verify", help="校验 .brainpack 文件内容")
    p_ver.add_argument("--file", type=Path, required=True)
    p_hf = sub.add_parser("export-hf", help=".brainpack → 标准 HuggingFace 模型目录（单侧）")
    p_hf.add_argument("--file", type=Path, required=True)
    p_hf.add_argument("--side", choices=("left", "right"), default="left")
    p_hf.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.state_dir, args.out)
    elif args.command == "import":
        import_brain(args.file, args.state_dir, args.force)
    elif args.command == "verify":
        verify(args.file)
    elif args.command == "export-hf":
        export_hf(args.file, args.side, args.out)
    return 0
    p_exp = sub.add_parser("export", help="state 目录 → 单一 .brainpack 文件")
    p_exp.add_argument("--state-dir", type=Path, default=ROOT / "state_dual")
    p_exp.add_argument("--out", type=Path, required=True)
    p_imp = sub.add_parser("import", help="单一 .brainpack 文件 → state 目录")
    p_imp.add_argument("--file", type=Path, required=True)
    p_imp.add_argument("--state-dir", type=Path, default=ROOT / "state_dual")
    p_imp.add_argument("--force", action="store_true", help="覆盖已存在的大脑")
    p_ver = sub.add_parser("verify", help="校验 .brainpack 文件内容")
    p_ver.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.state_dir, args.out)
    elif args.command == "import":
        import_brain(args.file, args.state_dir, args.force)
    elif args.command == "verify":
        verify(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
