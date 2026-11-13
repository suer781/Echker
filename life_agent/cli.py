"""Console entry point; ordinary operation requires no commands beyond chatting."""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

import torch

from .agent import LifeAgent
from .config import AppConfig


def configure_logging(state_dir: Path, verbose: bool = False) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            state_dir / "life.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        ),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local dual-hemisphere byte-level continual learner")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--fresh", action="store_true", help="ignore existing state without deleting it")
    parser.add_argument("--once", type=str, help="run one interaction and exit")
    parser.add_argument("--cpu-threads", type=int, default=0, help="override PyTorch CPU thread count")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.state_dir, args.verbose)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    config = AppConfig.load(args.config) if args.config.exists() else AppConfig.default()
    config.validate()
    agent = LifeAgent(config, args.state_dir)
    try:
        restored = False if args.fresh else agent.load()
        agent.start()
        device = agent.status()["training_device"]
        print("本地持续学习智能体已启动。直接输入文字即可交互；/help 查看可选命令。")
        print(f"推理设备: cpu | 训练设备: {device} | 检查点: {'已恢复' if restored else '全新初始化'}")
        if args.once is not None:
            print(agent.interact(args.once))
            return 0
        while True:
            try:
                user = input("\n你> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user in ("/quit", "/exit"):
                break
            if user == "/help":
                print("普通文字：交互并学习\n/feed <文件路径>：以原始字节帧观察任意文件\n/status：运行状态\n/switch：立即切换\n/save：保存\n/quit：保存并退出")
                continue
            if user.startswith("/feed "):
                path = Path(user[6:].strip().strip('"')).expanduser()
                result = agent.feed_file(path)
                print("已观察字节帧：" + json.dumps(result, ensure_ascii=False))
                continue
            if user == "/status":
                print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
                continue
            if user == "/switch":
                agent.switch()
                print(f"已切换，当前半球：{agent.active}")
                continue
            if user == "/save":
                print(f"已保存：{agent.save()}")
                continue
            response = agent.interact(user)
            print(f"智能体> {response}")
        return 0
    except KeyboardInterrupt:
        print("\n收到停止信号，正在保存。")
        return 130
    finally:
        agent.close()


if __name__ == "__main__":
    sys.exit(main())
