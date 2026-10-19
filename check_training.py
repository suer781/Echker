# -*- coding: utf-8 -*-
"""训练一键监控：进程 / 存档新鲜度 / 关键指标（cycle、probe、ability、源、金发姑娘）。

用法：python check_training.py
"""
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"c:\Users\13682\Documents\自学习AI"
os.chdir(ROOT)

# 1. 进程检查（PowerShell 一行命令返回 feed.py 的 PID）
ps_cmd = (
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and "
    "$_.CommandLine -match 'feed\\.py' } | Select-Object -First 1 -ExpandProperty ProcessId"
)
try:
    pid = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", ps_cmd], timeout=30
    ).decode("utf-8", errors="replace").strip()
    if pid:
        print(f"[进程] feed.py PID={pid} 运行中")
    else:
        print("[进程] WARN: feed.py 未运行！")
except Exception as e:
    print(f"[进程] 检测失败: {e!r}")

# 2. 存档
ck = os.path.join(ROOT, "dolphin", "fed_state.pt")
if os.path.exists(ck):
    mtime = os.path.getmtime(ck)
    age = int(time.time() - mtime)
    size_mb = os.path.getsize(ck) / 1e6
    print(f"[存档] {time.strftime('%H:%M:%S', time.localtime(mtime))} （{age} 秒前） {size_mb:.1f}MB")
    # 2.1 存档内关键计数器（直接读档，避免日志格式变化/文件名排序漏报）
    try:
        import torch
        _ck = torch.load(ck, map_location="cpu", weights_only=True)
        print(f"[存档指标] cycle={_ck.get('cycle')} "
              f"ability={float(_ck.get('ability', 0.0)):.4f} "
              f"跳过={_ck.get('skipped_too_easy', 0)} "
              f"挂起={_ck.get('suspended_too_hard', 0)}")
    except Exception as e:
        print(f"[存档指标] 读取失败（忽略）：{e!r}")
else:
    print("[存档] WARN: fed_state.pt 不存在！")

# 3. 日志关键指标（按 mtime 取最新训练日志，避免按文件名排序误读旧日志）
logs_dir = os.path.join(ROOT, "实验记录")
logs = [f for f in os.listdir(logs_dir)
        if f.startswith("训练日志") and not f.endswith("_err.txt") and not f.endswith(".err")]
if logs:
    log = os.path.join(logs_dir, max(logs, key=lambda f: os.path.getmtime(os.path.join(logs_dir, f))))
else:
    log = None

if log and os.path.exists(log):
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    text = "".join(lines[-150:])

    def last(pattern):
        ms = re.findall(pattern, text)
        return ms[-1] if ms else None

    cyc = last(r"cycle=(\d+)")
    pr = last(r"probe ([\d.]+)→([\d.]+)")
    ab = last(r"能力=([\d.]+)")
    src = last(r"已喂 \d+ 条（当前源 (\w+)）")
    fed = last(r"已喂 (\d+) 条")
    gl = last(r"跳过=(\d+) 挂起=(\d+)")
    print(f"[周期] cycle={cyc}")
    if pr:
        print(f"[probe] {pr[0]} -> {pr[1]}")
    if ab:
        print(f"[ability] {ab}")
    if src:
        print(f"[源] {src}  已喂 {fed} 条")
    if gl:
        print(f"[金发姑娘] 跳过={gl[0]}  挂起={gl[1]}")
else:
    print("[日志] 未找到训练日志")