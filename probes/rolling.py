"""试卷池自滚动（律 L8 的供给侧机制，2026-10-03 G2-3 施工）。

问题：换卷原先必须人工重跑 build_logic_probe.py——体检卷终身制，
模型长期成长后旧卷会失去分辨力，而人工换卷在自学习循环里等于不存在。

机制（2026-10-04 起候选流已接通，供给侧见 probes/candidates.py）：
- probes/probe_pool/  候选目录：从未进过卷面的题面按文件序排队等待转正；
- 当前卷 = probes/probe.txt（门控正在批改的卷子）；
- rotate()：把候选流的一小节转正入卷，同时把卷内最旧一段退休到
  probes/retired/——退休段可回流训练粮：卷子永不训练，但卷池内部滚动，
  L8"探测集永不训练"不破（一段内容要么在卷上、要么在粮里，从不同时）。
- 指针 pool_pointer.json 记录候选流消费进度（文件名/字节偏移）与退休序号，
  rotate 跨重启可续，候选绝不重复进场。

安全栏：
- 候选池耗尽 → 不动卷（只退休不补入会让卷面缩水，卷须保持在 30-100KB 合同区间）；
- 卷面不足两段 → 不动卷（不许退休唯一一段）；
- 换卷先写临时文件再 os.replace，崩溃不会留下半截卷面；
- 所有切段位置回退到 UTF-8 字符边界，多字节字符零割裂。

候选内容与训练粮/当前卷的重叠抽查（L8 硬性要求 + 入池质检）属于候选流入
池时的质检，由 probes/candidates.py 在入池时强制调用公共引擎 probes/qc.py
完成，不在本模块职责内。

用法：
  python probes/rolling.py --status    # 查看卷面/池/退休档/候选源现状
  python probes/rolling.py --rotate    # 手动滚动一次
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_PATH = os.path.join(ROOT, "probes", "probe.txt")
POOL_DIR = os.path.join(ROOT, "probes", "probe_pool")
RETIRED_DIR = os.path.join(ROOT, "probes", "retired")
POINTER_NAME = "pool_pointer.json"  # 指针住在池目录里：整个池可整体搬走/隔离


def _align(data: bytes, pos: int) -> int:
    """把字节位置回退到 UTF-8 字符边界（连续字节的最高两位是 10 则属字符中腹）。"""
    while pos > 0 and pos < len(data) and (data[pos] & 0xC0) == 0x80:
        pos -= 1
    return pos


def _split(data: bytes, section_bytes: int):
    """不重叠切段（尾段可短），每个内部切点都对齐字符边界。"""
    out = []
    s = 0
    while s < len(data):
        e = _align(data, min(s + section_bytes, len(data)))
        if e <= s:  # 防御：section_bytes 小于一个字符宽度时保底推进
            e = min(s + section_bytes, len(data))
        out.append(data[s:e])
        s = e
    return out


def _load_pointer(pool_dir):
    p = os.path.join(pool_dir, POINTER_NAME)
    if os.path.exists(p):
        try:
            ptr = json.load(open(p, encoding="utf-8"))
            # 2026-10-04 修复：旧指针用 file_idx（排序列表下标），池文件集合一变就错位。
            # 新指针改用 file（文件名）定位当前文件。兼容旧档：把 file_idx 转成 file。
            if "file" not in ptr and "file_idx" in ptr:
                files = _pool_files(pool_dir)
                idx = int(ptr.get("file_idx", 0))
                ptr["file"] = files[idx] if 0 <= idx < len(files) else None
            ptr.setdefault("file", None)
            ptr.setdefault("offset", 0)
            ptr.setdefault("retired_seq", 0)
            ptr.setdefault("exhausted", False)
            return ptr
        except (json.JSONDecodeError, OSError):
            pass  # 指针损坏从头消费：宁可重来不可跳卷
    return {"file": None, "offset": 0, "retired_seq": 0, "exhausted": False}


def _save_pointer(pool_dir, ptr):
    os.makedirs(pool_dir, exist_ok=True)
    tmp = os.path.join(pool_dir, POINTER_NAME + ".writing")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ptr, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(pool_dir, POINTER_NAME))


def _pool_files(pool_dir):
    if not os.path.isdir(pool_dir):
        return []
    return sorted(f for f in os.listdir(pool_dir)
                  if os.path.isfile(os.path.join(pool_dir, f)) and f != POINTER_NAME
                  and not f.endswith(".writing"))


def _next_section(pool_dir, ptr, section_bytes):
    """从候选流取下一小节（可跨文件），推进指针。返回 (字节, 来源名) 或 (None, None)。

    2026-10-04 修复：
    1. 用文件名定位当前文件，而不是排序下标——运行中新增/改名/重排池文件，
       也能从指针记录的文件名正确续读，不会错位到别的文件。
    2. 已消费完的文件移入 .consumed/ 子目录，不再出现在候选池中——
       否则指针重置（或新增文件）后已消费文件会"复活"被重复消费/计数。
    """
    files = _pool_files(pool_dir)
    consumed_dir = os.path.join(pool_dir, ".consumed")
    # 从指针记录的文件开始（若已被移走/删除，则从头开始——剩余的都是未消费）
    start = 0
    if ptr.get("file") is not None and ptr["file"] in files:
        start = files.index(ptr["file"])
    i = start
    while i < len(files):
        fname = files[i]
        data = open(os.path.join(pool_dir, fname), "rb").read()
        # 仅当这就是指针记录的文件时，从 offset 续读；否则从头读
        pos = ptr.get("offset", 0) if fname == ptr.get("file") else 0
        pos = _align(data, min(pos, len(data)))
        if pos >= len(data):
            # 当前文件已读完 → 移入 .consumed，避免之后复活被重复消费
            if fname == ptr.get("file"):
                ptr["file"] = None
            ptr["offset"] = 0
            os.makedirs(consumed_dir, exist_ok=True)
            try:
                os.replace(os.path.join(pool_dir, fname), os.path.join(consumed_dir, fname))
            except FileNotFoundError:
                pass  # 已被并发移除，忽略
            files = _pool_files(pool_dir)  # 刷新（当前文件已移走）
            continue  # 从当前位置继续（下一个文件现在占据 i）
        end = _align(data, min(pos + section_bytes, len(data)))
        if end <= pos:
            end = len(data)
        ptr["file"] = fname
        ptr["offset"] = end
        ptr["exhausted"] = False  # 消费到新内容 → 不再是耗尽态
        return data[pos:end], fname
    # 全部文件已耗尽
    ptr["file"] = None
    ptr["offset"] = 0
    ptr["exhausted"] = True
    return None, None


def rotate(probe_path=None, pool_dir=None, retired_dir=None, section_bytes=8192):
    """滚动一次：最旧一段退休出卷，候选流的一小节转正入卷。

    返回报告 dict（rotated=False 时带 reason）。池空/卷过短均不动卷。
    """
    probe_path = probe_path or PROBE_PATH
    pool_dir = pool_dir or POOL_DIR
    retired_dir = retired_dir or RETIRED_DIR

    if not os.path.exists(probe_path):
        return {"rotated": False, "reason": "当前卷不存在"}
    if not _pool_files(pool_dir):
        return {"rotated": False, "reason": "候选池为空或不存在"}

    data = open(probe_path, "rb").read()
    sections = _split(data, section_bytes)
    if len(sections) < 2:
        return {"rotated": False, "reason": "卷面不足两段，拒绝退休唯一一段"}

    ptr = _load_pointer(pool_dir)
    new_sec, src = _next_section(pool_dir, ptr, section_bytes)
    if new_sec is None:
        _save_pointer(pool_dir, ptr)
        return {"rotated": False, "reason": "候选池已耗尽"}

    # 先落退休档，再原子换卷：中途崩溃最多多一份退休档，卷面永不残缺
    os.makedirs(retired_dir, exist_ok=True)
    rname = f"retired_{ptr['retired_seq']:04d}.txt"
    with open(os.path.join(retired_dir, rname), "wb") as f:
        f.write(sections[0])
    ptr["retired_seq"] += 1

    tmp = probe_path + ".rotating"
    with open(tmp, "wb") as f:
        f.write(b"".join(sections[1:]) + new_sec)
    os.replace(tmp, probe_path)
    _save_pointer(pool_dir, ptr)

    return {"rotated": True, "retired": rname, "retired_bytes": len(sections[0]),
            "promoted_bytes": len(new_sec), "source": src,
            "probe_bytes": len(data) - len(sections[0]) + len(new_sec),
            "sections": len(sections)}  # 段数守恒：退一进一


def status(probe_path=None, pool_dir=None, retired_dir=None, section_bytes=8192):
    """卷面/候选池/退休档现状一览（供查房与下一棒接手）。"""
    probe_path = probe_path or PROBE_PATH
    pool_dir = pool_dir or POOL_DIR
    retired_dir = retired_dir or RETIRED_DIR

    probe_bytes = os.path.getsize(probe_path) if os.path.exists(probe_path) else 0
    sections = len(_split(open(probe_path, "rb").read(), section_bytes)) if probe_bytes else 0

    ptr = _load_pointer(pool_dir) if os.path.isdir(pool_dir) else {"file": None, "offset": 0, "exhausted": False}
    files = _pool_files(pool_dir)
    pool_left = 0
    if ptr.get("exhausted"):
        # 全部候选已消费完：剩余 0（候选池持续流入后由 _next_section 清除标志）
        pool_left = 0
    else:
        cur = ptr.get("file", None)
        seen_cur = False
        for f in files:
            n = os.path.getsize(os.path.join(pool_dir, f))
            if cur is None:
                # 无指针 → 全部候选都未消费
                pool_left += n
            elif f == cur:
                seen_cur = True
                pool_left += max(0, n - ptr.get("offset", 0))
            elif not seen_cur:
                # 还没到指针文件 → 之前的文件都已消费完，不再计入
                continue
            else:
                # 已过指针文件 → 后续文件全部未消费
                pool_left += n
    retired_files = sorted(f for f in os.listdir(retired_dir)) if os.path.isdir(retired_dir) else []
    retired_bytes = sum(os.path.getsize(os.path.join(retired_dir, f)) for f in retired_files)
    return {"probe_bytes": probe_bytes, "probe_sections": sections,
            "pool_files": len(files), "pool_bytes_left": max(0, pool_left),
            "retired_files": len(retired_files), "retired_bytes": retired_bytes,
            "pointer": {"file": ptr.get("file", None), "offset": ptr.get("offset", 0),
                        "retired_seq": ptr.get("retired_seq", 0),
                        "exhausted": ptr.get("exhausted", False)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="试卷池自滚动")
    ap.add_argument("--rotate", action="store_true", help="滚动一次（默认只看现状）")
    ap.add_argument("--status", action="store_true",
                    help="查看现状（与默认行为相同；补上交接文档承诺的显式旗标）")
    ap.add_argument("--section-kb", type=int, default=8, help="切段大小 KB")
    args = ap.parse_args()
    if args.rotate:
        print(rotate(section_bytes=args.section_kb * 1024))
    print(json.dumps(status(section_bytes=args.section_kb * 1024), ensure_ascii=False, indent=2))
    # 候选源对账（2026-10-04 接入）：台账累计（入池供给侧的账，candidates.py 维护）。
    # 缺模块/缺数据盘都不拖垮卷面对账——对账宁可少一节，不可整表崩掉。
    try:
        from probes.candidates import ledger_report
        led = ledger_report()
        per = {}
        for run in led["runs"]:
            for src, cnt in run.get("sources", {}).items():
                per[src] = per.get(src, 0) + cnt.get("admitted", 0)
        print("[候选源] 台账：" + json.dumps({
            "填充次数": led["fills"], "累计入池题干指纹": led["stems_total"],
            "各来源累计": per}, ensure_ascii=False))
    except Exception as e:
        print(f"[候选源] 台账对账不可用（{e!r}）——卷面/池/退休档对账不受影响")
