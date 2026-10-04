"""成长配套测试：ward 真实状态查房 + 候选流接池 + QC 全量抽查（2026-10-04）。

对应"自成长缺口"三件套的常备回归，风格仿 tests/test_conformance.py 但独立
运行（python tests/test_growth_support.py，纯 CPU，数秒级；含 O: 盘真实
候选源解析的用例在数据盘不在场时显式记 SKIP，不判红）：

- probes/qc.py          重叠抽查公共引擎：覆盖率全量化（2.25% → 100%）、
                        ≥50 字符共享子串必被抓到、命中归因、预算模式。
- probes/candidates.py  候选流解析（与 build_logic_probe 同源规则）、入池
                        质检（撞卷候选按条拒绝）、台账去重、rolling 对账。
- probes/rolling.py     rotate 全流程（入池→转正→退休→卷更新→池尽收口）。
- ward.py               fed_state.pt 优先加载 + 回退明示"未部署状态"。

红线自守：本文件对生产资产零接触——所有写操作落在 tempfile 临时目录；
真实数据源只读；生产 probes/probe_pool、probes/retired、probe.txt、
dolphin/fed_state.pt 一律不在测试里写。
"""
import atexit
import contextlib
import io
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.model import Config
import probes.build_logic_probe as build_logic_probe
from probes import candidates, rolling
from probes.qc import COVERAGE_TARGET, GRAM, STEP, WIN, overlap_check, slice_windows
import ward

PASS = []
SKIP = []


def check(name, cond, detail="", skip=False):
    if skip:
        SKIP.append(name)
        PASS.append(True)
        print(f"  ⏭ {name}  {detail}（本环境不适用，跳过）")
        return
    PASS.append(cond)
    print(f"  {'✓' if cond else '✗'} {name}  {detail}")


@contextlib.contextmanager
def _tmpdir(prefix="dolphin_growth_"):
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==================== probes/qc.py：重叠抽查公共引擎 ====================

def t_qc_engine():
    """滑窗切片覆盖数学 + ≥GRAM 共享子串必被抓 + 阈值下不误抓 + 归因。"""
    text = "甲乙丙丁戊" * 48  # 240 字符
    ws = slice_windows(text)
    check("qc 滑窗切片：头从 0 步长推进",
          len(ws) == (240 - 1) // STEP + 1 and ws[0] == text[:WIN],
          f"{len(ws)} 窗（win={WIN} step={STEP}）")
    # 覆盖保证：任何 ≥GRAM 的子串必完整落入某窗（含跨窗边界、非对齐起点）
    missed = 0
    for start in range(0, 240 - GRAM + 1, 7):  # 步进 7 系统性扫非对齐位置
        run = text[start:start + GRAM]
        if not any(run in w for w in ws):
            missed += 1
    check("qc 覆盖数学：任意 ≥GRAM 子串必落入某窗（含非对齐起点）", missed == 0,
          f"抽查 {len(range(0, 240 - GRAM + 1, 7))} 个起点，漏 {missed}")

    src = ("这是一段独一无二的来源正文内容，包含一个必须被查出的特征句子："
           "青雀从不畏惧风浪，因为它记得母亲唱过的歌谣至今仍在深谷之中回响"
           "不止，从未消散。其余部分是无关填充句，用于把来源撑到足够长度。")
    filler = "填充字符" * 400
    mid = src[40:100]  # 60 字符、被查文本中起点 73（非 step 对齐）
    r_hit = overlap_check([filler[:73] + mid + filler[133:]],
                          [("测试来源", [src])])
    check("qc 共享 ≥GRAM 子串必被抓（非对齐跨窗起点）",
          not r_hit["passed"] and r_hit["hit_items"] == [0],
          f"grams={r_hit['hit_grams']} items={r_hit['hit_items']}")
    r_edge = overlap_check([filler[:73] + src[40:90] + filler[123:]],
                           [("测试来源", [src])])
    check("qc 阈值下限：恰好 GRAM 长共享也抓到", not r_edge["passed"])
    r_below = overlap_check([filler[:73] + src[40:89] + filler[122:]],
                            [("测试来源", [src])])
    check("qc 阈值下 1 字符不误抓（检测灵敏度契约）", r_below["passed"])
    r_clean = overlap_check(["完全不相干的被查文本，与来源毫无交集。谁谓河广，一苇杭之。" * 6],
                            [("测试来源", [src])])
    check("qc 干净文本零命中且覆盖率达标",
          r_clean["passed"] and r_clean["coverage"] >= COVERAGE_TARGET,
          f"覆盖率 {r_clean['coverage']:.0%}")
    # 多条被查的命中归因：只有夹带私货的那条被点名
    texts = ["干净文本一，与来源毫无关系，但长度要足够产生窗口才行。" * 4,
             "这段文本夹带了私货：" + mid + "，后面继续凑长度避免边界效应。" * 2,
             "干净文本二，同样毫无关系，凑够窗口长度。" * 8]
    r_attr = overlap_check(texts, [("测试来源", [src])])
    check("qc 命中归因到具体被查条目", r_attr["hit_items"] == [1],
          f"items={r_attr['hit_items']}")
    check("qc 报告含窗口/覆盖率字段",
          r_attr["windows_total"] > 0 and r_attr["windows_checked"] == r_attr["windows_total"]
          and r_attr["coverage"] == 1.0,
          f"{r_attr['windows_checked']}/{r_attr['windows_total']}")
    # 预算模式：覆盖率如实打折（指标不是恒等式，是可被限制的真实测量）
    r_budget = overlap_check(texts, [("测试来源", [src])], max_windows=1)
    check("qc 预算模式覆盖率如实打折",
          r_budget["coverage"] == 1 / r_budget["windows_total"]
          and r_budget["coverage"] < 1.0,
          f"{r_budget['windows_checked']}/{r_budget['windows_total']} = "
          f"{r_budget['coverage']:.3f}")
    # 来源异常不掩盖其余来源（沿用旧抽查语义）
    def bad_stream():
        yield "正常记录一段，长度足够。"
        raise OSError("数据盘抖动")
    r_err = overlap_check([mid + filler[:100]], [("坏来源", bad_stream()),
                                                 ("好来源", [src])])
    check("qc 单来源异常不掩盖其他来源",
          not r_err["passed"]
          and any(s["error"] for s in r_err["sources"])
          and any(s["name"] == "好来源" and not s["error"] for s in r_err["sources"]))


def t_qc_vs_training_semantics():
    """build_logic_probe.overlap_check 已移交 qc 引擎（零复制，可被假粮驱动）。"""
    orig_sources, orig_stream = build_logic_probe.TRAIN_SOURCES, build_logic_probe.grain_stream
    try:
        grain = "训练粮独有句子，青雀记得母亲的歌谣在深谷回响不止，从未消散。" * 10
        build_logic_probe.TRAIN_SOURCES = ["假粮"]
        build_logic_probe.grain_stream = lambda name: iter([grain])
        ok = build_logic_probe.overlap_check([("假源", "干净题面，与训练粮毫无交集，长度足够。" * 8, "答案")])
        check("qc 委托：干净题面通过（零命中）", ok is True)
        planted = "题干夹带：" + grain[10:75] + "，其余为无关凑长内容。" * 3
        ok2 = build_logic_probe.overlap_check([("假源", planted, "答案")])
        check("qc 委托：与训练粮共享 ≥GRAM 片段判失败", ok2 is False)
    finally:
        build_logic_probe.TRAIN_SOURCES, build_logic_probe.grain_stream = orig_sources, orig_stream


# ==================== probes/candidates.py：候选流接池 ====================

def t_candidates_parse():
    """真实候选流解析：与 build_logic_probe 同源规则，条数 > 0（O: 缺席则跳过）。"""
    need = [build_logic_probe.LLMEVAL_BASE, build_logic_probe.LLMEVAL_HARD,
            build_logic_probe.LOGIQA2_TRAIN,
            os.path.join(build_logic_probe.LOGIQA1_DIR, "zh_test.txt"),
            os.path.join(build_logic_probe.LOGIQA1_DIR, "zh_eval.txt"),
            build_logic_probe.CMATH_TEST, build_logic_probe.CMATH_DEV]
    if not all(os.path.exists(p) for p in need):
        check("候选解析（真实数据源）", True, "O:/数据集 不在场", skip=True)
        return
    items, dropped = candidates.iter_candidate_items()
    per = {}
    for src, _, _ in items:
        per[src] = per.get(src, 0) + 1
    check("候选解析三源齐备且条数 > 0",
          per.get("llmeval_logic", 0) > 0 and per.get("logiqa1_test", 0) > 0
          and per.get("cmath_test", 0) > 0,
          f"{per}  合计 {len(items)}")
    check("候选解析防撞预过滤生效（LogiQA2 train / cmath_dev 撞题剔除）",
          dropped.get("logiqa2_train", 0) > 0 and dropped.get("cmath_dev", 0) >= 0,
          f"{dropped}")
    check("候选条目卷面格式与 build_probe 同源",
          all(candidates.item_text(q, a).startswith("问：") and "\n答：" in candidates.item_text(q, a)
              for _, q, a in items[:50]))


def _synthetic_probe_body():
    """合成卷面（与真实卷同构：卷面即条目文本的拼接，inspectable）。"""
    p1 = ("卷面第一题：这是一道足够长的合成探测题干，用来占据卷面并充当撞车"
          "比对面，内容唯一且不会与候选池的干净候选重合。甲乙丙丁戊己庚辛。")
    p2 = "卷面短题：三加五等于八吗？"
    return candidates.item_text(p1, "答案一。") + candidates.item_text(p2, "八。")


def _synthetic_items(probe_body):
    """合成候选：7 条干净（两来源）+ 2 条与卷面撞车（长/短各一）+ 1 条重复。

    长撞车（≥GRAM 共享）由 n-gram 通道抓；短撞车（整条 < GRAM，gram 机制
    失明）由全文逐字通道抓——分别对应两条质检机理。
    """
    mk = lambda tag, i: (f"这是{tag}来源的第{i}号合成候选题干，内容各不相同且足够长，"
                         f"用于验证滑窗切片与入池质检的计数账目是否对得上。{tag}-{i}号唯一指纹尾巴。")
    items = [("甲源", mk("甲", i), f"答案甲{i}，同样各不相同避免题干指纹撞车。") for i in range(4)]
    items += [("乙源", mk("乙", i), f"答案乙{i}，同样各不相同避免题干指纹撞车。") for i in range(3)]
    # 长撞车：卷面第一题的题干主体（≥GRAM 共享片段 → n-gram 通道）
    items.append(("丙源", "撞车长题：" + probe_body[:96], "撞车答案"))
    # 短撞车：整条与卷面短题同文（< GRAM → 只有全文逐字通道能抓）
    short_probe_item = [seg for seg in probe_body.split("\n\n") if seg][1]
    items.append(("丙源", short_probe_item[len("问："):].split("\n答：")[0], "八。"))
    items.append(("甲源", items[0][1], items[0][2]))  # 本轮重复
    return items


def t_pool_fill_qc_status():
    """入池质检按条拒绝撞车候选 + 台账去重 + rolling.status 对账一致（全临时目录）。"""
    with _tmpdir() as tmp:
        probe = os.path.join(tmp, "probe.txt")
        probe_body = _synthetic_probe_body()
        with open(probe, "w", encoding="utf-8", newline="\n") as f:
            f.write(probe_body)
        pool = os.path.join(tmp, "pool")
        retired = os.path.join(tmp, "retired")
        ledger = os.path.join(tmp, "ledger.json")

        rep = candidates.fill_pool(pool_dir=pool, probe_path=probe,
                                   retired_dir=retired, ledger_path=ledger,
                                   items=_synthetic_items(probe_body))
        check("入池：解析 10 条（含 2 撞卷 + 1 重复）", rep["parsed"] == 10, f"{rep['parsed']}")
        check("入池：撞卷候选被质检按条拒绝（长 1 + 短 1）", rep["qc_rejected"] == 2,
              f"rejected={rep['qc_rejected']} coverage={rep['qc_coverage']}")
        check("入池：短撞车经全文逐字通道抓到（gram 机制对 <GRAM 条目失明）",
              rep["qc_verbatim_hits"] >= 1, f"verbatim={rep['qc_verbatim_hits']}")
        check("入池：本轮重复去重", rep["dup_run"] == 1)
        check("入池：7 条干净候选全部入池", rep["admitted"] == 7,
              f"admitted={rep['admitted']} bytes={rep['bytes']}")
        check("入池：每来源一个池文件", len(rep["files"]) == 2
              and {f["source"] for f in rep["files"]} == {"甲源", "乙源"},
              f"{[f['file'] for f in rep['files']]}")
        written = sum(f["bytes"] for f in rep["files"])
        st = rolling.status(probe_path=probe, pool_dir=pool, retired_dir=retired)
        check("对账：池文件数/字节数与入池报告一致",
              st["pool_files"] == 2 and st["pool_bytes_left"] == written,
              f"池 {st['pool_files']} 文件 / {st['pool_bytes_left']}B vs 报告 {written}B")
        check("对账：卷面零改动（律 L8：入池不碰卷）",
              st["probe_bytes"] == len(probe_body.encode("utf-8"))
              and st["retired_files"] == 0)
        check("台账：7 条题干指纹入账", len(candidates.load_ledger(ledger)["stems"]) == 7)

        # 重复填充：台账拦截已入池者（7 条指纹 + 本轮重复者同指纹 = 8）；
        # 两条撞车者不在台账、再次被质检拒绝；零重复入池
        rep2 = candidates.fill_pool(pool_dir=pool, probe_path=probe,
                                    retired_dir=retired, ledger_path=ledger,
                                    items=_synthetic_items(probe_body))
        check("重复填充：台账去重生效、零重复入池",
              rep2["dup_ledger"] == 8 and rep2["qc_rejected"] == 2
              and rep2["admitted"] == 0 and not rep2["files"],
              f"dup_ledger={rep2['dup_ledger']} rejected={rep2['qc_rejected']} "
              f"admitted={rep2['admitted']}")
        st2 = rolling.status(probe_path=probe, pool_dir=pool, retired_dir=retired)
        check("重复填充：池账目不变", st2["pool_bytes_left"] == written,
              f"{st2['pool_bytes_left']}B")
        check("指针文件不占池位（rolling 指针协议未被破坏）",
              rolling.POINTER_NAME not in rolling._pool_files(pool))


def t_rotate_full_flow():
    """入池→转正→退休→卷更新→池尽收口（rolling.rotate 全流程，临时目录）。"""
    with _tmpdir() as tmp:
        probe = os.path.join(tmp, "probe.txt")
        probe_body = _synthetic_probe_body()
        with open(probe, "w", encoding="utf-8", newline="\n") as f:
            f.write(probe_body)
        pool = os.path.join(tmp, "pool")
        retired = os.path.join(tmp, "retired")
        ledger = os.path.join(tmp, "ledger.json")
        candidates.fill_pool(pool_dir=pool, probe_path=probe,
                             retired_dir=retired, ledger_path=ledger,
                             items=_synthetic_items(probe_body))
        sec = 100
        n_rot, prev_bytes = 0, len(probe_body.encode("utf-8"))
        while True:
            r = rolling.rotate(probe_path=probe, pool_dir=pool,
                               retired_dir=retired, section_bytes=sec)
            if not r["rotated"]:
                check("rotate：池尽收口（不动卷，原因=候选池已耗尽）",
                      r["reason"] == "候选池已耗尽"
                      and os.path.getsize(probe) == prev_bytes, f"{r}")
                break
            n_rot += 1
            now = os.path.getsize(probe)
            check(f"rotate 第{n_rot}次：字节账目 退一进一守恒",
                  now == prev_bytes - r["retired_bytes"] + r["promoted_bytes"],
                  f"{prev_bytes}-{r['retired_bytes']}+{r['promoted_bytes']}={now}")
            prev_bytes = now
            if n_rot > 200:
                break  # 防御：账目异常时跳出而非死循环
        st = rolling.status(probe_path=probe, pool_dir=pool, retired_dir=retired)
        check("rotate 全流程：卷面仍在合同语义内（非空、可按段切）",
              st["probe_bytes"] > 0 and st["probe_sections"] >= 1,
              f"卷 {st['probe_bytes']}B / {st['probe_sections']} 段")
        check("rotate 全流程：退休档笔数 = 转正次数", st["retired_files"] == n_rot,
              f"退 {st['retired_files']} / 转 {n_rot}")
        check("rotate 全流程：池清零且已消费文件移入 .consumed（不复活）",
              st["pool_bytes_left"] == 0 and st["pool_files"] == 0
              and os.path.isdir(os.path.join(pool, ".consumed")),
              f"池 {st['pool_files']} 文件 / {st['pool_bytes_left']}B")
        check("rotate 全流程：转正段进卷、退休段出卷（卷≠原卷）",
              open(probe, "rb").read() != probe_body.encode("utf-8"))
        # 多字节边界：卷面切段零割裂（_align 语义在 candidates 入池的内容上仍成立）
        with open(probe, "rb") as f:
            raw = f.read()
        try:
            raw.decode("utf-8")
            ok_utf8 = True
        except UnicodeDecodeError:
            ok_utf8 = False
        check("rotate 全流程：卷面始终是合法 UTF-8（多字节零割裂）", ok_utf8)


# ==================== ward.py：fed_state 真实状态查房 ====================

def t_ward_fed_state():
    """fed_state.pt 优先加载、游标/周期随档复原、回退路径明示未部署状态。"""
    with _tmpdir() as tmp:
        probe = os.path.join(tmp, "probe.txt")
        seg = "成长配套探测片段：体检评分专用，永不进训练粮。"
        with open(probe, "wb") as f:
            f.write((seg.encode() * 8)[:8 * 64])  # 512B = 8 块（block_size=64）
        cfg = Config(d_model=32, n_layers=1, n_heads=2, block_size=64)
        d = ward.Dolphin(cfg=cfg, device="cpu", probe_path=probe)
        d.memory = ward.MemoryStore(cold_path=os.path.join(tmp, "cold.jsonl"))
        d.cycle = 3
        d.feed_cursor = {"sources": ["甲", "乙"], "source_idx": 1, "record_idx": 5}
        d.learn("查房冒烟记录一，内容足够长。")
        d.learn("查房冒烟记录二，内容同样足够长。")
        fed = os.path.join(tmp, "fed_state.pt")
        d.save(fed)

        d2, info = ward.load_dolphin(dev="cpu", fed_state_path=fed,
                                     birth_path=os.path.join(tmp, "无胎教.pt"),
                                     old_state_path=os.path.join(tmp, "无旧档.pt"),
                                     cold_path=os.path.join(tmp, "cold.jsonl"),
                                     probe_path=probe)
        check("ward：fed_state 优先加载（deployed=True）",
              info["deployed"] is True and info["version"] == 3, f"{info['note']}")
        check("ward：喂食游标随档复原", d2.feed_cursor == d.feed_cursor,
              f"{d2.feed_cursor}")
        check("ward：周期随档复原", d2.cycle == 3, f"cycle={d2.cycle}")
        check("ward：记忆库随档复原", len(d2.memory.entries) == len(d.memory.entries),
              f"热层 {len(d2.memory.entries)} 条")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ward.report(d2, "查房 · 测试", info)
        out = buf.getvalue()
        check("ward：报告含状态来源/游标/热冷层/learn_total 新字段",
              all(k in out for k in ("状态来源", "喂食游标", "热层", "冷层", "learn_total")),
              "字段齐全")

        # 回退路径：存档不存在 → 明示未部署状态
        d3, info3 = ward.load_dolphin(dev="cpu", fed_state_path=os.path.join(tmp, "不存在.pt"),
                                      birth_path=os.path.join(tmp, "无胎教.pt"),
                                      old_state_path=os.path.join(tmp, "无旧档.pt"),
                                      cold_path=os.path.join(tmp, "cold2.jsonl"),
                                      probe_path=probe)
        check("ward：存档缺失回退并明示未部署状态",
              info3["deployed"] is False and "未部署状态" in info3["note"],
              info3["note"][:60])

        # 回退路径：存档损坏 → 警报 + 回退，不崩查房
        bad = os.path.join(tmp, "bad.pt")
        with open(bad, "wb") as f:
            f.write("这不是一个合法的 torch 存档".encode("utf-8"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            d4, info4 = ward.load_dolphin(dev="cpu", fed_state_path=bad,
                                          birth_path=os.path.join(tmp, "无胎教.pt"),
                                          old_state_path=os.path.join(tmp, "无旧档.pt"),
                                          cold_path=os.path.join(tmp, "cold3.jsonl"),
                                          probe_path=probe)
        check("ward：存档损坏回退不崩（stderr 醒目警报）",
              info4["deployed"] is False and "加载失败" in info4["note"]
              and "警报" in err.getvalue(), info4["note"][:60])


if __name__ == "__main__":
    print("== 成长配套测试（ward 真实状态 / 候选流接池 / QC 全量抽查） ==")
    t_qc_engine()
    t_qc_vs_training_semantics()
    t_candidates_parse()
    t_pool_fill_qc_status()
    t_rotate_full_flow()
    t_ward_fed_state()
    n_ok = sum(PASS)
    if SKIP:
        print(f"（跳过 {len(SKIP)} 项：{'; '.join(SKIP)}）")
    print(f"\n== 结论：{n_ok}/{len(PASS)} 绿 ==")
    sys.exit(0 if n_ok == len(PASS) else 1)
