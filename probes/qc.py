"""重叠抽查公共引擎（律 L8 质检与试卷池入池质检的同一实现）。

历史缺陷（2026-10-03 build_logic_probe.py 首版）：抽查只随机取 5 个 100 字符
片段在训练粮里搜索——约 61.9KB 卷面只查了 ≈2.25%，其余 97.75% 的卷面内容
与训练粮的重叠从未被检查过，L8 防泄漏抽查可能整段漏检。

本引擎的升级（2026-10-04）：
- 全量滑窗切片：对每条被查文本按 100 字符滑窗（步长 50）切片，任何 ≥GRAM
  长度的共享子串必完整落入至少一个窗口（win=2×step 的覆盖数学，见
  slice_windows 注释），覆盖率从 2.25% 提升到 100%（报告里给出实测数字）。
- n-gram 集合比对：被查侧构建 n-gram 集合，来源侧单遍流式扫描——每条来源
  记录逐位置取 n-gram 查集合，命中即上报。资源有向是有意的设计：n-gram 集
  合建在被查文本一侧（量小：卷面/候选流，至多几 MB 字符量级），来源侧（训
  练粮 ~9 亿字符）只流式过一遍、内存 O(被查文本)。反向建"训练粮全集 n-gram
  集合"在数学上等效但内存爆炸（数亿条 n-gram），不可行。
- 命中归因：报告命中发生在哪些被查条目上（hit_items），供调用方按条拒绝
  （入池质检拒绝撞卷候选）或整体判失败（构建期 L8 检查）。
- 全文逐字通道（verbatim_against）：n-gram 机制对短于 GRAM 的被查条目天然
  失明，对当前卷/退休档这类 KB 级小额比对面补一条"整条候选原文逐字子串"
  检查——这是入池质检最关心的撞车形态，开销可忽略（见 overlap_check 参数注）。

检测灵敏度契约：本引擎保证抓到"长度 ≥ GRAM 字符的共享子串"；短于 GRAM 的
共享片段（样板话术量级）不抓——GRAM=50 的取值理由：≥50 字符的共享子串在
中文语料里不可能是巧合（题干级撞车），而 <50 字符的公共套话（"根据上述材
料回答下列问题"之类）不会误报。比旧实现（只查 100 字符整段逐字）更灵敏，
又不至于被套话淹没。短于 GRAM 的窗口（尾窗/短题）内部不存在 ≥GRAM 的子
串，按此契约没有可查内容，计入"被查"不虚减覆盖率。

用法（被 build_logic_probe.py 与 probes/candidates.py 共用，勿在两处复刻）：
  from probes.qc import overlap_check
  report = overlap_check(texts, [("训练粮:logiqa", record_text_stream), ...])
  report["passed"]  # True=零命中
"""
import time

# 值：自成（理由见模块注释）。三数绑定关系：GRAM ≤ WIN/2（即 WIN ≥ 2×STEP
# 且 STEP ≥ GRAM），破坏任一条都会丢"≥GRAM 共享子串必被抓到"的覆盖保证。
WIN = 100    # 滑窗窗口（字符），与旧抽查的片段粒度一致
STEP = 50    # 滑窗步长（字符），WIN/2：保证跨窗边界的共享子串仍落入某窗
GRAM = 50    # n-gram 长度（字符）：检测灵敏度下限
COVERAGE_TARGET = 0.9   # 任务定标：覆盖率目标 ≥90%（本引擎全量切片应达 100%）

MAX_HIT_GRAMS = 20000    # 命中 n-gram 收集上限：来源被污染时命中会呈风暴，
                         # 截断收集并判失败，不撑爆内存也不谎报"零命中"
MAX_HIT_DETAIL = 50      # 报告里详列的命中明细条数上限（总数照实统计）


def slice_windows(text, win=WIN, step=STEP):
    """对整段文本滑窗全量切片：窗口头从 0 起按 step 推进直到文本尾。

    覆盖保证：win ≥ 2×step 时，任何长度 ≥ win/2 的子串必完整落入至少一个
    窗口——设子串起点为 p，取 k=floor(p/step)×step，则窗口 k 覆盖
    [k, k+win) ⊇ [p, p+win/2)。尾窗可短于 win（保真不丢弃）。
    返回窗口列表；空文本返回空列表（无窗口可查）。
    """
    if not text:
        return []
    out = []
    s = 0
    n = len(text)
    while s < n:
        out.append(text[s:s + win])
        s += step
    return out


def _iter_texts(stream):
    """来源文本流归一化：单个 str 视为单条记录；其余按可迭代处理。"""
    if isinstance(stream, str):
        yield stream
        return
    for t in stream:
        yield t


def overlap_check(text_fragments, sources, win=WIN, step=STEP, gram=GRAM,
                  max_windows=None, verbatim_against=None, log=None):
    """全量重叠抽查：被查文本 vs 来源流，返回覆盖率报告 dict。

    参数：
      text_fragments : str 或 list[str]——被查文本，每条独立滑窗全量切片。
      sources        : 可迭代的 (来源名, 文本流)；文本流是 str 或 str 可迭代
                       （生成器即可，引擎只单遍流式消费，不整仓进内存）。
      win/step/gram  : 滑窗与 n-gram 参数（默认值的关系约束见模块注释）。
      max_windows    : 检查预算（None=不限）。设上限时只查前 N 个窗口，
                       覆盖率如实按 checked/total 计——用于超大被查侧的
                       预算受限模式；默认全量，覆盖率恒 100%。
      verbatim_against : 可选 [(来源名, 整篇文本 str)]——小额比对面（当前卷、
                       退休档这类 KB 级文本）。n-gram 机制对短于 GRAM 的被查
                       条目（如 CMATH 短题）天然失明，这里用"条目全文逐字
                       子串"补盲区：整条候选原文出现在比对面里即命中。调用方
                       只应把 KB 级文本放进来（逐条目 × 整篇 in 搜索，大仓
                       会拖垮扫描），训练粮走 sources 流式通道。
      log            : 打印函数（None=静默；结论照常进返回值）。

    返回（覆盖率 = 被查窗口数 / 应查窗口数，任务定标目标 ≥0.9）：
      {items, windows_total, windows_checked, coverage, win, step, gram,
       sources: [{name, records, chars, error}],
       hit_grams, hit_records, hits: [{source, gram, context}],
       hit_items, passed, elapsed_s}
    """
    t0 = time.time()
    if isinstance(text_fragments, str):
        text_fragments = [text_fragments]
    text_fragments = list(text_fragments)

    # ① 被查侧全量滑窗切片：全部窗口都进比对（覆盖率不作假）
    windows = []       # (item_idx, window_text)
    for i, text in enumerate(text_fragments):
        for w in slice_windows(text, win, step):
            windows.append((i, w))
    windows_total = len(windows)
    if max_windows is not None:
        windows = windows[:max_windows]   # 预算受限：如实按 checked/total 计

    # ② 被查侧 n-gram 集合（资源有向：集合建在小量侧，见模块注释）
    gram_set = set()
    for _, w in windows:
        for p in range(len(w) - gram + 1):
            gram_set.add(w[p:p + gram])

    # ③ 来源侧单遍流式扫描
    src_stats = []
    hit_grams = set()          # 命中的 n-gram（去重，封顶防风暴）
    hit_detail = []            # 命中明细（封顶）
    hit_records = 0
    hit_gram_overflow = False
    g = gram
    for src in sources:
        name, stream = src
        n_rec = n_char = 0
        rec_hit = 0            # 本来源中至少命中一条的记录数
        err = None
        try:
            for text in _iter_texts(stream):
                n_rec += 1
                n_char += len(text)
                if len(text) < g:
                    continue   # 短于 gram 的记录装不下任何待查 gram
                gs = gram_set
                bumped = False
                for p in range(len(text) - g + 1):
                    sl = text[p:p + g]
                    if sl in gs:
                        bumped = True
                        if len(hit_grams) < MAX_HIT_GRAMS:
                            hit_grams.add(sl)
                            if len(hit_detail) < MAX_HIT_DETAIL:
                                ctx_a = max(0, p - 15)
                                hit_detail.append({
                                    "source": name, "gram": sl,
                                    "context": text[ctx_a:p + g + 15]})
                        else:
                            hit_gram_overflow = True
                if bumped:
                    rec_hit += 1
        except Exception as e:  # 单来源缺失/损坏不掩盖其他来源（沿用旧抽查语义）
            err = repr(e)
            if log:
                log(f"  [警告] 来源 {name} 检查中断：{err}")
        hit_records += rec_hit
        src_stats.append({"name": name, "records": n_rec, "chars": n_char,
                          "error": err, "hit_records": rec_hit})
        if log:
            log(f"  来源 {name:<24} 扫描 {n_rec} 条 / {n_char:,} 字符"
                + (f"  <命中记录 {rec_hit}>" if rec_hit else ""))

    # ④ 命中归因到被查条目：hit_grams 通常极少，对已查窗口一遍轻扫描即可
    hit_items = set()
    if hit_grams:
        hg = hit_grams
        for i, w in windows:
            for p in range(0, max(0, len(w) - gram + 1)):
                if w[p:p + gram] in hg:
                    hit_items.add(i)
                    break

    # ⑤ 全文逐字通道（补盲区）：短于 GRAM 的条目无法进 n-gram 集合，但"整条
    #    候选原文出现在当前卷/退休档里"恰恰是入池质检最关心的撞车——对小额
    #    比对面做逐条目 in 搜索（条目数 × KB 级文本，开销可忽略）
    verbatim_hits = 0
    for name, text in (verbatim_against or []):
        for i, frag in enumerate(text_fragments):
            if frag and frag in text:
                hit_items.add(i)
                verbatim_hits += 1
                if len(hit_detail) < MAX_HIT_DETAIL:
                    p = text.find(frag)
                    hit_detail.append({
                        "source": f"{name}（全文逐字）", "gram": frag[:gram],
                        "context": text[max(0, p - 15):p + min(len(frag), gram) + 15]})
                if len(hit_grams) < MAX_HIT_GRAMS:
                    hit_grams.add(frag)  # 借同一计数：整条原文即"共享片段"
                else:
                    hit_gram_overflow = True
                hit_records += 1

    report = {
        "items": len(text_fragments),
        "windows_total": windows_total,
        "windows_checked": len(windows),   # 全量切片：每窗都进了比对（见契约）
        "coverage": (len(windows) / windows_total) if windows_total else 1.0,
        "win": win, "step": step, "gram": gram,
        "sources": src_stats,
        "hit_grams": len(hit_grams),
        "hit_records": hit_records,
        "verbatim_hits": verbatim_hits,
        "hits": hit_detail,
        "hit_items": sorted(hit_items),
        "passed": not hit_grams and not hit_gram_overflow,
        "elapsed_s": round(time.time() - t0, 2),
    }
    if hit_gram_overflow:
        report["note"] = (f"命中 n-gram 超过收集上限 {MAX_HIT_GRAMS}，已截断——"
                          f"视为发现重大重叠")
    if log:
        log(f"\n[重叠抽查] 被查 {report['items']} 条文本 → 滑窗全量切片 "
            f"{report['windows_total']} 个窗口（win={win} step={step} gram={gram}）")
        log(f"[重叠抽查] 覆盖率 {report['coverage']:.0%}（被查 "
            f"{report['windows_checked']}/{report['windows_total']} 窗口，"
            f"目标 ≥{COVERAGE_TARGET:.0%}）  耗时 {report['elapsed_s']}s")
        if report["passed"]:
            log("[重叠抽查] 全部来源零共享片段 ✓")
        else:
            log(f"[重叠抽查] 发现命中：{report['hit_grams']} 个共享片段（含全文逐字 "
                f"{report['verbatim_hits']}）/ {hit_records} 条来源记录 / "
                f"涉及被查条目 {report['hit_items']}")
            for h in hit_detail[:10]:
                log(f"  !! {h['source']}  片段「{h['gram'][:24]}…」"
                    f"  上下文「{h['context'][:48]}…」")
    return report
