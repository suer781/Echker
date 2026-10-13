"""记忆库（律 L9 / L10）。

- 滞留：没进传输预算的经验降级到这里，可检索、不进权重。
- 预支（快通道）：醒脑服务时检索引用，即时可用。
- 复活（间隔重复）：被频繁检索命中的条目升权，进入下一轮睡眠选拔。
- 冷层（2026-10-03 G1 施工）：逐出=降级落盘 memory_cold.jsonl（append-only 日志），
  不是删除——被逐出条目跨周期仍可检索、可复活，L9"可检索可复活"对冷层同样成立。
- 做梦（同上）：dream() 按检索分数自我回忆——喂食模式没有人工对话供给检索
  命中，hits 由做梦代偿，复活通道保持可用。
"""
import array
import bisect
import json
import os
import threading
import time


class MemoryEntry:
    __slots__ = ("text", "score", "hits", "cycle", "kind", "created", "cold_seq")

    def __init__(self, text: str, score: float, cycle: int, kind: str):
        self.text = text
        self.score = float(score)
        self.hits = 0
        self.cycle = int(cycle)
        self.kind = kind
        self.created = time.time()
        # 冷层稳定序号（2026-10-12 倒排索引）：冷层条目在 RAM 中的唯一身份，
        # 由 _load_cold_locked/_demote 分配，resurrect 升回热层后复位为 -1。
        # 不参与序列化（to_json/from_json 忽略），仅服务于内存倒排索引。
        self.cold_seq = -1

    def to_json(self):
        return {"text": self.text, "score": self.score, "hits": self.hits,
                "cycle": self.cycle, "kind": self.kind, "created": self.created}

    @classmethod
    def from_json(cls, d):
        en = cls(d["text"], d["score"], d["cycle"], d["kind"])
        en.hits = int(d.get("hits", 0))
        en.created = float(d.get("created", time.time()))
        return en


def _grams(s: str, n=4):
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


_MASK64 = 0xFFFFFFFFFFFFFFFF


def _hash64(s: str) -> int:
    """64 位字符串哈希（倒排索引用）。

    使用 Python 内置 hash()（进程内稳定、跨进程随机种子不同——索引每次进程
    启动都重建，一致性只要求进程内）。掩码到 64 位无符号，供 array('Q') 存储。
    """
    return hash(s) & _MASK64


class _ColdInvertedIndex:
    """冷层 4-gram 倒排索引（紧凑平行数组 + 增量 delta）。

    - hashes: array('Q')，按 64 位 gram 哈希升序排列（主数组）。
    - seqs:   array('I')，与 hashes 平行的冷层条目稳定序号。
    - delta:  dict[int, list[int]]，主数组构建后新进入冷层的条目（_demote
      追加）的增量 posting list——避免每次逐出都触发全量重建。
    - max_seq: 主数组覆盖的最大 cold_seq，用于构建完成后 catch-up。
    """

    __slots__ = ("hashes", "seqs", "delta", "max_seq")

    def __init__(self, hashes, seqs, max_seq):
        self.hashes = hashes
        self.seqs = seqs
        self.delta = {}
        self.max_seq = max_seq

    def add_entry(self, entry):
        """把新进入冷层的条目追加进 delta（增量更新，不触发全量重建）。"""
        seq = entry.cold_seq
        if seq < 0:
            return
        seen = set()
        for g in _grams(entry.text):
            h = _hash64(g)
            if h in seen:
                continue
            seen.add(h)
            lst = self.delta.get(h)
            if lst is None:
                self.delta[h] = [seq]
            else:
                lst.append(seq)
        if seq > self.max_seq:
            self.max_seq = seq


# 冷层小于等于该条数时同步构建倒排索引（测试确定性、毫秒级）；
# 生产冷层（数十万条）走后台线程，避免首周期阻塞。
_SYNC_BUILD_THRESHOLD = 5000


class MemoryStore:
    def __init__(self, cap_entries=500, promote_hits=3, cold_path=None):
        self.entries = []                 # 热层
        self.cap = cap_entries
        self.promote_hits = promote_hits  # 值：自成
        pkg = os.path.dirname(os.path.abspath(__file__))
        self.cold_path = cold_path or os.path.join(pkg, "memory_cold.jsonl")
        self.cold_entries = []            # 冷层影子索引：落盘为准，RAM 仅供检索
        self._cold_loaded = False
        self._cold_lock = threading.Lock()  # serve(主线程检索) 与 训练线程逐出 的冷层互斥
        # 冷层倒排索引（2026-10-12 吞吐提速）：懒构建，后台线程构建。
        # 主结构为按 64 位 gram 哈希排序的平行数组（hashes: array('Q')，
        # seqs: array('I')），配合 _demote 增量 delta 与 resurrect 的
        # cold_seq 失效标记，避免每次周期全量重建。
        self._cold_inverted = None          # None=未构建；否则为 _ColdInvertedIndex
        self._cold_inverted_building = False
        self._cold_inverted_ready = threading.Event()
        self._cold_by_seq = []              # cold_seq -> MemoryEntry（resurrect 后置 None）
        self._cold_next_seq = 0

    # ---------- 冷层（G1：逐出降级落盘，跨周期可检索可复活） ----------

    def _load_cold_locked(self):
        if self._cold_loaded:
            return
        self.cold_entries = []
        if os.path.exists(self.cold_path):
            # 2026-10-04 修复：旧实现逐行 append，反刍把同一条反复逐出，
            # 会导致同一 text 落盘成百上千行（生产实况 16316 行仅 153 条
            # 唯一 text）。加载时按 text 去重：保留 hits 最大、created 最早
            # 的一条（信息量最大且保留审计链），一次加载即修复旧档。
            #
            # 2026-10-11 根治：旧实现用 next() 线性扫描已加载列表做去重，
            # 对 75 万行是 O(n²)（≈2.8 万亿次字符串比较），加载卡死数十分钟。
            # 改为 dict 按 text 哈希去重，O(n)：遍历时 seen 累积唯一条目，
            # 重复 text 合并字段，语义与旧实现完全一致（hits 取 max、created 取
            # min、score 取 min），遍历结束后一次转列表。dict 保持插入顺序，
            # 列表顺序仍为各唯一 text 首次出现的顺序，与旧实现一致。
            seen = {}
            with open(self.cold_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:  # 崩溃尾行的半行日志不影响整体加载
                        en = MemoryEntry.from_json(json.loads(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    existing = seen.get(en.text)
                    if existing is None:
                        seen[en.text] = en
                    else:
                        if en.hits > existing.hits:
                            existing.hits = en.hits
                        if en.created < existing.created:
                            existing.created = en.created
                        # 保留更低的 score（逐出优先级 min 取低分，早逐出者应优先保留）
                        if en.score < existing.score:
                            existing.score = en.score
            self.cold_entries = list(seen.values())
        self._cold_loaded = True
        # 分配冷层稳定序号并登记 seq -> entry 映射（倒排索引的地址空间）。
        # 仅首次加载时执行（_cold_loaded 守卫保证只跑一次）。
        for en in self.cold_entries:
            en.cold_seq = self._cold_next_seq
            self._cold_by_seq.append(en)
            self._cold_next_seq += 1
        self._maybe_start_index_build()

    def _ensure_cold(self):
        with self._cold_lock:
            self._load_cold_locked()

    # ---------- 冷层倒排索引（2026-10-12 吞吐提速） ----------

    def _maybe_start_index_build(self):
        """冷层加载完成后启动倒排索引构建（幂等）。

        小冷层（≤5000 条）同步构建（测试确定性、毫秒级）；大冷层（生产实况
        数十万条）放后台线程，构建完成前 dream() 回退全量扫描，不阻塞部署。
        """
        if self._cold_inverted is not None or self._cold_inverted_building:
            return
        if not self.cold_entries:
            return
        if len(self.cold_entries) <= _SYNC_BUILD_THRESHOLD:
            idx = self._build_cold_inverted()
            max_seq = idx.max_seq
            for en in list(self.cold_entries):
                if en.cold_seq > max_seq:
                    idx.add_entry(en)
            self._cold_inverted = idx
            self._cold_inverted_ready.set()
            return
        self._cold_inverted_building = True
        t = threading.Thread(target=self._build_cold_inverted_bg, daemon=True,
                               name="cold-inverted-build")
        t.start()

    def _build_cold_inverted_bg(self):
        try:
            idx = self._build_cold_inverted()
            # 构建期间 _demote 可能追加了新条目：主数组只覆盖快照，
            # 把快照之后进入冷层的条目补进 delta（增量正确性）。
            max_seq = idx.max_seq
            for en in list(self.cold_entries):
                if en.cold_seq > max_seq:
                    idx.add_entry(en)
            self._cold_inverted = idx
        finally:
            self._cold_inverted_building = False
            self._cold_inverted_ready.set()

    def _build_cold_inverted(self):
        """同步构建冷层倒排索引（测试与后台构建共用）。"""
        snapshot = list(self.cold_entries)
        packed = []
        append = packed.append
        for en in snapshot:
            seq = en.cold_seq
            if seq < 0:
                continue
            # 每条目 gram 哈希去重（对应 _grams 的集合语义）。
            for h in {_hash64(g) for g in _grams(en.text)}:
                append((h << 32) | seq)
        packed.sort()
        hashes = array.array("Q", (v >> 32 for v in packed))
        seqs = array.array("I", (v & 0xFFFFFFFF for v in packed))
        max_seq = max((en.cold_seq for en in snapshot), default=-1)
        del packed
        return _ColdInvertedIndex(hashes, seqs, max_seq)

    def _ensure_cold_index(self):
        """同步确保倒排索引已构建（测试用；生产路径由后台线程懒构建）。

        若后台构建已在进行，等待其完成；若后台构建失败（理论上不发生），
        回退为同步构建兜底。
        """
        self._ensure_cold()
        if self._cold_inverted is None:
            if self._cold_inverted_building:
                self._cold_inverted_ready.wait()
            if self._cold_inverted is None:
                idx = self._build_cold_inverted()
                # 同步构建同样要补 delta（与后台构建一致）。
                max_seq = idx.max_seq
                for en in list(self.cold_entries):
                    if en.cold_seq > max_seq:
                        idx.add_entry(en)
                self._cold_inverted = idx
                self._cold_inverted_ready.set()
        return self._cold_inverted

    def _dream_cold_indexed(self, qs, k):
        """用倒排索引对冷层做 dream 打分（精确语义：count/len(q) == len(q&g)/len(q)）。

        对每个查询的 4-gram 集合，从倒排索引取候选条目序号（主数组 + delta），
        累计共享 gram 数，只对超过 5% 阈值的候选打分，排序取 top-k。
        """
        idx = self._cold_inverted
        best = {}  # seq -> (ov, entry)
        for q in qs:
            qh = {_hash64(g) for g in q}
            lq = len(q)
            if lq <= 0:
                continue
            thresh = lq * 0.05
            counts = {}
            for h in qh:
                lo = bisect.bisect_left(idx.hashes, h)
                hi = bisect.bisect_left(idx.hashes, h + 1)
                for j in range(lo, hi):
                    seq = idx.seqs[j]
                    counts[seq] = counts.get(seq, 0) + 1
                for seq in idx.delta.get(h, ()):
                    counts[seq] = counts.get(seq, 0) + 1
            for seq, cnt in counts.items():
                if cnt > thresh:
                    en = self._cold_by_seq[seq] if seq < len(self._cold_by_seq) else None
                    if en is None or en.hits >= self.promote_hits:
                        continue
                    ov = cnt / lq
                    prev = best.get(seq)
                    if prev is None or ov > prev[0]:
                        best[seq] = (ov, en)
        return list(best.values())

    def flush_cold(self):
        """把冷层影子索引（含检索命中计数）写回落盘，供存档时调用——
        冷层命中进度跨重启不丢。缓存从未加载过则落盘文件为唯一依据，不做改动；
        冷层从未启用过（无逐出、文件也不存在）则零足迹，不额外创建文件。"""
        with self._cold_lock:
            if not self._cold_loaded or (not self.cold_entries
                                         and not os.path.exists(self.cold_path)):
                return
            d = os.path.dirname(self.cold_path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.cold_path + ".flushing"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                for en in self.cold_entries:
                    f.write(json.dumps(en.to_json(), ensure_ascii=False) + "\n")
            os.replace(tmp, self.cold_path)

    # ---------- 增与逐出 ----------

    def add(self, data: bytes, score: float, cycle: int, kind="residue"):
        text = data.decode("utf-8", errors="replace")
        entry = MemoryEntry(text, score, cycle, kind)
        self.entries.append(entry)
        if len(self.entries) > self.cap:
            self._evict(protect=entry)

    def _evict(self, protect=None):
        """逐出优先级（G1）：零命中 > 低分 > 最旧（note 与 residue 同规则）。

        - 候选=零命中条目，且排除本次刚写入的新条目：它还没活过一个周期，
          逐出它会使 add 的净效果变为删除（审计确认的新条目自删缺陷）。
        - 零命中集合为空（全员被检索过）→ 不逐出，扩容一档：cap 由事件驱动
          上调（值：自成）——旧知识不因新知识到来而被移除。
        - 逐出=降级进冷层落盘，不是删除：L9"可检索可复活"跨周期成立。
        """
        candidates = [e for e in self.entries if e.hits == 0 and e is not protect]
        if not candidates:
            self.cap = max(int(self.cap * 1.5), len(self.entries))  # 值：自成（事件驱动扩容）
            return
        victim = min(candidates, key=lambda e: (e.score, e.created))
        self.entries.remove(victim)
        self._demote(victim)

    def _demote(self, entry):
        """把热层条目降级进冷层。2026-10-04 修复：若同 text 已在冷层，不再追加
        新行（旧实现 append-only 导致反刍把同一条反复逐出→冷层无限膨胀，生产实况
        16316 行仅 153 条唯一 text）。改为更新已有行的 hits/score，保留最早
        created 作为审计链起点——落盘行数与唯一记忆数一致，且历史可追溯。"""
        with self._cold_lock:
            self._load_cold_locked()
            self._append_cold_locked(entry)

    def _append_cold_locked(self, entry):
        """在 _cold_lock 内把条目写进冷层（调用方须已持锁且已 _load_cold_locked）。

        去重合并 + append 落盘 + 倒排索引增量，语义即原 _demote 的核心：
        同 text 已在冷层则合并 hits/score 并保留最早 created，不追加新行；
        否则追加文件行、登记 cold_seq 并增量更新倒排索引。
        """
        d = os.path.dirname(self.cold_path)
        if d:
            os.makedirs(d, exist_ok=True)
        existing = next((x for x in self.cold_entries if x.text == entry.text), None)
        if existing is not None:
            # 同 text 已在冷层：合并命中计数与分数，保留最早 created。
            # hits 取 max（合并两次逐出之间的检索进度）。
            existing.hits = max(existing.hits, entry.hits)
            existing.score = min(existing.score, entry.score)
            existing.cycle = max(existing.cycle, entry.cycle)
            if entry.created < existing.created:
                existing.created = entry.created
            # 不需要追加文件行；flush_cold 时统一按合并后的状态重写。
            return
        with open(self.cold_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
        self.cold_entries.append(entry)
        # 增量更新倒排索引（若已构建）：新条目分配稳定序号并追加到 delta。
        entry.cold_seq = self._cold_next_seq
        self._cold_next_seq += 1
        self._cold_by_seq.append(entry)
        idx = self._cold_inverted
        if idx is not None:
            idx.add_entry(entry)

    def add_cold(self, data: bytes, score: float, cycle: int, kind="suspended"):
        """金发姑娘挂起（律 L4）：太怪数据不进热层，直接进冷层。

        与 add() 的区别：add() 走热层 append + cap 逐出（_evict/_demote），
        逐出才会降级冷层；add_cold 跳过热层，直接复用 _append_cold_locked
        落冷层——挂起数据永不占热层容量、永不触发热层逐出，也不影响现有
        cap=500 逐出逻辑。语义与 _demote 一致（同 text 去重合并）。
        """
        text = data.decode("utf-8", errors="replace")
        entry = MemoryEntry(text, score, cycle, kind)
        with self._cold_lock:
            self._load_cold_locked()
            self._append_cold_locked(entry)

    # ---------- 检索 / 复活 / 做梦 ----------

    def retrieve(self, query: bytes, k=1):
        q = _grams(query.decode("utf-8", errors="replace"))
        if not q:
            return []
        self._ensure_cold()  # 冷层同场参与检索：G1 修复前被逐出条目从此无法检索到
        scored = []
        # 2026-10-04 修复：训练线程（_evict/_demote/resurrect）会并发修改
        # self.entries / self.cold_entries，直接迭代会跳过元素/漏检索（语义错误）。
        # 迭代前取快照，保证检索看到一致视图，且不会在遍历中触发列表长度突变。
        for en in list(self.entries):
            g = _grams(en.text)
            if g:
                ov = len(q & g) / len(q)
                if ov > 0.05:
                    scored.append((ov, en))
        for en in list(self.cold_entries):
            g = _grams(en.text)
            if g:
                ov = len(q & g) / len(q)
                if ov > 0.05:
                    scored.append((ov, en))
        scored.sort(key=lambda t: -t[0])
        top = [en for _, en in scored[:k]]
        for en in top:
            en.hits += 1
        return top

    def resurrect(self):
        """命中达阈值的条目复活，重回睡眠选拔（间隔重复）。

        G1：冷层同样参与——冷层条目命中达标则升回热层（跨周期复活的完整闭环）。
        2026-10-04 修复两处：
        - 先 _ensure_cold()：与 retrieve()/dream() 一致。部署喂食路径
          （feed.py --resume → life.run_cycle(feeding=True)）只调 resurrect()、
          不调 serve()/retrieve()，缺这一步则新进程冷层为空，上一进程逐出的知识
          重启后完全不可见，违反 L9「可检索可复活」= 数据静默丢失。
        - 遍历副本：原实现边遍历 cold_entries 边 remove()，索引左移导致隔一条漏一条，
          间隔重复通道吞吐减半、且残留条目 hits 已达标却未被消费（状态不一致）。
        锁：全程不持 _cold_lock——_ensure_cold() 内部自行取放，且下方 _evict()→_demote()
        会再次取 _cold_lock（Lock 不可重入，整体加锁必然死锁）。_cold_lock 是叶锁：
        永不持有其它锁时再取、且不取任何锁，故与 feed_lock 只能形成 feed_lock→_cold_lock
        单向顺序，不存在 AB-BA 环路。
        """
        out = []
        self._ensure_cold()  # 冷层不加载 = 重启后不可用（retrieve/dream 早有，此处曾遗漏）
        for en in self.entries:
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                en.hits = 0
        promoted = []
        for en in list(self.cold_entries):  # 副本：原实现边遍历边删除 → 隔一条漏一条
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                self.cold_entries.remove(en)
                self.entries.append(en)
                en.hits = 0
                # 倒排索引失效标记：seq 槽位置 None，查询时跳过已升回热层的条目。
                # 主数组/ delta 中的 (hash, seq) 保留（惰性清理），不影响正确性。
                if 0 <= en.cold_seq < len(self._cold_by_seq):
                    self._cold_by_seq[en.cold_seq] = None
                en.cold_seq = -1
                promoted.append(en)
        while len(self.entries) > self.cap:  # 冷层升回可能超过容量：热层规则不变
            self._evict(protect=promoted[-1] if promoted else None)
        return out

    def resurrect_light(self, cold_sample=32):
        """浅睡轻量复活（2026-10-12）：只扫热层 + 冷层随机小批量抽样。

        语义与 resurrect() 一致——命中达阈值的条目升回热层、hits 复位、
        冷层升回触发热层容量保护——但冷层只随机抽样 cold_sample 条参与检查，
        避免浅睡周期背负 24.9 万条冷层的 O(n) 全量扫描（浅睡轻量快速）。
        热层永远全量检查（≤cap≈500，毫秒级）。抽样每次调用种子不同，
        保证跨周期覆盖面（不同冷层条目轮流获得复活机会）。
        仍调用 _ensure_cold()：冷层已加载、跨重启不失聪（L9 不破）。
        """
        out = []
        self._ensure_cold()
        for en in self.entries:
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                en.hits = 0
        # 冷层随机抽样（浅睡不背全量）：抽样后再遍历，避免复制整表。
        cold_pool = self.cold_entries
        if len(cold_pool) > cold_sample:
            import random as _random
            sample_idx = set(_random.sample(range(len(cold_pool)), cold_sample))
            cold_pool = [cold_pool[i] for i in sorted(sample_idx)]
        promoted = []
        for en in list(cold_pool):  # 副本：语义与 resurrect() 一致
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                self.cold_entries.remove(en)
                self.entries.append(en)
                en.hits = 0
                if 0 <= en.cold_seq < len(self._cold_by_seq):
                    self._cold_by_seq[en.cold_seq] = None
                en.cold_seq = -1
                promoted.append(en)
        while len(self.entries) > self.cap:  # 冷层升回可能超过容量：热层规则不变
            self._evict(protect=promoted[-1] if promoted else None)
        return out

    def dream(self, queries, k=5):
        """做梦（G1）：按检索分数从记忆库抽样旧经验，供周期头部注入缓冲。

        喂食模式没有人工对话供给检索命中——用近期经验做联想线索自我回忆，
        hits 由此增长，复活通道不再依赖外部检索。注入量封顶 k（≤5），
        防旧知识占用新经验的选拔预算（审计 G11：满带权挤压）。
        只抽 hits < promote_hits 的条目：达标者是 resurrect() 的职责，
        两条复活通道不重复注入同一条。返回 MemoryEntry 列表（命中计数 +1）。

        2026-10-12 吞吐提速：冷层改用倒排索引加速（count-based 精确打分）。
        索引未就绪（后台构建中/未触发）时回退原全量扫描，行为完全一致。
        """
        qs = [_grams(q.decode("utf-8", errors="replace")) for q in (queries or [])]
        qs = [g for g in qs if g]
        if not qs or k <= 0:
            return []
        self._ensure_cold()
        # 热层全量打分（≤cap≈500，毫秒级）。
        pool = [en for en in list(self.entries) if en.hits < self.promote_hits]
        scored = []
        for en in pool:
            g = _grams(en.text)
            if not g:
                continue
            ov = max(len(q & g) / len(q) for q in qs)
            if ov > 0.05:
                scored.append((ov, en))
        # 冷层：索引优先，未就绪则全量扫描（语义一致）。
        idx = self._cold_inverted
        if idx is not None:
            scored.extend(self._dream_cold_indexed(qs, k))
        else:
            for en in list(self.cold_entries):
                if en.hits >= self.promote_hits:
                    continue
                g = _grams(en.text)
                if not g:
                    continue
                ov = max(len(q & g) / len(q) for q in qs)
                if ov > 0.05:
                    scored.append((ov, en))
        scored.sort(key=lambda t: (-t[0], t[1].created))
        top = [en for _, en in scored[:k]]
        for en in top:
            en.hits += 1
        return top

    def dream_indexed(self, queries, k=5):
        """倒排索引加速版做梦：语义与 dream() 完全一致，但强制使用冷层索引。

        与 dream() 的唯一区别：若索引尚未构建，会同步构建（测试/需要确定性的
        场景使用；生产路径由 dream() 在索引就绪后自动走加速）。
        """
        qs = [_grams(q.decode("utf-8", errors="replace")) for q in (queries or [])]
        qs = [g for g in qs if g]
        if not qs or k <= 0:
            return []
        self._ensure_cold_index()
        pool = [en for en in list(self.entries) if en.hits < self.promote_hits]
        scored = []
        for en in pool:
            g = _grams(en.text)
            if not g:
                continue
            ov = max(len(q & g) / len(q) for q in qs)
            if ov > 0.05:
                scored.append((ov, en))
        scored.extend(self._dream_cold_indexed(qs, k))
        scored.sort(key=lambda t: (-t[0], t[1].created))
        top = [en for _, en in scored[:k]]
        for en in top:
            en.hits += 1
        return top

    def dream_light(self, queries, k=5, cold_sample=32):
        """浅睡轻量做梦（2026-10-12）：热层全量打分 + 冷层随机小批量抽样。

        语义与 dream() 一致——按 4-gram 检索分数从记忆库抽样旧经验、只抽
        hits < promote_hits 的条目、命中计数 +1、注入量封顶 k——但冷层只随机
        抽样 cold_sample 条参与打分，避免浅睡周期背负冷层全量 O(n) 扫描。
        热层永远全量打分（≤cap≈500，毫秒级）。抽样每次调用种子不同，
        保证跨周期覆盖面。仍 _ensure_cold()：冷层已加载、跨重启不失聪。
        """
        qs = [_grams(q.decode("utf-8", errors="replace")) for q in (queries or [])]
        qs = [g for g in qs if g]
        if not qs or k <= 0:
            return []
        self._ensure_cold()
        # 热层全量 + 冷层随机抽样（浅睡不背全量）。
        pool = [en for en in list(self.entries) if en.hits < self.promote_hits]
        cold_pool = self.cold_entries
        if len(cold_pool) > cold_sample:
            import random as _random
            sample_idx = set(_random.sample(range(len(cold_pool)), cold_sample))
            cold_pool = [cold_pool[i] for i in sorted(sample_idx)]
        pool += [en for en in cold_pool if en.hits < self.promote_hits]
        scored = []
        for en in pool:
            g = _grams(en.text)
            if not g:
                continue
            ov = max(len(q & g) / len(q) for q in qs)
            if ov > 0.05:
                scored.append((ov, en))
        scored.sort(key=lambda t: (-t[0], t[1].created))
        top = [en for _, en in scored[:k]]
        for en in top:
            en.hits += 1
        return top
