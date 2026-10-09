"""记忆库（律 L9 / L10）。

- 滞留：没进传输预算的经验降级到这里，可检索、不进权重。
- 预支（快通道）：醒脑服务时检索引用，即时可用。
- 复活（间隔重复）：被频繁检索命中的条目升权，进入下一轮睡眠选拔。
- 冷层（2026-10-03 G1 施工）：逐出=降级落盘 memory_cold.jsonl（append-only 日志），
  不是删除——被逐出条目跨周期仍可检索、可复活，L9"可检索可复活"对冷层同样成立。
- 做梦（同上）：dream() 按检索分数自我回忆——喂食模式没有人工对话供给检索
  命中，hits 由做梦代偿，复活通道保持可用。
"""
import json
import os
import threading
import time


class MemoryEntry:
    __slots__ = ("text", "score", "hits", "cycle", "kind", "created")

    def __init__(self, text: str, score: float, cycle: int, kind: str):
        self.text = text
        self.score = float(score)
        self.hits = 0
        self.cycle = int(cycle)
        self.kind = kind
        self.created = time.time()

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

    # ---------- 冷层（G1：逐出降级落盘，跨周期可检索可复活） ----------

    def _load_cold_locked(self):
        if self._cold_loaded:
            return
        self.cold_entries = []
        if os.path.exists(self.cold_path):
            with open(self.cold_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:  # 崩溃尾行的半行日志不影响整体加载
                        en = MemoryEntry.from_json(json.loads(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    # 2026-10-04 修复：旧实现逐行 append，反刍把同一条反复逐出，
                    # 会导致同一 text 落盘成百上千行（生产实况 16316 行仅 153 条
                    # 唯一 text）。加载时按 text 去重：保留 hits 最大、created 最早
                    # 的一条（信息量最大且保留审计链），一次加载即修复旧档。
                    existing = next((x for x in self.cold_entries if x.text == en.text), None)
                    if existing is None:
                        self.cold_entries.append(en)
                    else:
                        if en.hits > existing.hits:
                            existing.hits = en.hits
                        if en.created < existing.created:
                            existing.created = en.created
                        # 保留更低的 score（逐出优先级 min 取低分，早逐出者应优先保留）
                        if en.score < existing.score:
                            existing.score = en.score
        self._cold_loaded = True

    def _ensure_cold(self):
        with self._cold_lock:
            self._load_cold_locked()

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
        """
        qs = [_grams(q.decode("utf-8", errors="replace")) for q in (queries or [])]
        qs = [g for g in qs if g]
        if not qs or k <= 0:
            return []
        self._ensure_cold()
        # 2026-10-04 修复：与 retrieve 同理，训练线程并发修改热/冷层列表，
        # 取快照保证一致视图（+ 操作虽然创建新列表，但两个取值点之间可能被插入）。
        pool = [en for en in list(self.entries) + list(self.cold_entries)
                if en.hits < self.promote_hits]
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
