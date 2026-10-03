"""记忆库（律 L9 / L10）。

- 滞留：没进传输预算的经验降级到这里，可检索、不进权重。
- 预支（快通道）：醒脑服务时检索引用，即时可用。
- 复活（间隔重复）：被频繁检索命中的条目升权，进入下一轮睡眠选拔。
- 冷层（2026-10-03 G1 施工）：逐出=降级落盘 memory_cold.jsonl（append-only 日志），
  不是删除——被逐出条目跨周期仍可检索、可复活，L9"可检索可复活"对冷层同样成立。
- 做梦（同上）：dream() 按检索分数自我回忆——喂食模式没有人工对话供给检索
  命中，hits 由做梦代偿，复活通道不再死亡。
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
                    try:  # 崩溃尾行的半行日志不拖垮整卷
                        self.cold_entries.append(MemoryEntry.from_json(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
        self._cold_loaded = True

    def _ensure_cold(self):
        with self._cold_lock:
            self._load_cold_locked()

    def flush_cold(self):
        """把冷层影子索引（含检索命中计数）写回落盘，供存档时调用——
        冷层命中进度跨重启不丢。缓存从未加载过则落盘文件即真相，不动；
        冷层从未启用过（无逐出、文件也不存在）则零足迹，不凭空造文件。"""
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
          逐出它=add 的净效果是删除（审计实锤的新条目自杀 bug）。
        - 零命中集合为空（全员被检索过）→ 不逐出，扩容一档：cap 由事件驱动
          上调（值：自成）——旧知识不因新知识到来而被处决。
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
        with self._cold_lock:
            self._load_cold_locked()
            d = os.path.dirname(self.cold_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.cold_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
            self.cold_entries.append(entry)

    # ---------- 检索 / 复活 / 做梦 ----------

    def retrieve(self, query: bytes, k=1):
        q = _grams(query.decode("utf-8", errors="replace"))
        if not q:
            return []
        self._ensure_cold()  # 冷层同场竞技：G1 修复前被逐出条目从此永远查无此人
        scored = []
        for en in self.entries:
            g = _grams(en.text)
            if g:
                ov = len(q & g) / len(q)
                if ov > 0.05:
                    scored.append((ov, en))
        for en in self.cold_entries:
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
        """
        out = []
        for en in self.entries:
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                en.hits = 0
        promoted = []
        for en in self.cold_entries:
            if en.hits >= self.promote_hits:
                out.append(en.text.encode("utf-8", errors="replace"))
                self.cold_entries.remove(en)
                self.entries.append(en)
                en.hits = 0
                promoted.append(en)
        while len(self.entries) > self.cap:  # 冷层升回可能顶满：热层规矩照旧
            self._evict(protect=promoted[-1] if promoted else None)
        return out

    def dream(self, queries, k=5):
        """做梦（G1）：按检索分数从记忆库抽样旧事，供周期头部注入缓冲。

        喂食模式没有人工对话供给检索命中——用近期经验做联想线索自我回忆，
        hits 由此增长，复活通道不再依赖有人来聊。注入量封顶 k（≤5），
        防旧知识挤占新经验的选拔预算（审计 G11：满带权挤压）。
        只抽 hits < promote_hits 的条目：达标者是 resurrect() 的业务，
        两条复活通道不重复注入同一条。返回 MemoryEntry 列表（命中计数 +1）。
        """
        qs = [_grams(q.decode("utf-8", errors="replace")) for q in (queries or [])]
        qs = [g for g in qs if g]
        if not qs or k <= 0:
            return []
        self._ensure_cold()
        pool = [en for en in self.entries + self.cold_entries
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
