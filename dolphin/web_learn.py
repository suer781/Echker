"""模型自主联网学习控制器（2026-10-15）。

让项目自己学：当模型出现"知识饥饿"信号（训练缓冲空、数据惊讶度低、数据太熟
被大量跳过）时，自动联网搜索 → 抓取网页 → 清洗正文 → 喂入 `d.learn()`，
进入既有睡眠训练管线。全程无需人工干预。

- **自主判断**：`should_fetch()` 读取模型状态（缓冲/压力/跳过计数）判断是否需要联网；
- **自主选题**：`decide_queries()` 从最近记忆/自评估日志提取知识缺口，无信号时轮换通用主题；
- **闲时上网**：`idle_browse_loop()` 守护线程，在训练空闲时自动"上网冲浪"补充新知识；
- **容错**：网络/结构/访问限制异常由爬虫后端统一重试，控制器层再兜底跳过。

用法（在 feed.py 训练进程内）：
    from dolphin.web_learn import WebLearner
    learner = WebLearner(d)          # d 为已加载的 Dolphin 实例
    learner.start()                  # 启动闲时上网守护线程
    ...
    learner.stop()
"""
import logging
import random
import threading
import time
from urllib.parse import urlparse

from .webcrawler import CrawlerError, clean_text, make_crawler

log = logging.getLogger("web_learn")

# 可观测性：feed.py 的运行日志只收 stdout（logging 未配 handler 会静默丢弃），
# 这里直连打印保证联网活动在训练日志中可见（[联网] 标签，flush 防缓冲丢失）。
_LOG_LOCK = threading.Lock()


def _emit(msg: str):
    with _LOG_LOCK:
        print(f"[联网] {msg}", flush=True)


DEFAULT_TOPICS = [
    "量子计算的基本原理", "人工智能最新研究进展", "太阳系行星的特征",
    "光合作用的过程", "互联网的发展历史", "机器学习中的注意力机制",
    "海洋生物多样性", "中国古代四大发明", "气候变化的原因与影响",
    "太空探索的最新发现", "DNA 与基因遗传", "可再生能源技术",
    "计算机网络的组成", "人类大脑的记忆机制", "数学中的分形几何",
    "机器人技术的应用", "火山喷发的原因", "世界经济运行规律",
]


class WebLearner:
    """自主联网学习控制器。

    配置项（config dict）：
    - check_interval: 空闲检查间隔（秒），默认 60
    - fetch_interval: 触发式联网轮次最小间隔（秒），默认 600
    - quiet_interval: 绝对冷却（秒），默认 300——**仅在系统真正无事可做时**，
      距上次联网超过该时长即自动冲浪一次（"闲着没事就上网"的兜底节奏）
    - idle_buffer_bytes: 缓冲低于此值视为"训练空闲"，默认 4096
    - bored_pressure: 缓冲压力低于此值视为"无聊"，默认 3.5
    - idle_learn_delta: 检查间隔内新增学习量低于此值视为"没有活干"，默认 5
    - skip_delta_threshold: 两次检查间"太熟跳过"增长超过此值触发联网，默认 50
    - max_queries_per_fetch: 每轮最多搜索词数，默认 2
    - max_pages_per_fetch: 每个搜索词最多抓取页数，默认 2
    - min_content_len: 有效正文最短字节数，默认 200
    - max_content_len: 清洗后正文最大字符数，默认 4000
    - topics: 通用主题轮换列表（无知识缺口信号时使用）
    """

    def __init__(self, dolphin, crawler=None, config=None):
        self.d = dolphin
        cfg = dict(config or {})
        self.check_interval = float(cfg.get("check_interval", 60))
        self.fetch_interval = float(cfg.get("fetch_interval", 600))
        self.quiet_interval = float(cfg.get("quiet_interval", 300))
        self.idle_buffer_bytes = int(cfg.get("idle_buffer_bytes", 4096))
        self.bored_pressure = float(cfg.get("bored_pressure", 3.5))
        self.idle_learn_delta = int(cfg.get("idle_learn_delta", 5))
        self.skip_delta_threshold = int(cfg.get("skip_delta_threshold", 50))
        self.max_queries_per_fetch = int(cfg.get("max_queries_per_fetch", 2))
        self.max_pages_per_fetch = int(cfg.get("max_pages_per_fetch", 2))
        self.min_content_len = int(cfg.get("min_content_len", 200))
        self.max_content_len = int(cfg.get("max_content_len", 4000))
        self.topics = list(cfg.get("topics") or DEFAULT_TOPICS)
        self.crawler = crawler if crawler is not None else make_crawler()
        self._topic_idx = random.randrange(len(self.topics))
        self._last_fetch_ts = 0.0
        self._last_skip = getattr(self.d, "skipped_too_easy", 0)
        # 负面信号基线（2026-10-17 用户定案）：出现新异常事件（被指出问题/
        # 自己没做好）→ 自发上网查正确知识。随检查点更新。
        self._last_anomaly = len(getattr(self.d, "anomaly_log", []) or [])
        # 自我怀疑基线（2026-10-17 用户定案）：模型自评估发现最近回答可能
        # 错了/不好（self_eval_log 新增负奖励）→ 主动上网查知识补救。
        self._last_self_eval = len(getattr(self.d, "self_eval_log", []) or [])
        # 自我怀疑触发的冷却（秒）：避免自评估连续负分时反复联网刷屏。
        self.doubt_interval = float(cfg.get("doubt_interval", 60))
        # 活跃喂入基线：learn_total 增量超过 idle_learn_delta 视为"有活干"。
        self._last_learn_total = getattr(self.d, "learn_total", 0)
        self._stop = threading.Event()
        self._thread = None
        self.stats = {"rounds": 0, "searches": 0, "fetched": 0,
                      "learned": 0, "errors": 0, "snippet_fallbacks": 0,
                      "last_error": None}
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------ 生命周期
    def start(self):
        """启动闲时上网守护线程（幂等）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self.crawler.start()
        except Exception as e:  # 爬虫后端不可用不阻塞训练
            _emit(f"爬虫启动失败（联网学习降级为仅靠本地数据）：{e!r}")
            log.error("爬虫启动失败（联网学习降级为仅靠本地数据）：%r", e)
            self._bump_stats(error=str(e))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="web-learn",
                                        daemon=True)
        self._thread.start()
        _emit(f"WebLearner 已启动：每 {self.check_interval:.0f}s 检查一次，"
              f"静默冷却 {self.quiet_interval:.0f}s 到期即自主冲浪")
        log.info("WebLearner 已启动：每 %.0fs 检查一次联网学习", self.check_interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            self.crawler.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ 状态
    def _bump_stats(self, **kw):
        with self._stats_lock:
            for k, v in kw.items():
                if k == "error":
                    self.stats["errors"] += 1
                    self.stats["last_error"] = v
                elif k in self.stats:
                    self.stats[k] += v

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self.stats)

    # ------------------------------------------------------------------ 自主判断
    def _system_idle(self) -> bool:
        """系统是否真正无事可做（2026-10-17 用户定案）。

        必须同时满足：
        - 没有积压内容：缓冲字节数 ≤ idle_buffer_bytes 且压力 ≤ bored_pressure；
        - 没有活跃喂入：距上次检查新增学习量 ≤ idle_learn_delta
          （数据源正在大量喂入时，即使缓冲被训练线程抽空也不算无事可做）。
        """
        d = self.d
        buf_bytes = getattr(d.buffer, "bytes", 0)
        pressure = d.buffer.pressure()
        if buf_bytes > self.idle_buffer_bytes or pressure > self.bored_pressure:
            return False
        lt = getattr(d, "learn_total", 0)
        return (lt - self._last_learn_total) <= self.idle_learn_delta

    def should_fetch(self) -> bool:
        """模型自主判断是否需要联网（四通道，2026-10-17 用户定案）。

        通道③（被指出问题/自己失败，最优先）：出现新的异常事件（用户指出
        问题、自己没做好/任务失败）→ **无条件**自发上网查正确知识/操作——
        即使当前训练繁忙也立即查（知错就改、求知补救）。
        通道④（自我怀疑）：模型自评估发现最近回答可能错了/不好（self_eval_log
        新增负奖励）→ 主动上网查知识补救（知不足而学）；带短冷却防刷屏。
        通道①（真正闲时兜底）：系统**真正无事可做**（无积压、无活跃喂入）
        且距上次联网 ≥ quiet_interval → 自动冲浪。
        通道②（知识饥饿）：系统真正无事可做且距上次联网 ≥ fetch_interval，
        且数据太熟（跳过计数增量 ≥ skip_delta_threshold）→ 联网补充新知识。
        """
        if self._stop.is_set():
            return False
        d = self.d
        # 通道③：新异常事件 → 无条件触发（无论是否繁忙）
        anomaly = len(getattr(d, "anomaly_log", []) or [])
        if anomaly - self._last_anomaly > 0:
            return True
        # 通道④：自我怀疑 → 自评估新增负奖励则触发（带短冷却防刷屏）
        self_eval = len(getattr(d, "self_eval_log", []) or [])
        if self_eval - self._last_self_eval > 0:
            new_evals = (getattr(d, "self_eval_log", []) or [])[
                self._last_self_eval:]
            if any(e.get("reward", 0) < 0 for e in new_evals):
                since_any = time.time() - self._last_fetch_ts
                if since_any >= self.doubt_interval:
                    return True
        # 其余通道都要求"真正无事可做"
        if not self._system_idle():
            return False
        since = time.time() - self._last_fetch_ts
        # 通道①：真正闲时兜底
        if self.quiet_interval > 0 and since >= self.quiet_interval:
            return True
        # 通道②：知识饥饿（数据太熟）
        if since < self.fetch_interval:
            return False
        skip = getattr(d, "skipped_too_easy", 0)
        skip_delta = skip - self._last_skip
        return skip_delta >= self.skip_delta_threshold

    def _note_checkpoint(self):
        """每次检查后记录基线（供增量判断）。"""
        self._last_skip = getattr(self.d, "skipped_too_easy", 0)
        self._last_anomaly = len(getattr(self.d, "anomaly_log", []) or [])
        self._last_self_eval = len(getattr(self.d, "self_eval_log", []) or [])
        self._last_learn_total = getattr(self.d, "learn_total", 0)

    # ------------------------------------------------------------------ 自主选题
    def decide_queries(self, n: int = 2) -> list:
        """生成搜索词。

        依次尝试：
        1. 从热层记忆抽样一条文本，提取其中的连续中英文字符片段作为"好奇主题"；
        2. 从最近自评估日志（若有 eid 且 reward<0）反查缓冲文本作为知识缺口；
        3. 回退：轮换通用主题列表。
        """
        queries = []
        # ① 记忆热层好奇心种子
        try:
            mem = list(getattr(self.d.memory, "entries", []) or [])
            if mem:
                seed = random.choice(mem).text
                frag = self._extract_query(seed)
                if frag:
                    queries.append(frag)
        except Exception:
            pass
        # ② 知识缺口：最近负奖励交互
        try:
            logs = list(getattr(self.d, "self_eval_log", []) or [])
            for entry in reversed(logs[-30:]):
                if entry.get("reward", 0) >= 0:
                    continue
                eid = entry.get("eid")
                item = next((e for e in self.d.buffer.items if e.id == eid), None)
                if item is not None:
                    frag = self._extract_query(item.data.decode("utf-8", "replace"))
                    if frag and frag not in queries:
                        queries.append(frag)
                if len(queries) >= n:
                    break
        except Exception:
            pass
        # ③ 回退：轮换通用主题
        while len(queries) < n:
            topic = self.topics[self._topic_idx % len(self.topics)]
            self._topic_idx += 1
            if topic not in queries:
                queries.append(topic)
        return queries[:n]

    @staticmethod
    def _extract_query(text: str, max_len: int = 24) -> str:
        """从文本中提取连续中英文字符片段作为搜索词（去标点/空白）。"""
        if not text:
            return ""
        import re
        frags = re.findall(r"[\u4e00-\u9fff]{4,}|[A-Za-z][A-Za-z0-9 _\-]{4,}", text)
        if not frags:
            return ""
        frag = frags[0].strip()
        return frag[:max_len]

    # ------------------------------------------------------------------ 联网学习
    def fetch_and_learn(self) -> int:
        """联网搜索→抓取→清洗→learn()。返回成功喂入条数。

        页面抓取失败时启用**摘要兜底**：把该查询的搜索结果标题+摘要聚合为一条
        学习文本喂入（保证在目标站反爬/网络受限环境下仍能学到真实网络信息）。
        """
        queries = self.decide_queries(self.max_queries_per_fetch)
        learned = 0
        for q in queries:
            try:
                results = self.crawler.search(q, n=self.max_pages_per_fetch)
            except CrawlerError as e:
                self._bump_stats(error=f"search {q!r}: {e}")
                continue
            self._bump_stats(searches=1)
            if not results:
                continue
            page_learned = 0
            snippets = []
            for r in results[:self.max_pages_per_fetch]:
                url = r.get("url", "")
                title = (r.get("title") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                if title or snippet:
                    snippets.append(f"{title}。{snippet}")
                if not url:
                    continue
                try:
                    html = self.crawler.fetch(url)
                except CrawlerError as e:
                    self._bump_stats(error=f"fetch {url}: {e}")
                    continue
                self._bump_stats(fetched=1)
                text = clean_text(html, max_len=self.max_content_len)
                if len(text.encode("utf-8", errors="replace")) < self.min_content_len:
                    continue
                domain = urlparse(url).netloc or "web"
                try:
                    eid = self.d.learn("【联网资料】" + text, source=f"web:{domain}")
                except Exception as e:
                    self._bump_stats(error=f"learn {url}: {e}")
                    continue
                if eid is not None:
                    page_learned += 1
            # 摘要兜底：本查询没有任何页面成功喂入，但搜索结果摘要含真实信息
            if page_learned == 0 and snippets:
                agg = "\n".join(snippets)
                agg = clean_text(agg, max_len=self.max_content_len)
                if len(agg.encode("utf-8", errors="replace")) >= self.min_content_len:
                    try:
                        eid = self.d.learn("【联网资料】" + agg, source="web:search")
                    except Exception as e:
                        self._bump_stats(error=f"learn snippet {q!r}: {e}")
                    else:
                        if eid is not None:
                            page_learned += 1
                            self._bump_stats(snippet_fallbacks=1)
            learned += page_learned  # 每个查询的贡献即时累计（须在查询循环内）
        self._last_fetch_ts = time.time()
        self._note_checkpoint()
        self._bump_stats(rounds=1, learned=learned)
        if learned:
            _emit(f"学到 {learned} 条新知识（查询：{queries[0]!r}）"
                  f"{'，含摘要兜底' if self.stats.get('snippet_fallbacks') else ''}")
            log.info("联网学习完成：%d 条新知识入缓冲（查询 %s）", learned, queries[:1])
        else:
            _emit(f"本轮无新知识入缓冲（查询 {queries!r}，"
                  f"可能被金发姑娘太熟跳过或内容不足），累计错误 {self.stats['errors']}")
        return learned

    # ------------------------------------------------------------------ 闲时上网
    def _run_loop(self):
        while not self._stop.wait(self.check_interval):
            try:
                if self.should_fetch():
                    quiet_due = time.time() - self._last_fetch_ts >= self.quiet_interval
                    reason = "静默冷却到期" if quiet_due else "知识饥饿"
                    _emit(f"{reason}触发联网学习（缓冲 {self.d.buffer.bytes}B "
                          f"压力 {self.d.buffer.pressure():.2f}）")
                    learned = self.fetch_and_learn()
                    _emit(f"本轮联网完成：喂入 {learned} 条，累计 "
                          f"轮次 {self.stats['rounds']} 页 {self.stats['fetched']} "
                          f"知识 {self.stats['learned']} 错误 {self.stats['errors']}")
                self._note_checkpoint()
            except Exception as e:  # 控制器异常绝不能拖垮训练线程
                self._bump_stats(error=f"loop: {e}")
                _emit(f"循环异常（已容错继续）：{e!r}")