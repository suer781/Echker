"""自主联网学习机制测试（2026-10-15）。

覆盖：
- 爬虫稳定性：RateLimiter 频率控制、网络错误重试、访问限制 UA 轮换、结构变更兜底。
- 信息采集准确性：clean_text 正确剥离 HTML/脚本、提取正文。
- 自主学习有效性：WebLearner 自主判断联网时机、选题、抓取并喂入 Dolphin 训练管线。
- 控制器生命周期：start/stop 守护线程安全退出。
隔离纪律：全部用 FakeCrawler（无真实网络）；Dolphin 用沙箱冷层+合成探测卷。
"""
import atexit
import os
import shutil
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin.dolphin import Dolphin
from dolphin.memory import MemoryStore
from dolphin.model import Config
from dolphin.web_learn import WebLearner
from dolphin.webcrawler import (AccessDeniedError, BaseCrawler, NetworkError,
                                RateLimiter, clean_text, make_crawler,
                                parse_rss_results)

# ---------- 冷层隔离 ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROD_COLD = os.path.realpath(os.path.join(_ROOT, "dolphin", "memory_cold.jsonl"))
_SANDBOX = tempfile.mkdtemp(prefix="dolphin_web_sandbox_")
atexit.register(shutil.rmtree, _SANDBOX, True)
_sandbox_seq = 0


def _sandbox_cold_path():
    global _sandbox_seq
    _sandbox_seq += 1
    return os.path.join(_SANDBOX, f"cold_{_sandbox_seq:04d}.jsonl")


_orig_ms_init = MemoryStore.__init__


def _quarantined_ms_init(self, cap_entries=500, promote_hits=3, cold_path=None):
    resolved = os.path.realpath(cold_path) if cold_path else _PROD_COLD
    if resolved == _PROD_COLD:
        cold_path = _sandbox_cold_path()
    _orig_ms_init(self, cap_entries, promote_hits, cold_path)


MemoryStore.__init__ = _quarantined_ms_init

_SMALL_CFG = Config(d_model=32, n_layers=2, n_heads=2, block_size=64)
_PROBE = os.path.join(_SANDBOX, "probe_web.txt")


@pytest.fixture()
def d():
    body = ("逻辑推理探测片段：若甲高于乙则甲高于丙；金属导电铁是金属故铁导电；"
            "此段仅供体检评分永不进训练粮。").encode("utf-8")
    with open(_PROBE, "wb") as f:
        f.write(body * 4)
    dol = Dolphin(cfg=_SMALL_CFG, device="cpu", probe_path=_PROBE)
    dol.param_adaptive_enabled = False
    dol.memory = MemoryStore(cold_path=_sandbox_cold_path())
    return dol


# ==================== 爬虫稳定性 ====================

def test_rate_limiter_enforces_min_interval():
    rl = RateLimiter(min_interval=0.05, jitter=0.0)
    t0 = time.time()
    rl.wait()
    rl.wait()
    dt = time.time() - t0
    assert dt >= 0.05 - 1e-3, f"最小间隔未生效：{dt:.4f}s"


def test_clean_text_strips_html_and_scripts():
    html = """
    <html><head><script>var x=1;</script><style>body{}</style></head>
    <body><nav>导航</nav><h1>标题：测试页面</h1>
    <p>第一段正文内容，用于验证正文提取。</p>
    <p>第二段正文内容。</p><footer>页脚</footer></body></html>
    """
    text = clean_text(html, max_len=10000)
    assert "标题：测试页面" in text
    assert "第一段正文内容" in text
    assert "第二段正文内容" in text
    assert "script" not in text.lower() and "var x" not in text
    assert "导航" not in text  # nav 被剔除


def test_parse_rss_results_extracts_items():
    xml = """<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>量子计算入门</title><link>https://example.com/quantum</link>
    <description>&lt;p&gt;量子计算介绍文本&lt;/p&gt;</description></item>
    <item><title>第二个结果</title><link>https://example.com/two</link>
    <description>第二个</description></item>
    </channel></rss>"""
    res = parse_rss_results(xml, limit=2)
    assert len(res) == 2
    assert res[0]["title"] == "量子计算入门"
    assert res[0]["url"] == "https://example.com/quantum"
    assert "量子计算介绍" in res[0]["snippet"]
    assert res[1]["title"] == "第二个结果"


def test_crawler_retries_network_error():
    class FlakyCrawler(BaseCrawler):
        def __init__(self):
            super().__init__(min_interval=0.0, max_retries=3)
            self.calls = 0

        def _do_fetch(self, url, timeout, **kw):
            self.calls += 1
            if self.calls < 3:
                raise NetworkError("boom")
            return "<html><body><p>ok</p></body></html>"

        def search(self, query, n=5):
            return []

    c = FlakyCrawler()
    c.start()
    try:
        out = c.fetch("http://example.com")
        assert "ok" in out
        assert c.calls == 3  # 失败 2 次后成功
    finally:
        c.close()


def test_crawler_access_denied_rotates_ua():
    class DeniedCrawler(BaseCrawler):
        def __init__(self):
            super().__init__(min_interval=0.0, max_retries=2)
            self.calls = 0

        def _do_fetch(self, url, timeout, **kw):
            self.calls += 1
            raise AccessDeniedError("403")

        def search(self, query, n=5):
            return []

    c = DeniedCrawler()
    c.start()
    ua_before = c._ua_idx
    with pytest.raises(AccessDeniedError):
        c.fetch("http://blocked.example")
    assert c._ua_idx > ua_before  # UA 已轮换
    assert c.calls == c.max_retries
    c.close()


def test_make_crawler_returns_base():
    """无论 Playwright 浏览器是否可用，工厂都应返回可用后端（不抛错）。"""
    c = make_crawler("auto", min_interval=0.0, max_retries=1)
    assert isinstance(c, BaseCrawler)
    c.close()


# ==================== 自主学习有效性 ====================

class FakeCrawler(BaseCrawler):
    """无网络假爬虫：返回固定搜索结果与页面 HTML。"""

    def __init__(self, results=None):
        super().__init__(min_interval=0.0, max_retries=1)
        self.results = results or [
            {"title": "量子计算", "url": "https://wiki.example/quantum",
             "snippet": "量子计算利用量子叠加与纠缠。"},
        ]
        self.page_html = ("<html><body><h1>量子计算</h1>"
                          "<p>量子计算是利用量子力学原理进行计算的新范式。</p>"
                          "<p>它在特定问题上具有经典计算机无法比拟的优势。</p>"
                          "</body></html>")
        self.search_calls = 0
        self.fetch_calls = 0

    def search(self, query, n=5):
        self.search_calls += 1
        return self.results[:n]

    def _do_fetch(self, url, timeout, **kw):
        self.fetch_calls += 1
        return self.page_html


def _fresh_learner(d, results=None):
    return WebLearner(d, crawler=FakeCrawler(results), config={
        "check_interval": 1.0,
        "fetch_interval": 0.0,
        "idle_buffer_bytes": 4096,
        "bored_pressure": 100.0,  # 恒触发：压力必然低于此值
        "max_queries_per_fetch": 1,
        "max_pages_per_fetch": 1,
        "min_content_len": 10,
        "max_content_len": 4000,
    })


def test_should_fetch_when_idle(d):
    d.buffer.items = []
    d.buffer.bytes = 0
    wl = _fresh_learner(d)
    wl._last_fetch_ts = 0.0  # 超过最小间隔
    assert wl.should_fetch() is True


def test_should_fetch_false_within_interval(d):
    wl = _fresh_learner(d)
    wl._last_fetch_ts = time.time()  # 刚联网过
    wl.fetch_interval = 600.0
    wl.quiet_interval = 300.0
    assert wl.should_fetch() is False


def test_quiet_interval_not_enough_when_busy(d):
    """静默冷却到期但系统繁忙（有积压内容）→ 不触发（2026-10-17 用户定案：
    只有真正无事可做时才上网冲浪）。"""
    wl = _fresh_learner(d)
    # 模拟训练繁忙：缓冲满、压力高、无跳过增量
    d.buffer.bytes = 999999
    d.buffer.items = []
    wl._last_fetch_ts = time.time() - 301.0  # 静默冷却(300s)已过期
    wl.fetch_interval = 600.0  # 知识饥饿通道尚未到期
    wl._last_skip = getattr(d, "skipped_too_easy", 0)
    assert wl.should_fetch() is False, "繁忙时有积压内容，不应上网冲浪"


def test_quiet_interval_fires_when_truly_idle(d):
    """静默冷却到期且系统真正无事可做 → 触发冲浪。"""
    wl = _fresh_learner(d)
    d.buffer.bytes = 0
    d.buffer.items = []
    wl._last_fetch_ts = time.time() - 301.0
    wl.fetch_interval = 600.0
    wl._last_skip = getattr(d, "skipped_too_easy", 0)
    wl._last_learn_total = getattr(d, "learn_total", 0)  # 无活跃喂入
    assert wl.should_fetch() is True, "真正无事可做且冷却到期，应触发冲浪"


def test_anomaly_triggers_fetch_even_when_busy(d):
    """出现新异常事件（被指出问题/自己失败）→ 即使训练繁忙也无条件上网查资料。"""
    wl = _fresh_learner(d)
    # 模拟训练繁忙
    d.buffer.bytes = 999999
    d.buffer.items = []
    wl._last_fetch_ts = time.time()  # 刚联网过
    wl.fetch_interval = 600.0
    wl.quiet_interval = 300.0
    wl._last_skip = getattr(d, "skipped_too_easy", 0)
    wl._last_anomaly = len(getattr(d, "anomaly_log", []) or [])
    # 新增一条异常事件（用户指出问题/自己失败）
    d.anomaly_log.append({"kind": "user_correction", "detail": "答错了",
                          "ts": time.time()})
    assert wl.should_fetch() is True, "出现异常事件必须无条件触发联网补救"
    # 检查点更新后，同一异常不重复触发
    wl._note_checkpoint()
    assert wl.should_fetch() is False, "已处理的异常不应重复触发"


def test_self_doubt_triggers_fetch(d):
    """通道④自我怀疑（2026-10-17 用户定案）：模型自评估发现最近回答可能
    错了/不好（self_eval_log 新增负奖励）→ 主动上网查知识补救。"""
    wl = _fresh_learner(d)
    # 模拟训练繁忙（自我怀疑通道不要求空闲）
    d.buffer.bytes = 999999
    d.buffer.items = []
    wl._last_fetch_ts = 0.0  # 冷却已过
    wl.doubt_interval = 60.0
    wl._last_self_eval = len(getattr(d, "self_eval_log", []) or [])
    # 新增负奖励自评估（模型觉得自己答错了）
    d.self_eval_log.append({"eid": 1, "reward": -1.0, "dim": {}})
    assert wl.should_fetch() is True, "自我怀疑（负奖励）应触发联网"
    # 检查点更新后不重复触发
    wl._note_checkpoint()
    assert wl.should_fetch() is False, "已处理的自我怀疑不应重复触发"
    # 正奖励自评估不触发
    d.self_eval_log.append({"eid": 2, "reward": 1.0, "dim": {}})
    assert wl.should_fetch() is False, "正奖励不应触发自我怀疑上网"


def test_decide_queries_fallback_topic(d):
    wl = _fresh_learner(d)
    qs = wl.decide_queries(n=2)
    assert len(qs) == 2
    assert all(isinstance(q, str) and q for q in qs)


def test_fetch_and_learn_feeds_dolphin(d):
    """端到端：搜索→抓取→清洗→learn() → 模型缓冲增长（知识更新）。"""
    wl = _fresh_learner(d)
    learned = wl.fetch_and_learn()
    assert learned >= 1, f"应至少喂入 1 条，实际 {learned}"
    assert wl.crawler.search_calls >= 1
    assert wl.crawler.fetch_calls >= 1
    assert d.learn_total >= 1
    assert len(d.buffer.items) >= 1
    stats = wl.get_stats()
    assert stats["learned"] >= 1
    assert stats["rounds"] >= 1


def test_fetch_and_learn_skips_short_content(d):
    """清洗后正文过短（< min_content_len）→ 跳过不喂入。"""
    fc = FakeCrawler()
    fc.page_html = "<html><body><p>短</p></body></html>"
    wl = WebLearner(d, crawler=fc, config={
        "fetch_interval": 0.0, "min_content_len": 100, "max_content_len": 4000,
        "max_queries_per_fetch": 1, "max_pages_per_fetch": 1,
    })
    learned = wl.fetch_and_learn()
    assert learned == 0
    assert d.learn_total == 0


def test_idle_loop_start_stop(d):
    wl = _fresh_learner(d)
    wl.start()
    time.sleep(1.2)  # 至少一个检查周期
    wl.stop()
    assert wl._thread is None or not wl._thread.is_alive()


def test_learner_error_does_not_crash(d):
    """爬虫异常时控制器容错，不抛错、不拖垮训练。"""
    class BoomCrawler(BaseCrawler):
        def __init__(self):
            super().__init__(min_interval=0.0, max_retries=1)

        def search(self, query, n=5):
            raise NetworkError("search down")

        def _do_fetch(self, url, timeout, **kw):
            raise NetworkError("fetch down")

    wl = WebLearner(d, crawler=BoomCrawler(), config={"fetch_interval": 0.0})
    learned = wl.fetch_and_learn()
    assert learned == 0
    assert wl.get_stats()["errors"] >= 1


def test_snippet_fallback_when_page_fetch_fails(d):
    """页面抓取失败（反爬/网络受限）时，搜索结果摘要聚合兜底喂入模型。"""
    class SnippetOnlyCrawler(BaseCrawler):
        def __init__(self):
            super().__init__(min_interval=0.0, max_retries=1)
            self.results = [
                {"title": "量子计算（遵循量子力学规律的计算模式）",
                 "url": "https://blocked.example/quantum",
                 "snippet": "量子计算是一种遵循量子力学规律的新型计算模式，"
                            "利用量子叠加与量子纠缠实现并行计算。"},
                {"title": "量子计算的未来",
                 "url": "https://blocked.example/future",
                 "snippet": "量子计算在密码学、材料模拟等领域具有巨大潜力。"},
            ]

        def search(self, query, n=5):
            return self.results[:n]

        def _do_fetch(self, url, timeout, **kw):
            raise NetworkError("blocked")

    wl = WebLearner(d, crawler=SnippetOnlyCrawler(), config={
        "fetch_interval": 0.0,
        "max_queries_per_fetch": 1,
        "max_pages_per_fetch": 2,
        "min_content_len": 20,
        "max_content_len": 4000,
    })
    learned = wl.fetch_and_learn()
    assert learned >= 1, f"摘要兜底应喂入至少 1 条，实际 {learned}"
    assert wl.get_stats()["snippet_fallbacks"] >= 1
    assert d.learn_total >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))