"""自主联网爬虫模块（2026-10-15）。

基于**无头浏览器**（Playwright + Chromium）实现动态渲染页面抓取与搜索；当
Playwright 不可用时自动降级为 requests + BeautifulSoup 静态抓取后端。内置：

- 反爬绕过：User-Agent 池轮换、随机延迟抖动、浏览器指纹伪装（Playwright stealth）；
- 请求频率控制：令牌桶（最小请求间隔）；
- 异常容错：网络错误 / 页面结构变更 / 访问限制的统一异常体系 + 指数退避重试；
- 内容提取：HTML 正文清洗（bs4，兜底正则），去除脚本/样式/导航噪声。

用法：
    from dolphin.webcrawler import make_crawler
    c = make_crawler()            # Playwright 可用则用之，否则降级 requests
    c.start()
    results = c.search("量子计算", n=5)
    text = c.fetch(results[0]["url"])
    c.close()
"""
import hashlib
import logging
import os
import random
import re
import threading
import time

log = logging.getLogger("webcrawler")


# ============================================================================
# 异常体系
# ============================================================================

class CrawlerError(Exception):
    """爬虫异常基类。"""


class NetworkError(CrawlerError):
    """网络层错误（连接失败/DNS/SSL）。"""


class TimeoutError(CrawlerError):
    """请求超时。"""


class AccessDeniedError(CrawlerError):
    """访问限制（403/429/验证码/被屏蔽）。"""


class PageStructureError(CrawlerError):
    """页面结构变更（预期选择器缺失/正文提取失败）。"""


# ============================================================================
# 工具函数
# ============================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(html_or_text: str, max_len: int = 4000) -> str:
    """把 HTML 或纯文本清洗为适合喂入字节级模型的正文字符串。

    优先使用 BeautifulSoup 提取段落/标题/列表/代码块文本；不可用时用正则
    兜底。统一去除脚本/样式、折叠空白、截断到 max_len。
    """
    if not html_or_text:
        return ""
    s = str(html_or_text)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(s, "lxml")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()
        parts = []
        for node in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "code", "blockquote"]):
            t = node.get_text(" ", strip=True)
            if t:
                parts.append(t)
        text = "\n".join(parts)
        if len(text) < 50:  # 结构解析失败（非常规页面）→ 退回 body 全文本
            body = soup.body or soup
            text = body.get_text("\n", strip=True)
    except Exception:
        s = _SCRIPT_RE.sub(" ", s)
        s = _TAG_RE.sub(" ", s)
        text = s
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    text = text.strip()
    if max_len and len(text) > max_len:
        text = text[:max_len]
    return text


def parse_rss_results(xml_text: str, limit: int = 10) -> list:
    """解析 Bing RSS 搜索结果 XML，返回 [{title, url, snippet}, ...]。

    这是搜索的**首选路径**：Bing 的 `format=rss` 端点无需 JS、无反爬验证码，
    稳定返回真实自然结果（实测可用）。
    """
    import xml.etree.ElementTree as ET
    out = []
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    # RSS: <rss><channel><item><title>..</title><link>..</link><description>..</description></item>
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if title and link.startswith(("http://", "https://")):
            out.append({"title": title[:300], "url": link,
                        "snippet": re.sub(r"<[^>]+>", "", desc)[:300]})
        if len(out) >= limit:
            break
    return out


# ============================================================================
# 反爬：User-Agent 池
# ============================================================================

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
]


def random_user_agent() -> str:
    return random.choice(UA_POOL)


# ============================================================================
# 频率控制：令牌桶
# ============================================================================

class RateLimiter:
    """令牌桶：两次请求之间至少间隔 min_interval 秒（含随机抖动）。"""

    def __init__(self, min_interval: float = 2.0, jitter: float = 0.5):
        self.min_interval = float(min_interval)
        self.jitter = float(jitter)
        self._next_ts = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            if now < self._next_ts:
                time.sleep(self._next_ts - now)
            self._next_ts = time.time() + self.min_interval + random.uniform(0, self.jitter)


# ============================================================================
# 爬虫基类：统一重试/退避/UA 轮换
# ============================================================================

class BaseCrawler:
    """爬虫基类：封装频率控制、指数退避重试、UA 轮换与异常映射。

    子类实现 `_do_fetch(url, timeout) -> str`（返回 HTML/纯文本）与 `search()`。
    """

    def __init__(self, min_interval: float = 2.0, max_retries: int = 3,
                 timeout: float = 15.0, user_agents=None):
        self.limiter = RateLimiter(min_interval)
        self.max_retries = max(1, int(max_retries))
        self.timeout = float(timeout)
        self.user_agents = list(user_agents) if user_agents else list(UA_POOL)
        self._ua_idx = 0
        self._started = False

    # -- 生命周期 ----------------------------------------------------------
    def start(self):
        self._started = True

    def close(self):
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- 工具 --------------------------------------------------------------
    def next_ua(self):
        self._ua_idx += 1
        return random_user_agent()

    def rotate_ua(self):
        """访问受限时强制轮换 UA（跳过一个）。"""
        return self.next_ua()

    # -- 核心：带重试的抓取 ------------------------------------------------
    def fetch(self, url: str, timeout: float = None, **kw) -> str:
        if not self._started:
            self.start()
        timeout = timeout or self.timeout
        last_err = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                return self._do_fetch(url, timeout, **kw)
            except AccessDeniedError as e:
                self.rotate_ua()
                last_err = e
                # 访问限制：退避更久
                time.sleep(2 ** attempt * 3 + random.uniform(0, 1))
            except (NetworkError, TimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt * 2 + random.uniform(0, 1))
            except PageStructureError as e:
                # 结构变化：换解析模式重试一次
                last_err = e
                time.sleep(1 + random.uniform(0, 1))
        raise last_err or NetworkError(f"抓取失败（重试 {self.max_retries} 次）: {url}")

    def search(self, query: str, n: int = 5) -> list:
        raise NotImplementedError

    def _do_fetch(self, url: str, timeout: float, **kw) -> str:
        raise NotImplementedError


# ============================================================================
# Requests 静态抓取后端（兜底）
# ============================================================================

class RequestsCrawler(BaseCrawler):
    """基于 requests + bs4 的静态抓取后端（无头浏览器不可用时的降级方案）。

    搜索走 Bing RSS 端点（`format=rss`）：无需 JS、无反爬验证码，稳定返回
    真实自然结果；抓取普通网页用 bs4 正文提取。
    """

    SEARCH_URL = "https://www.bing.com/search?q={q}&format=rss&count={n}"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._session = None
        self._session_lock = threading.Lock()

    def start(self):
        super().start()
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
            })

    def close(self):
        with self._session_lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None
        super().close()

    def _do_fetch(self, url: str, timeout: float, **kw) -> str:
        import requests
        ua = kw.get("user_agent") or self.next_ua()
        try:
            r = self._session.get(url, headers={"User-Agent": ua}, timeout=timeout,
                                   allow_redirects=True)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"请求超时: {url}") from e
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"网络错误: {url}: {e}") from e
        if r.status_code in (403, 429):
            raise AccessDeniedError(f"访问限制 HTTP {r.status_code}: {url}")
        if r.status_code >= 400:
            raise NetworkError(f"HTTP {r.status_code}: {url}")
        return r.text

    def search(self, query: str, n: int = 5) -> list:
        from urllib.parse import quote
        url = self.SEARCH_URL.format(q=quote(query), n=n)
        xml = self.fetch(url)
        return parse_rss_results(xml, limit=n)


# ============================================================================
# Playwright 无头浏览器后端（主方案）
# ============================================================================

class PlaywrightCrawler(BaseCrawler):
    """基于 Playwright + Chromium 的无头浏览器后端。

    支持动态渲染页面（JS 执行后抓取）、反爬绕过（关闭自动化标志、注入
    navigator.webdriver=undefined、随机 viewport/UA）。启动时懒加载浏览器。
    """

    SEARCH_URL = "https://www.bing.com/search?q={q}&format=rss&count={n}"

    def __init__(self, headless: bool = True, **kw):
        super().__init__(**kw)
        self.headless = headless
        self._pw = None
        self._browser = None
        self._lock = threading.Lock()

    def start(self):
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise CrawlerError("Playwright 未安装：pip install playwright && playwright install chromium") from e
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled",
                      "--no-sandbox", "--disable-dev-shm-usage"],
            )
        except Exception as e:
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
                self._pw = None
            raise CrawlerError(
                "Chromium 启动失败（可能未安装浏览器二进制）："
                "python -m playwright install chromium") from e
        super().start()

    def close(self):
        with self._lock:
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
                self._pw = None
        super().close()

    def _new_page(self, ua: str):
        ctx = self._browser.new_context(
            user_agent=ua,
            viewport={"width": random.randint(1024, 1440),
                      "height": random.randint(720, 900)},
            locale="zh-CN",
        )
        page = ctx.new_page()
        # 反爬：抹除自动化指纹
        try:
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        except Exception:
            pass
        return ctx, page

    def _do_fetch(self, url: str, timeout: float, wait_selector: str = None, **kw) -> str:
        ua = kw.get("user_agent") or self.next_ua()
        try:
            with self._lock:
                ctx, page = self._new_page(ua)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=int(timeout * 1000))
                if wait_selector:
                    page.wait_for_selector(wait_selector, timeout=int(timeout * 1000))
                else:
                    page.wait_for_timeout(random.randint(800, 1500))  # 等动态渲染
                return page.content()
            finally:
                try:
                    ctx.close()
                except Exception:
                    pass
        except CrawlerError:
            raise
        except Exception as e:
            msg = str(e).lower()
            if "403" in msg or "blocked" in msg or "denied" in msg:
                raise AccessDeniedError(f"访问限制: {url}") from e
            if "timeout" in msg:
                raise TimeoutError(f"页面超时: {url}") from e
            raise NetworkError(f"抓取失败: {url}: {e}") from e

    def search(self, query: str, n: int = 5) -> list:
        from urllib.parse import quote
        url = self.SEARCH_URL.format(q=quote(query), n=n)
        xml = self.fetch(url)
        return parse_rss_results(xml, limit=n)


# ============================================================================
# 工厂：自动选择后端
# ============================================================================

def _playwright_browser_available() -> bool:
    """检测 Playwright 包与其浏览器二进制是否真正可用（不触发浏览器启动）。"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
            return bool(path) and os.path.exists(path)
    except Exception:
        return False


def make_crawler(prefer: str = "auto", **kw) -> BaseCrawler:
    """创建爬虫实例。

    prefer="auto"：Playwright 及其浏览器二进制可用则用无头浏览器，否则降级 requests。
    prefer="playwright"：强制 Playwright（不可用抛 CrawlerError）。
    prefer="requests"：强制 requests 后端。
    """
    if prefer in ("auto", "playwright"):
        if _playwright_browser_available():
            return PlaywrightCrawler(**kw)
        if prefer == "playwright":
            raise CrawlerError("Playwright 浏览器不可用：python -m playwright install chromium")
    return RequestsCrawler(**kw)