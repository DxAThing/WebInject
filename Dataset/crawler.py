# ============================================================
# crawler.py — 真实网页采集器
# ============================================================
# 功能：
#   1. 通过 Google / Bing / DuckDuckGo 搜索获取目标 URL
#      - Google / Bing 使用 Selenium 可见浏览器，遇到人机验证暂停让用户操作
#      - DuckDuckGo 使用 requests（无需浏览器）
#   2. 使用 single-file-cli 将网页下载为单一 HTML 文件
#   3. 维护 URL 映射表（url_mapping.json）
#   4. 支持断点续传（已下载的文件自动跳过）
# ============================================================

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import platform as _platform_mod
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

import config

# 平台检测
_IS_WINDOWS = sys.platform == "win32"
_SYSTEM = _platform_mod.system()  # "Windows", "Darwin", "Linux"

# ============================================================
# Selenium 搜索浏览器管理（Google / Bing）
# ============================================================
# 使用可见浏览器，遇到人机验证时用户可直接在浏览器窗口完成
# 同一个浏览器会话在整个爬虫生命周期复用（保持登录/Cookie）

_search_driver = None   # 全局复用的 Selenium WebDriver


def _get_search_driver():
    """获取或创建用于搜索的可见浏览器（非 headless）。"""
    global _search_driver
    if _search_driver is not None:
        try:
            _ = _search_driver.title  # 检查 session 是否还活着
            return _search_driver
        except Exception:
            _search_driver = None

    from selenium import webdriver

    _NON_HEADLESS_ARGS = (
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    )

    # ---------- 尝试 Edge ----------
    edge_candidates: list[str] = []
    if _SYSTEM == "Windows":
        edge_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
    elif _SYSTEM == "Darwin":
        edge_candidates = [
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]

    edge_bin = (
        shutil.which("msedge")
        or shutil.which("microsoft-edge")
        or shutil.which("microsoft-edge-stable")
    )
    if edge_bin is None:
        for p in edge_candidates:
            if os.path.isfile(p):
                edge_bin = p
                break

    if edge_bin:
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService
        from monitor_simulator import _ensure_edge_driver

        driver_path = _ensure_edge_driver(edge_bin)
        opts = EdgeOptions()
        opts.binary_location = edge_bin
        for arg in _NON_HEADLESS_ARGS:
            opts.add_argument(arg)
        # 防止被检测为自动化
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        service = EdgeService(executable_path=driver_path)
        service.log_output = subprocess.DEVNULL
        _search_driver = webdriver.Edge(service=service, options=opts)
        # 隐藏 webdriver 标识
        _search_driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        print(f"[Search] 可见 Edge 浏览器已启动 ({edge_bin})")
        return _search_driver

    # ---------- 尝试 Chrome ----------
    chrome_candidates: list[str] = []
    if _SYSTEM == "Windows":
        chrome_candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
    elif _SYSTEM == "Darwin":
        chrome_candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ]

    chrome_bin = (
        shutil.which("chrome")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium-browser")
        or shutil.which("chromium")
    )
    if chrome_bin is None:
        for p in chrome_candidates:
            if os.path.isfile(p):
                chrome_bin = p
                break

    if chrome_bin:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        opts = ChromeOptions()
        opts.binary_location = chrome_bin
        for arg in _NON_HEADLESS_ARGS:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        _search_driver = webdriver.Chrome(options=opts)
        _search_driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        print(f"[Search] 可见 Chrome 浏览器已启动 ({chrome_bin})")
        return _search_driver

    raise RuntimeError("[Search] 未找到 Chrome 或 Edge 浏览器，请安装其中之一。")


def _close_search_driver():
    """关闭搜索浏览器。"""
    global _search_driver
    if _search_driver is not None:
        try:
            _search_driver.quit()
        except Exception:
            pass
        _search_driver = None
        print("[Search] 搜索浏览器已关闭。")


def _wait_for_captcha(driver, check_fn, timeout: Optional[int] = None) -> bool:
    """
    检测人机验证并等待用户完成。

    参数:
        driver   : Selenium WebDriver
        check_fn : 检测函数，返回 True 表示仍在验证页面
        timeout  : 最大等待时间（秒），默认使用 config.CAPTCHA_WAIT_TIMEOUT

    返回:
        True = 用户已完成验证或无需验证
        False = 超时
    """
    if timeout is None:
        timeout = config.CAPTCHA_WAIT_TIMEOUT

    if not check_fn(driver):
        return True  # 没有触发验证

    print()
    print("=" * 60)
    print("[CAPTCHA] 检测到人机验证! 请在浏览器窗口中手动完成验证。")
    print(f"    等待超时: {timeout} 秒")
    print("=" * 60)

    start = time.time()
    while check_fn(driver):
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"[Search] 人机验证等待超时 ({timeout}s)!")
            return False
        remaining = int(timeout - elapsed)
        print(f"\r[Search] 等待验证完成... 剩余 {remaining}s  ", end="", flush=True)
        time.sleep(2)

    print(f"\n[Search] 验证已通过! 继续搜索...")
    return True


# ============================================================
# Google 搜索（Selenium 可见浏览器）
# ============================================================
def _is_google_captcha(driver) -> bool:
    """检测是否在 Google 人机验证页面。"""
    try:
        url = driver.current_url.lower()
        page = driver.page_source.lower()
        if "sorry/index" in url or "/recaptcha/" in url:
            return True
        if "unusual traffic" in page or "我们的系统检测到" in page:
            return True
        if "recaptcha" in page and "search" not in url:
            return True
    except Exception:
        pass
    return False


def _google_search(query: str, num_results: int = 10) -> list:
    """通过 Google 搜索获取 URL（Selenium 可见浏览器）。"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = _get_search_driver()
    urls: list[str] = []

    try:
        from urllib.parse import quote as url_quote
        search_url = f"https://www.google.com/search?q={url_quote(query)}&num={num_results + 5}"
        driver.get(search_url)
        time.sleep(2)

        # 检测并等待人机验证
        if not _wait_for_captcha(driver, _is_google_captcha):
            return []

        # 等待搜索结果加载
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#search a[href]"))
            )
        except Exception:
            pass

        # 提取搜索结果链接
        links = driver.find_elements(By.CSS_SELECTOR, "div#search a[href]")
        for link in links:
            try:
                href = link.get_attribute("href") or ""
                if not href.startswith("http"):
                    continue
                parsed = urlparse(href)
                # 排除 Google 自身的链接
                if any(d in parsed.netloc for d in [
                    "google.com", "google.co", "googleapis.com",
                    "gstatic.com", "youtube.com", "webcache.",
                ]):
                    continue
                # 排除 Google 翻译/缓存
                if "/translate?" in href or "webcache.googleusercontent" in href:
                    continue
                if href not in urls:
                    urls.append(href)
                if len(urls) >= num_results:
                    break
            except Exception:
                continue

    except Exception as e:
        print(f"[Google] [WARN] 搜索异常: {e}")

    return urls[:num_results]


# ============================================================
# Bing 搜索（Selenium 可见浏览器）
# ============================================================
def _is_bing_captcha(driver) -> bool:
    """检测是否在 Bing 人机验证页面。"""
    try:
        url = driver.current_url.lower()
        page = driver.page_source.lower()
        if "captcha" in url or "/challenge/" in url:
            return True
        if "人机验证" in page or "verify you are human" in page:
            return True
        if "blocked" in page and "bing.com" in url:
            return True
    except Exception:
        pass
    return False


def _bing_search(query: str, num_results: int = 10) -> list:
    """通过 Bing 搜索获取 URL（Selenium 可见浏览器）。"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = _get_search_driver()
    urls: list[str] = []

    try:
        from urllib.parse import quote as url_quote
        search_url = f"https://www.bing.com/search?q={url_quote(query)}&count={num_results + 5}"
        driver.get(search_url)
        time.sleep(2)

        # 检测并等待人机验证
        if not _wait_for_captcha(driver, _is_bing_captcha):
            return []

        # 等待搜索结果加载
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#b_results li.b_algo a[href]"))
            )
        except Exception:
            pass

        # 提取搜索结果链接
        links = driver.find_elements(By.CSS_SELECTOR, "#b_results li.b_algo h2 a[href]")
        for link in links:
            try:
                href = link.get_attribute("href") or ""
                if not href.startswith("http"):
                    continue
                parsed = urlparse(href)
                if any(d in parsed.netloc for d in [
                    "bing.com", "microsoft.com", "msn.com", "live.com",
                ]):
                    continue
                if href not in urls:
                    urls.append(href)
                if len(urls) >= num_results:
                    break
            except Exception:
                continue

    except Exception as e:
        print(f"[Bing] [WARN] 搜索异常: {e}")

    return urls[:num_results]


# ============================================================
# DuckDuckGo 搜索（requests，无需浏览器）
# ============================================================
def _duckduckgo_search(query: str, num_results: int = 10) -> list:
    """通过 DuckDuckGo HTML 版搜索获取 URL（国内可直接访问）。"""
    urls: list[str] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    }

    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[DDG] [WARN] 请求失败: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    for link in soup.select("a.result__a"):
        href = str(link.get("href", ""))
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            real_urls = qs.get("uddg", [])
            if real_urls:
                real_url = unquote(real_urls[0])
                if "duckduckgo.com/y.js" in real_url:
                    continue
                if real_url not in urls:
                    urls.append(real_url)
        elif href.startswith("http"):
            if "duckduckgo.com" not in href and href not in urls:
                urls.append(href)

        if len(urls) >= num_results:
            break

    return urls[:num_results]


def _get_target_urls(category: str, num_results: Optional[int] = None) -> list:
    """获取指定类别的目标 URL（根据 config.SEARCH_ENGINE 选择引擎）。"""
    if num_results is None:
        num_results = config.NUM_REAL_PAGES

    queries = config.SEARCH_QUERIES.get(category, [])
    urls: list[str] = []

    engine = config.SEARCH_ENGINE.lower()
    engine_names = {"google": "Google", "bing": "Bing", "duckduckgo": "DuckDuckGo"}
    engine_name = engine_names.get(engine, engine)

    print(f"[Crawler] 使用 {engine_name} 搜索获取 {category} 类别 URL...")

    for query in queries:
        if len(urls) >= num_results:
            break
        try:
            if engine == "google":
                results = _google_search(query, num_results=10)
            elif engine == "bing":
                results = _bing_search(query, num_results=10)
            else:
                results = _duckduckgo_search(query, num_results=10)

            for url in results:
                if url not in urls:
                    urls.append(url)
                if len(urls) >= num_results:
                    break
            time.sleep(config.SEARCH_INTERVAL)
        except Exception as e:
            print(f"[Crawler] [WARN] 搜索 '{query}' 失败: {e}")
            continue

    if urls:
        print(f"[Crawler] 搜索成功获取 {len(urls)} 个 URL")
        for i, u in enumerate(urls[:num_results], 1):
            print(f"    [{i:2d}] {u}")
        return urls[:num_results]

    # Fallback（已禁用，仅使用搜索结果）
    # fallback = FALLBACK_URLS.get(category, [])
    # if fallback:
    #     print(f"[Crawler] 回退 Fallback ({category}): {len(fallback)} 个")
    # return fallback[:num_results]
    print(f"[Crawler] [WARN] 搜索未找到 {category} 类别的 URL")
    return []


# ============================================================
# 单页下载
# ============================================================
def download_page(url: str, output_path: str) -> bool:
    """使用 single-file-cli 下载网页（Popen + 强制超时终止）。"""
    cmd = [
        config.SINGLE_FILE_BIN,
        url,
        output_path,
        "--browser-args", '["--no-sandbox", "--disable-gpu"]',
    ]

    for attempt in range(1, config.MAX_RETRIES + 1):
        proc = None
        try:
            print(f"[Crawler] 下载中 (尝试 {attempt}/{config.MAX_RETRIES}): {url}")

            # 不捕获管道输出以避免 Windows 管道死锁
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _IS_WINDOWS else 0,
            )

            try:
                proc.wait(timeout=config.DOWNLOAD_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"[Crawler] 超时 ({config.DOWNLOAD_TIMEOUT}s)，强制终止进程...")
                _kill_proc_tree(proc)
                print(f"[Crawler]   进程已终止")
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.REQUEST_DELAY)
                continue

            if proc.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 100:
                    print(f"[Crawler] 下载成功: {os.path.basename(output_path)} ({file_size:,} bytes)")
                    return True

            print(f"[Crawler]   返回码: {proc.returncode}")

        except FileNotFoundError:
            print(f"[Crawler] [FAIL] 找不到 single-file-cli: {config.SINGLE_FILE_BIN}")
            return False
        except Exception as e:
            print(f"[Crawler] [FAIL] 异常: {e}")
            if proc is not None:
                _kill_proc_tree(proc)

        if attempt < config.MAX_RETRIES:
            time.sleep(config.REQUEST_DELAY)

    return False


def _kill_proc_tree(proc: subprocess.Popen):
    """强制终止进程及其子进程树。"""
    try:
        if _IS_WINDOWS:
            # Windows: taskkill /T 可终止整个进程树
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[attr-defined]  # Linux only
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


# ============================================================
# URL 映射表
# ============================================================
def _load_url_mapping() -> dict:
    if os.path.exists(config.URL_MAPPING_FILE):
        try:
            with open(config.URL_MAPPING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_url_mapping(mapping: dict):
    os.makedirs(os.path.dirname(config.URL_MAPPING_FILE), exist_ok=True)
    with open(config.URL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[Crawler] URL 映射已保存 ({len(mapping)} 条)")


# ============================================================
# 主调度（并发下载）
# ============================================================
def run_crawler() -> dict:
    """
    爬虫主调度：
      1. 先为所有类别收集 URL 并构建下载任务列表
      2. 跳过已存在的文件（断点续传）
      3. 使用 ThreadPoolExecutor 并发下载
      4. 完成后关闭搜索浏览器
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "=" * 60)
    print(f"Phase 0: 真实网页采集 (并发数={config.CONCURRENT_DOWNLOADS})")
    print("=" * 60)

    url_mapping = _load_url_mapping()
    stats: dict[str, dict[str, int]] = {}

    # ---------- 1. 收集所有下载任务 ----------
    tasks: list[tuple[str, str, str, str]] = []  # (category, filename, url, output_path)
    skipped = 0

    for category in config.DOMAINS:
        category_dir = os.path.join(config.RAW_HTML_DIR, category)
        os.makedirs(category_dir, exist_ok=True)

        print(f"\n--- 类别: {category} ---")
        urls = _get_target_urls(category)
        if not urls:
            stats[category] = {"success": 0, "fail": 0}
            continue

        stats[category] = {"success": 0, "fail": 0}

        for idx, url in enumerate(urls):
            filename = f"{category.lower()}_real_{idx + 1}.html"
            output_path = os.path.join(category_dir, filename)
            rel_key = f"{category}/{filename}"

            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                print(f"[Crawler] [SKIP] 已存在: {filename}")
                url_mapping[rel_key] = url
                stats[category]["success"] += 1
                skipped += 1
            else:
                tasks.append((category, filename, url, output_path))

    pending = len(tasks)
    print(f"\n[Crawler] 任务汇总: {pending} 待下载, {skipped} 已跳过")

    if not tasks:
        _save_url_mapping(url_mapping)
        return stats

    # ---------- 2. 并发下载 ----------
    completed = 0

    def _do_download(task: tuple) -> tuple:
        """线程工作函数，返回 (category, rel_key, url, success)。"""
        cat, fname, url, out_path = task
        rel_key = f"{cat}/{fname}"
        ok = download_page(url, out_path)
        return (cat, rel_key, url, ok)

    with ThreadPoolExecutor(max_workers=config.CONCURRENT_DOWNLOADS) as pool:
        futures = {pool.submit(_do_download, t): t for t in tasks}

        for future in as_completed(futures):
            completed += 1
            try:
                cat, rel_key, url, ok = future.result()
            except Exception as e:
                cat = futures[future][0]
                print(f"[Crawler] [FAIL] 线程异常: {e}")
                stats[cat]["fail"] += 1
                continue

            if ok:
                url_mapping[rel_key] = url
                stats[cat]["success"] += 1
            else:
                stats[cat]["fail"] += 1

            print(f"[Crawler] 进度: {completed}/{pending}")

    _save_url_mapping(url_mapping)

    # ---------- 3. 打印汇总 ----------
    total_success = sum(s["success"] for s in stats.values())
    total_fail = sum(s["fail"] for s in stats.values())

    print("\n" + "-" * 40)
    for cat, s in stats.items():
        print(f"  {cat:15s}: OK {s['success']:3d}  FAIL {s['fail']:3d}")
    print(f"  {'合计':15s}: OK {total_success:3d}  FAIL {total_fail:3d}")

    # ---------- 4. 关闭搜索浏览器 ----------
    _close_search_driver()

    return stats
