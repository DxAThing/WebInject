# ============================================================
# monitor_simulator.py — 核心：显示器渲染模拟器
# ============================================================
# 复现论文中"从网页源码到显示器截图"的不可微渲染过程。
# 关键步骤：
#   1. 通过 headless Edge/Chrome 加载 HTML
#   2. 使用 Selenium 截图获取渲染像素
#   3. 在 Python 端应用 ICC 色彩配置文件变换，模拟真实显示器色域
# ============================================================

import os
import io
import re
import shutil
import stat
import time
import zipfile
import urllib.request
import subprocess
import platform as _platform_mod

import numpy as np
from PIL import Image, ImageCms
from selenium import webdriver

import config

# ============================================================
# 平台检测
# ============================================================

_SYSTEM = _platform_mod.system()     # "Windows", "Darwin", "Linux"
_MACHINE = _platform_mod.machine()   # "x86_64", "AMD64", "arm64", ...


def _driver_platform_tag() -> str:
    """返回 Edge Driver 下载包的平台标识。"""
    if _SYSTEM == "Windows":
        return "win64"
    elif _SYSTEM == "Darwin":
        return "mac64_m1" if _MACHINE in ("arm64", "aarch64") else "mac64"
    else:
        return "linux64"


def _driver_exe_name() -> str:
    """返回 Edge Driver 可执行文件名。"""
    return "msedgedriver.exe" if _SYSTEM == "Windows" else "msedgedriver"


# ============================================================
# 驱动自动下载
# ============================================================

_DRIVER_DIR = os.path.join(os.path.dirname(__file__), "data", "driver")

# 镜像源列表 — {ver} = 版本号, {plat} = 平台标识
_EDGE_DRIVER_MIRRORS = [
    "https://registry.npmmirror.com/-/binary/edgedriver/{ver}/edgedriver_{plat}.zip",
    "https://cdn.npmmirror.com/binaries/edgedriver/{ver}/edgedriver_{plat}.zip",
    "https://msedgedriver.azureedge.net/{ver}/edgedriver_{plat}.zip",
]


def _get_edge_version(edge_bin: str) -> str:
    """从 Edge 可执行文件获取版本号（跨平台）。"""
    # Windows：读取文件版本信息
    if _SYSTEM == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 f'(Get-Item "{edge_bin}").VersionInfo.FileVersion'],
                capture_output=True, text=True, timeout=10,
            )
            ver = result.stdout.strip()
            if re.match(r"\d+\.\d+\.\d+\.\d+", ver):
                return ver
        except Exception:
            pass

    # macOS：读取 Info.plist
    if _SYSTEM == "Darwin":
        try:
            # Edge.app/Contents/MacOS/Microsoft Edge -> Edge.app/Contents/Info.plist
            contents_dir = os.path.dirname(os.path.dirname(edge_bin))
            plist_path = os.path.join(contents_dir, "Info.plist")
            if os.path.isfile(plist_path):
                result = subprocess.run(
                    ["/usr/libexec/PlistBuddy", "-c",
                     "Print CFBundleShortVersionString", plist_path],
                    capture_output=True, text=True, timeout=10,
                )
                ver = result.stdout.strip()
                if re.match(r"\d+\.\d+\.\d+", ver):
                    # macOS plist 可能是 3 段版本号，补 .0
                    if ver.count(".") == 2:
                        ver += ".0"
                    return ver
        except Exception:
            pass

    # 通用回退：--version
    try:
        result = subprocess.run(
            [edge_bin, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
        if m:
            return m.group(1)
    except Exception:
        pass

    raise RuntimeError(f"无法获取 Edge 版本号: {edge_bin}")


def _ensure_edge_driver(edge_bin: str) -> str:
    """
    确保本地有与 Edge 版本匹配的 msedgedriver。
    如果不存在则自动从镜像下载。返回 driver 路径。
    """
    driver_exe = _driver_exe_name()
    driver_path = os.path.join(_DRIVER_DIR, driver_exe)

    edge_ver = _get_edge_version(edge_bin)

    # 检查已有的 driver 版本是否匹配
    if os.path.isfile(driver_path):
        try:
            result = subprocess.run(
                [driver_path, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if m and m.group(1) == edge_ver:
                print(f"[Driver] msedgedriver {edge_ver} 已就绪")
                return driver_path
            else:
                print(f"[Driver] 版本不匹配 (driver={m.group(1) if m else '?'}, edge={edge_ver})，重新下载")
        except Exception:
            print("[Driver] 无法检测已有 driver 版本，重新下载")

    # 下载
    os.makedirs(_DRIVER_DIR, exist_ok=True)
    plat = _driver_platform_tag()
    print(f"[Driver] 需要下载 msedgedriver {edge_ver} ({plat}) ...")

    for mirror_tpl in _EDGE_DRIVER_MIRRORS:
        url = mirror_tpl.format(ver=edge_ver, plat=plat)
        try:
            print(f"[Driver]   尝试: {url}")
            resp = urllib.request.urlopen(url, timeout=60)
            data = resp.read()
            print(f"[Driver]   下载完成 ({len(data)} bytes)")

            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(_DRIVER_DIR)

            # 搜索解压后的可执行文件
            found_path = None
            if os.path.isfile(driver_path):
                found_path = driver_path
            else:
                for root, _, files in os.walk(_DRIVER_DIR):
                    for f in files:
                        if f.lower() == driver_exe.lower():
                            found_path = os.path.join(root, f)
                            break
                    if found_path:
                        break

            if found_path:
                # macOS / Linux: 确保可执行权限
                if _SYSTEM != "Windows":
                    os.chmod(found_path, os.stat(found_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                print(f"[Driver] msedgedriver {edge_ver} 已安装到 {found_path}")
                return found_path

            raise FileNotFoundError(f"解压后未找到 {driver_exe}")
        except Exception as e:
            print(f"[Driver]   失败: {e}")
            continue

    raise RuntimeError(
        f"无法下载 msedgedriver {edge_ver}，所有镜像均失败。"
        f"请手动下载并放到 {_DRIVER_DIR}"
    )


# ============================================================
# WebDriver 创建
# ============================================================

def _create_driver():
    """
    自动检测可用浏览器并创建 WebDriver。
    - 自动下载匹配版本的 driver
    - 支持 Windows / macOS / Linux
    - 优先级: Chrome > Edge > 报错
    """
    _HEADLESS_ARGS = (
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--force-device-scale-factor=1",
        "--hide-scrollbars",
    )

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
        for arg in _HEADLESS_ARGS:
            opts.add_argument(arg)
        driver = webdriver.Chrome(options=opts)
        print(f"[MonitorSimulator] headless Chrome 已启动 ({chrome_bin})")
        return driver

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

        # 自动确保 driver 存在
        driver_path = _ensure_edge_driver(edge_bin)

        opts = EdgeOptions()
        opts.binary_location = edge_bin
        for arg in _HEADLESS_ARGS:
            opts.add_argument(arg)

        service = EdgeService(executable_path=driver_path)
        driver = webdriver.Edge(service=service, options=opts)
        print(f"[MonitorSimulator] headless Edge 已启动 ({edge_bin})")
        return driver

    raise RuntimeError("未找到 Chrome 或 Edge 浏览器，请安装其中之一。")


class MonitorSimulator:
    """
    显示器渲染模拟器。
    利用 Selenium + ICC Profile 模拟"HTML → 真实显示器截图"的不可微渲染管线。
    """

    def __init__(self):
        """
        初始化 headless 浏览器 WebDriver（自动检测 Chrome / Edge）。
        """
        self.driver = _create_driver()

    # ----------------------------------------------------------
    # 核心渲染方法
    # ----------------------------------------------------------
    def render(self, html_path: str, monitor_config: dict) -> tuple[Image.Image, Image.Image]:
        """
        将指定 HTML 文件渲染为截图。

        参数:
            html_path      : HTML 文件的绝对路径
            monitor_config : 显示器配置字典，包含 width / height / icc_file

        返回:
            (raw_image, icc_image) 元组:
              - raw_image : sRGB 原始截图 (ICC 变换前，可作为 U-Net 输入)
              - icc_image : 经过 ICC 色彩变换后的截图
        """
        width = monitor_config["width"]
        height = monitor_config["height"]
        icc_file = monitor_config["icc_file"]

        # ---------- Step 1: 设置窗口大小并加载网页 ----------
        self.driver.set_window_size(width, height)

        # 将路径转换为 file:// URI
        abs_path = os.path.abspath(html_path)
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        self.driver.get(file_url)

        # 等待页面加载完成（document.readyState + 短暂延迟让资源渲染）
        for _ in range(30):
            ready = self.driver.execute_script("return document.readyState")
            if ready == "complete":
                break
            time.sleep(0.2)
        # 额外等待确保 CSS / 内联资源渲染完毕
        time.sleep(0.5)

        print(f"[Render] 已加载: {os.path.basename(html_path)}  窗口: {width}×{height}")

        # ---------- Step 2: 使用 CDP 全页截图获取真实渲染像素 ----------
        # CDP 方式可以得到 device pixel 级别的精确截图
        try:
            # 使用 Selenium CDP 命令获取截图（不受 viewport 限制）
            result = self.driver.execute_cdp_cmd(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            import base64
            png_data = base64.b64decode(result["data"])
            raw_image = Image.open(io.BytesIO(png_data)).convert("RGB")
            print(f"[Render]   CDP 截图成功 ({raw_image.size[0]}×{raw_image.size[1]})")
        except Exception as e:
            # 回退：使用标准 Selenium 截图
            print(f"[Render]   CDP 截图失败 ({e})，使用 Selenium 截图")
            png_data = self.driver.get_screenshot_as_png()
            raw_image = Image.open(io.BytesIO(png_data)).convert("RGB")
            print(f"[Render]   Selenium 截图成功 ({raw_image.size[0]}×{raw_image.size[1]})")

        # 如果截图尺寸与目标不匹配（因 DPI 缩放），进行 resize
        if raw_image.size != (width, height):
            raw_image = raw_image.resize((width, height), Image.LANCZOS)

        # ---------- Step 3: ICC 色彩配置文件变换 ----------
        icc_image = self._apply_icc_transform(raw_image, icc_file)

        return raw_image, icc_image

    # ----------------------------------------------------------
    # ICC 色彩变换
    # ----------------------------------------------------------
    def _apply_icc_transform(self, raw_image: Image.Image, icc_filename: str) -> Image.Image:
        """
        应用 ICC Profile 变换，将 sRGB 图像转换到目标显示器色彩空间。

        参数:
            raw_image    : 原始 sRGB 截图 (PIL Image)
            icc_filename : ICC 配置文件名（位于 config.ICC_PROFILE_DIR 下）

        返回:
            变换后的 PIL Image 或 None
        """
        icc_path = os.path.join(config.ICC_PROFILE_DIR, icc_filename)

        # 如果 ICC 文件不存在，跳过变换并给出警告
        if not os.path.exists(icc_path):
            print(f"[ICC] [WARN] ICC 文件不存在: {icc_path}，跳过色彩变换。")
            return raw_image

        try:
            # 创建源色彩配置文件（标准 sRGB）
            src_profile = ImageCms.createProfile("sRGB")

            # 加载目标显示器的 ICC 配置文件
            dst_profile = ImageCms.getOpenProfile(icc_path)

            # 构建从 sRGB → 目标色彩空间的变换对象
            transform = ImageCms.buildTransformFromOpenProfiles(
                src_profile, dst_profile, "RGB", "RGB"
            )

            # 应用变换
            final_screenshot = ImageCms.applyTransform(raw_image, transform)
            assert final_screenshot is not None, "ICC applyTransform 返回 None"
            print(f"[ICC] 色彩变换完成: sRGB → {icc_filename}")
            return final_screenshot

        except Exception as e:
            print(f"[ICC] [WARN] ICC 变换失败 ({e})，返回原始图像。")
            return raw_image

    # ----------------------------------------------------------
    # 资源清理
    # ----------------------------------------------------------
    def close(self):
        """关闭 WebDriver。"""
        if self.driver:
            self.driver.quit()
            print("[MonitorSimulator] WebDriver 已关闭。")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
