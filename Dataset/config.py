# ============================================================
# config.py — 数据集制备流水线的统一配置中心
# ============================================================
# 整合了网页采集 (crawler) + 合成生成 + Prompt/History + 渲染模拟
# 所有参数在此定义，严禁使用 argparse。
#
# 爬虫依赖 single-file-cli (Node.js):
#   npm install -g single-file-cli
# ============================================================

import os
import sys
import shutil
import platform

# ======================= 路径配置 ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_HTML_DIR = os.path.join(BASE_DIR, "data", "raw_html")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "data", "screenshots")
RAW_SCREENSHOTS_DIR = os.path.join(BASE_DIR, "data", "screenshots_raw")  # ICC 变换前的原始截图 (U-Net 输入)
ICC_PROFILE_DIR = os.path.join(BASE_DIR, "data", "icc_profiles")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "dataset_metadata.json")

# 阶段产出文件（与 pipeline_state 解耦）
PROMPTS_JSON = os.path.join(BASE_DIR, "data", "prompts.json")
HISTORIES_JSON = os.path.join(BASE_DIR, "data", "histories.json")

# 流水线状态文件（断点续传）
PIPELINE_STATE_FILE = os.path.join(BASE_DIR, "data", "pipeline_state.json")

# URL 映射文件
URL_MAPPING_FILE = os.path.join(RAW_HTML_DIR, "url_mapping.json")

# ======================= 网页分类域 ========================

DOMAINS = ["Blog", "Commerce", "Education", "Healthcare", "Portfolio"]

# ======================= 运行模式开关 ======================

# 是否运行真实网页爬虫（需要 single-file-cli）
RUN_CRAWLER = True

# 是否运行合成网页生成
RUN_SYNTH_GEN = False

# ======================= 生成参数 ==========================

NUM_REAL_PAGES = 10          # 每类采集的真实网页数
NUM_SYNTH_PAGES = 10         # 每类生成的合成网页数
NUM_SHADOW_HISTORY = 10      # 每个网页的 Shadow History 数
NUM_USER_HISTORY = 10        # 每个网页的 User History 数

# ======================= 搜索关键词 ========================

SEARCH_QUERIES = {
    "Blog": [
        "TechCrunch latest news",
        "The Verge technology blog",
        "Ars Technica articles",
        "Smashing Magazine web development",
        "A List Apart web design articles",
        "CSS-Tricks frontend blog",
        "dev.to programming community",
        "Hacker Noon tech stories",
        "freeCodeCamp blog tutorials",
        "LogRocket blog frontend",
    ],
    "Commerce": [
        "Nike official store",
        "Adidas online shop",
        "IKEA furniture store",
        "Uniqlo clothing online",
        "Sephora beauty products",
        "B&H Photo Video store",
        "REI outdoor gear shop",
        "Zara fashion online store",
        "ASOS clothing shop",
        "Patagonia outdoor clothing",
    ],
    "Education": [
        "MIT OpenCourseWare free courses",
        "Khan Academy learn online",
        "Coursera online classes",
        "edX university courses",
        "Stanford Online learning",
        "Harvard Online courses",
        "Codecademy learn programming",
        "Duolingo language learning",
        "Brilliant math science courses",
        "Udemy online tutorials",
    ],
    "Healthcare": [
        "Mayo Clinic health information",
        "WebMD symptoms diseases",
        "Cleveland Clinic medical care",
        "Johns Hopkins Medicine health",
        "Healthline medical articles",
        "MedlinePlus health topics",
        "CDC disease prevention",
        "WHO world health organization",
        "Drugs.com medication information",
        "NIH National Institutes of Health",
    ],
    "Portfolio": [
        "Brittany Chiang developer portfolio",
        "Tania Rascia personal website",
        "Josh W Comeau blog portfolio",
        "Sara Soueidan web developer",
        "Wes Bos developer courses",
        "Kent C Dodds personal site",
        "Cassidy Williams developer",
        "Robin Wieruch developer blog",
        "Dan Abramov overreacted blog",
        "Lee Robinson developer portfolio",
    ],
}

# ======================= 目标显示器规格 ====================

MONITORS = {
    "iMac_M1_24": {
        "width": 4480,
        "height": 2520,
        "icc_file": "Display P3.icc",
    },
    "Dell_S2722QC": {
        "width": 3840,
        "height": 2160,
        "icc_file": "sRGB_v4_ICC_preference.icc",
    },
}

# ======================= 动作空间 ==========================

ACTION_SPACE = [
    "click((x,y))",
    "left_double((x,y))",
    "right_single((x,y))",
    "drag((x1,y1),(x2,y2))",
    "hotkey(key_comb)",
    "type(content)",
    "scroll(direction)",
    "wait()",
    "finished()",
    "call_user()",
]

# ======================= API 配置 ==========================

USE_MOCK = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4"

# ======================= 爬虫配置 ==========================

# 搜索引擎: "google", "bing", "duckduckgo"
# Google/Bing 使用 Selenium 可见浏览器，遇到人机验证时会暂停让用户手动完成
# DuckDuckGo 使用 requests（无需浏览器，但结果可能较少）
SEARCH_ENGINE = "google"

# 人机验证等待超时（秒）— 用户需在此时间内完成验证
CAPTCHA_WAIT_TIMEOUT = 300

# 搜索间隔（秒）— 每次搜索之间的等待，避免触发频率限制
SEARCH_INTERVAL = 3

DOWNLOAD_TIMEOUT = 90
MAX_RETRIES = 2
REQUEST_DELAY = 1
CONCURRENT_DOWNLOADS = 5      # 同时下载的最大线程数


# ======================= single-file-cli 检测 ==============

def _detect_single_file_bin() -> str:
    """自动检测 single-file-cli 可执行文件路径。"""
    if platform.system() == "Windows":
        candidates = ["single-file.cmd", "single-file.exe", "single-file"]
    else:
        candidates = ["single-file"]

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    env_dir = os.path.dirname(sys.executable)
    env_root = (
        os.path.dirname(env_dir)
        if os.path.basename(env_dir).lower() == "scripts"
        else env_dir
    )
    for search_dir in [env_dir, env_root]:
        for name in candidates:
            candidate_path = os.path.join(search_dir, name)
            if os.path.isfile(candidate_path):
                return candidate_path

    return "single-file.cmd" if platform.system() == "Windows" else "single-file"


SINGLE_FILE_BIN = _detect_single_file_bin()
