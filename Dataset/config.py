# ============================================================
# config.py — 数据集制备流水线的统一配置中心
# ============================================================
# 弃用 U-Net 和 ICC 色彩变换，假设浏览器原始像素即为用户看到的。
# 弃用 OpenAI API，使用本地 7B 级别大模型（Qwen2.5-7B-Instruct）。
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
# 每个阶段独立开关，设为 False 则跳过该阶段。
# 已完成的阶段即使为 True 也会被断点续传跳过。
#
# 工作流程（本地 + 云端分离）：
#   第 1 步：本地运行 python main.py
#            → Phase 0 采集真实网页，Phase 3 生成历史
#            → Phase 1/2 自动跳过（云端完成）
#   第 2 步：将 data/raw_html/ 上传到云端 deploy/webinject_prompt/data/
#   第 3 步：云端运行 bash start.sh（Phase 1 合成 + Phase 2 指令生成）
#   第 4 步：从云端下载 raw_html/ (含合成网页) + prompts.json 回本地 data/
#   第 5 步：删除 data/pipeline_state.json，将 RUN_RENDER 和 RUN_METADATA
#            改为 True，再次运行 python main.py

RUN_CRAWLER    = True    # Phase 0: 真实网页采集（本地，需要 single-file-cli + 网络）
RUN_SYNTH_GEN  = False   # Phase 1: ← 云端完成（Qwen3-7B-Instruct）
RUN_PROMPT_GEN = False   # Phase 2: ← 云端完成（Qwen3-7B-Instruct）
RUN_HISTORY    = True    # Phase 3: 历史生成（本地，随机采样，无需大模型）
RUN_RENDER     = True    # Phase 4: 截图渲染（本地）
RUN_METADATA   = True    # Phase 5: 元数据汇总（本地）

# ======================= 生成参数 ==========================

NUM_REAL_PAGES = 50          # 每类采集的真实网页数
NUM_SYNTH_PAGES = 10         # 每类生成的合成网页数
NUM_TARGET_PROMPTS = 10      # 每个网页生成的 Target Prompt 数
NUM_SHADOW_HISTORY = 10      # 每个网页的 Shadow History 数
NUM_USER_HISTORY = 10        # 每个网页的 User History 数
HISTORY_MIN_STEPS = 3        # 每条历史的最小动作步数
HISTORY_MAX_STEPS = 5        # 每条历史的最大动作步数

# ======================= 本地大模型配置 ====================
# 使用本地 Qwen2.5-7B-Instruct 替代 OpenAI API
# 必须使用 bfloat16 精度 + flash_attention_2 以优化显存和推理速度

LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # HuggingFace 模型标识
LOCAL_MODEL_DTYPE = "bfloat16"                   # 推理精度：bfloat16
LOCAL_MODEL_ATTN = "flash_attention_2"           # 注意力实现：FlashAttention-2
LOCAL_MODEL_MAX_NEW_TOKENS = 4096                # 最大生成 token 数
LOCAL_MODEL_TEMPERATURE = 0.7                    # 采样温度
LOCAL_MODEL_TOP_P = 0.9                          # nucleus 采样概率阈值
LOCAL_MODEL_DEVICE = "auto"                      # 设备映射: "auto" 让 accelerate 自动分配

# ======================= HTML 压缩配置 ====================
# DOM 压缩器用于精简 HTML，防止长文档撑爆大模型显存

HTML_COMPRESS_MAX_CHARS = 6000       # 压缩后 HTML 的最大字符数
HTML_COMPRESS_KEEP_TAGS = [          # 保留的核心交互标签
    "a", "button", "input", "select", "textarea", "form",
    "nav", "header", "footer", "main", "article", "section",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "span", "div", "ul", "ol", "li",
    "img", "video", "audio",
    "table", "tr", "td", "th",
    "label", "option",
]
HTML_COMPRESS_REMOVE_TAGS = [        # 强制移除的标签（脚本 / 样式 / 注释）
    "script", "style", "noscript", "svg", "path",
    "meta", "link",
]
HTML_COMPRESS_MAX_TEXT_LEN = 80      # 单个文本节点的最大保留长度
HTML_COMPRESS_MAX_ATTR_LEN = 60      # 单个属性值的最大保留长度
HTML_COMPRESS_KEEP_ATTRS = [         # 始终保留的属性名
    "id", "class", "href", "src", "alt", "title", "name",
    "type", "value", "placeholder", "action", "method",
    "role", "aria-label", "data-testid",
]

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
        "Wired technology articles",
        "Engadget tech news blog",
        "Gizmodo gadgets technology",
        "9to5Mac Apple news blog",
        "Android Authority mobile blog",
        "Mashable tech culture blog",
        "ZDNet enterprise technology",
        "TechRadar reviews blog",
        "Lifehacker productivity tips blog",
        "Daring Fireball Apple blog",
        "Krebs on Security cybersecurity blog",
        "Wait But Why long form blog",
        "Brain Pickings culture blog",
        "Coding Horror programming blog",
        "Martin Fowler software blog",
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
        "Amazon online shopping",
        "eBay auction marketplace",
        "Etsy handmade marketplace",
        "Walmart online grocery",
        "Target department store online",
        "Best Buy electronics store",
        "Wayfair furniture home decor",
        "Nordstrom fashion clothing",
        "Costco wholesale shopping",
        "Home Depot tools hardware",
        "Lowes home improvement store",
        "Shopify example store",
        "Apple Store online",
        "Samsung electronics shop",
        "Zappos shoes online",
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
        "Skillshare creative classes",
        "Pluralsight technology courses",
        "LinkedIn Learning professional",
        "FutureLearn university courses",
        "OpenLearn free courses",
        "Treehouse web development courses",
        "DataCamp data science learning",
        "W3Schools web tutorials",
        "MDN Web Docs documentation",
        "GeeksforGeeks programming tutorials",
        "LeetCode coding practice",
        "HackerRank coding challenges",
        "Kaggle data science competitions",
        "Class Central course aggregator",
        "Swayam India online courses",
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
        "Medical News Today health articles",
        "Verywell Health wellness guide",
        "Health.com fitness nutrition",
        "Everyday Health medical information",
        "Kaiser Permanente health care",
        "Mount Sinai patient care",
        "NHS health conditions treatments",
        "American Heart Association heart health",
        "American Cancer Society cancer info",
        "Diabetes.org diabetes management",
        "Mental Health America resources",
        "Psychology Today mental health",
        "NAMI mental illness support",
        "Planned Parenthood health info",
        "GoodRx prescription drug prices",
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
        "Adham Dannaway UI designer portfolio",
        "Jack Jeznach creative portfolio",
        "Dejan Markovic UX portfolio",
        "Matthew Williams designer portfolio",
        "Bruno Simon creative developer",
        "Lynn Fisher web developer portfolio",
        "Sarah Drasner developer portfolio",
        "Jhey Tompkins creative developer",
        "Tobias van Schneider designer",
        "Olaolu Olawuyi developer portfolio",
        "best developer portfolio websites 2024",
        "creative web design portfolio examples",
        "freelance designer portfolio website",
        "UI UX designer portfolio showcase",
        "frontend developer personal website examples",
    ],
}

# ======================= 截图分辨率 ====================
# 假定浏览器原始像素即为用户看到的，不再区分显示器型号

SCREENSHOT_WIDTH = 1920
SCREENSHOT_HEIGHT = 1080

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

# ======================= 爬虫配置 ==========================

# 搜索引擎: "google", "bing", "duckduckgo"
SEARCH_ENGINE = "google"

# 人机验证等待超时（秒）
CAPTCHA_WAIT_TIMEOUT = 300

# 搜索间隔（秒）
SEARCH_INTERVAL = 3

DOWNLOAD_TIMEOUT = 90
MAX_RETRIES = 2
REQUEST_DELAY = 1
CONCURRENT_DOWNLOADS = 5


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
