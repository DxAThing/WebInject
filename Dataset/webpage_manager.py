# ============================================================
# webpage_manager.py — 网页管理：合成生成 + 文件工具
# ============================================================
# 功能：
#   1. 调用 OpenAI GPT-4 / Mock 生成合成 HTML 页面
#   2. 提供 list_html_files / load_html 等文件级工具函数
#   3. 支持断点续传（已存在的文件自动跳过）
# ============================================================

import os
import glob
from typing import Optional

import config

# ---------------------- 论文原文 Prompt ----------------------
SYNTHETIC_PROMPT = (
    "Generate a highly realistic HTML page for a {category} website. "
    "Include detailed and modern HTML and CSS directly in the file, "
    "using advanced layouts (e.g., grid, flexbox) and professional-level styling. "
    "Add responsive design elements to make the page look polished on both "
    "desktop and mobile devices. The page should be unique and specific to the "
    "category, with placeholder images and realistic content. "
    "Only include the HTML and CSS content, without any additional text, "
    "explanations, or surrounding code blocks like '```html'."
)


# ============================================================
# OpenAI API 调用
# ============================================================
def _call_openai(prompt: str) -> str:
    """调用 OpenAI Chat Completion API 生成 HTML。"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content
        content = raw.strip() if raw else ""

        if content.startswith("```html"):
            content = content[len("```html"):].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        return content

    except ImportError:
        print("[SyntheticGen] [WARN] openai 库未安装，请运行: pip install openai")
        return ""
    except Exception as e:
        print(f"[SyntheticGen] [FAIL] OpenAI API 调用失败: {e}")
        return ""


# ============================================================
# Mock HTML 生成（含 CSS Grid / Flexbox）
# ============================================================
def _generate_mock_html(category: str, index: int) -> str:
    """生成一个含 CSS Grid 布局的 Mock HTML 页面。"""
    color_themes = {
        "Blog": ("#1a1a2e", "#16213e", "#0f3460", "#e94560"),
        "Commerce": ("#2d3436", "#636e72", "#00b894", "#fdcb6e"),
        "Education": ("#2c3e50", "#3498db", "#2ecc71", "#e74c3c"),
        "Healthcare": ("#00b4d8", "#0077b6", "#90e0ef", "#caf0f8"),
        "Portfolio": ("#0d1b2a", "#1b263b", "#415a77", "#e0e1dd"),
    }

    bg, secondary, accent, highlight = color_themes.get(
        category, ("#333", "#555", "#007bff", "#ffc107")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mock {category} Page {index + 1}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        header {{
            background: {bg};
            color: #fff;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        header h1 {{ font-size: 1.4rem; letter-spacing: 1px; }}
        nav {{ display: flex; gap: 1.5rem; }}
        nav a {{
            color: rgba(255,255,255,0.85);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }}
        nav a:hover {{ color: {highlight}; }}
        .hero {{
            background: linear-gradient(135deg, {bg}, {secondary});
            color: #fff;
            text-align: center;
            padding: 5rem 2rem;
        }}
        .hero h2 {{ font-size: 2.5rem; margin-bottom: 1rem; }}
        .hero p {{ font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem; opacity: 0.9; }}
        .hero .btn {{
            display: inline-block;
            padding: 0.8rem 2rem;
            background: {highlight};
            color: {bg};
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
        }}
        .hero .btn:hover {{ transform: translateY(-2px); }}
        .grid-section {{
            max-width: 1200px;
            margin: 3rem auto;
            padding: 0 2rem;
        }}
        .grid-section h3 {{
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            color: {bg};
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }}
        .card-img {{
            width: 100%;
            height: 180px;
            background: linear-gradient(45deg, {accent}, {secondary});
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.2rem;
            font-weight: 600;
        }}
        .card-body {{ padding: 1.2rem; }}
        .card-body h4 {{ margin-bottom: 0.5rem; color: {bg}; }}
        .card-body p {{ font-size: 0.9rem; color: #666; }}
        .features {{
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
            max-width: 1200px;
            margin: 3rem auto;
            padding: 0 2rem;
        }}
        .feature {{
            flex: 1 1 250px;
            text-align: center;
            padding: 2rem;
        }}
        .feature .icon {{
            width: 60px; height: 60px;
            background: {accent};
            border-radius: 50%;
            margin: 0 auto 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.5rem;
        }}
        footer {{
            background: {bg};
            color: rgba(255,255,255,0.7);
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
        }}
        @media (max-width: 768px) {{
            header {{ flex-direction: column; gap: 0.5rem; }}
            .hero h2 {{ font-size: 1.8rem; }}
            .features {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>{category} Hub</h1>
        <nav>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Services</a>
            <a href="#">Blog</a>
            <a href="#">Contact</a>
        </nav>
    </header>
    <section class="hero">
        <h2>Welcome to {category} Page {index + 1}</h2>
        <p>This is a synthetically generated mock page for the WebInject dataset.
           Built with modern CSS Grid and Flexbox layouts for a professional look.</p>
        <a href="#" class="btn">Explore Now</a>
    </section>
    <section class="grid-section">
        <h3>Featured Content</h3>
        <div class="card-grid">
            <div class="card">
                <div class="card-img">Image Placeholder 1</div>
                <div class="card-body">
                    <h4>Article Title One</h4>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                       Vivamus lacinia odio vitae vestibulum vestibulum.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-img">Image Placeholder 2</div>
                <div class="card-body">
                    <h4>Article Title Two</h4>
                    <p>Cras pulvinar mattis nunc sed blandit. Pellentesque
                       habitant morbi tristique senectus et netus.</p>
                </div>
            </div>
            <div class="card">
                <div class="card-img">Image Placeholder 3</div>
                <div class="card-body">
                    <h4>Article Title Three</h4>
                    <p>Praesent commodo cursus magna, vel scelerisque nisl
                       consectetur et. Donec sed odio dui.</p>
                </div>
            </div>
        </div>
    </section>
    <section class="features">
        <div class="feature">
            <div class="icon">⚡</div>
            <h4>Fast Performance</h4>
            <p>Optimized for speed and reliability across all devices.</p>
        </div>
        <div class="feature">
            <div class="icon">🎨</div>
            <h4>Modern Design</h4>
            <p>Clean, professional layouts using the latest CSS techniques.</p>
        </div>
        <div class="feature">
            <div class="icon">📱</div>
            <h4>Responsive</h4>
            <p>Looks great on desktops, tablets, and mobile devices.</p>
        </div>
    </section>
    <footer>
        <p>&copy; 2026 {category} Mock Page {index + 1} — WebInject Dataset Pipeline</p>
    </footer>
</body>
</html>"""


# ============================================================
# 单页面生成
# ============================================================
def generate_one(category: str, index: int, use_mock: Optional[bool] = None) -> str:
    """
    生成一个合成 HTML 页面并保存到磁盘。
    如果目标文件已存在则跳过（断点续传）。

    返回:
        保存后的文件绝对路径
    """
    if use_mock is None:
        use_mock = config.USE_MOCK

    category_dir = os.path.join(config.RAW_HTML_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    filename = f"{category.lower()}_synth_{index + 1}.html"
    filepath = os.path.join(category_dir, filename)

    # 断点续传：已存在的文件跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        print(f"[SyntheticGen] [SKIP] 已存在: {filename}")
        return filepath

    if use_mock:
        html_content = _generate_mock_html(category, index)
        print(f"[SyntheticGen] Mock 生成: {category} #{index + 1}")
    else:
        prompt = SYNTHETIC_PROMPT.format(category=category)
        html_content = _call_openai(prompt)
        if not html_content:
            print(f"[SyntheticGen] API 返回空，回退 Mock: {category} #{index + 1}")
            html_content = _generate_mock_html(category, index)
        else:
            print(f"[SyntheticGen] API 生成成功: {category} #{index + 1}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath


# ============================================================
# 批量生成
# ============================================================
def generate_all() -> list:
    """
    批量生成所有类别的合成网页。
    返回所有生成/已存在文件的路径列表。
    """
    print("\n" + "=" * 60)
    print("Phase 1: 合成网页生成 (Synthetic Webpage Generation)")
    print("=" * 60)

    all_paths: list[str] = []
    stats: dict[str, int] = {}

    for category in config.DOMAINS:
        print(f"\n--- 类别: {category} ---")
        cat_paths: list[str] = []

        for idx in range(config.NUM_SYNTH_PAGES):
            path = generate_one(category, idx)
            cat_paths.append(path)

        stats[category] = len(cat_paths)
        all_paths.extend(cat_paths)

    print("\n" + "=" * 60)
    print("合成生成汇总")
    print("=" * 60)
    for cat, count in stats.items():
        print(f"  {cat:15s}: {count:3d} 个页面")
    print(f"  {'合计':15s}: {len(all_paths):3d} 个页面")
    print("=" * 60)

    return all_paths


# ============================================================
# 文件工具函数
# ============================================================
def list_html_files() -> list:
    """
    列出 RAW_HTML_DIR 下所有 .html 文件路径。
    按 (类别, 文件名) 排序。
    """
    html_files: list[str] = []
    for category in config.DOMAINS:
        cat_dir = os.path.join(config.RAW_HTML_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for f in sorted(glob.glob(os.path.join(cat_dir, "*.html"))):
            html_files.append(f)
    return html_files


def load_html(html_path: str) -> str:
    """读取一个 HTML 文件的内容。"""
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()
