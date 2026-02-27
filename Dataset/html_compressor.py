# ============================================================
# html_compressor.py — DOM 压缩器
# ============================================================
# 利用 BeautifulSoup 提取网页核心交互节点，精简 HTML 后再喂给大模型。
# 目的：防止长 HTML 撑爆大模型显存。
#
# 压缩策略：
#   1. 移除所有 <script>、<style>、<noscript>、<svg> 等非内容标签
#   2. 只保留与用户交互相关的核心标签（a, button, input, form 等）
#   3. 截断过长的文本节点和属性值
#   4. 移除无关属性，只保留 id/class/href/src 等语义属性
#   5. 最终输出字符数上限硬性截断
# ============================================================

import re
from typing import Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

import config


def compress_html(html_content: str, max_chars: Optional[int] = None) -> str:
    """
    将原始 HTML 压缩为仅包含核心交互节点的精简版本。

    参数:
        html_content : 原始 HTML 字符串
        max_chars    : 最终输出的最大字符数（默认读取 config.HTML_COMPRESS_MAX_CHARS）

    返回:
        压缩后的 HTML 字符串

    压缩流程:
        ① 解析 HTML 为 DOM 树
        ② 移除强制删除标签（script/style/svg 等）
        ③ 移除所有 HTML 注释
        ④ 清理无关属性，只保留语义属性
        ⑤ 截断过长的文本节点和属性值
        ⑥ 递归移除空的非交互容器标签
        ⑦ 硬性截断到 max_chars
    """
    if max_chars is None:
        max_chars = config.HTML_COMPRESS_MAX_CHARS

    # ----------------------------------------------------------------
    # 第一步：解析 HTML
    # ----------------------------------------------------------------
    soup = BeautifulSoup(html_content, "html.parser")

    # ----------------------------------------------------------------
    # 第二步：移除强制删除标签（script / style / svg 等）
    # 这些标签包含大量无用内容，是显存的主要"杀手"
    # ----------------------------------------------------------------
    for tag_name in config.HTML_COMPRESS_REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()  # 彻底从 DOM 树中删除

    # ----------------------------------------------------------------
    # 第三步：移除所有 HTML 注释节点
    # 注释可能非常长（模板引擎生成的调试信息等）
    # ----------------------------------------------------------------
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # ----------------------------------------------------------------
    # 第四步：清理属性 — 只保留语义相关属性
    # 大量的 data-*、style、on* 事件属性对大模型理解页面结构无帮助
    # ----------------------------------------------------------------
    keep_attrs = set(config.HTML_COMPRESS_KEEP_ATTRS)
    max_attr_len = config.HTML_COMPRESS_MAX_ATTR_LEN

    for tag in soup.find_all(True):  # 遍历所有标签
        if not isinstance(tag, Tag):
            continue

        # 收集需要删除的属性名
        attrs_to_remove = []
        for attr_name in list(tag.attrs.keys()):
            # 保留白名单中的属性
            if attr_name in keep_attrs:
                # 截断过长的属性值（如超长的 class 列表）
                val = tag.attrs[attr_name]
                if isinstance(val, str) and len(val) > max_attr_len:
                    tag.attrs[attr_name] = val[:max_attr_len] + "..."
                elif isinstance(val, list):
                    # class 属性返回 list，合并后截断
                    joined = " ".join(val)
                    if len(joined) > max_attr_len:
                        tag.attrs[attr_name] = joined[:max_attr_len] + "..."
            else:
                attrs_to_remove.append(attr_name)

        for attr_name in attrs_to_remove:
            del tag.attrs[attr_name]

    # ----------------------------------------------------------------
    # 第五步：截断过长的文本节点
    # 某些页面内联了大量文本（文章正文、JSON-LD 等），需要截断
    # ----------------------------------------------------------------
    max_text_len = config.HTML_COMPRESS_MAX_TEXT_LEN

    for text_node in soup.find_all(string=True):
        if isinstance(text_node, Comment):
            continue
        if not isinstance(text_node, NavigableString):
            continue

        # 清理多余空白
        cleaned = re.sub(r"\s+", " ", str(text_node)).strip()

        if len(cleaned) > max_text_len:
            cleaned = cleaned[:max_text_len] + "..."

        if cleaned != str(text_node):
            text_node.replace_with(cleaned)

    # ----------------------------------------------------------------
    # 第六步：递归移除空的非交互容器标签
    # 清理后很多 <div><span> 变成了空标签，可以安全移除以减小体积
    # ----------------------------------------------------------------
    _remove_empty_tags(soup)

    # ----------------------------------------------------------------
    # 第七步：输出并硬性截断
    # ----------------------------------------------------------------
    compressed = soup.prettify()

    # 压缩连续空行为单个换行
    compressed = re.sub(r"\n{3,}", "\n\n", compressed)

    # 硬性截断到 max_chars
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars] + "\n<!-- [HTML 已截断] -->"

    return compressed


def _remove_empty_tags(soup: BeautifulSoup, max_passes: int = 3):
    """
    递归移除空的非交互容器标签。
    需要多轮扫描，因为移除内层空标签后，外层标签也可能变空。

    参数:
        soup       : BeautifulSoup 对象（原地修改）
        max_passes : 最大扫描轮数，防止无限循环
    """
    # 核心交互标签不应被移除，即使它们暂时为空
    interactive_tags = {
        "a", "button", "input", "select", "textarea", "form",
        "img", "video", "audio", "iframe",
        "nav", "header", "footer", "main",
    }

    for _ in range(max_passes):
        removed = False
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            # 跳过交互标签
            if tag.name in interactive_tags:
                continue
            # 如果标签内没有任何文本内容且没有子标签，移除
            text = tag.get_text(strip=True)
            if not text and not tag.find_all(True):
                tag.decompose()
                removed = True

        # 如果本轮没有移除任何标签，提前退出
        if not removed:
            break


def get_page_summary(html_content: str) -> str:
    """
    提取页面的简要结构摘要，用于日志和调试。

    参数:
        html_content : 原始 HTML 字符串

    返回:
        页面摘要字符串（标题、链接数、表单数等）
    """
    soup = BeautifulSoup(html_content, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "(无标题)"

    num_links = len(soup.find_all("a"))
    num_buttons = len(soup.find_all("button"))
    num_inputs = len(soup.find_all("input"))
    num_forms = len(soup.find_all("form"))
    num_images = len(soup.find_all("img"))

    return (
        f"标题: {title} | "
        f"链接: {num_links} | 按钮: {num_buttons} | "
        f"输入框: {num_inputs} | 表单: {num_forms} | 图片: {num_images}"
    )
