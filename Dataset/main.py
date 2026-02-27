# ============================================================
# main.py — 数据集制备流水线主入口
# ============================================================
# 6 阶段流水线，支持断点续传：
#   Phase 0 : 真实网页采集 (Crawler)      — 可选
#   Phase 1 : 合成网页生成 (SyntheticGen)  — 可选（使用本地大模型）
#   Phase 2 : Target / User Prompt 生成   — 使用本地大模型 + DOM 压缩
#   Phase 3 : Shadow / User History 生成  — 随机采样
#   Phase 4 : 显示器渲染截图             — Selenium headless
#   Phase 5 : 元数据 JSON 汇总
#
# 核心设计：
#   - 弃用 U-Net 和 ICC 色彩变换，假设原始像素即为用户所见
#   - 使用本地 Qwen2.5-7B-Instruct 替代 OpenAI API
#   - HTML 压缩（html_compressor）防止长文档撑爆大模型显存
#
# 运行方式:
#   cd Dataset && python main.py
# ============================================================

import json
import os
import sys
import time

import config
import logger
import pipeline_state
import webpage_manager


# ============================================================
# Phase 0 — 真实网页采集
# ============================================================
def phase0_crawl(state: dict):
    """采集真实网页（需要 single-file-cli + 网络）。"""
    phase = "phase0_crawl"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_CRAWLER:
        print(f"\n[SKIP] {phase} 未启用 (RUN_CRAWLER=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    import crawler
    stats = crawler.run_crawler()
    pipeline_state.mark_completed(state, phase, stats)


# ============================================================
# Phase 1 — 合成网页生成（使用本地大模型）
# ============================================================
def phase1_synth(state: dict):
    """使用本地大模型生成合成 HTML 页面。"""
    phase = "phase1_synth"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_SYNTH_GEN:
        print(f"\n[SKIP] {phase} 未启用 (RUN_SYNTH_GEN=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    paths = webpage_manager.generate_all()
    pipeline_state.mark_completed(state, phase, {"count": len(paths)})


# ============================================================
# Phase 2 — Prompt 生成（本地大模型 + DOM 压缩）
# ============================================================
def phase2_prompts(state: dict):
    """
    为每个 HTML 页面生成 Target Prompts + User Prompts。

    流程：
      1. 遍历所有 HTML 文件
      2. 对每个 HTML 使用 html_compressor 压缩后喂给本地大模型
      3. 大模型输出 Target Prompts（动作导向任务列表）
      4. 再逐条改写为 User Prompts（语义等价版本）
      5. 结果持久化到 prompts.json

    断点续传：整个阶段完成后才标记，未完成则下次重跑。
    """
    phase = "phase2_prompts"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_PROMPT_GEN:
        print(f"\n[SKIP] {phase} 未启用 (RUN_PROMPT_GEN=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    import prompt_generator

    print("\n" + "=" * 60)
    print("Phase 2: Prompt 生成（本地大模型 + DOM 压缩）")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()
    if not html_files:
        print("[Phase2] [WARN] 未找到 HTML 文件，跳过。")
        pipeline_state.mark_completed(state, phase, {"count": 0})
        return

    # ----------------------------------------------------------------
    # 加载已有的 prompts（实现页面级断点续传）
    # 如果之前已经为某些页面生成了 prompts，则跳过这些页面
    # ----------------------------------------------------------------
    all_prompts: dict[str, dict] = {}
    if os.path.exists(config.PROMPTS_JSON):
        try:
            with open(config.PROMPTS_JSON, "r", encoding="utf-8") as f:
                all_prompts = json.load(f)
            print(f"[Phase2] 已加载 {len(all_prompts)} 个页面的已有 Prompt 数据")
        except (json.JSONDecodeError, IOError):
            all_prompts = {}

    for idx, html_path in enumerate(html_files):
        rel_name = os.path.relpath(html_path, config.RAW_HTML_DIR)
        rel_key = rel_name.replace(os.sep, "/")

        # 页面级断点续传：如果该页面已有完整的 prompt 数据则跳过
        if rel_key in all_prompts and all_prompts[rel_key].get("target_prompts"):
            print(f"[Phase2] [SKIP] 已存在: {rel_key}")
            continue

        print(f"\n[Phase2] [{idx + 1}/{len(html_files)}] 处理: {rel_key}")
        target_prompts = prompt_generator.generate_target_prompts(html_path)
        user_prompts = prompt_generator.generate_user_prompts(target_prompts)
        all_prompts[rel_key] = {
            "target_prompts": target_prompts,
            "user_prompts": user_prompts,
        }

        # 每处理一个页面就保存一次，防止中断丢失进度
        os.makedirs(os.path.dirname(config.PROMPTS_JSON), exist_ok=True)
        with open(config.PROMPTS_JSON, "w", encoding="utf-8") as f:
            json.dump(all_prompts, f, ensure_ascii=False, indent=2)

    print(f"\n[Phase2] Prompt 数据已保存: {config.PROMPTS_JSON}")
    print(f"[Phase2] 共为 {len(all_prompts)} 个页面生成 Prompt。")
    pipeline_state.mark_completed(state, phase, {"count": len(all_prompts)})


# ============================================================
# Phase 3 — History 生成
# ============================================================
def phase3_history(state: dict):
    """为数据集生成 Shadow History 和 User History。"""
    phase = "phase3_history"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_HISTORY:
        print(f"\n[SKIP] {phase} 未启用 (RUN_HISTORY=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    import history_generator

    print("\n" + "=" * 60)
    print("Phase 3: 动作历史生成")
    print("=" * 60)

    shadow = history_generator.generate_shadow_histories()
    user = history_generator.generate_user_histories()

    histories = {
        "shadow_histories": shadow,
        "user_histories": user,
    }
    os.makedirs(os.path.dirname(config.HISTORIES_JSON), exist_ok=True)
    with open(config.HISTORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)
    print(f"[Phase3] History 数据已保存: {config.HISTORIES_JSON}")

    pipeline_state.mark_completed(state, phase, {
        "shadow_count": len(shadow),
        "user_count": len(user),
    })


# ============================================================
# Phase 4 — 渲染截图（浏览器原始像素即为最终截图）
# ============================================================
def phase4_render(state: dict):
    """
    使用 headless 浏览器将每个 HTML 页面渲染为截图。

    假设浏览器原始像素即为用户看到的，每页一图。
    """
    phase = "phase4_render"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_RENDER:
        print(f"\n[SKIP] {phase} 未启用 (RUN_RENDER=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    import monitor_simulator

    print("\n" + "=" * 60)
    print("Phase 4: 网页截图渲染")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()
    if not html_files:
        print("[Phase4] [WARN] 未找到 HTML 文件，跳过。")
        pipeline_state.mark_completed(state, phase, {"count": 0})
        return

    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
    screenshot_count = 0
    render_cfg = {"width": config.SCREENSHOT_WIDTH, "height": config.SCREENSHOT_HEIGHT}

    try:
        sim = monitor_simulator.MonitorSimulator()

        for idx, html_path in enumerate(html_files):
            basename = os.path.splitext(os.path.basename(html_path))[0]
            png_name = f"{basename}.png"
            png_path = os.path.join(config.SCREENSHOTS_DIR, png_name)

            # 断点续传：截图已存在则跳过
            if os.path.exists(png_path) and os.path.getsize(png_path) > 100:
                print(f"[Render] [SKIP] 已存在: {png_name}")
                screenshot_count += 1
                continue

            try:
                img = sim.render(html_path, render_cfg)
                img.save(png_path, "PNG")
                screenshot_count += 1
                print(f"[Render] [{idx+1}/{len(html_files)}] 已保存: {png_name}")
            except Exception as e:
                print(f"[Render] [FAIL] 失败 ({png_name}): {e}")

        sim.close()
    except Exception as e:
        print(f"[Phase4] [FAIL] MonitorSimulator 初始化失败: {e}")

    print(f"\n[Phase4] 共生成 {screenshot_count} 张截图。")
    pipeline_state.mark_completed(state, phase, {"count": screenshot_count})


# ============================================================
# Phase 5 — 元数据汇总
# ============================================================
def phase5_metadata(state: dict):
    """将所有生成结果汇总为一个 JSON 文件。"""
    phase = "phase5_metadata"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    if not config.RUN_METADATA:
        print(f"\n[SKIP] {phase} 未启用 (RUN_METADATA=False)，跳过。")
        pipeline_state.mark_completed(state, phase, {"skipped": True})
        return

    print("\n" + "=" * 60)
    print("Phase 5: 元数据 JSON 汇总")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()

    # 读取 prompts 数据
    prompts_data: dict = {}
    if os.path.exists(config.PROMPTS_JSON):
        try:
            with open(config.PROMPTS_JSON, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[Phase5] [WARN] prompts.json 读取失败")

    # 读取 history 数据
    history_data: dict = {}
    if os.path.exists(config.HISTORIES_JSON):
        try:
            with open(config.HISTORIES_JSON, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[Phase5] [WARN] histories.json 读取失败")

    # 加载 URL 映射
    url_mapping: dict = {}
    if os.path.exists(config.URL_MAPPING_FILE):
        try:
            with open(config.URL_MAPPING_FILE, "r", encoding="utf-8") as f:
                url_mapping = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    records: list[dict] = []

    for html_path in html_files:
        rel_path = os.path.relpath(html_path, config.RAW_HTML_DIR)
        rel_key = rel_path.replace(os.sep, "/")
        basename = os.path.splitext(os.path.basename(html_path))[0]

        # 该页面的截图
        png = f"{basename}.png"
        png_path = os.path.join(config.SCREENSHOTS_DIR, png)
        screenshot = png if os.path.exists(png_path) else None

        # 该页面的 Prompts
        page_prompts = prompts_data.get(rel_key, {})

        record = {
            "html_file": rel_key,
            "url": url_mapping.get(rel_key, ""),
            "screenshot": screenshot,
            "target_prompts": page_prompts.get("target_prompts", []),
            "user_prompts": page_prompts.get("user_prompts", []),
            "shadow_histories": history_data.get("shadow_histories", []),
            "user_histories": history_data.get("user_histories", []),
        }
        records.append(record)

    metadata = {
        "total_html_files": len(html_files),
        "total_screenshots": sum(1 for r in records if r["screenshot"]),
        "screenshot_resolution": f"{config.SCREENSHOT_WIDTH}x{config.SCREENSHOT_HEIGHT}",
        "domains": config.DOMAINS,
        "records": records,
    }

    os.makedirs(os.path.dirname(config.OUTPUT_JSON), exist_ok=True)
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Phase5] 元数据已保存: {config.OUTPUT_JSON}")
    print(f"         HTML 文件:  {metadata['total_html_files']}")
    print(f"         截图:       {metadata['total_screenshots']}")
    pipeline_state.mark_completed(state, phase, {
        "total_html": metadata["total_html_files"],
        "total_screenshots": metadata["total_screenshots"],
    })


# ============================================================
# 主函数
# ============================================================
def main():
    # 启动双通道日志（控制台 + 文件）
    log_path = logger.setup_logging(config.LOG_DIR)

    start_time = time.time()

    print("=" * 60)
    print("WebInject 数据集制备流水线")
    print("=" * 60)
    print(f"  RUN_CRAWLER     = {config.RUN_CRAWLER}")
    print(f"  RUN_SYNTH_GEN   = {config.RUN_SYNTH_GEN}")
    print(f"  RUN_PROMPT_GEN  = {config.RUN_PROMPT_GEN}")
    print(f"  RUN_HISTORY     = {config.RUN_HISTORY}")
    print(f"  RUN_RENDER      = {config.RUN_RENDER}")
    print(f"  RUN_METADATA    = {config.RUN_METADATA}")
    print(f"  LOCAL_MODEL     = {config.LOCAL_MODEL_NAME}")
    print(f"  DOMAINS         = {config.DOMAINS}")
    print(f"  SINGLE_FILE     = {config.SINGLE_FILE_BIN}")
    print(f"  LOG_FILE        = {log_path}")
    print("=" * 60)

    # 加载断点状态
    state = pipeline_state.load_state()

    # ================================================================
    # 依次执行 6 个阶段（断点续传机制确保已完成的阶段自动跳过）
    # ================================================================
    phase0_crawl(state)       # 真实网页采集
    phase1_synth(state)       # 合成网页生成
    phase2_prompts(state)     # Prompt 生成
    phase3_history(state)     # History 生成
    phase4_render(state)      # 渲染截图
    phase5_metadata(state)    # 元数据汇总

    # ================================================================
    # 流水线完成后释放大模型显存
    # ================================================================
    try:
        import local_llm
        local_llm.unload_model()
    except Exception:
        pass

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"全部 6 个阶段完成! 耗时: {elapsed:.1f} 秒")
    print(f"日志文件: {log_path}")
    print("=" * 60)

    # 关闭日志
    logger.shutdown_logging()


if __name__ == "__main__":
    main()
