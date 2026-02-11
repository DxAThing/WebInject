# ============================================================
# main.py — 数据集制备流水线主入口
# ============================================================
# 6 阶段流水线，支持断点续传：
#   Phase 0 : 真实网页采集 (Crawler)      — 可选
#   Phase 1 : 合成网页生成 (SyntheticGen)  — 可选
#   Phase 2 : Target / User Prompt 生成
#   Phase 3 : Shadow / User History 生成
#   Phase 4 : 显示器渲染模拟
#   Phase 5 : 元数据 JSON 汇总
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
# Phase 1 — 合成网页生成
# ============================================================
def phase1_synth(state: dict):
    """生成合成 HTML 页面。"""
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
# Phase 2 — Prompt 生成
# ============================================================
def phase2_prompts(state: dict):
    """为每个 HTML 页面生成 Target Prompts + User Prompts。"""
    phase = "phase2_prompts"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    import prompt_generator

    print("\n" + "=" * 60)
    print("Phase 2: Prompt 生成")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()
    if not html_files:
        print("[Phase2] [WARN] 未找到 HTML 文件，跳过。")
        pipeline_state.mark_completed(state, phase, {"count": 0})
        return

    all_prompts: dict[str, dict] = {}

    for html_path in html_files:
        rel_name = os.path.relpath(html_path, config.RAW_HTML_DIR)
        target_prompts = prompt_generator.generate_target_prompts(html_path)
        user_prompts = prompt_generator.generate_user_prompts(target_prompts)
        all_prompts[rel_name] = {
            "target_prompts": target_prompts,
            "user_prompts": user_prompts,
        }

    # 产出数据写入独立文件
    os.makedirs(os.path.dirname(config.PROMPTS_JSON), exist_ok=True)
    with open(config.PROMPTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    print(f"[Phase2] Prompt 数据已保存: {config.PROMPTS_JSON}")

    print(f"\n[Phase2] 共为 {len(all_prompts)} 个页面生成 Prompt。")
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

    import history_generator

    print("\n" + "=" * 60)
    print("Phase 3: 动作历史生成")
    print("=" * 60)

    shadow = history_generator.generate_shadow_histories()
    user = history_generator.generate_user_histories()

    # 产出数据写入独立文件
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
# Phase 4 — 渲染模拟
# ============================================================
def phase4_render(state: dict):
    """
    使用 headless 浏览器 + ICC Profile 渲染每个 HTML 页面。
    每个页面 x 每种显示器规格 → 一张 ICC 截图 + 一张原始截图 (U-Net 输入)。
    """
    phase = "phase4_render"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    import monitor_simulator

    print("\n" + "=" * 60)
    print("Phase 4: 显示器渲染模拟")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()
    if not html_files:
        print("[Phase4] [WARN] 未找到 HTML 文件，跳过。")
        pipeline_state.mark_completed(state, phase, {"count": 0})
        return

    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(config.RAW_SCREENSHOTS_DIR, exist_ok=True)
    screenshot_paths: list[str] = []
    raw_screenshot_paths: list[str] = []

    try:
        sim = monitor_simulator.MonitorSimulator()

        for html_path in html_files:
            basename = os.path.splitext(os.path.basename(html_path))[0]

            for monitor_name, monitor_cfg in config.MONITORS.items():
                icc_name = f"{basename}_{monitor_name}.png"
                raw_name = f"{basename}_{monitor_name}_raw.png"
                icc_path = os.path.join(config.SCREENSHOTS_DIR, icc_name)
                raw_path = os.path.join(config.RAW_SCREENSHOTS_DIR, raw_name)

                # 断点续传：两张图都已存在则跳过
                icc_exists = os.path.exists(icc_path) and os.path.getsize(icc_path) > 100
                raw_exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 100
                if icc_exists and raw_exists:
                    print(f"[Render] [SKIP] 已存在: {icc_name}")
                    screenshot_paths.append(icc_path)
                    raw_screenshot_paths.append(raw_path)
                    continue

                try:
                    raw_img, icc_img = sim.render(html_path, monitor_cfg)
                    icc_img.save(icc_path, "PNG")
                    raw_img.save(raw_path, "PNG")
                    screenshot_paths.append(icc_path)
                    raw_screenshot_paths.append(raw_path)
                    print(f"[Render] 已保存: {icc_name} + {raw_name}")
                except Exception as e:
                    print(f"[Render] [FAIL] 失败 ({icc_name}): {e}")

        sim.close()
    except Exception as e:
        print(f"[Phase4] [FAIL] MonitorSimulator 初始化失败: {e}")

    print(f"\n[Phase4] 共生成 {len(screenshot_paths)} 张 ICC 截图 + {len(raw_screenshot_paths)} 张原始截图。")
    pipeline_state.mark_completed(state, phase, {
        "icc_count": len(screenshot_paths),
        "raw_count": len(raw_screenshot_paths),
    })


# ============================================================
# Phase 5 — 元数据汇总
# ============================================================
def phase5_metadata(state: dict):
    """将所有生成结果汇总为一个 JSON 文件。"""
    phase = "phase5_metadata"
    if pipeline_state.is_completed(state, phase):
        print(f"\n[SKIP] {phase} 已完成，跳过。")
        return

    print("\n" + "=" * 60)
    print("Phase 5: 元数据 JSON 汇总")
    print("=" * 60)

    html_files = webpage_manager.list_html_files()

    # 从独立的产出文件读取 prompts / histories
    prompts_data: dict = {}
    if os.path.exists(config.PROMPTS_JSON):
        try:
            with open(config.PROMPTS_JSON, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[Phase5] [WARN] prompts.json 读取失败")

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

        # 该页面的截图（ICC 变换后 + 原始 sRGB）
        screenshots: list[str] = []
        raw_screenshots: list[str] = []
        for monitor_name in config.MONITORS:
            png = f"{basename}_{monitor_name}.png"
            png_path = os.path.join(config.SCREENSHOTS_DIR, png)
            if os.path.exists(png_path):
                screenshots.append(png)

            raw_png = f"{basename}_{monitor_name}_raw.png"
            raw_path = os.path.join(config.RAW_SCREENSHOTS_DIR, raw_png)
            if os.path.exists(raw_path):
                raw_screenshots.append(raw_png)

        # 该页面的 Prompts
        page_prompts = prompts_data.get(rel_key, {})

        record = {
            "html_file": rel_key,
            "url": url_mapping.get(rel_key, ""),
            "screenshots": screenshots,
            "raw_screenshots": raw_screenshots,
            "target_prompts": page_prompts.get("target_prompts", []),
            "user_prompts": page_prompts.get("user_prompts", []),
            "shadow_histories": history_data.get("shadow_histories", []),
            "user_histories": history_data.get("user_histories", []),
        }
        records.append(record)

    metadata = {
        "total_html_files": len(html_files),
        "total_screenshots": sum(len(r["screenshots"]) for r in records),
        "total_raw_screenshots": sum(len(r["raw_screenshots"]) for r in records),
        "domains": config.DOMAINS,
        "monitors": list(config.MONITORS.keys()),
        "records": records,
    }

    os.makedirs(os.path.dirname(config.OUTPUT_JSON), exist_ok=True)
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Phase5] 元数据已保存: {config.OUTPUT_JSON}")
    print(f"         HTML 文件: {metadata['total_html_files']}")
    print(f"         ICC 截图:  {metadata['total_screenshots']}")
    print(f"         原始截图: {metadata['total_raw_screenshots']}")
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
    print(f"  RUN_CRAWLER   = {config.RUN_CRAWLER}")
    print(f"  RUN_SYNTH_GEN = {config.RUN_SYNTH_GEN}")
    print(f"  USE_MOCK      = {config.USE_MOCK}")
    print(f"  DOMAINS       = {config.DOMAINS}")
    print(f"  SINGLE_FILE   = {config.SINGLE_FILE_BIN}")
    print(f"  LOG_FILE      = {log_path}")
    print("=" * 60)

    # 加载断点状态
    state = pipeline_state.load_state()

    # 依次执行 6 个阶段
    phase0_crawl(state)
    phase1_synth(state)
    phase2_prompts(state)
    phase3_history(state)
    phase4_render(state)
    phase5_metadata(state)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"流水线完成! 耗时: {elapsed:.1f} 秒")
    print(f"日志文件: {log_path}")
    print("=" * 60)

    # 关闭日志
    logger.shutdown_logging()


if __name__ == "__main__":
    main()
