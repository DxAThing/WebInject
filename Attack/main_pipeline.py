# ============================================================
# main_pipeline.py — 攻击-评估流水线编排脚本
# ============================================================
# 提供四种运行模式:
#   - pack:     将截图 + 元数据打包为 LMDB 二进制数据库
#   - attack:   遍历未处理的网页，执行 PGD 优化生成 δ
#   - evaluate: 加载已生成的 δ，计算 ASR 并输出 JSON 报告
#   - all:      pack → attack → evaluate (完整流水线)
#
# 用法:
#   python main_pipeline.py           # 默认执行 attack 模式
#   python main_pipeline.py pack      # 打包 LMDB
#   python main_pipeline.py attack    # 攻击模式
#   python main_pipeline.py evaluate  # 评估模式
#   python main_pipeline.py all       # 完整流水线
# ============================================================

import logging
import os
import sys
import time

import torch
from tqdm import tqdm

from config import ATTACK_CONFIG, EVAL_CONFIG, LMDB_PATH
from dataset import AttackDataset
from attacker import PGDAttacker
from evaluator import Evaluator

# ======================= 日志配置 ==========================

def setup_logging() -> None:
    """配置统一的日志格式。"""
    log_dir = ATTACK_CONFIG["LOG_DIR"]
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ======================= GPU 信息报告 =======================

def report_gpu_info() -> None:
    """输出 GPU 设备信息和显存状态。"""
    if not torch.cuda.is_available():
        logger.info("GPU: 不可用, 使用 CPU")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        logger.info(
            f"GPU {i}: {props.name} | "
            f"显存: {total_gb:.1f} GB | "
            f"SM 数: {props.multi_processor_count} | "
            f"Compute: {props.major}.{props.minor}"
        )

    # 当前显存使用
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.info(f"显存状态: 已分配 {allocated:.2f} GB | 已保留 {reserved:.2f} GB")


def report_optimization_config() -> None:
    """输出并行优化配置摘要。"""
    logger.info("并行优化配置:")
    logger.info(f"  梯度累积步数:     {ATTACK_CONFIG.get('GRAD_ACCUM_STEPS', 1)}")
    logger.info(f"  总 forward 预算:  {ATTACK_CONFIG.get('TOTAL_PAIRS_PER_STEP', 0)} (0=禁用自动缩放)")
    logger.info(f"  动量 PGD (μ):     {ATTACK_CONFIG.get('MOMENTUM_DECAY', 0.0)}")
    logger.info(f"  torch.compile:    {ATTACK_CONFIG.get('COMPILE_MODEL', False)}")
    logger.info(f"  梯度检查点:       {ATTACK_CONFIG.get('GRADIENT_CHECKPOINTING', False)}")
    logger.info(f"  GPU Prefetch:     {ATTACK_CONFIG.get('PREFETCH_TO_GPU', False)}")
    logger.info(f"  并行网页数:       {ATTACK_CONFIG.get('PARALLEL_WEBPAGES', 1)}")
    logger.info(f"  评估批量大小:     {EVAL_CONFIG.get('EVAL_BATCH_SIZE', 1)}")
    logger.info(f"  LMDB 路径:        {LMDB_PATH}")


# ======================= 打包模式 ==========================

def run_pack() -> None:
    """打包模式: 将截图 + 元数据打包为 LMDB 二进制数据库。"""
    logger.info("=" * 60)
    logger.info("启动打包模式 (LMDB Packing)")
    logger.info("=" * 60)

    if os.path.isdir(LMDB_PATH):
        logger.info(f"LMDB 已存在: {LMDB_PATH}, 跳过打包 (如需重新打包请先删除)")
        return

    from pack_data import pack_attack_lmdb
    pack_attack_lmdb()
    logger.info("打包完成。")


# ======================= 攻击模式 ==========================

def run_attack() -> None:
    """
    攻击模式: 遍历未处理的网页，执行 PGD 优化生成 δ。

    优化特性:
        - LMDB 数据源 + GPU Prefetch: 零 I/O 延迟
        - 梯度累积: 每步 N 组 pair, 吃满显存
        - 动量 PGD: 稳定梯度信号
        - 断点续传: 已生成 delta 的网页自动跳过
    """
    logger.info("=" * 60)
    logger.info("启动攻击模式 (PGD Optimization)")
    logger.info("=" * 60)

    # ---- 前置检查: LMDB 必须存在 ----
    if not os.path.isdir(LMDB_PATH):
        logger.error(
            f"LMDB 数据库不存在: {LMDB_PATH}\n"
            f"请先在本地运行 'python pack_data.py' 打包数据, 或确认部署包包含 data/attack_dataset.lmdb/"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    report_gpu_info()
    report_optimization_config()

    # ---- 初始化数据加载器 ----
    dataset = AttackDataset()

    # ---- GPU Prefetch: 所有截图一次性加载到 GPU ----
    if device == "cuda" and ATTACK_CONFIG.get("PREFETCH_TO_GPU", False):
        dataset.prefetch_all_to_gpu(device=device)
        report_gpu_info()

    # ---- 获取待处理列表 (断点续传) ----
    unprocessed = dataset.get_unprocessed_webpages()
    total = len(dataset.get_all_webpages())

    logger.info(
        f"网页总数: {total} | 待处理: {len(unprocessed)} | "
        f"已完成: {total - len(unprocessed)}"
    )

    if not unprocessed:
        logger.info("所有网页已处理完毕，无需再次攻击。")
        return

    # ---- 初始化 MLLM ----
    from mllm_wrapper import MLLMWrapper
    logger.info(f"正在加载 MLLM: {ATTACK_CONFIG['MLLM_MODEL_PATH']} ...")
    try:
        mllm = MLLMWrapper(
            model_path=ATTACK_CONFIG["MLLM_MODEL_PATH"],
            device=device,
        )
    except Exception as e:
        logger.error(
            f"MLLM 加载失败: {e}\n"
            f"请检查: 1) 模型路径是否正确  2) 显存是否充足  3) transformers 版本是否匹配",
            exc_info=True,
        )
        sys.exit(1)
    logger.info("MLLM 加载完成。")
    report_gpu_info()

    # ---- 初始化 PGD 攻击器 ----
    attacker = PGDAttacker(mllm=mllm)

    # ---- 并行优化参数 ----
    parallel_pages = ATTACK_CONFIG.get("PARALLEL_WEBPAGES", 1)
    logger.info(f"并行网页数: {parallel_pages}")

    # ---- 按 batch 优化 δ (按需加载 + 多图并行) ----
    for batch_start in range(0, len(unprocessed), parallel_pages):
        batch_records = unprocessed[batch_start : batch_start + parallel_pages]
        batch_ids = [r.webpage_id for r in batch_records]
        batch_desc = ", ".join(batch_ids)

        logger.info(
            f"\n{'='*40}\n"
            f"Batch [{batch_start // parallel_pages + 1}/"
            f"{(len(unprocessed) + parallel_pages - 1) // parallel_pages}]  "
            f"网页: {batch_desc}\n"
            f"{'='*40}"
        )

        # ---- 按需加载: 仅将当前 batch 的截图送入 GPU ----
        batch_images = []
        batch_shadows = []
        batch_prompts = []
        valid_ids = []
        skip_count = 0

        for record in batch_records:
            try:
                image_tensor = dataset.load_screenshot_tensor(record, device=device)
                batch_images.append(image_tensor)
                batch_shadows.append(record.shadow_histories)
                batch_prompts.append(record.target_prompts)
                valid_ids.append(record.webpage_id)
            except FileNotFoundError as e:
                logger.error(f"✗ {record.webpage_id} 跳过 (截图缺失): {e}")
                skip_count += 1
                continue

        if not valid_ids:
            logger.warning(f"Batch 中无有效网页, 跳过")
            continue

        # ---- 执行批量 PGD 优化 ----
        try:
            attacker.attack_batch(
                webpage_ids=valid_ids,
                image_tensors=batch_images,
                shadow_histories_list=batch_shadows,
                target_prompts_list=batch_prompts,
            )
            for wid in valid_ids:
                logger.info(f"✓ {wid} 优化完成")

        except torch.OutOfMemoryError:
            logger.error(
                f"✗ Batch OOM ({batch_desc}) — "
                f"建议减小 PARALLEL_WEBPAGES (当前: {parallel_pages})"
            )
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # OOM 回退: 逐个网页重试
            logger.info("回退到单网页模式逐个重试...")
            for i, wid in enumerate(valid_ids):
                try:
                    attacker.attack_webpage(
                        webpage_id=wid,
                        image_tensor=batch_images[i],
                        shadow_histories=batch_shadows[i],
                        target_prompts=batch_prompts[i],
                    )
                    logger.info(f"✓ {wid} (回退) 优化完成")
                except torch.OutOfMemoryError:
                    logger.error(f"✗ {wid} 单页也 OOM, 跳过")
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"✗ {wid} (回退) 失败: {e}", exc_info=True)
                finally:
                    mllm.cleanup()

        except Exception as e:
            logger.error(f"✗ Batch 失败 ({batch_desc}): {e}", exc_info=True)

        finally:
            # ---- 释放当前 batch 的 GPU 张量和 CPU 字节缓存 ----
            del batch_images
            for wid in valid_ids:
                dataset.release_gpu_cache(wid)
                dataset.evict_image_cache(wid)
            mllm.cleanup()

    logger.info("\n攻击模式完成。")


# ======================= 评估模式 ==========================

def run_evaluate() -> None:
    """
    评估模式: 加载已生成的 δ，在测试集上计算 ASR。

    优化特性:
        - 批量推理: EVAL_BATCH_SIZE 组 prompt 并行生成
        - GPU Prefetch: 截图从显存缓存读取
    """
    logger.info("=" * 60)
    logger.info("启动评估模式 (ASR Evaluation)")
    logger.info("=" * 60)

    # ---- 前置检查: LMDB 必须存在 ----
    if not os.path.isdir(LMDB_PATH):
        logger.error(
            f"LMDB 数据库不存在: {LMDB_PATH}\n"
            f"请确认部署包包含 data/attack_dataset.lmdb/"
        )
        sys.exit(1)

    # ---- 前置检查: 至少要有 delta 文件 ----
    delta_dir = ATTACK_CONFIG["DELTA_OUTPUT_DIR"]
    if not os.path.isdir(delta_dir) or not any(
        f.endswith(".pt") and not f.endswith(".tmp")
        for f in os.listdir(delta_dir)
    ):
        logger.error(
            f"未找到可用的 delta 文件 (目录: {delta_dir})\n"
            f"请先运行 attack 模式生成 delta"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    report_gpu_info()

    # ---- 初始化数据加载器 ----
    dataset = AttackDataset()

    # GPU Prefetch
    if device == "cuda" and ATTACK_CONFIG.get("PREFETCH_TO_GPU", False):
        dataset.prefetch_all_to_gpu(device=device)

    # ---- 初始化 MLLM ----
    from mllm_wrapper import MLLMWrapper
    logger.info(f"正在加载 MLLM: {ATTACK_CONFIG['MLLM_MODEL_PATH']} ...")
    try:
        mllm = MLLMWrapper(
            model_path=ATTACK_CONFIG["MLLM_MODEL_PATH"],
            device=device,
        )
    except Exception as e:
        logger.error(
            f"MLLM 加载失败: {e}\n"
            f"请检查: 1) 模型路径是否正确  2) 显存是否充足  3) transformers 版本是否匹配",
            exc_info=True,
        )
        sys.exit(1)
    logger.info("MLLM 加载完成。")

    # ---- 初始化评估器 ----
    evaluator = Evaluator(mllm=mllm, dataset=dataset)

    # ---- 执行评估 ----
    results = evaluator.evaluate_all()

    # ---- 输出摘要 ----
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("评估完成 — 最终报告:")
    logger.info(f"  评估网页数:   {summary['num_webpages_evaluated']}")
    logger.info(f"  总查询数:     {summary['total_queries']}")
    logger.info(f"  攻击成功数:   {summary['total_success']}")
    logger.info(f"  总体 ASR:     {summary['overall_asr']:.2%}")
    logger.info(f"  结果文件:     {EVAL_CONFIG['EVAL_RESULTS_PATH']}")
    logger.info("=" * 60)


# ======================= 入口 ==============================

def main() -> None:
    """解析运行模式并执行对应流水线。"""
    setup_logging()

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "attack"

    valid_modes = {"pack", "attack", "evaluate", "eval", "all"}
    if mode not in valid_modes:
        logger.error(
            f"未知模式: '{mode}'。可选: {', '.join(sorted(valid_modes))}"
        )
        sys.exit(1)

    start_time = time.time()

    if mode == "pack":
        run_pack()

    elif mode == "attack":
        run_attack()

    elif mode in ("evaluate", "eval"):
        run_evaluate()

    elif mode == "all":
        run_pack()
        run_attack()
        run_evaluate()

    elapsed = time.time() - start_time
    logger.info(f"\n总用时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
