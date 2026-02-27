# ============================================================
# config.py — 攻击-评估流水线的统一配置中心
# ============================================================
# 所有路径、超参数在此定义，严禁使用 argparse。
# ============================================================

import os

# ======================= 路径配置 ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset")

# 上一阶段产出
DATASET_METADATA_JSON = os.path.join(DATASET_DIR, "data", "dataset_metadata.json")
PROMPTS_JSON = os.path.join(DATASET_DIR, "data", "prompts.json")
SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "data", "screenshots")

# LMDB 打包路径 (pack_data.py 输出)
LMDB_PATH = os.path.join(BASE_DIR, "data", "attack_dataset.lmdb")

# ======================= 攻击配置 ==========================

ATTACK_CONFIG = {
    # MLLM 模型路径 (必须是多模态 VL 模型, 纯文本模型无法处理截图)
    # Qwen2.5-VL-7B-Instruct: 7B 参数多模态模型, 支持图像+文本输入
    "MLLM_MODEL_PATH": "./model",

    # L∞ 扰动约束
    "EPSILON": 16 / 255.0,

    # PGD 步长
    "ALPHA": 0.3 / 255.0,

    # PGD 迭代次数
    "PGD_STEPS": 2500,

    # 默认攻击目标动作
    "TARGET_ACTION": 'click((500, 500))',

    # 评估生成后是否执行 DSL 规范化后处理。
    # True: 将自然语言输出尽量规整为 click/type/scroll/wait DSL。
    # False: 保留模型原始输出 (用于 A/B 对照与诊断)。
    "NORMALIZE_GENERATION_TO_DSL": True,

    # 优化后 delta 的输出目录
    "DELTA_OUTPUT_DIR": os.path.join(BASE_DIR, "data", "optimized_deltas"),

    # 日志目录
    "LOG_DIR": os.path.join(BASE_DIR, "data", "logs"),

    # 每 N 步打印一次 loss
    "LOG_INTERVAL": 50,

    # =================== 并行与显存优化 ===================

    # 梯度累积步数: 每个 PGD step 内对 N 组 (prompt, history) 累积梯度
    # 相当于 effective_batch_size = GRAD_ACCUM_STEPS
    # 吃满显存的核心参数: 增大此值直到 OOM 再回退一步
    "GRAD_ACCUM_STEPS": 8,

    # 每个 PGD step 的总序列数预算 (跨所有并行网页)
    # 当 PARALLEL_WEBPAGES 增大时, 每页的 grad_accum_steps 自动缩小
    # 保证: effective_accum = max(1, TOTAL_PAIRS_PER_STEP // PARALLEL_WEBPAGES)
    # 设 0 则禁用自动缩放, 始终使用 GRAD_ACCUM_STEPS
    "TOTAL_PAIRS_PER_STEP": 32,

    # GPU 批量前向大小: 每次 model.forward() 并行处理的序列数
    # 将多个 (image, prompt) 打包为一个 batch, 一次 GPU forward 并行计算
    # 32GB 显存推荐 8, 若 OOM 可降为 4
    "FORWARD_BATCH_SIZE": 12,

    # Momentum PGD (MI-FGSM): 动量衰减系数, 0 = 标准 PGD
    # 动量有助于在大缩放比 (4K→448) 时稳定梯度信号
    "MOMENTUM_DECAY": 0.9,

    # torch.compile 加速 (需要 PyTorch 2.0+, 首次编译有数分钟耗时)
    # ⚠ Qwen2.5-VL 视觉编码器使用动态 torch.split(lengths.tolist()),
    #   Inductor 无法处理 aten._local_scalar_dense, 必须设为 False
    "COMPILE_MODEL": False,

    # 梯度检查点: 以计算时间换显存, 允许更大 GRAD_ACCUM_STEPS
    "GRADIENT_CHECKPOINTING": True,

    # GPU prefetch: 启动时将所有截图一次性加载到 GPU
    # 670 × 1920×1080×3×4 bytes ≈ 15GB, 若显存不足会自动回退到按需加载
    # 默认关闭: 改用按需加载策略, 仅在 PGD 需要时才从 LMDB 读取并送入 GPU
    "PREFETCH_TO_GPU": False,

    # 多网页并行优化: 同一 PGD 循环内同时优化多个网页的 δ
    # 增大此值可提升 GPU 利用率, 但显存消耗线性增长
    # 32GB 显存推荐 8, 若 OOM 可降为 4
    "PARALLEL_WEBPAGES": 8,

    # 禁止在 PGD 内循环调用 empty_cache (会导致 CPU-GPU 同步停顿)
    # 仅在切换网页时清理一次
    "EMPTY_CACHE_PER_STEP": False,
}

# ======================= 评估配置 ==========================

EVAL_CONFIG = {
    # 评估结果 JSON 输出路径
    "EVAL_RESULTS_PATH": os.path.join(BASE_DIR, "data", "eval_results.json"),

    # 评估时批量生成的 prompt 组数 (吃满显存)
    "EVAL_BATCH_SIZE": 32,
}

# ======================= 网页分类域 ========================

DOMAINS = ["Blog", "Commerce", "Education", "Healthcare", "Portfolio"]
