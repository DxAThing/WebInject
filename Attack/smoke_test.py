# ============================================================
# smoke_test.py — 攻击-评估流水线冒烟测试
# ============================================================
# 使用 Mock MLLM (不下载真实模型) 验证流水线逻辑:
#   1. config.py 配置验证
#   2. pack_data.py LMDB 打包
#   3. dataset.py 解析 (LMDB + 文件双模式) + GPU Prefetch
#   4. attacker.py PGD 优化 (梯度累积 + 动量)
#   5. evaluator.py ASR 评估 (批量推理)
#   6. 断点续传
#
# 用法: python smoke_test.py
# ============================================================

import logging
import os
import sys
import shutil

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ======================= 测试 1: Config =======================

def test_config():
    logger.info("=" * 50)
    logger.info("TEST 1: config.py")
    from config import ATTACK_CONFIG, EVAL_CONFIG, DATASET_METADATA_JSON, SCREENSHOTS_DIR, LMDB_PATH
    assert ATTACK_CONFIG["EPSILON"] > 0
    assert ATTACK_CONFIG["ALPHA"] > 0
    assert ATTACK_CONFIG["PGD_STEPS"] > 0
    assert ATTACK_CONFIG["GRAD_ACCUM_STEPS"] >= 1
    assert 0 <= ATTACK_CONFIG["MOMENTUM_DECAY"] < 1
    # 云端部署时 Dataset 目录可能不存在, 仅在有 Dataset 时检查
    if os.path.exists(DATASET_METADATA_JSON):
        logger.info(f"  数据集元数据: {DATASET_METADATA_JSON} ✓")
    else:
        logger.warning(f"  数据集元数据不存在 (云端部署正常): {DATASET_METADATA_JSON}")
    # LMDB 必须存在 (云端部署的核心数据)
    if os.path.isdir(LMDB_PATH):
        logger.info(f"  LMDB 数据库: {LMDB_PATH} ✓")
    else:
        logger.warning(f"  LMDB 数据库不存在: {LMDB_PATH} (需要先 pack)")
    logger.info(f"  EPSILON={ATTACK_CONFIG['EPSILON']:.4f}, ALPHA={ATTACK_CONFIG['ALPHA']:.6f}")
    logger.info(f"  PGD_STEPS={ATTACK_CONFIG['PGD_STEPS']}")
    logger.info(f"  GRAD_ACCUM_STEPS={ATTACK_CONFIG['GRAD_ACCUM_STEPS']}")
    logger.info(f"  MOMENTUM_DECAY={ATTACK_CONFIG['MOMENTUM_DECAY']}")
    logger.info(f"  COMPILE_MODEL={ATTACK_CONFIG.get('COMPILE_MODEL', False)}")
    logger.info(f"  GRADIENT_CHECKPOINTING={ATTACK_CONFIG.get('GRADIENT_CHECKPOINTING', False)}")
    logger.info(f"  PREFETCH_TO_GPU={ATTACK_CONFIG.get('PREFETCH_TO_GPU', False)}")
    logger.info(f"  EVAL_BATCH_SIZE={EVAL_CONFIG.get('EVAL_BATCH_SIZE', 1)}")
    logger.info(f"  LMDB_PATH={LMDB_PATH}")
    logger.info("  ✓ config.py OK")


# ======================= 测试 2: LMDB 打包 ====================

def test_pack():
    logger.info("=" * 50)
    logger.info("TEST 2: pack_data.py (LMDB 打包)")
    from config import LMDB_PATH, DATASET_METADATA_JSON

    # 云端部署时 Dataset 目录可能不存在, 此时跳过 pack 测试
    if not os.path.exists(DATASET_METADATA_JSON):
        logger.warning("  ⚠ Dataset 目录不存在 (云端部署), 跳过 pack 测试")
        # 使用已有 LMDB 进行后续测试
        if os.path.isdir(LMDB_PATH):
            logger.info(f"  使用已有 LMDB: {LMDB_PATH}")
            return LMDB_PATH
        else:
            logger.error("  LMDB 也不存在, 无法继续测试")
            raise RuntimeError("既无 Dataset 目录也无 LMDB 数据库")

    test_lmdb_path = os.path.join(
        os.path.dirname(__file__), "data", "_smoke_test_lmdb.lmdb"
    )
    # 清理旧测试数据
    if os.path.isdir(test_lmdb_path):
        shutil.rmtree(test_lmdb_path)

    # 临时修改 LMDB_PATH 进行打包
    import config
    original_lmdb = config.LMDB_PATH
    config.LMDB_PATH = test_lmdb_path

    try:
        from pack_data import pack_attack_lmdb
        # 注意: pack_data 使用 from config import LMDB_PATH, 所以需要 reload
        import importlib
        import pack_data
        pack_data.LMDB_PATH = test_lmdb_path
        pack_data.pack_attack_lmdb()

        assert os.path.isdir(test_lmdb_path), "LMDB 目录未创建"
        assert os.path.isfile(os.path.join(test_lmdb_path, "data.mdb")), "data.mdb 不存在"

        # 验证 LMDB 内容
        import lmdb
        import pickle
        env = lmdb.open(test_lmdb_path, readonly=True, lock=False)
        with env.begin(buffers=True) as txn:
            raw_keys = txn.get(b"__keys__")
            assert raw_keys is not None, "LMDB 中无 __keys__"
            keys = pickle.loads(raw_keys)
            logger.info(f"  LMDB 包含 {len(keys)} 条记录")
            assert len(keys) > 0, "LMDB 记录为空"

            # 验证第一条记录
            first_key = keys[0]
            raw = txn.get(first_key.encode("utf-8"))
            entry = pickle.loads(raw)
            assert "image_bytes" in entry, "缺少 image_bytes"
            assert "target_prompts" in entry, "缺少 target_prompts"
            assert "shadow_histories" in entry, "缺少 shadow_histories"
            logger.info(f"  第一条: {first_key}, image: {len(entry['image_bytes'])} bytes")
        env.close()

        logger.info("  ✓ pack_data.py OK")
    finally:
        config.LMDB_PATH = original_lmdb

    return test_lmdb_path


# ======================= 测试 3: Dataset (LMDB 模式) ==========

def test_dataset_lmdb(lmdb_path):
    logger.info("=" * 50)
    logger.info("TEST 3: dataset.py (LMDB 模式)")
    from dataset import AttackDataset

    ds = AttackDataset(lmdb_path=lmdb_path, prefetch_to_gpu=False)
    all_pages = ds.get_all_webpages()
    logger.info(f"  网页总数: {len(all_pages)}")
    assert len(all_pages) > 0, "未找到任何网页记录"

    rec = all_pages[0]
    logger.info(f"  第一条: id={rec.webpage_id}")
    logger.info(f"    shadow_histories: {len(rec.shadow_histories)}")
    logger.info(f"    target_prompts: {len(rec.target_prompts)}")

    # 从 LMDB 加载截图
    tensor = ds.load_screenshot_tensor(rec)
    logger.info(f"    截图 Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    assert tensor.min() >= 0 and tensor.max() <= 1, "截图 Tensor 不在 [0, 1]"

    # 测试 get_batch_pairs
    pairs = ds.get_batch_pairs(rec, batch_size=4, pair_type="shadow")
    assert len(pairs) == 4, f"期望 4 组 pair, 得到 {len(pairs)}"
    logger.info(f"    batch_pairs(4): {len(pairs)} 组 OK")

    # 断点续传测试
    unprocessed = ds.get_unprocessed_webpages()
    assert len(unprocessed) == len(all_pages), "首次运行应全部为未处理"

    ds.close()
    logger.info("  ✓ dataset.py (LMDB) OK")

    return all_pages


# ======================= 测试 3b: Dataset (文件模式) ===========

def test_dataset_file():
    logger.info("=" * 50)
    logger.info("TEST 3b: dataset.py (文件回退模式)")
    from config import DATASET_METADATA_JSON

    # 云端部署时 Dataset 目录可能不存在, 跳过文件模式测试
    if not os.path.exists(DATASET_METADATA_JSON):
        logger.warning("  ⚠ Dataset 目录不存在 (云端部署), 跳过文件模式测试")
        return None, []

    from dataset import AttackDataset

    ds = AttackDataset(lmdb_path="/nonexistent/path.lmdb", prefetch_to_gpu=False)
    all_pages = ds.get_all_webpages()
    logger.info(f"  网页总数: {len(all_pages)}")
    assert len(all_pages) > 0, "文件模式下未找到任何网页记录"

    rec = all_pages[0]
    if os.path.exists(rec.screenshot_path):
        tensor = ds.load_screenshot_tensor(rec)
        logger.info(f"    截图 Tensor shape: {tensor.shape}")
    else:
        logger.warning(f"    ⚠ 截图文件不存在 (跳过): {rec.screenshot_path}")

    ds.close()
    logger.info("  ✓ dataset.py (文件回退) OK")

    return ds, all_pages


# ======================= Mock MLLM ===========================

class MockMLLMWrapper:
    """
    模拟 MLLMWrapper, 不加载真实模型。
    支持新增的 compute_loss_batch 和 generate_batch 接口。
    """

    def __init__(self, target_action="click((500, 500))"):
        self.target_action = target_action
        self.device = "cpu"
        self._linear = nn.Linear(3, 1, bias=False)

    def compute_loss(self, image_tensor, prompt_text, history, target_action):
        loss = image_tensor.mean() * self._linear.weight.sum()
        return loss

    def compute_loss_batch(self, image_tensor, pairs, target_action):
        total_loss = torch.tensor(0.0)
        n = len(pairs)
        for prompt, history in pairs:
            loss = self.compute_loss(image_tensor, prompt, history, target_action)
            scaled = loss / n
            scaled.backward(retain_graph=True)
            total_loss += loss.detach()
        return total_loss / n

    def generate(self, image_tensor, prompt_text, history, max_new_tokens=128):
        import random
        if random.random() < 0.5:
            return self.target_action
        return "scroll(down)"

    def generate_batch(self, image_tensor, pairs, max_new_tokens=128):
        return [
            self.generate(image_tensor, p, h, max_new_tokens)
            for p, h in pairs
        ]

    def cleanup(self):
        pass


# ======================= 测试 4: Attacker (梯度累积 + 动量) ====

def test_attacker(lmdb_path):
    logger.info("=" * 50)
    logger.info("TEST 4: attacker.py (梯度累积 + 动量 PGD, 10 steps)")
    from config import ATTACK_CONFIG
    from dataset import AttackDataset

    ds = AttackDataset(lmdb_path=lmdb_path, prefetch_to_gpu=False)
    all_pages = ds.get_all_webpages()

    test_output_dir = os.path.join(os.path.dirname(__file__), "data", "_smoke_test_deltas")
    os.makedirs(test_output_dir, exist_ok=True)

    mock_mllm = MockMLLMWrapper(target_action=ATTACK_CONFIG["TARGET_ACTION"])

    from attacker import PGDAttacker
    attacker = PGDAttacker(
        mllm=mock_mllm,
        pgd_steps=10,
        output_dir=test_output_dir,
        model_input_size=(64, 64),
        log_interval=5,
        grad_accum_steps=3,       # 测试梯度累积
        momentum_decay=0.9,       # 测试动量 PGD
    )

    # 找第一个有截图数据的网页
    rec = None
    for page in all_pages:
        try:
            _ = ds.load_screenshot_tensor(page)
            rec = page
            break
        except FileNotFoundError:
            continue

    if rec is None:
        logger.warning("  ⚠ 无可用截图，跳过攻击测试")
        ds.close()
        return test_output_dir

    logger.info(f"  测试网页: {rec.webpage_id}")
    image_tensor = ds.load_screenshot_tensor(rec)

    delta = attacker.attack_webpage(
        webpage_id=rec.webpage_id,
        image_tensor=image_tensor,
        shadow_histories=rec.shadow_histories,
        target_prompts=rec.target_prompts,
    )

    logger.info(f"  delta shape: {delta.shape}, L∞: {delta.abs().max().item():.6f}")

    delta_file = os.path.join(test_output_dir, f"delta_{rec.webpage_id}.pt")
    assert os.path.exists(delta_file), f"delta 文件未生成: {delta_file}"
    logger.info(f"  ✓ delta 文件已生成")

    # 断点续传验证
    ds2 = AttackDataset(lmdb_path=lmdb_path, prefetch_to_gpu=False)
    unprocessed = ds2.get_unprocessed_webpages(output_dir=test_output_dir)
    logger.info(f"  断点续传: 未处理 {len(unprocessed)}/{len(all_pages)}")
    ds2.close()

    ds.close()
    logger.info("  ✓ attacker.py (梯度累积 + 动量) OK")

    return test_output_dir


# ======================= 测试 5: Evaluator (批量推理) ==========

def test_evaluator(lmdb_path, delta_dir):
    logger.info("=" * 50)
    logger.info("TEST 5: evaluator.py (批量推理)")
    from config import ATTACK_CONFIG
    from dataset import AttackDataset

    ds = AttackDataset(lmdb_path=lmdb_path, prefetch_to_gpu=False)

    mock_mllm = MockMLLMWrapper(target_action=ATTACK_CONFIG["TARGET_ACTION"])

    test_results_path = os.path.join(delta_dir, "smoke_eval_results.json")

    from evaluator import Evaluator
    evaluator = Evaluator(
        mllm=mock_mllm,
        dataset=ds,
        delta_dir=delta_dir,
        results_path=test_results_path,
        eval_batch_size=4,  # 测试批量推理
    )

    results = evaluator.evaluate_all()

    logger.info(f"  评估网页数: {results['summary']['num_webpages_evaluated']}")
    logger.info(f"  总查询数:   {results['summary']['total_queries']}")
    logger.info(f"  攻击成功数: {results['summary']['total_success']}")
    logger.info(f"  总体 ASR:   {results['summary']['overall_asr']:.2%}")

    assert os.path.exists(test_results_path), "评估结果文件未生成"
    logger.info(f"  ✓ 评估结果已保存")

    ds.close()
    logger.info("  ✓ evaluator.py (批量推理) OK")


# ======================= 清理 ================================

def cleanup(*dirs):
    logger.info("=" * 50)
    import gc
    gc.collect()
    for d in dirs:
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d)
                logger.info(f"已清理: {d}")
            except PermissionError:
                logger.warning(f"无法清理 (文件锁定): {d}")


# ======================= 主入口 ===============================

def main():
    logger.info("🔧 攻击-评估流水线冒烟测试 (含 LMDB 打包 + 并行优化)")
    logger.info("(使用 Mock MLLM, 不下载真实模型)\n")

    lmdb_path = None
    delta_dir = None
    try:
        test_config()
        lmdb_path = test_pack()
        test_dataset_lmdb(lmdb_path)
        test_dataset_file()
        delta_dir = test_attacker(lmdb_path)
        test_evaluator(lmdb_path, delta_dir)

        logger.info("\n" + "=" * 50)
        logger.info("✅ 全部冒烟测试通过!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        sys.exit(1)

    finally:
        cleanup(lmdb_path, delta_dir)


if __name__ == "__main__":
    main()
