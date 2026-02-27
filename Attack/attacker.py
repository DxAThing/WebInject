# ============================================================
# attacker.py — PGD 核心优化器 (梯度累积 + 动量 PGD + 多图并行)
# ============================================================
# 为单个或多个网页执行 PGD (Projected Gradient Descent) 优化，
# 寻找通用对抗扰动 δ，使得 MLLM 在多种 prompt + history 下
# 都生成目标动作 TARGET_ACTION。
#
# ====================== 并行与显存优化 ======================
# 1. 梯度累积 (Gradient Accumulation):
#    每个 PGD step 内, 对 GRAD_ACCUM_STEPS 组 (prompt, history)
#    分别计算 loss 并 backward, 梯度自动累积到 delta.grad。
#    等效于 batch_size = GRAD_ACCUM_STEPS 的 mini-batch SGD,
#    梯度信号更稳定, 对抗扰动更通用。
#
# 2. 动量 PGD (MI-FGSM):
#    引入动量 g_t = μ * g_{t-1} + grad / ||grad||_1
#    delta = delta - α * sign(g_t)
#    动量有助于在大缩放比 (4K→448) 时稳定梯度信号。
#
# 3. 移除 PGD 内循环 empty_cache:
#    empty_cache() 会导致 CPU-GPU 同步停顿, 在高频调用时
#    是严重的性能杀手。仅在切换网页时清理一次。
#
# 4. 自动缩放梯度累积步数:
#    当并行网页数较大时, 每页的 accum_steps 自动减小,
#    保证每步总 forward 次数 = TOTAL_PAIRS_PER_STEP (e.g. 64),
#    避免 N×8 = 512 次串行 forward 导致单步耗时过长。
#
# 5. CUDA Stream 流水线:
#    不同网页的 forward+backward 分配到不同 Stream,
#    重叠计算与 CPU 逻辑, 提升 GPU 利用率。
#
# =============== 关于非微分图像 Resize 的处理 ===============
# δ 在原始分辨率初始化 (如 1920×1080), 通过 F.interpolate(bilinear)
# 可微 Resize 到模型输入尺寸 (如 448×448)。梯度通过 interpolate
# 无缝回传到原始分辨率的 δ。
# ============================================================

import atexit
import logging
import os
import random
import signal
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ATTACK_CONFIG
from dataset import AttackDataset, WebpageRecord

logger = logging.getLogger(__name__)


class PGDAttacker:
    """
    PGD 对抗攻击优化器 (带梯度累积 + 动量)。

    对每个网页独立优化一个通用扰动 δ，使得 MLLM 在该网页截图上
    对多种 prompt + history 组合都生成目标动作。

    参数:
        mllm: MLLMWrapper 实例。
        epsilon: L∞ 扰动约束。
        alpha: PGD 步长。
        pgd_steps: 优化迭代次数。
        target_action: 目标动作字符串。
        output_dir: delta 文件输出目录。
        model_input_size: MLLM vision encoder 所需的输入尺寸 (H, W)。
        log_interval: 每 N 步打印一次 loss。
        grad_accum_steps: 每 PGD step 内梯度累积的 (prompt, history) 组数。
        momentum_decay: 动量衰减系数, 0 = 标准 PGD。
    """

    def __init__(
        self,
        mllm,
        epsilon: float = ATTACK_CONFIG["EPSILON"],
        alpha: float = ATTACK_CONFIG["ALPHA"],
        pgd_steps: int = ATTACK_CONFIG["PGD_STEPS"],
        target_action: str = ATTACK_CONFIG["TARGET_ACTION"],
        output_dir: str = ATTACK_CONFIG["DELTA_OUTPUT_DIR"],
        model_input_size: tuple = (448, 448),
        log_interval: int = ATTACK_CONFIG["LOG_INTERVAL"],
        grad_accum_steps: int = ATTACK_CONFIG.get("GRAD_ACCUM_STEPS", 1),
        momentum_decay: float = ATTACK_CONFIG.get("MOMENTUM_DECAY", 0.0),
    ):
        self.mllm = mllm
        self.epsilon = epsilon
        self.alpha = alpha
        self.pgd_steps = pgd_steps
        self.target_action = target_action
        self.output_dir = output_dir
        self.model_input_size = model_input_size
        self.log_interval = log_interval
        self.grad_accum_steps = grad_accum_steps
        self.momentum_decay = momentum_decay

        self.checkpoint_interval = ATTACK_CONFIG.get("CHECKPOINT_INTERVAL", 200)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ---- 中断信号处理: 意外退出时保存检查点 ----
        self._interrupt_requested = False
        self._active_checkpoint_state = None  # 当前 PGD 循环的状态引用
        self._original_sigint = None
        self._original_sigterm = None

    # ==================== 检查点管理 ========================

    def _install_signal_handlers(self) -> None:
        """安装信号处理器, 捕获 SIGINT/SIGTERM 以优雅保存检查点。"""
        def _handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.warning(f"[Checkpoint] 收到 {sig_name}, 将在当前 step 结束后保存检查点并退出")
            self._interrupt_requested = True

        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _restore_signal_handlers(self) -> None:
        """恢复原始信号处理器。"""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        self._interrupt_requested = False

    def _checkpoint_path(self, *webpage_ids: str) -> str:
        """生成检查点文件路径。"""
        tag = "_".join(sorted(webpage_ids))
        # 避免文件名过长
        if len(tag) > 100:
            import hashlib
            tag = hashlib.md5(tag.encode()).hexdigest()
        return os.path.join(self.checkpoint_dir, f"ckpt_{tag}.pt")

    def _save_checkpoint(
        self,
        webpage_ids: List[str],
        step: int,
        deltas: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        """保存 PGD 检查点 (原子写入)。"""
        ckpt_path = self._checkpoint_path(*webpage_ids)
        tmp_path = ckpt_path + ".tmp"

        state = {
            "webpage_ids": webpage_ids,
            "step": step,
            "deltas": [d.data.cpu() for d in deltas],
            "momentum_buffers": [m.cpu() for m in momentum_buffers],
        }
        torch.save(state, tmp_path)
        os.replace(tmp_path, ckpt_path)

        logger.info(
            f"[Checkpoint] 已保存 step {step}/{self.pgd_steps} → {ckpt_path}"
        )

    def _load_checkpoint(
        self,
        webpage_ids: List[str],
    ) -> Optional[Dict]:
        """尝试加载检查点, 如果存在且网页匹配则返回状态。"""
        ckpt_path = self._checkpoint_path(*webpage_ids)
        if not os.path.exists(ckpt_path):
            return None

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            saved_ids = sorted(state["webpage_ids"])
            expected_ids = sorted(webpage_ids)
            if saved_ids != expected_ids:
                logger.warning(
                    f"[Checkpoint] 网页 ID 不匹配, 忽略检查点: "
                    f"{saved_ids} vs {expected_ids}"
                )
                return None

            logger.info(
                f"[Checkpoint] 恢复检查点: step {state['step']}/{self.pgd_steps} "
                f"({len(state['deltas'])} 个 delta)"
            )
            return state
        except Exception as e:
            logger.warning(f"[Checkpoint] 加载失败, 从头开始: {e}")
            return None

    def _delete_checkpoint(self, *webpage_ids: str) -> None:
        """训练完成后删除检查点文件。"""
        ckpt_path = self._checkpoint_path(*webpage_ids)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            logger.info(f"[Checkpoint] 已清理: {ckpt_path}")

    # ==================== 单图 PGD 优化 ======================

    def attack_webpage(
        self,
        webpage_id: str,
        image_tensor: torch.Tensor,
        shadow_histories: List[List[str]],
        target_prompts: List[str],
    ) -> torch.Tensor:
        """
        对单个网页执行 PGD 优化 (梯度累积 + 动量)。
        支持检查点保存/恢复: 意外中断后自动从上次位置继续。

        参数:
            webpage_id: 网页标识符。
            image_tensor: 原始截图, shape (3, H, W), [0, 1], 在 GPU 上。
            shadow_histories: 攻击训练用的历史动作序列列表。
            target_prompts: 攻击训练用的提示文本列表。

        返回:
            delta: 优化后的扰动 Tensor, shape (3, H, W)。
        """
        device = image_tensor.device
        _, H, W = image_tensor.shape

        logger.info(
            f"[PGD] 开始优化 {webpage_id} | "
            f"图像尺寸: ({H}, {W}) | "
            f"步数: {self.pgd_steps} | "
            f"ε={self.epsilon:.4f} | α={self.alpha:.6f} | "
            f"梯度累积: {self.grad_accum_steps} | "
            f"动量: {self.momentum_decay}"
        )

        # ---- Step 1: 初始化 δ (或从检查点恢复) ----
        start_step = 0
        delta = torch.zeros(
            1, 3, H, W, device=device, dtype=torch.float32, requires_grad=True
        )
        image = image_tensor.unsqueeze(0).float()  # (1, 3, H, W)
        momentum_buffer = torch.zeros_like(delta)

        ckpt = self._load_checkpoint([webpage_id])
        if ckpt is not None:
            start_step = ckpt["step"] + 1
            delta.data.copy_(ckpt["deltas"][0].to(device))
            momentum_buffer.copy_(ckpt["momentum_buffers"][0].to(device))
            logger.info(f"[PGD] 从 step {start_step} 恢复, 跳过已完成步骤")
            del ckpt

        if start_step >= self.pgd_steps:
            logger.info(f"[PGD] {webpage_id} 已完成全部 {self.pgd_steps} 步, 跳过")
            delta_result = delta.data.squeeze(0).cpu()
            self._save_delta_atomic(webpage_id, delta_result)
            self._delete_checkpoint(webpage_id)
            return delta_result

        # 确保有可用的 prompts 和 histories
        if not target_prompts:
            logger.warning(f"[PGD] {webpage_id} 没有 target_prompts, 使用默认 prompt")
            target_prompts = ["Perform the next action on the webpage."]
        if not shadow_histories:
            logger.warning(f"[PGD] {webpage_id} 没有 shadow_histories, 使用空历史")
            shadow_histories = [[]]

        # ---- Step 2: PGD 迭代优化 (梯度累积 + 动量) ----
        self._install_signal_handlers()
        pbar = tqdm(
            range(start_step, self.pgd_steps),
            desc=f"PGD {webpage_id}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            initial=start_step,
            total=self.pgd_steps,
        )
        for step in pbar:

            # ---- 2a: 随机采样 grad_accum_steps 组 (prompt, history) ----
            pairs = [
                (random.choice(target_prompts), random.choice(shadow_histories))
                for _ in range(self.grad_accum_steps)
            ]

            # ---- 2b + 2c: 梯度累积 (吃满显存的核心) ----
            # 关键优化: 每次迭代独立构建 adv_image → 独立计算图,
            #   backward() 后立即释放, 避免 retain_graph=True
            #   retain_graph 会同时保留 N-1 份激活图 (每份约 2-4 GB),
            #   在 32 GB 显存上极易 OOM。
            total_loss = 0.0
            n = len(pairs)

            for pair_idx, (prompt, history) in enumerate(pairs):
                # 每次重建对抗图像, 创建独立于其他 pair 的计算图
                adv_image = torch.clamp(image + delta, 0.0, 1.0)
                adv_image_resized = F.interpolate(
                    adv_image,
                    size=self.model_input_size,
                    mode="bilinear",
                    align_corners=False,
                )

                loss = self.mllm.compute_loss(
                    image_tensor=adv_image_resized.squeeze(0),
                    prompt_text=prompt,
                    history=history,
                    target_action=self.target_action,
                )
                scaled_loss = loss / n
                scaled_loss.backward()  # 无需 retain_graph, 立即释放激活
                total_loss += loss.item()

                # 显式释放当前迭代的计算图
                del loss, scaled_loss, adv_image, adv_image_resized

            avg_loss = total_loss / n

            # ---- 2d: 动量更新 + 符号法 (MI-FGSM) ----
            with torch.no_grad():
                grad = delta.grad
                
                assert grad is not None, "梯度为 None, 请检查 compute_loss 是否正确返回标量 loss"

                if self.momentum_decay > 0:
                    grad_norm = grad / (grad.abs().mean() + 1e-12)
                    momentum_buffer.mul_(self.momentum_decay).add_(grad_norm)
                    update_direction = momentum_buffer.sign()
                else:
                    update_direction = grad.sign()

                delta_data = delta.data - self.alpha * update_direction
                delta_data = torch.clamp(delta_data, -self.epsilon, self.epsilon)
                delta_data = torch.clamp(
                    delta_data, -image.data, 1.0 - image.data
                )
                delta.data.copy_(delta_data)

            delta.grad.zero_()

            # ---- tqdm 进度条实时更新 ----
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                linf=f"{delta.data.abs().max().item():.4f}",
                accum=self.grad_accum_steps,
            )

            # ---- 定期保存检查点 ----
            if (
                self.checkpoint_interval > 0
                and (step + 1) % self.checkpoint_interval == 0
                and step + 1 < self.pgd_steps
            ):
                self._save_checkpoint(
                    [webpage_id], step, [delta], [momentum_buffer]
                )

            # ---- 中断处理 ----
            if self._interrupt_requested:
                logger.warning(
                    f"[PGD] 中断! 保存检查点 step {step}/{self.pgd_steps}"
                )
                self._save_checkpoint(
                    [webpage_id], step, [delta], [momentum_buffer]
                )
                self._restore_signal_handlers()
                raise KeyboardInterrupt(
                    f"用户中断, 检查点已保存 (step {step})"
                )

        # ---- Step 3: 保存 delta (原子写入) ----
        self._restore_signal_handlers()
        delta_result = delta.data.squeeze(0).cpu()  # (3, H, W)
        self._save_delta_atomic(webpage_id, delta_result)
        self._delete_checkpoint(webpage_id)  # 训练完成, 清理检查点

        logger.info(f"[PGD] 完成 {webpage_id} | 最终 δ L∞: {delta_result.abs().max().item():.6f}")

        return delta_result

    def _save_delta_atomic(self, webpage_id: str, delta: torch.Tensor) -> None:
        """
        原子写入 delta 文件。

        先保存为 .tmp 文件，成功后再 rename 为正式文件。
        这样即使进程被中断，也不会留下损坏的 .pt 文件，
        断点续传时不会误判为已完成。
        """
        final_path = os.path.join(self.output_dir, f"delta_{webpage_id}.pt")
        tmp_path = final_path + ".tmp"

        torch.save(delta, tmp_path)
        os.replace(tmp_path, final_path)

        logger.info(f"  [Save] delta 已保存: {final_path}")

    # ============================================================
    # 多图并行 PGD 优化 (Batched PGD + Batched Forward)
    # ============================================================
    # 同一 PGD 循环内同时维护 N 个网页的 δ。
    #
    # 核心优化 — 真正的 GPU 并行:
    #   将多个 (image, prompt) 打包为一个 batch, 通过一次
    #   model.forward() 并行计算, GPU 的 SM 和显存带宽同时
    #   服务所有序列。
    #
    #   与串行版本 (K 次 compute_loss) 相比:
    #     串行: 64 次 forward, 每次 ~0.3s = ~21s/step
    #     批量: 4 次 forward (batch=16), 每次 ~1s = ~4s/step
    #
    # 梯度正确性:
    #   PGD 使用 sign(梯度) 更新, 与梯度绝对值无关。
    #   batch 平均 loss 的梯度方向与逐条累积一致,
    #   因此批量与串行等价。
    # ============================================================

    def _compute_effective_accum(self, num_pages: int) -> int:
        """
        根据并行网页数自动计算每页的梯度累积步数。

        总 forward 预算 = TOTAL_PAIRS_PER_STEP (默认 64)。
        当 num_pages 大时, 每页 accum 自动减少, 避免:
          64 页 × 8 accum = 512 次串行 forward → 单步 2分钟。

        设 TOTAL_PAIRS_PER_STEP=0 则禁用自动缩放, 始终使用 GRAD_ACCUM_STEPS。
        """
        budget = ATTACK_CONFIG.get("TOTAL_PAIRS_PER_STEP", 0)
        if budget <= 0:
            return self.grad_accum_steps
        return max(1, budget // num_pages)

    def attack_batch(
        self,
        webpage_ids: List[str],
        image_tensors: List[torch.Tensor],
        shadow_histories_list: List[List[List[str]]],
        target_prompts_list: List[List[str]],
    ) -> List[torch.Tensor]:
        """
        对一批网页同时进行 PGD 优化 (批量 GPU Forward)。

        性能特性:
          - 批量 Forward: 多个 (image, prompt) 打包为一个 batch,
            一次 model.forward() 并行计算, 真正利用 GPU 并行性
          - 自动缩放梯度累积: 每步总序列数 ≤ TOTAL_PAIRS_PER_STEP
          - 子批处理: 显存不足时自动分割为 FORWARD_BATCH_SIZE 大小的子批
          - 预采样: 所有 step 的 pair 一次性生成

        参数:
            webpage_ids:          网页标识符列表, 长度为 N。
            image_tensors:        原始截图列表, 每个 shape (3, H_i, W_i), [0,1], GPU。
            shadow_histories_list: 每个网页的历史动作序列列表。
            target_prompts_list:  每个网页的 target_prompts 列表。

        返回:
            deltas: 优化后的扰动列表, 每个 shape (3, H_i, W_i), CPU。
        """
        N = len(webpage_ids)
        assert N == len(image_tensors) == len(shadow_histories_list) == len(target_prompts_list), \
            f"batch 参数长度不一致: {N}, {len(image_tensors)}, {len(shadow_histories_list)}, {len(target_prompts_list)}"

        if N == 0:
            return []

        if N == 1:
            delta = self.attack_webpage(
                webpage_id=webpage_ids[0],
                image_tensor=image_tensors[0],
                shadow_histories=shadow_histories_list[0],
                target_prompts=target_prompts_list[0],
            )
            return [delta]

        device = image_tensors[0].device
        fwd_batch_size = ATTACK_CONFIG.get("FORWARD_BATCH_SIZE", 16)

        # ---- 自动缩放梯度累积步数 ----
        effective_accum = self._compute_effective_accum(N)
        total_fwd_per_step = N * effective_accum

        logger.info(
            f"[BatchPGD] 启动批量优化 | 网页数: {N} | "
            f"步数: {self.pgd_steps} | "
            f"ε={self.epsilon:.4f} | α={self.alpha:.6f} | "
            f"动量: {self.momentum_decay}"
        )
        logger.info(
            f"[BatchPGD] 梯度累积自动缩放: "
            f"GRAD_ACCUM_STEPS={self.grad_accum_steps} → "
            f"effective_accum={effective_accum} | "
            f"每步总序列: {total_fwd_per_step} | "
            f"Forward batch: {fwd_batch_size}"
        )
        for i, wid in enumerate(webpage_ids):
            _, H, W = image_tensors[i].shape
            logger.info(f"  [{i}] {wid}: ({H}, {W})")

        # ---- Step 1: 为每个网页初始化 δ, 动量缓冲, 图像 ----
        deltas = []
        images = []
        momentum_buffers = []
        prompts_per_page = []
        histories_per_page = []

        for i in range(N):
            _, H, W = image_tensors[i].shape
            d = torch.zeros(1, 3, H, W, device=device, dtype=torch.float32, requires_grad=True)
            deltas.append(d)
            images.append(image_tensors[i].unsqueeze(0).float())  # (1,3,H,W)
            momentum_buffers.append(torch.zeros_like(d))

            tp = target_prompts_list[i] or ["Perform the next action on the webpage."]
            sh = shadow_histories_list[i] or [[]]
            prompts_per_page.append(tp)
            histories_per_page.append(sh)

        # ---- Step 1b: 构建唯一 (prompt, history) 组合集 ----
        # 从可用 prompts × histories 的笛卡尔积构建,
        # 而非预采样 2500×32=80000 个元组 (节省 CPU 内存 + VRAM 碑片)
        unique_pairs = set()
        for idx in range(N):
            for p in prompts_per_page[idx]:
                for h in histories_per_page[idx]:
                    unique_pairs.add((p, tuple(h)))

        # ---- Step 1c: 预计算 Token 缓存 (消除 PGD 循环内 CPU 阻塞) ----
        token_cache = self.mllm.precompute_token_cache(
            self.model_input_size, list(unique_pairs), self.target_action
        )
        del unique_pairs

        # 为每个网页构建 token 索引列表 (PGD 循环内只用整数采样)
        pair_to_idx = token_cache["pair_to_idx"]
        page_token_indices = []  # page_token_indices[i] = [cache_idx, ...]
        for idx in range(N):
            page_indices = []
            for p in prompts_per_page[idx]:
                for h in histories_per_page[idx]:
                    page_indices.append(pair_to_idx[(p, tuple(h))])
            page_token_indices.append(page_indices)

        # 释放 CPU 临时对象
        del prompts_per_page, histories_per_page, pair_to_idx

        # ---- Step 1d: 尝试从检查点恢复 ----
        start_step = 0
        ckpt = self._load_checkpoint(webpage_ids)
        if ckpt is not None:
            start_step = ckpt["step"] + 1
            for idx in range(N):
                deltas[idx].data.copy_(ckpt["deltas"][idx].to(device))
                momentum_buffers[idx].copy_(
                    ckpt["momentum_buffers"][idx].to(device)
                )
            logger.info(f"[BatchPGD] 从 step {start_step} 恢复")
            del ckpt

        if start_step >= self.pgd_steps:
            logger.info(f"[BatchPGD] 已完成全部 {self.pgd_steps} 步, 直接保存")
            results = []
            for idx in range(N):
                delta_result = deltas[idx].data.squeeze(0).cpu()
                self._save_delta_atomic(webpage_ids[idx], delta_result)
                results.append(delta_result)
            self._delete_checkpoint(*webpage_ids)
            return results

        # ---- 释放初始化阶段 CUDA 缓存碑片 ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Step 2: PGD 迭代 (批量 GPU Forward) ----
        self._install_signal_handlers()
        step_start_time = time.time()
        max_linf = 0.0  # 先初始化, 避免首次未定义
        loss_acc = torch.zeros(1, device=device)  # 预分配, 避免循环内反复创建
        avg_loss = 0.0
        pbar = tqdm(
            range(start_step, self.pgd_steps),
            desc=f"BatchPGD ×{N}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            initial=start_step,
            total=self.pgd_steps,
        )
        for step in pbar:
            # ---- On-the-fly 采样 (纯整数, flat 列表) ----
            page_ix = []
            cache_ix = []
            for idx in range(N):
                indices = page_token_indices[idx]
                for _ in range(effective_accum):
                    page_ix.append(idx)
                    cache_ix.append(random.choice(indices))
            total_count = len(cache_ix)

            # 每步仅两次小张量分配 (总共 ~256 bytes)
            step_page_t = torch.tensor(
                page_ix, dtype=torch.long, device=device
            )
            step_cache_t = torch.tensor(
                cache_ix, dtype=torch.long, device=device
            )
            loss_acc.zero_()

            # ---- Per-step: adv + diff_process 仅一次 ----
            # 同一 step 内 delta 不变, 避免子批重复计算
            target_h = token_cache["diff_target_h"]
            target_w = token_cache["diff_target_w"]
            per_page_pvs = []
            per_page_grids = []
            for idx in range(N):
                adv = torch.clamp(
                    images[idx] + deltas[idx], 0.0, 1.0
                )
                adv_r = F.interpolate(
                    adv, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                pv, grid = self.mllm._differentiable_image_process(
                    adv_r
                )
                per_page_pvs.append(pv)
                per_page_grids.append(grid.squeeze(0))  # (3,)

            # Stack → (N, patches, dim) / (N, 3)
            stacked_pvs = torch.stack(
                per_page_pvs, dim=0
            ).to(dtype=self.mllm.model_dtype)
            stacked_grids = torch.stack(
                per_page_grids, dim=0
            ).to(device)
            patch_dim = stacked_pvs.shape[2]

            # ---- 2a: 子批 forward + backward (纯 GPU 切片) ----
            for bs in range(0, total_count, fwd_batch_size):
                be = min(bs + fwd_batch_size, total_count)
                sub_size = be - bs

                # GPU 切片 (零分配, 零 CPU 开销)
                sub_page = step_page_t[bs:be]
                sub_cache = step_cache_t[bs:be]

                # GPU 索引构建 pixel_values
                sub_pvs = stacked_pvs[sub_page]
                pixel_values = sub_pvs.reshape(-1, patch_dim)
                image_grid_thw = stacked_grids[sub_page]

                # 极简前向 (纯 GPU 索引 + model.forward)
                loss = self.mllm.forward_precomputed(
                    pixel_values, image_grid_thw,
                    sub_cache, token_cache,
                )

                weight = sub_size / total_count
                is_last_sub = (be >= total_count)
                (loss * weight).backward(retain_graph=not is_last_sub)
                loss_acc.add_(loss.detach() * sub_size)
                del loss

            # 释放计算图 (adv → diff_process 的中间张量)
            del stacked_pvs, stacked_grids
            del per_page_pvs, per_page_grids
            del step_page_t, step_cache_t

            avg_loss = loss_acc.item() / total_count

            # ---- 2b: 统一批量更新所有 δ (动量 PGD) ----
            with torch.no_grad():
                for idx in range(N):
                    delta = deltas[idx]
                    image = images[idx]
                    grad = delta.grad

                    assert grad is not None, \
                        f"梯度为 None ({webpage_ids[idx]}), 请检查 compute_loss"

                    if self.momentum_decay > 0:
                        grad_norm = grad / (grad.abs().mean() + 1e-12)
                        momentum_buffers[idx].mul_(self.momentum_decay).add_(grad_norm)
                        update_direction = momentum_buffers[idx].sign()
                    else:
                        update_direction = grad.sign()

                    delta_data = delta.data - self.alpha * update_direction
                    delta_data = torch.clamp(delta_data, -self.epsilon, self.epsilon)
                    delta_data = torch.clamp(delta_data, -image.data, 1.0 - image.data)
                    delta.data.copy_(delta_data)
                    delta.grad.zero_()

            # ---- 进度条 (减少 CUDA 同步) ----
            # loss_acc.item() 已触发一次同步, linf 每 10 步才同步
            if (step + 1) % 10 == 0 or step == start_step:
                max_linf = max(d.data.abs().max().item() for d in deltas)
            elapsed = time.time() - step_start_time
            steps_done = step - start_step + 1
            speed = steps_done / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                linf=f"{max_linf:.4f}",
                batch=fwd_batch_size,
                spd=f"{speed:.2f}it/s",
            )

            # ---- 定期保存检查点 ----
            if (
                self.checkpoint_interval > 0
                and (step + 1) % self.checkpoint_interval == 0
                and step + 1 < self.pgd_steps
            ):
                self._save_checkpoint(
                    webpage_ids, step, deltas, momentum_buffers
                )

            # ---- 中断处理 ----
            if self._interrupt_requested:
                logger.warning(
                    f"[BatchPGD] 中断! 保存检查点 step {step}/{self.pgd_steps}"
                )
                self._save_checkpoint(
                    webpage_ids, step, deltas, momentum_buffers
                )
                self._restore_signal_handlers()
                raise KeyboardInterrupt(
                    f"用户中断, 检查点已保存 (step {step})"
                )

        # ---- Step 3: 保存所有 delta ----
        self._restore_signal_handlers()
        total_time = time.time() - step_start_time
        actual_steps = self.pgd_steps - start_step
        logger.info(
            f"[BatchPGD] 全部完成 | 总耗时: {total_time:.1f}s | "
            f"平均: {total_time / max(1, actual_steps):.2f}s/step | "
            f"GPU forward 次数/step: {(total_fwd_per_step + fwd_batch_size - 1) // fwd_batch_size}"
        )

        results = []
        for idx in range(N):
            delta_result = deltas[idx].data.squeeze(0).cpu()  # (3, H, W)
            self._save_delta_atomic(webpage_ids[idx], delta_result)
            results.append(delta_result)
            logger.info(
                f"[BatchPGD] 完成 {webpage_ids[idx]} | "
                f"最终 δ L∞: {delta_result.abs().max().item():.6f}"
            )

        self._delete_checkpoint(*webpage_ids)  # 训练完成, 清理检查点
        return results
