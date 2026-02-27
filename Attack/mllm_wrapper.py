# ============================================================
# mllm_wrapper.py — 大模型白盒计算引擎 (VL 视觉模型兼容)
# ============================================================
# 封装 HuggingFace MLLM, 提供可微的 Loss 计算和文本生成。
#
# =================== VL 模型兼容性 (关键) ====================
# 必须使用 AutoModelForVision2Seq (而非 AutoModelForCausalLM)
# 加载视觉语言模型。AutoModelForCausalLM 不会加载视觉编码器,
# 导致 pixel_values 未参与计算图, loss.backward() 时
# delta.grad 为 NoneType。
#
# ================== 可微图像预处理 ============================
# VL 模型的图像预处理包含:
#   1. CLIP 归一化 (mean/std)
#   2. Patch 重组 (reshape + permute)
#   3. 时序维度扩展 (单帧图像 → temporal_patch_size 帧)
#
# 以上所有步骤均使用 PyTorch 张量运算实现, 确保梯度完整
# 回传路径: loss → logits → LM → ViT → normalize → δ
#
# ================== 显存管理最佳实践 =========================
# 1. torch.bfloat16: 模型以半精度加载, 显存减半
# 2. 冻结参数: model.requires_grad_(False), 仅计算输入梯度
# 3. gradient_checkpointing: 以计算换显存
# 4. torch.amp.autocast: 混合精度前向传播
# 5. 推理时 torch.no_grad(): 避免显存泄漏
# ============================================================

import logging
import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from config import ATTACK_CONFIG

logger = logging.getLogger(__name__)


class MLLMWrapper:
    """
    MLLM 白盒包装器 — 兼容 VL 视觉模型。

    自动检测 VL 模型并使用正确的模型类 (AutoModelForVision2Seq)
    加载, 确保视觉编码器参与计算图, 梯度可回传到输入扰动 δ。

    参数:
        model_path: HuggingFace 模型路径或本地路径。
        device: 运行设备 ("cuda" 或 "cpu")。
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.normalize_generation_to_dsl = ATTACK_CONFIG.get(
            "NORMALIZE_GENERATION_TO_DSL", True
        )

        # ---- 加载模型与处理器 ----
        self._load_model(model_path, device)

        # ---- 冻结所有模型参数 (仅需输入梯度, 不需参数梯度) ----
        # 这可节省 ~14 GB 梯度显存 (7B bfloat16 模型)
        self.model.requires_grad_(False)

        # ---- 梯度检查点: 以计算时间换显存 ----
        # ⚠ 必须在 eval() 之前启用, 且保持 model.train() 模式!
        #   HuggingFace 的 forward() 仅在 self.training=True 时
        #   执行梯度检查点逻辑。eval() 会禁用它, 导致 28 层激活
        #   全部保留在显存中 (~5 GB → 无检查点 vs ~1.5 GB → 有检查点)。
        #   Qwen2.5-VL 的 dropout=0.0, train/eval 模式行为完全一致。
        if ATTACK_CONFIG.get("GRADIENT_CHECKPOINTING", False):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                # 保持 training 模式以确保检查点生效
                self.model.train()
                logger.info(
                    "[MLLM] 梯度检查点已启用 (use_reentrant=False, training=True)"
                )
            except Exception as e:
                logger.warning(f"[MLLM] 梯度检查点启用失败, 回退到 eval 模式: {e}")
                self.model.eval()
        else:
            self.model.eval()

        # ---- torch.compile 加速 ----
        # ⚠ Qwen2.5-VL 视觉编码器的注意力层使用动态 torch.split
        #   (lengths.tolist()), Inductor 后端无法处理
        #   aten._local_scalar_dense 算子, 会产生大量 graph break。
        #   因此对 Qwen VL 系列自动跳过 torch.compile。
        if ATTACK_CONFIG.get("COMPILE_MODEL", False):
            _model_name = getattr(self.model.config, "_name_or_path", model_path).lower()
            _is_qwen_vl = "qwen" in _model_name and "vl" in _model_name
            if _is_qwen_vl:
                logger.warning(
                    "[MLLM] 检测到 Qwen VL 模型, 跳过 torch.compile "
                    "(视觉编码器动态 split 与 Inductor 不兼容)"
                )
            else:
                try:
                    self.model = torch.compile(self.model)
                    logger.info("[MLLM] torch.compile 已启用")
                except Exception as e:
                    logger.warning(f"[MLLM] torch.compile 失败 (忽略): {e}")

        # ---- 设置视觉配置 (patch_size, 归一化参数等) ----
        self._setup_vision_config()

        logger.info(
            f"[MLLM] 初始化完成 | 模型: {model_path} | "
            f"设备: {device} | 精度: {self.model_dtype} | "
            f"输出DSL规范化: {self.normalize_generation_to_dsl}"
        )

    # ==================== 模型加载 ============================

    def _load_model(self, model_path: str, device: str) -> None:
        """
        加载模型与处理器。

        关键: 必须使用包含视觉编码器的模型类加载 VL 模型,
        否则 pixel_values 不参与计算图, backward() 时
        delta.grad 为 NoneType。

        兼容策略 (按优先级):
          1. Qwen2_5_VLForConditionalGeneration (Qwen2.5-VL 专用)
          2. Qwen2VLForConditionalGeneration     (Qwen2-VL 专用)
          3. AutoModelForImageTextToText          (transformers ≥ 4.48)
          4. AutoModelForVision2Seq               (transformers ≥ 4.34)
        """
        from transformers import AutoProcessor

        # ---- 按优先级尝试导入 VL 模型类 ----
        VLModelClass = None
        _tried = []
        for cls_name in (
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            try:
                VLModelClass = getattr(
                    __import__("transformers", fromlist=[cls_name]), cls_name
                )
                logger.info(f"[MLLM] 使用模型类: {cls_name}")
                break
            except (ImportError, AttributeError):
                _tried.append(cls_name)

        if VLModelClass is None:
            raise ImportError(
                f"无法导入任何 VL 模型类 (已尝试: {', '.join(_tried)})。"
                f"请升级 transformers: pip install -U transformers"
            )

        logger.info(f"[MLLM] 正在加载处理器: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Decoder-only 生成模型要求左填充, 否则 batch generate 会出现
        # right-padding 警告并可能影响生成质量。
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
            logger.info("[MLLM] tokenizer.padding_side 已设置为 left")


        use_cuda = device != "cpu" and torch.cuda.is_available()
        self.model_dtype = torch.bfloat16 if use_cuda else torch.float32

        logger.info(
            f"[MLLM] 正在加载模型: {model_path} "
            f"(dtype={self.model_dtype}, device_map={'auto' if use_cuda else 'N/A'})"
        )

        if use_cuda:
            self.model = VLModelClass.from_pretrained(
                model_path,
                torch_dtype=self.model_dtype,
                device_map="auto",
            )
        else:
            self.model = VLModelClass.from_pretrained(
                model_path,
                torch_dtype=self.model_dtype,
            )

        logger.info(f"[MLLM] 模型加载完成, 参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

    # ==================== 视觉配置 ============================

    def _setup_vision_config(self) -> None:
        """
        从模型配置中提取视觉编码器参数,
        用于可微图像预处理 (patch 重组)。
        """
        config = self.model.config

        # 尝试从 vision_config 获取 patch 参数
        vision_config = getattr(config, "vision_config", None)

        if vision_config is not None:
            self.patch_size = getattr(vision_config, "patch_size", 14)
            self.temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        else:
            # 回退默认值 (Qwen2.5-VL 标准配置)
            self.patch_size = 14
            self.temporal_patch_size = 2
            logger.warning(
                "[MLLM] 未找到 vision_config, 使用默认 patch 参数 "
                f"(patch_size={self.patch_size}, temporal_patch_size={self.temporal_patch_size})"
            )

        # CLIP 归一化常量 (OpenAI CLIP 标准值)
        self._image_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]
        ).view(3, 1, 1)
        self._image_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]
        ).view(3, 1, 1)

        logger.info(
            f"[MLLM] 视觉配置: patch_size={self.patch_size}, "
            f"temporal_patch_size={self.temporal_patch_size}"
        )

    # ==================== 消息格式化 ==========================

    def _format_messages(
        self, prompt_text: str, history: List[str]
    ) -> list:
        """
        构造 Chat 消息格式 (兼容 Qwen2.5-VL 等 VL 模型)。

        参数:
            prompt_text: 当前任务指令。
            history: 历史动作序列 (可为空列表)。

        返回:
            messages: Chat 消息列表。
        """
        # 拼接历史动作
        if history and len(history) > 0:
            history_text = "\n".join(f"- {action}" for action in history)
            user_text = (
                f"Previous actions:\n{history_text}\n\n"
                f"Current task: {prompt_text}"
            )
        else:
            user_text = prompt_text

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a web browsing agent. Based on the screenshot "
                    "and the user's instruction, determine and output ONLY the "
                    "next action in a strict DSL format. "
                    "Valid outputs are exactly one action line such as: "
                    "click((x, y)) | type(\"text\") | scroll(\"down\") | wait(). "
                    "Do not output explanations, prefixes, suffixes, or natural language."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return messages

    def _default_click_action(self) -> str:
        """返回 click 的默认 DSL 动作，优先复用配置中的 TARGET_ACTION。"""
        target_action = ATTACK_CONFIG.get("TARGET_ACTION", "click((500, 500))")
        target_norm = re.sub(r"\s+", "", str(target_action).strip().lower())
        if target_norm.startswith("click"):
            m = re.search(
                r"click\s*\(\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*\)",
                str(target_action),
                re.IGNORECASE,
            )
            if m is not None:
                return f"click(({int(m.group(1))}, {int(m.group(2))}))"
        return "click((500, 500))"

    def _normalize_generated_action(self, raw_text: str) -> str:
        """
        将模型自然语言输出规范为单行 DSL。

        优先提取合法 DSL；若失败则按关键词回退到可评测动作。
        """
        text = (raw_text or "").strip()

        # 1) 优先提取 click((x, y)) / click(x, y)
        m_click = re.search(
            r"click\s*\(\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*\)",
            text,
            re.IGNORECASE,
        )
        if m_click is not None:
            x = int(m_click.group(1))
            y = int(m_click.group(2))
            return f"click(({x}, {y}))"

        # 2) 提取 type("...") / type('...') / type(...)
        m_type_quoted = re.search(
            r"type\s*\(\s*([\"'])(.*?)\1\s*\)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if m_type_quoted is not None:
            content = m_type_quoted.group(2).replace('"', "\\\"").strip()
            return f'type("{content}")'

        m_type_raw = re.search(r"type\s*\(\s*([^\)]*?)\s*\)", text, re.IGNORECASE)
        if m_type_raw is not None:
            content = m_type_raw.group(1).strip().strip('"\'')
            content = content.replace('"', "\\\"")
            return f'type("{content}")'

        # 3) 提取 scroll("down"|"up")
        m_scroll = re.search(r"scroll\s*\(\s*([\"'])?(down|up)\1\s*\)", text, re.IGNORECASE)
        if m_scroll is not None:
            direction = m_scroll.group(2).lower()
            return f'scroll("{direction}")'

        # 4) 提取 wait()
        if re.search(r"\bwait\s*\(\s*\)", text, re.IGNORECASE):
            return "wait()"

        # 5) 关键词回退（处理 click here / scroll to ...）
        lower = text.lower()
        if "click" in lower:
            return self._default_click_action()
        if "type" in lower:
            return 'type("")'
        if "scroll" in lower:
            direction = "up" if "up" in lower else "down"
            return f'scroll("{direction}")'
        if "wait" in lower:
            return "wait()"

        # 6) 全失败兜底，保持可解析 DSL
        return "wait()"

    # ==================== 可微图像预处理 ======================

    def _differentiable_image_process(
        self, image_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        可微图像预处理: 将 (3, H, W) 图像张量转换为 VL 模型的
        pixel_values 格式, 全程使用 PyTorch 运算保持梯度图。

        梯度回传路径:
            pixel_values → reshape/permute → expand (temporal) →
            normalize (CLIP) → image_tensor → δ

        参数:
            image_tensor: (3, H, W), [0, 1], float32, 可带梯度。

        返回:
            pixel_values: (num_patches, patch_dim), 保持梯度图。
            image_grid_thw: (1, 3), 图像网格信息 [grid_t, grid_h, grid_w]。
        """
        C, H, W = image_tensor.shape
        ps = self.patch_size

        # ---- Step 1: 确保尺寸可被 patch_size 整除 ----
        new_h = (H // ps) * ps
        new_w = (W // ps) * ps

        if new_h != H or new_w != W:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            new_h, new_w = H, W

        # ---- Step 2: CLIP 归一化 (可微) ----
        mean = self._image_mean.to(
            device=image_tensor.device, dtype=image_tensor.dtype
        )
        std = self._image_std.to(
            device=image_tensor.device, dtype=image_tensor.dtype
        )
        normalized = (image_tensor - mean) / std  # (C, H, W), 保持梯度

        # ---- Step 3: 时序维度扩展 ----
        # VL 模型将单帧图像视为 temporal_patch_size 帧
        # expand 共享内存, backward 时梯度自动累加
        tps = self.temporal_patch_size
        frames = normalized.unsqueeze(0).expand(tps, -1, -1, -1)
        # frames: (temporal_patch_size, C, new_h, new_w)

        # ---- Step 4: Patch 重组 ----
        grid_t = 1  # 单帧图像
        grid_h = new_h // ps
        grid_w = new_w // ps

        # (tps, C, new_h, new_w)
        #   → (grid_t, tps, C, grid_h, ps, grid_w, ps)
        patches = frames.reshape(
            grid_t, tps, C,
            grid_h, ps,
            grid_w, ps,
        )

        # → (grid_t, grid_h, grid_w, tps, ps, ps, C)
        patches = patches.permute(0, 3, 5, 1, 4, 6, 2)

        # → (grid_t * grid_h * grid_w, tps * ps * ps * C)
        pixel_values = patches.reshape(
            grid_t * grid_h * grid_w,
            tps * ps * ps * C,
        )

        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]], dtype=torch.long
        )

        return pixel_values, image_grid_thw

    # ==================== 核心方法: compute_loss ===============

    def compute_loss(
        self,
        image_tensor: torch.Tensor,
        prompt_text: str,
        history: List[str],
        target_action: str,
    ) -> torch.Tensor:
        """
        计算目标动作的交叉熵损失 (可微)。

        梯度通过以下完整路径回传:
            loss → logits → LM forward → vision encoder (ViT) →
            pixel_values → differentiable_normalize → image_tensor → δ

        关键: 使用可微的图像预处理替换 processor 的 PIL 预处理,
        确保梯度图从 loss 连通到输入 image_tensor。

        参数:
            image_tensor: (3, H, W), [0, 1], float32, 来自 attacker 的
                          对抗图像 (包含 δ 的梯度图)。
            prompt_text: 当前任务指令。
            history: 历史动作序列。
            target_action: 目标动作字符串 (如 "click((500, 500))")。

        返回:
            loss: 标量张量, 保持完整梯度图, 可直接 backward()。
        """
        # ---- 1. 获取 PIL 图像 (仅用于 processor 确定 token 结构) ----
        with torch.no_grad():
            pil_img = to_pil_image(image_tensor.cpu().float().clamp(0, 1))

        # ---- 2. 格式化消息 ----
        messages = self._format_messages(prompt_text, history)

        # ---- 3. 使用 processor 获取 input_ids (含正确数量的 image_pad token) ----
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img],
            return_tensors="pt", padding=True,
        )

        # ---- 4. 可微图像预处理 (替换 processor 的非微分结果) ----
        diff_pv, diff_grid = self._differentiable_image_process(image_tensor)

        # 验证形状一致 (processor 与可微处理应产生相同的 patch 结构)
        proc_pv_shape = inputs["pixel_values"].shape
        if diff_pv.shape != proc_pv_shape:
            logger.warning(
                f"[MLLM] pixel_values 形状不匹配: "
                f"可微={diff_pv.shape} vs processor={proc_pv_shape}. "
                f"尝试调整图像尺寸..."
            )
            # 从 processor 的 grid_thw 反推目标尺寸
            proc_grid = inputs["image_grid_thw"][0]
            target_h = proc_grid[1].item() * self.patch_size
            target_w = proc_grid[2].item() * self.patch_size
            resized = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            diff_pv, diff_grid = self._differentiable_image_process(resized)

        # ---- 5. tokenize target_action ----
        target_token_ids = self.processor.tokenizer.encode(
            target_action, add_special_tokens=False
        )
        target_ids = torch.tensor(
            [target_token_ids], dtype=torch.long
        )

        # ---- 6. 拼接 prompt + target ----
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(target_ids)], dim=1
        )

        # ---- 7. 构造 labels: -100 (忽略 prompt) + target token ids ----
        prompt_len = input_ids.shape[1]
        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        # ---- 8. 移至目标设备 ----
        full_input_ids = full_input_ids.to(self.device)
        full_attention_mask = full_attention_mask.to(self.device)
        labels = labels.to(self.device)

        # pixel_values 保持梯度图, 仅转换 dtype
        pixel_values = diff_pv.to(self.device, dtype=self.model_dtype)
        image_grid_thw = diff_grid.to(self.device)

        # ---- 9. Forward pass (混合精度) ----
        use_autocast = self.device != "cpu" and torch.cuda.is_available()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )

        return outputs.loss

    # ==================== 核心方法: compute_loss_batch =========

    def compute_loss_batch(
        self,
        image_tensors: List[torch.Tensor],
        prompt_texts: List[str],
        histories: List[List[str]],
        target_action: str,
    ) -> torch.Tensor:
        """
        真正的 GPU 批量前向: 多个 (image, prompt, history) 组在
        一次 model.forward() 中并行计算 loss。

        与 compute_loss (逐条前向) 相比, 将 K 条查询打包为一个 batch,
        GPU 的 SM 和显存带宽同时服务 K 条序列, 吐量提升 K 倍。

        梯度回传路径:
            loss → logits → LM → ViT → pixel_values →
            differentiable_normalize → image_tensor → δ
        每个 image_tensor 可能出自不同的 δ, autograd 自动
        将梯度分发到对应 δ.grad。

        参数:
            image_tensors: K 个图像, 每个 (3, H, W), [0,1], 可带梯度。
                           同一 δ 的多个 prompt 可重复引用同一 tensor。
            prompt_texts:  K 个 prompt 字符串。
            histories:     K 个 history 列表。
            target_action: 目标动作字符串。

        返回:
            loss: 标量张量 (batch 平均 CE), 保持完整梯度图。
        """
        K = len(image_tensors)
        assert K == len(prompt_texts) == len(histories), \
            f"batch 长度不一致: {K} vs {len(prompt_texts)} vs {len(histories)}"

        if K == 1:
            return self.compute_loss(
                image_tensors[0], prompt_texts[0], histories[0], target_action
            )

        # ---- 1. 批量 CPU 转换 (相同 tensor 只转一次) ----
        pil_cache = {}  # id(tensor) -> PIL Image
        with torch.no_grad():
            for img in image_tensors:
                tid = id(img)
                if tid not in pil_cache:
                    pil_cache[tid] = to_pil_image(
                        img.detach().float().clamp(0, 1).cpu()
                    )

        # ---- 2. 可微图像预处理 + Tokenize (相同 tensor 缓存) ----
        diff_cache = {}  # id(tensor) -> (diff_pv, diff_grid)
        target_token_ids = self.processor.tokenizer.encode(
            target_action, add_special_tokens=False
        )

        all_diff_pvs = []
        all_diff_grids = []
        all_full_ids = []
        all_full_masks = []
        all_labels = []

        for i in range(K):
            img = image_tensors[i]
            tid = id(img)

            # 可微预处理 (缓存)
            if tid not in diff_cache:
                diff_cache[tid] = self._differentiable_image_process(img)
            diff_pv, diff_grid = diff_cache[tid]

            # Processor (获取 input_ids 中正确数量的 image_pad token)
            messages = self._format_messages(prompt_texts[i], histories[i])
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[pil_cache[tid]],
                return_tensors="pt", padding=True,
            )

            # 形状校验 (每个唯一图像仅需一次)
            proc_pv_shape = inputs["pixel_values"].shape
            if diff_pv.shape != proc_pv_shape:
                proc_grid = inputs["image_grid_thw"][0]
                target_h = proc_grid[1].item() * self.patch_size
                target_w = proc_grid[2].item() * self.patch_size
                resized = F.interpolate(
                    img.unsqueeze(0), size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                diff_pv, diff_grid = self._differentiable_image_process(resized)
                diff_cache[tid] = (diff_pv, diff_grid)

            all_diff_pvs.append(diff_pv)
            all_diff_grids.append(diff_grid)

            # 拼接 prompt + target
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            target_ids = torch.tensor(
                [target_token_ids], dtype=torch.long
            )

            full_ids = torch.cat([input_ids, target_ids], dim=1)
            full_mask = torch.cat(
                [attention_mask, torch.ones_like(target_ids)], dim=1
            )

            prompt_len = input_ids.shape[1]
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            all_full_ids.append(full_ids)
            all_full_masks.append(full_mask)
            all_labels.append(labels)

        # ---- 3. 左填充对齐 (causal LM 标准做法) ----
        max_len = max(ids.shape[1] for ids in all_full_ids)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        padded_ids = []
        padded_masks = []
        padded_labels = []

        for ids, mask, lab in zip(all_full_ids, all_full_masks, all_labels):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_id))
                padded_masks.append(F.pad(mask, (pad_len, 0), value=0))
                padded_labels.append(F.pad(lab, (pad_len, 0), value=-100))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
                padded_labels.append(lab)

        # ---- 4. 组装批量张量 ----
        batch_input_ids = torch.cat(padded_ids, dim=0).to(self.device)
        batch_attention_mask = torch.cat(padded_masks, dim=0).to(self.device)
        batch_labels = torch.cat(padded_labels, dim=0).to(self.device)

        # pixel_values: 按序连接各图像的 patch (Qwen VL 格式)
        pixel_values = torch.cat(
            all_diff_pvs, dim=0
        ).to(self.device, dtype=self.model_dtype)
        image_grid_thw = torch.cat(
            all_diff_grids, dim=0
        ).to(self.device)

        # ---- 5. 单次 GPU Forward (所有 K 条序列并行) ----
        use_autocast = self.device != "cpu" and torch.cuda.is_available()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=batch_labels,
            )

        return outputs.loss

    # ==================== 预计算 Token 缓存 ====================

    def precompute_token_cache(
        self,
        model_input_size: Tuple[int, int],
        unique_pairs: List[Tuple[str, tuple]],
        target_action: str,
    ) -> dict:
        """
        在 PGD 循环外一次性完成 **全部** CPU 密集型工作:
        PIL 转换、Processor 调用、Tokenization、左填充、GPU 传输。

        PGD 内循环只需按索引取 GPU 张量行, 零 CPU 开销。

        参数:
            model_input_size: (H, W) 模型输入尺寸。
            unique_pairs: 所有唯一的 (prompt_text, history_tuple) 组合。
            target_action: 目标动作字符串。

        返回:
            cache: 字典, 包含 GPU 端预构建张量和索引映射。
        """
        logger.info(
            f"[MLLM] 预计算 token 缓存: "
            f"{len(unique_pairs)} 个唯一 (prompt, history) 组合"
        )

        # 1. 创建 dummy PIL (仅用于 processor 确定 image_pad token 数)
        dummy_pil = Image.new(
            "RGB", (model_input_size[1], model_input_size[0])
        )

        # 2. Tokenize target_action (仅一次)
        target_token_ids = self.processor.tokenizer.encode(
            target_action, add_special_tokens=False
        )
        target_ids_t = torch.tensor(
            [target_token_ids], dtype=torch.long
        )

        # 3. 从 processor 获取图像网格信息 (仅依赖图像尺寸, 与像素无关)
        dummy_msg = self._format_messages("x", [])
        dummy_text = self.processor.apply_chat_template(
            dummy_msg, tokenize=False, add_generation_prompt=True
        )
        dummy_inputs = self.processor(
            text=[dummy_text], images=[dummy_pil],
            return_tensors="pt", padding=True,
        )
        proc_grid = dummy_inputs["image_grid_thw"][0]
        diff_target_h = proc_grid[1].item() * self.patch_size
        diff_target_w = proc_grid[2].item() * self.patch_size

        # 4. 逐个预 tokenize (prompt, history) 组合
        pair_list = list(unique_pairs)
        pair_to_idx = {}
        raw_ids = []
        raw_masks = []
        raw_labels = []

        for i, (prompt_text, history_tuple) in enumerate(pair_list):
            history = list(history_tuple)
            messages = self._format_messages(prompt_text, history)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[dummy_pil],
                return_tensors="pt", padding=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            full_ids = torch.cat([input_ids, target_ids_t], dim=1)
            full_mask = torch.cat(
                [attention_mask, torch.ones_like(target_ids_t)], dim=1
            )

            prompt_len = input_ids.shape[1]
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            raw_ids.append(full_ids.squeeze(0))       # (seq_len,)
            raw_masks.append(full_mask.squeeze(0))
            raw_labels.append(labels.squeeze(0))
            pair_to_idx[(prompt_text, history_tuple)] = i

        # 5. 全局左填充 + 堆叠 + 一次性送入 GPU
        pad_id = self.processor.tokenizer.pad_token_id or 0
        global_max_len = max(ids.shape[0] for ids in raw_ids)

        padded_ids_list = []
        padded_masks_list = []
        padded_labels_list = []

        for ids, mask, lab in zip(raw_ids, raw_masks, raw_labels):
            pad_len = global_max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids_list.append(F.pad(ids, (pad_len, 0), value=pad_id))
                padded_masks_list.append(F.pad(mask, (pad_len, 0), value=0))
                padded_labels_list.append(F.pad(lab, (pad_len, 0), value=-100))
            else:
                padded_ids_list.append(ids)
                padded_masks_list.append(mask)
                padded_labels_list.append(lab)

        # (num_pairs, global_max_len) — 常驻 GPU
        gpu_ids = torch.stack(padded_ids_list, dim=0).to(self.device)
        gpu_masks = torch.stack(padded_masks_list, dim=0).to(self.device)
        gpu_labels = torch.stack(padded_labels_list, dim=0).to(self.device)

        logger.info(
            f"[MLLM] Token 缓存完成 | "
            f"Processor 目标分辨率: {diff_target_h}×{diff_target_w} | "
            f"缓存条目: {len(pair_list)} | "
            f"序列长度: {global_max_len} | "
            f"GPU 张量: {gpu_ids.shape}"
        )

        return {
            "pair_to_idx": pair_to_idx,
            "gpu_ids": gpu_ids,
            "gpu_masks": gpu_masks,
            "gpu_labels": gpu_labels,
            "diff_target_h": diff_target_h,
            "diff_target_w": diff_target_w,
        }

    def compute_loss_batch_cached(
        self,
        image_tensors: List[torch.Tensor],
        pair_indices: List[int],
        token_cache: dict,
    ) -> torch.Tensor:
        """
        零 CPU 开销批量前向 — token 已预填充在 GPU 上,
        仅做 GPU 索引 + 可微图像处理 + model.forward()。

        每步 CPU 工作量: ~几十微秒 (构造索引列表),
        消除左填充 / torch.cat / CPU→GPU 传输。

        参数:
            image_tensors: K 个 (3, H, W) 图像, 可带梯度。
            pair_indices:  K 个整数, 对应 token_cache 中的行索引。
            token_cache:   precompute_token_cache() 的返回值。

        返回:
            loss: 标量张量, 保持完整梯度图。
        """
        K = len(image_tensors)
        target_h = token_cache["diff_target_h"]
        target_w = token_cache["diff_target_w"]

        # 1. 可微图像预处理 (纯 GPU, 缓存同一 tensor)
        diff_cache = {}
        all_diff_pvs = []
        all_diff_grids = []

        for i in range(K):
            img = image_tensors[i]
            tid = id(img)
            if tid not in diff_cache:
                if img.shape[1] != target_h or img.shape[2] != target_w:
                    img_r = F.interpolate(
                        img.unsqueeze(0), size=(target_h, target_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                else:
                    img_r = img
                diff_cache[tid] = self._differentiable_image_process(img_r)

            pv, grid = diff_cache[tid]
            all_diff_pvs.append(pv)
            all_diff_grids.append(grid)

        # 2. GPU 索引取 token (纯整数, 零 dict 查找)
        idx_t = torch.tensor(
            pair_indices, dtype=torch.long, device=self.device
        )
        batch_input_ids = token_cache["gpu_ids"][idx_t]
        batch_attention_mask = token_cache["gpu_masks"][idx_t]
        batch_labels = token_cache["gpu_labels"][idx_t]

        # 3. 拼接 pixel_values (GPU 端)
        pixel_values = torch.cat(
            all_diff_pvs, dim=0
        ).to(self.device, dtype=self.model_dtype)
        image_grid_thw = torch.cat(
            all_diff_grids, dim=0
        ).to(self.device)

        # 4. 单次 GPU Forward (无任何 CPU 阻塞)
        use_autocast = self.device != "cpu" and torch.cuda.is_available()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=batch_labels,
            )

        return outputs.loss

    # ==================== 极简 GPU 前向 =======================

    def forward_precomputed(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        pair_indices: torch.Tensor,
        token_cache: dict,
    ) -> torch.Tensor:
        """
        极简前向: 接受上游预构建的 pixel_values 和 GPU 索引张量。

        所有预处理 (adv 构建、diff_process、采样) 在上游完成,
        本方法只做: GPU 索引取 token + model.forward(),
        从 CPU 视角几乎零开销, GPU 保持满载。

        参数:
            pixel_values:   (total_patches, patch_dim), GPU, 可带梯度。
            image_grid_thw: (K, 3), GPU, 图像网格信息。
            pair_indices:   (K,), long, GPU, token_cache 行索引。
            token_cache:    precompute_token_cache() 的返回值。

        返回:
            loss: 标量张量, 保持完整梯度图。
        """
        batch_input_ids = token_cache["gpu_ids"][pair_indices]
        batch_attention_mask = token_cache["gpu_masks"][pair_indices]
        batch_labels = token_cache["gpu_labels"][pair_indices]

        use_autocast = self.device != "cpu" and torch.cuda.is_available()
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=batch_labels,
            )
        return outputs.loss

    # ==================== 核心方法: generate ==================

    def generate(
        self,
        image_tensor: torch.Tensor,
        prompt_text: str,
        history: List[str],
        max_new_tokens: int = 128,
    ) -> str:
        """
        生成模型响应 (用于评估, 不需要梯度)。

        参数:
            image_tensor: (3, H, W), [0, 1], 对抗图像。
            prompt_text: 当前任务指令。
            history: 历史动作序列。
            max_new_tokens: 最大生成 token 数。

        返回:
            response: 模型生成的字符串。
        """
        with torch.no_grad():
            # 转换为 PIL 图像
            pil_img = to_pil_image(
                image_tensor.cpu().float().clamp(0, 1)
            )

            # 格式化消息
            messages = self._format_messages(prompt_text, history)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 使用 processor 处理 (非微分, 无需手动 patch)
            inputs = self.processor(
                text=[text], images=[pil_img], return_tensors="pt"
            )
            inputs = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            # 生成
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪心解码, 保证可复现
            )

            # 截取生成部分 (去掉 prompt)
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]
            response = self.processor.decode(
                output_ids[0], skip_special_tokens=True
            )
            if self.normalize_generation_to_dsl:
                return self._normalize_generated_action(response)
            return response.strip()

    # ==================== 批量生成 ============================

    def generate_batch(
        self,
        image_tensor: torch.Tensor,
        pairs: List[Tuple[str, List[str]]],
        max_new_tokens: int = 128,
    ) -> List[str]:
        """
        批量生成 (用于评估)。

        注意: 这里所有 pair 共享同一张 image_tensor (同一网页),
        因此图像 token 长度一致, 可安全执行真正的 batch generate。

        参数:
            image_tensor: (3, H, W), 对抗图像 (所有 pair 共享)。
            pairs: [(prompt, history), ...] 列表。
            max_new_tokens: 最大生成 token 数。

        返回:
            responses: 生成的字符串列表。
        """
        if not pairs:
            return []

        with torch.no_grad():
            pil_img = to_pil_image(image_tensor.cpu().float().clamp(0, 1))

            messages_list = [
                self._format_messages(prompt, history)
                for prompt, history in pairs
            ]
            texts = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_list
            ]

            inputs = self.processor(
                text=texts,
                images=[pil_img for _ in pairs],
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            responses = []
            for i, input_len in enumerate(input_lens):
                output_ids = generated_ids[i, int(input_len):]
                response = self.processor.decode(
                    output_ids, skip_special_tokens=True
                )
                if self.normalize_generation_to_dsl:
                    responses.append(self._normalize_generated_action(response))
                else:
                    responses.append(response.strip())

            return responses

    # ==================== 显存清理 ============================

    def cleanup(self) -> None:
        """释放 GPU 缓存 (在切换网页时调用)。"""
        import gc
        gc.collect()  # 先释放 Python 侧引用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
