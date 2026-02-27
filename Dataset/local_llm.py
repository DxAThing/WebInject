# ============================================================
# local_llm.py — 本地轻量化大模型推理封装
# ============================================================
# 使用 Qwen2.5-7B-Instruct 替代 OpenAI API，实现纯本地化推理。
# 关键优化：
#   - bfloat16 精度加载，减少显存占用
#   - flash_attention_2 加速长文本推理
#   - 单例模式避免重复加载模型
#   - 统一的 generate() 接口供其他模块调用
# ============================================================

import torch
from typing import Optional

import config

# ======================== 全局单例 ========================
# 模型和分词器只加载一次，后续调用复用同一实例
_model = None
_tokenizer = None


def _load_model():
    """
    懒加载本地大模型和分词器。
    使用 bfloat16 精度 + flash_attention_2 以优化显存与推理速度。

    此函数仅在首次调用时执行实际加载，后续调用直接返回缓存实例。
    """
    global _model, _tokenizer

    # 如果已加载则直接返回，避免重复初始化
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.LOCAL_MODEL_NAME
    print(f"[LocalLLM] 正在加载模型: {model_name}")
    print(f"[LocalLLM]   精度: {config.LOCAL_MODEL_DTYPE}")
    print(f"[LocalLLM]   注意力: {config.LOCAL_MODEL_ATTN}")
    print(f"[LocalLLM]   设备映射: {config.LOCAL_MODEL_DEVICE}")

    # 加载分词器
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 确定加载精度：必须使用 bfloat16
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.LOCAL_MODEL_DTYPE, torch.bfloat16)

    # 加载模型：开启 flash_attention_2 加速长文本推理
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config.LOCAL_MODEL_DEVICE,       # "auto" 让 accelerate 自动分配 GPU/CPU
        attn_implementation=config.LOCAL_MODEL_ATTN, # flash_attention_2
        trust_remote_code=True,
    )

    # 切换到推理模式，禁用梯度计算
    _model.eval()

    print(f"[LocalLLM] 模型加载完成: {model_name}")
    return _model, _tokenizer


def generate(prompt: str, system_prompt: Optional[str] = None,
             max_new_tokens: Optional[int] = None,
             temperature: Optional[float] = None) -> str:
    """
    调用本地大模型生成文本。

    参数:
        prompt         : 用户输入的提示文本
        system_prompt  : 系统级提示（可选，用于设定模型角色）
        max_new_tokens : 最大生成 token 数（默认读取 config）
        temperature    : 采样温度（默认读取 config）

    返回:
        模型生成的文本字符串（已去除输入部分）
    """
    if max_new_tokens is None:
        max_new_tokens = config.LOCAL_MODEL_MAX_NEW_TOKENS
    if temperature is None:
        temperature = config.LOCAL_MODEL_TEMPERATURE

    model, tokenizer = _load_model()

    # ----------------------------------------------------------------
    # 构建对话消息列表（Qwen2.5 使用 ChatML 格式）
    # ----------------------------------------------------------------
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # 使用 tokenizer 的 apply_chat_template 方法构建输入
    # 这会自动添加 <|im_start|> / <|im_end|> 等特殊标记
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ----------------------------------------------------------------
    # 分词并移至模型所在设备
    # ----------------------------------------------------------------
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    input_length = model_inputs["input_ids"].shape[1]

    # ----------------------------------------------------------------
    # 生成文本（禁用梯度计算以节省显存）
    # ----------------------------------------------------------------
    with torch.no_grad():
        # 根据温度选择采样或贪婪策略
        if temperature > 0:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=config.LOCAL_MODEL_TOP_P,
                do_sample=True,
            )
        else:
            # temperature=0 时使用贪婪解码
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    # ----------------------------------------------------------------
    # 解码输出：只取新生成的 token（去除输入部分）
    # ----------------------------------------------------------------
    output_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return response


def unload_model():
    """
    显式释放模型和分词器，回收 GPU 显存。
    在流水线结束或不再需要大模型时调用。
    """
    global _model, _tokenizer

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[LocalLLM] 模型已卸载，GPU 显存已释放")
