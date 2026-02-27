# ============================================================
# pipeline_state.py — 流水线状态管理（断点续传）
# ============================================================
# 使用 JSON 文件持久化每个 Phase 的完成状态，
# 重新运行时自动跳过已完成的阶段。
#
# 云端抢占式实例随时可能中断，此模块确保：
#   - 每个阶段完成后立即持久化状态
#   - 重启后自动从上次中断处继续
#   - 支持手动重置状态文件
# ============================================================

import json
import os

import config

# 所有阶段的名称
PHASES = [
    "phase0_crawl",       # 真实网页采集
    "phase1_synth",       # 合成网页生成
    "phase2_prompts",     # Prompt 生成
    "phase3_history",     # 历史生成
    "phase4_render",      # 渲染截图
    "phase5_metadata",    # 元数据汇总
]


def load_state() -> dict:
    """
    加载流水线状态。

    返回:
        {
          "completed_phases": ["phase0_crawl", ...],
          "phase_data": {<phase_name>: <any serializable data>}
        }

    断点续传核心逻辑：
      - 如果状态文件存在且合法，加载并返回
      - 如果状态文件损坏，打印警告并返回空状态（重新开始）
      - 如果状态文件不存在，返回空状态
    """
    if os.path.exists(config.PIPELINE_STATE_FILE):
        try:
            with open(config.PIPELINE_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                print(f"[State] 已加载断点状态: {config.PIPELINE_STATE_FILE}")
                completed = state.get("completed_phases", [])
                if completed:
                    print(f"[State] 已完成的阶段: {', '.join(completed)}")
                return state
        except (json.JSONDecodeError, IOError) as e:
            print(f"[State] [WARN] 状态文件损坏，将重新开始: {e}")

    return {"completed_phases": [], "phase_data": {}}


def save_state(state: dict):
    """
    将流水线状态保存到磁盘。
    每次阶段完成后必须调用，确保中断后可恢复。
    """
    os.makedirs(os.path.dirname(config.PIPELINE_STATE_FILE), exist_ok=True)
    with open(config.PIPELINE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def mark_completed(state: dict, phase_name: str, phase_data=None):
    """
    标记一个阶段为已完成，并立即持久化到磁盘。

    参数:
        state      : 当前状态 dict
        phase_name : 阶段名称
        phase_data : 该阶段产生的可序列化数据（可选）
    """
    if phase_name not in state["completed_phases"]:
        state["completed_phases"].append(phase_name)
    if phase_data is not None:
        state["phase_data"][phase_name] = phase_data
    save_state(state)
    print(f"[State] 阶段 {phase_name} 已标记完成并保存")


def is_completed(state: dict, phase_name: str) -> bool:
    """检查某个阶段是否已完成。"""
    return phase_name in state.get("completed_phases", [])


def reset_state():
    """重置流水线状态（删除状态文件），从头开始。"""
    if os.path.exists(config.PIPELINE_STATE_FILE):
        os.remove(config.PIPELINE_STATE_FILE)
        print("[State] 流水线状态已重置")
