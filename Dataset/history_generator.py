# ============================================================
# history_generator.py — 随机动作历史序列生成器
# ============================================================
# 功能：
#   从 config.ACTION_SPACE 中随机采样动作，
#   并为需要参数的动作填充随机坐标 / 内容，
#   生成 Shadow History 和 User History。
# ============================================================

import random
import string

import config


def _random_coord(max_x: int = 1920, max_y: int = 1080) -> str:
    """生成一个随机坐标字符串 (x, y)。"""
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return f"({x}, {y})"


def _random_content(length: int = 8) -> str:
    """生成一个随机字符串，模拟用户键入内容。"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _random_key_combo() -> str:
    """生成一个随机的快捷键组合。"""
    combos = ["ctrl+c", "ctrl+v", "ctrl+a", "ctrl+z", "alt+tab", "ctrl+s", "enter", "escape"]
    return random.choice(combos)


def _random_scroll_direction() -> str:
    """生成一个随机的滚动方向。"""
    return random.choice(["up", "down", "left", "right"])


def _fill_action(action_template: str) -> str:
    """
    根据动作模板填充具体的随机参数。

    参数:
        action_template : 动作模板字符串（来自 config.ACTION_SPACE）

    返回:
        带有具体参数的动作字符串
    """
    # 需要单组坐标的动作: click, left_double, right_single
    if action_template in ("click((x,y))", "left_double((x,y))", "right_single((x,y))"):
        action_name = action_template.split("(")[0]
        coord = _random_coord()
        return f"{action_name}({coord})"

    # 需要两组坐标的动作: drag
    if action_template == "drag((x1,y1),(x2,y2))":
        c1 = _random_coord()
        c2 = _random_coord()
        return f"drag({c1}, {c2})"

    # 需要快捷键组合: hotkey
    if action_template == "hotkey(key_comb)":
        combo = _random_key_combo()
        return f"hotkey({combo})"

    # 需要文本内容: type
    if action_template == "type(content)":
        content = _random_content()
        return f'type("{content}")'

    # 需要方向: scroll
    if action_template == "scroll(direction)":
        direction = _random_scroll_direction()
        return f"scroll({direction})"

    # 无需额外参数: wait, finished, call_user
    return action_template.replace("()", "()")


# ============================================================
# 公共接口
# ============================================================
def generate_history(num_steps: int = 3) -> list:
    """
    生成一个随机的动作序列。

    参数:
        num_steps : 动作序列长度（步数）

    返回:
        动作字符串列表，例如 ["scroll(down)", "click((200, 300))", "wait()"]
    """
    actions = []
    for _ in range(num_steps):
        template = random.choice(config.ACTION_SPACE)
        filled = _fill_action(template)
        actions.append(filled)
    return actions


def generate_shadow_histories(num_histories: int = None, steps_per_history: int = 3) -> list:
    """
    批量生成影子历史（Shadow History）。
    影子历史是攻击者伪造的、用于欺骗 Web Agent 的虚假动作序列。

    参数:
        num_histories     : 生成数量（默认读取 config.NUM_SHADOW_HISTORY）
        steps_per_history : 每条历史的步数

    返回:
        二维列表，每个元素为一个动作序列
    """
    if num_histories is None:
        num_histories = config.NUM_SHADOW_HISTORY

    histories = [generate_history(steps_per_history) for _ in range(num_histories)]
    print(f"[HistoryGen] 已生成 {num_histories} 条 Shadow History（每条 {steps_per_history} 步）。")
    return histories


def generate_user_histories(num_histories: int = None, steps_per_history: int = 3) -> list:
    """
    批量生成用户历史（User History）。
    用户历史是模拟真实用户操作的动作序列。

    参数:
        num_histories     : 生成数量（默认读取 config.NUM_USER_HISTORY）
        steps_per_history : 每条历史的步数

    返回:
        二维列表，每个元素为一个动作序列
    """
    if num_histories is None:
        num_histories = config.NUM_USER_HISTORY

    histories = [generate_history(steps_per_history) for _ in range(num_histories)]
    print(f"[HistoryGen] 已生成 {num_histories} 条 User History（每条 {steps_per_history} 步）。")
    return histories
