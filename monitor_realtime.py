#!/usr/bin/env python3
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple


def read_proc_stat() -> Tuple[int, int]:
    with open('/proc/stat', 'r', encoding='utf-8') as file:
        parts = file.readline().split()[1:]
    values = [int(item) for item in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def cpu_percent(prev_total: int, prev_idle: int) -> Tuple[float, int, int]:
    total, idle = read_proc_stat()
    total_delta = total - prev_total
    idle_delta = idle - prev_idle
    if total_delta <= 0:
        return 0.0, total, idle
    usage = (total_delta - idle_delta) / total_delta * 100.0
    return max(0.0, min(100.0, usage)), total, idle


def memory_info() -> Tuple[float, float, float]:
    mem_total = 0
    mem_available = 0
    with open('/proc/meminfo', 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1])
    used = max(0, mem_total - mem_available)
    return used / 1024 / 1024, mem_total / 1024 / 1024, (used / mem_total * 100.0 if mem_total else 0.0)


def read_net_bytes(iface: str = '') -> Dict[str, Tuple[int, int]]:
    result: Dict[str, Tuple[int, int]] = {}
    with open('/proc/net/dev', 'r', encoding='utf-8') as file:
        lines = file.readlines()[2:]
    for line in lines:
        name, data = line.split(':', 1)
        name = name.strip()
        if name == 'lo':
            continue
        if iface and name != iface:
            continue
        parts = data.split()
        rx = int(parts[0])
        tx = int(parts[8])
        result[name] = (rx, tx)
    return result


def pretty_rate(bytes_per_sec: float) -> str:
    units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
    value = max(0.0, bytes_per_sec)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f'{value:7.2f} {units[idx]}'


def get_gpu_info() -> List[str]:
    if not shutil.which('nvidia-smi'):
        return ['GPU: nvidia-smi 不可用']
    try:
        proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        lines = [item.strip() for item in proc.stdout.splitlines() if item.strip()]
        if not lines:
            return ['GPU: 未检测到设备']
        out = []
        for line in lines:
            parts = [item.strip() for item in line.split(',')]
            if len(parts) < 7:
                out.append(f'GPU: {line}')
                continue
            idx, name, util, used, total, temp, power = parts[:7]
            mem_pct = (float(used) / float(total) * 100.0) if float(total) > 0 else 0.0
            out.append(
                f'GPU{idx} {name[:28]:<28} | 利用率 {float(util):5.1f}% | 显存 {float(used):7.0f}/{float(total):.0f} MB ({mem_pct:5.1f}%) | 温度 {float(temp):4.0f}°C | 功耗 {float(power):6.1f} W'
            )
        return out
    except Exception as exc:
        return [f'GPU: 读取失败 ({exc})']


def get_gpu_utilizations() -> Dict[str, float]:
    """返回每块 GPU 的利用率 {gpu_id: util%}，失败时返回空字典。"""
    if not shutil.which('nvidia-smi'):
        return {}
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=2,
        )
        result: Dict[str, float] = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                result[parts[0]] = float(parts[1])
        return result
    except Exception:
        return {}


def render_gpu_chart(history: deque, width: int = 70, height: int = 10) -> List[str]:
    """
    用 Unicode 方块字符绘制 GPU 利用率变化曲线图。
    history: deque of float (0-100)
    返回多行字符串列表。
    """
    BLOCKS = ' ▁▂▃▄▅▆▇█'
    lines: List[str] = []

    data = list(history)
    if not data:
        lines.append('  (暂无数据)')
        return lines

    # 取最近 width 个数据点
    if len(data) > width:
        data = data[-width:]

    # 纵轴: 0% ~ 100%，分成 height 行
    y_max = 100.0
    y_min = 0.0
    row_range = (y_max - y_min) / height

    for row in range(height, 0, -1):
        row_top = y_min + row * row_range
        row_bot = y_min + (row - 1) * row_range
        # Y 轴标签
        if row == height:
            label = f'{y_max:5.0f}% │'
        elif row == 1:
            label = f'{y_min:5.0f}% │'
        elif row == height // 2 + 1:
            mid_val = (y_max + y_min) / 2
            label = f'{mid_val:5.0f}% │'
        else:
            label = '      │'

        row_chars = []
        for val in data:
            if val >= row_top:
                row_chars.append(BLOCKS[8])  # █ 全填充
            elif val <= row_bot:
                row_chars.append(BLOCKS[0])  # 空
            else:
                # 部分填充
                frac = (val - row_bot) / row_range
                idx = int(frac * 8)
                idx = max(1, min(8, idx))
                row_chars.append(BLOCKS[idx])

        lines.append(label + ''.join(row_chars))

    # X 轴线
    lines.append('      └' + '─' * len(data))

    # 时间标注线
    n = len(data)
    time_label = f'最近 {n} 个采样点'
    pad = max(0, 7 + n - len(time_label)) // 2
    lines.append(' ' * (7 + pad) + time_label)

    return lines


def clear_screen() -> None:
    sys.stdout.write('\033[2J\033[H')


def monitor(interval: float, iface: str = '', history_len: int = 60) -> None:
    last_total, last_idle = read_proc_stat()
    last_net = read_net_bytes(iface)
    last_time = time.time()

    # GPU 利用率历史（每块 GPU 独立记录）
    gpu_util_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_len))

    def _exit_handler(_sig, _frame):
        print('\n已退出监控。')
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    while True:
        time.sleep(interval)
        now = time.time()
        dt = max(0.001, now - last_time)

        cpu, last_total, last_idle = cpu_percent(last_total, last_idle)
        used_gb, total_gb, mem_pct = memory_info()

        current_net = read_net_bytes(iface)
        net_lines = []
        all_ifaces = sorted(current_net.keys())
        if not all_ifaces:
            net_lines.append('NET: 未检测到非 lo 网卡')
        else:
            for name in all_ifaces:
                prev_rx, prev_tx = last_net.get(name, current_net[name])
                cur_rx, cur_tx = current_net[name]
                rx_rate = (cur_rx - prev_rx) / dt
                tx_rate = (cur_tx - prev_tx) / dt
                net_lines.append(f'{name:<10} ↓ {pretty_rate(rx_rate)} | ↑ {pretty_rate(tx_rate)}')

        gpu_lines = get_gpu_info()

        # 采集 GPU 利用率并记录历史
        gpu_utils = get_gpu_utilizations()
        for gpu_id, util_val in gpu_utils.items():
            gpu_util_history[gpu_id].append(util_val)

        # clear_screen()
        os.system('clear')
        print('=== Real-time System Monitor ===')
        print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")} | Interval: {interval:.1f}s')
        print('-' * 90)
        print(f'CPU     : {cpu:6.2f}%')
        print(f'Memory  : {used_gb:6.2f}/{total_gb:.2f} GB ({mem_pct:5.1f}%)')
        print('-' * 90)
        print('Network :')
        for line in net_lines:
            print(f'  {line}')
        print('-' * 90)
        print('GPU     :')
        for line in gpu_lines:
            print(f'  {line}')

        # 显示 GPU 利用率变化曲线
        if gpu_util_history:
            print('-' * 90)
            print('GPU 利用率变化曲线 :')
            for gpu_id in sorted(gpu_util_history.keys()):
                hist = gpu_util_history[gpu_id]
                if len(hist) < 2:
                    continue
                cur_val = hist[-1] if hist else 0.0
                avg_val = sum(hist) / len(hist) if hist else 0.0
                max_val = max(hist) if hist else 0.0
                min_val = min(hist) if hist else 0.0
                print(f'  ┌─ GPU{gpu_id}  当前: {cur_val:5.1f}%  平均: {avg_val:5.1f}%  '
                      f'最大: {max_val:5.1f}%  最小: {min_val:5.1f}%')
                chart_lines = render_gpu_chart(hist, width=history_len, height=10)
                for cl in chart_lines:
                    print(f'  {cl}')
                print()

        print('按 Ctrl+C 退出')

        last_net = current_net
        last_time = now


def main() -> None:
    parser = argparse.ArgumentParser(description='实时监控 GPU/CPU/内存/网速（零依赖）')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='刷新间隔（秒），默认 1.0')
    parser.add_argument('--iface', type=str, default='', help='指定网卡名（如 eth0），默认显示全部非 lo 网卡')
    parser.add_argument('--history', type=int, default=60, help='GPU 曲线图保留的历史采样点数（默认 60）')
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit('interval 必须 > 0')
    monitor(args.interval, args.iface, args.history)


if __name__ == '__main__':
    main()