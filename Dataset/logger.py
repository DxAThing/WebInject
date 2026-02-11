# ============================================================
# logger.py -- 双通道日志输出（控制台 + 文件）
# ============================================================
# 将所有 print / stderr 输出同时写入控制台和带时间戳的日志文件。
# 每次运行自动生成独立日志文件，便于回溯和对比。
# ============================================================

import datetime
import os
import sys


class TeeStream:
    """同时写入多个流的包装器。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        # 返回第一个流（控制台）的 fd，兼容部分库的 isatty() 检查
        return self.streams[0].fileno()

    def isatty(self):
        return hasattr(self.streams[0], "isatty") and self.streams[0].isatty()


_log_file = None  # 保持引用防止 GC


def setup_logging(log_dir: str) -> str:
    """
    设置双通道日志：所有 print 输出同时写入控制台和日志文件。

    日志文件名包含时间戳，每次运行生成独立文件。

    参数:
        log_dir : 日志文件存放目录

    返回:
        日志文件的绝对路径
    """
    global _log_file

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    _log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = TeeStream(sys.__stdout__, _log_file)  # type: ignore[assignment]
    sys.stderr = TeeStream(sys.__stderr__, _log_file)  # type: ignore[assignment]

    return log_path


def shutdown_logging():
    """恢复标准输出并关闭日志文件。"""
    global _log_file

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None
