"""日志 / Logging.

一个进程内共享的 logger。**日志里永远不出现密钥** —— 云 API 适配器只打印
endpoint、模型名与状态码，不打印 Authorization 头（用户会截屏）。
"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME = "agentic_asr"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    level = os.environ.get("ASR_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"))
    root = logging.getLogger(_LOGGER_NAME)
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(suffix: str = "") -> logging.Logger:
    """取子 logger / get a child logger."""
    _configure()
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}" if suffix else _LOGGER_NAME)


__all__ = ["get_logger"]
