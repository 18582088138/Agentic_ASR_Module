"""异常体系 / Error hierarchy.

所有异常都继承 `ASRError`，于是 CLI / HTTP / GUI 只 catch 一个基类就能给出统一
的错误响应，而调用方仍可按子类精确处理。
All errors derive from `ASRError`, so the CLI / HTTP / GUI layers can catch one
base class for a uniform error response while callers still handle specifics.

内部约定 / internal rule：**能力不支持一律报错，绝不静默降级** ——
静默降级会让上层拿到一份看起来正常、实际上缺字段的结果，是最贵的一类 bug。
"""

from __future__ import annotations


class ASRError(Exception):
    """本模块所有异常的基类 / Base class of every error in this module."""


class ConfigError(ASRError):
    """配置缺失或非法 / Missing or invalid configuration."""


class CapabilityError(ASRError):
    """当前引擎不支持所请求的能力 / The current engine lacks the requested capability."""


class EngineError(ASRError):
    """引擎加载或推理失败 / Engine failed to load or to run."""


class DecodeError(ASRError):
    """音频解码 / 重采样失败 / Audio decoding or resampling failed."""


class GuardError(ASRError):
    """幻觉闸门拒绝处理该输入 / The hallucination guard refused this input."""


class APIError(ASRError):
    """云 API 调用失败 / A cloud API call failed.

    参数 / Args:
        status: HTTP 状态码；本地错误为 None。
        retryable: 是否值得重试（429 / 5xx 为 True）。
    """

    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


__all__ = [
    "APIError",
    "ASRError",
    "CapabilityError",
    "ConfigError",
    "DecodeError",
    "EngineError",
    "GuardError",
]
