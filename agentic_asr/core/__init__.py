"""核心层 / Core layer —— 配置、类型、异常、日志、引擎注册表."""

from agentic_asr.core.errors import (
    APIError,
    ASRError,
    CapabilityError,
    ConfigError,
    DecodeError,
    EngineError,
    GuardError,
)
from agentic_asr.core.logging import get_logger
from agentic_asr.core.registry import available, resolve
from agentic_asr.core.types import (
    AudioChunk,
    AudioInfo,
    Capability,
    ClipResult,
    Emotion,
    Segment,
    SegmentPiece,
    Transcript,
    Word,
)

__all__ = [
    "APIError",
    "ASRError",
    "AudioChunk",
    "AudioInfo",
    "Capability",
    "CapabilityError",
    "ClipResult",
    "ConfigError",
    "DecodeError",
    "Emotion",
    "EngineError",
    "GuardError",
    "Segment",
    "SegmentPiece",
    "Transcript",
    "Word",
    "available",
    "get_logger",
    "resolve",
]
