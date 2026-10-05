"""Agentic_ASR_Module —— 可插拔的本地优先 ASR 服务插件。

五个核心功能 / five core functions，每一个都能独立调用：

| 功能 | 入口 |
|---|---|
| ① 音频信息提取 | `ASRModule.probe()` |
| ② 音频分段 | `ASRModule.segment()` |
| ③ 字幕生成 | `ASRModule.transcribe()` / `Transcript` → `subtitle.write_subtitle()` |
| ④ 语音情绪识别 | `ASRModule.transcribe(emotion=True)` |
| ⑤ 参考音频截取 | `ASRModule.clip_reference()` |

四种用法共用同一个门面 / four usages over one facade：
Python 库 · CLI(`asr …`) · HTTP(`python -m agentic_asr.server`) · GUI(`python -m agentic_asr.gui`)。

快速开始 / quick start::

    from agentic_asr import ASRModule

    asr = ASRModule()
    info = asr.probe("demo.mp4")
    result = asr.transcribe("demo.mp4", emotion=True)
    result.to_srt("outputs/demo.srt") if hasattr(result, "to_srt") else None
    clips = asr.clip_reference("demo.mp4", top=3)
    asr.release()
"""

from agentic_asr.asr import ASRModule
from agentic_asr.core.config import ASRConfig, load_config
from agentic_asr.core.errors import (
    APIError,
    ASRError,
    CapabilityError,
    ConfigError,
    DecodeError,
    EngineError,
    GuardError,
)
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

__version__ = "0.1.0"

__all__ = [
    "APIError",
    "ASRConfig",
    "ASRError",
    "ASRModule",
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
    "__version__",
    "load_config",
]
