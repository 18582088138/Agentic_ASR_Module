"""引擎层 / Engine layer —— 抽象、注册表与具体引擎实现。

具体引擎通过 `core.registry` **延迟导入**，所以 `from agentic_asr.engines import available`
不会把 faster-whisper / sherpa-onnx 拉进来。
"""

from agentic_asr.core.registry import available, resolve
from agentic_asr.engines.base import ASREngine

__all__ = ["ASREngine", "available", "resolve"]
