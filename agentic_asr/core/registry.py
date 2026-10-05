"""引擎注册表 / Engine registry.

**延迟导入**是这里唯一重要的事：列一遍引擎名不该把 faster-whisper、ctranslate2、
sherpa-onnx 全都 import 进来（那要几秒，还会占内存）。
表里存的是 `模块:类名` 字符串，真正构造时才导入。
The table stores `module:Class` strings so listing engines stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from agentic_asr.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from agentic_asr.engines.base import ASREngine

# 引擎名 -> "模块:类名" / engine name -> "module:ClassName"
ENGINES: dict[str, str] = {
    "faster_whisper": "agentic_asr.engines.faster_whisper:FasterWhisperEngine",
    "sensevoice": "agentic_asr.engines.sensevoice:SenseVoiceEngine",
    "minimax": "agentic_asr.engines.minimax:MiniMaxEngine",
    "openai_compat": "agentic_asr.engines.openai_compat:OpenAICompatEngine",
    "mock": "agentic_asr.engines.mock:MockEngine",
}

# 零重依赖、无需权重的引擎 —— GUI/HTTP 开机与离线测试默认用它
# Depend-free engines usable without any weights; the GUI/HTTP default.
LIGHTWEIGHT = ("mock",)


def available() -> list[str]:
    """所有已登记的引擎名 / all registered engine names."""
    return sorted(ENGINES)


def resolve(name: str) -> type[ASREngine]:
    """按名字取引擎类 / resolve an engine class by name.

    抛出 / Raises:
        ConfigError: 名字未登记，或依赖缺失。
    """
    target = ENGINES.get(name)
    if target is None:
        raise ConfigError(
            f"未知引擎 / unknown engine {name!r}　可用 / available: {', '.join(available())}")
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # 依赖没装
        raise ConfigError(
            f"引擎 {name!r} 的依赖未安装 / missing dependency for {name!r}: {exc}"
            f"　提示 / hint: pip install -e \".[all]\"") from exc
    return getattr(module, class_name)


__all__ = ["ENGINES", "LIGHTWEIGHT", "available", "resolve"]
