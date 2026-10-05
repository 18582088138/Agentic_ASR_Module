"""引擎抽象 / Engine abstraction.

**一个引擎只需要会做一件事：一段音频 → 一份 `Transcript`。**
An engine does exactly one thing: audio in, a `Transcript` out.

为什么抽象面要这么窄 / why the interface is this narrow：
时间轴平移、切片合并、字幕渲染、无语音防护、情绪补齐这五件事**与引擎无关**。
一旦把它们下沉到各引擎，就会出现 N 份互相偷偷不一致的实现 —— 同一段音频
换个引擎得到不同的切法，调试时看不出是哪一份在起作用。
Everything engine-independent lives in the facade (`asr.py`) and is shared.

于是新增一个引擎的成本是：实现 `transcribe()` + 声明 `declared_capabilities()`。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentic_asr.core.errors import CapabilityError
from agentic_asr.core.types import AudioChunk, Capability, Transcript

if TYPE_CHECKING:  # pragma: no cover
    from agentic_asr.core.config import ASRConfig


class ASREngine(ABC):
    """一个 ASR 引擎 / One ASR engine.

    权重一律**懒加载** —— 构造实例不该花几秒，否则 GUI 开机就会卡住。
    Weights are always lazy-loaded; constructing an engine must be instant.
    """

    name: str = ""
    implemented: bool = True
    # 单次调用能吃的最长音频（秒）；None = 本地不限，由门面按需切片。
    # Longest audio this engine accepts per call; None means no local limit.
    max_audio_seconds: float | None = None
    # 是否已经加载了权重（门面据此决定要不要卸载）
    loaded: bool = False

    def __init__(self, config: ASRConfig) -> None:
        self.config = config

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def declared_capabilities(cls) -> set[Capability]:
        """静态能力声明 / static capability declaration.

        **必须不加载权重就能回答** —— GUI 与 HTTP 在开机时据此把不支持的开关
        置灰；不可能为了查能力先把几百 MB 权重载进来。
        Must be answerable without loading weights.
        """

    def capabilities(self) -> set[Capability]:
        """实例能力（默认等于静态声明）/ instance capabilities."""
        return self.declared_capabilities()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def require(self, capability: Capability, hint: str = "") -> None:
        """不支持就报错，**绝不静默降级** / fail loudly instead of degrading silently.

        抛出 / Raises:
            CapabilityError: 当前引擎不支持该能力。
        """
        if not self.supports(capability):
            supported = "、".join(sorted(c.value for c in self.capabilities())) or "（无）"
            raise CapabilityError(
                f"引擎 {self.name!r} 不支持 {capability.value}　已支持 / supported: {supported}"
                + (f"　{hint}" if hint else ""))

    # ── 转写 / transcription ────────────────────────────────────────────────

    @abstractmethod
    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        """把一段音频转成文本 / transcribe one chunk of audio.

        参数 / Args:
            chunk: 已解码归一的一段音频（单声道 float32）。**引擎不应再去读文件。**
            language: `auto` 或具体语种码（`zh` / `en` / `ja` …）。
            word_timestamps: 是否要求词/字级时间戳；不支持时应 `require()` 报错，
                而不是悄悄返回句级结果。

        返回 / Returns:
            `Transcript`，`segments[].start/end` 必须是**相对这段 chunk 的秒数**；
            门面负责加上 `chunk.offset` 平移回原始时间轴。

        抛出 / Raises:
            CapabilityError: 请求了引擎不具备的能力。
            EngineError: 加载或推理失败。
            APIError: 云调用失败。
        """

    def detect_emotion(self, chunk: AudioChunk) -> str | None:
        """可选：给一段音频打情绪标签 / optional per-chunk emotion label.

        默认返回 None（不具备该能力）。门面在主力引擎不支持情绪时，
        会**按 segment 逐段**调用支持情绪的引擎来补齐（组合拳）。
        """
        return None

    # ── 元信息与生命周期 / metadata & lifecycle ─────────────────────────────

    def languages(self) -> list[str]:
        """支持的语种（静态表打底）/ supported languages, statically."""
        return []

    def info(self) -> dict[str, Any]:
        """身份快照，进结果与 `/engines` / an identity snapshot."""
        return {
            "engine": self.name,
            "implemented": self.implemented,
            "loaded": self.loaded,
            "max_audio_seconds": self.max_audio_seconds,
            "capabilities": sorted(c.value for c in self.capabilities()),
        }

    def release(self) -> None:  # noqa: B027 - 可选钩子：默认无事可做，子类按需覆写
        """卸载权重、释放显存（默认无事可做）/ unload weights and free memory.

        **刻意不是 abstractmethod**：绝大多数引擎没有常驻状态要放，
        强制每个引擎写一个空实现只是噪音。
        Deliberately not abstract: most engines hold nothing to release.
        """

    def __enter__(self) -> ASREngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


__all__ = ["ASREngine"]
