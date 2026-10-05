"""离线占位引擎 / Offline placeholder engine.

**零重依赖、零权重**，让库 / CLI / HTTP / GUI / 单测在不加载任何模型的情况下
跑通整条链路。这不是玩具：`docs/02_dev_plan.md` 的 V1 验收就是
「全链路不加载权重可跑通」，靠的就是它。

输出是**确定性**的（同一段音频永远得到同样的结果），因此可以作为断言基准；
真实引擎的接入测试则用 `-m real` 标记单独跑。
Deterministic output makes it usable as a test baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agentic_asr.core.types import (
    AudioChunk,
    Capability,
    Emotion,
    Segment,
    Transcript,
    Word,
)
from agentic_asr.engines.base import ASREngine

# 情绪按段轮转，便于测试「情绪汇总」这条路径 / rotate emotions for summary tests
_EMOTION_CYCLE = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.SAD, Emotion.ANGRY]


class MockEngine(ASREngine):
    """确定性占位引擎 / deterministic placeholder engine."""

    name = "mock"
    implemented = True
    max_audio_seconds: float | None = None

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.loaded = True
        # 每段的目标时长，测试可覆盖 / segment length, overridable for tests
        self.segment_seconds = float(getattr(config, "mock_segment_seconds", 2.0) or 2.0)

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.TIMESTAMPS,
            Capability.WORD_TIMESTAMPS,
            Capability.LANGUAGE_ID,
            Capability.EMOTION,
            Capability.EVENTS,
        }

    def languages(self) -> list[str]:
        return ["zh", "en", "ja", "ko", "yue"]

    # ── 转写 / transcription ────────────────────────────────────────────────

    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        """把音频时长切成等长的段，填确定性文本 / deterministic fake transcript."""
        duration = chunk.duration
        lang = "zh" if language in ("auto", "") else language
        if duration <= 0:
            return Transcript.empty(self.name, 0.0, "empty audio")

        n_segments = max(1, int(round(duration / self.segment_seconds)))
        step = duration / n_segments
        segments: list[Segment] = []
        for i in range(n_segments):
            start = round(i * step, 3)
            end = round(min(duration, (i + 1) * step), 3)
            text = f"mock segment {i + 1}"
            seg = Segment(
                id=i, start=start, end=end, text=text,
                emotion=_EMOTION_CYCLE[i % len(_EMOTION_CYCLE)].value,
                words=self._fake_words(text, start, end) if word_timestamps else None,
            )
            segments.append(seg)

        return Transcript(
            text=" ".join(s.text for s in segments),
            language=lang,
            duration=round(duration, 3),
            segments=segments,
            engine=self.name,
            emotion_summary=_summarise(segments),
            events=["speech"],
            warnings=[],
        )

    def detect_emotion(self, chunk: AudioChunk) -> str | None:
        """按音频长度给一个确定性情绪 / deterministic emotion for a chunk."""
        if chunk.duration <= 0:
            return None
        idx = int(chunk.duration * 10) % len(_EMOTION_CYCLE)
        return _EMOTION_CYCLE[idx].value

    # ── 辅助 / helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _fake_words(text: str, start: float, end: float) -> list[Word]:
        """把文本按空格切成词，时间等分 / split text into evenly-timed words."""
        parts = text.split() or [text]
        span = max(0.0, end - start) / max(1, len(parts))
        return [Word(text=p, start=round(start + i * span, 3),
                     end=round(start + (i + 1) * span, 3))
                for i, p in enumerate(parts)]

    def info(self) -> dict[str, Any]:
        return {"engine": self.name, "loaded": True,
                "capabilities": sorted(c.value for c in self.capabilities())}

    def release(self) -> None:
        self.loaded = False


def _summarise(segments: list[Segment]) -> str | None:
    """情绪汇总：主导情绪 + 占比 / dominant emotion with share."""
    labels = [s.emotion for s in segments if s.emotion]
    if not labels:
        return None
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    top, n = max(counts.items(), key=lambda kv: kv[1])
    return f"{top} ({n}/{len(labels)})"


__all__ = ["MockEngine"]
