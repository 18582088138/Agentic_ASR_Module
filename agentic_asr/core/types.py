"""跨层数据类型 / Cross-layer data types.

这些 dataclass 是 **引擎 ↔ 门面 ↔ server/gui/cli** 之间的契约。
引擎只负责把一段音频变成 `Transcript`；切片、时间轴平移、字幕渲染、无语音防护
全部在门面完成 —— 所以这里的字段是**所有引擎的共同下限**。

**句级 segment 是最小必需契约**：word / speaker / emotion 一律可选。
真实世界里 `gpt-4o-transcribe`、硅基流动、百炼 `qwen3-asr-flash` 都**不返回时间戳**，
把词级设成必需会让适配器大面积不可用。
Sentence-level segments are the minimum contract; word timings, speakers and
emotion are optional because several real engines return none of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# ── 引擎能力 / engine capabilities ────────────────────────────────────────────


class Capability(StrEnum):
    """引擎可能具备的能力 / Capabilities an engine may declare.

    引擎必须**不加载权重**就能回答自己有哪些能力（GUI/HTTP 的开机选择器要用它
    把不支持的开关置灰），见 `engines/base.py`.
    """

    TIMESTAMPS = "timestamps"            # 句级时间戳
    WORD_TIMESTAMPS = "word_timestamps"  # 词/字级时间戳
    LANGUAGE_ID = "language_id"          # 自动语种识别
    DIARIZATION = "diarization"          # 说话人分离
    EMOTION = "emotion"                  # 情绪标签
    EVENTS = "events"                    # 音频事件（笑声/掌声/音乐…）


class Emotion(StrEnum):
    """SenseVoice 的 7 类情绪 / the 7 emotion labels of SenseVoice.

    引擎原始返回形如 `<|NEUTRAL|>`，门面负责解析（见 `engines/sensevoice.py`）。
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    UNKNOWN = "unknown"


# ── 转写结果 / transcription results ─────────────────────────────────────────


@dataclass
class Word:
    """一个词或字 / One word (or one CJK character)."""

    text: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end,
                "confidence": self.confidence}


@dataclass
class Segment:
    """一句（一段连续语音）/ One sentence-level segment."""

    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    emotion: str | None = None
    words: list[Word] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "start": self.start, "end": self.end,
                             "text": self.text, "speaker": self.speaker,
                             "emotion": self.emotion}
        if self.words is not None:
            d["words"] = [w.to_dict() for w in self.words]
        return d


@dataclass
class Transcript:
    """一次转写的完整结果 / The full result of one transcription."""

    text: str
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    engine: str = ""
    emotion_summary: str | None = None
    events: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # 幻觉闸门命中标记：调用方应据此人工复核，而不是当作正常结果使用
    # Set by the hallucination guard; callers should review rather than trust it.
    suspicious: bool = False
    raw: dict[str, Any] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self, words: bool = True) -> dict[str, Any]:
        segs = []
        for s in self.segments:
            d = s.to_dict()
            if not words:
                d.pop("words", None)
            segs.append(d)
        return {
            "text": self.text, "language": self.language, "duration": self.duration,
            "engine": self.engine, "segments": segs, "events": self.events,
            "emotion_summary": self.emotion_summary,
            "warnings": self.warnings, "suspicious": self.suspicious,
        }

    @classmethod
    def empty(cls, engine: str, duration: float, warning: str) -> Transcript:
        """构造一个「明确为空」的结果（幻觉闸门用）/ build an explicit empty result."""
        return cls(text="", language="", duration=duration, engine=engine,
                   warnings=[warning])


# ── 音频输入 / audio input ───────────────────────────────────────────────────


@dataclass
class AudioChunk:
    """已解码归一的一段音频 / A decoded and normalised piece of audio.

    引擎拿到的**永远**是这个形状，而不是文件路径 —— 这样做的直接原因是
    faster-whisper 1.2.1 与 av 19 不兼容（见 `docs/issues/001`），
    根因则是四条输入通路（文件/视频/字节流/麦克风）本就该归一成同一份表示。
    Engines always receive this, never a file path.
    """

    samples: np.ndarray        # float32, 单声道, `sample_rate` Hz
    sample_rate: int = 16000
    offset: float = 0.0        # 这段在原始音频中的起始秒数（切片合并用）
    source: str = ""           # 来源路径或标识，仅用于报错与元信息

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate or 1)


@dataclass
class AudioInfo:
    """音频/视频信息提取的结果 / Result of media probing."""

    path: str
    duration: float = 0.0
    container: str = ""
    bit_rate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str = ""
    bits_per_sample: int | None = None
    has_video: bool = False
    video: dict[str, Any] | None = None
    # 声学统计 / acoustic stats
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    estimated_snr_db: float | None = None
    speech_ratio: float | None = None      # VAD 语音占比
    clipping: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "duration": self.duration, "container": self.container,
            "bit_rate": self.bit_rate, "sample_rate": self.sample_rate,
            "channels": self.channels, "codec": self.codec,
            "bits_per_sample": self.bits_per_sample, "has_video": self.has_video,
            "video": self.video, "peak_dbfs": self.peak_dbfs, "rms_dbfs": self.rms_dbfs,
            "estimated_snr_db": self.estimated_snr_db, "speech_ratio": self.speech_ratio,
            "clipping": self.clipping, "warnings": self.warnings,
        }


# ── 音频分段 / segmentation ──────────────────────────────────────────────────


@dataclass
class SegmentPiece:
    """分段后的一段 / One piece produced by the segmenter."""

    index: int
    start: float
    end: float
    # True = 该段落在连续说话区、无静音点可切，只能按固定时长硬切
    # True when no silence was available and a hard cut at max_seconds was used.
    hard_cut: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": self.start, "end": self.end,
                "duration": self.duration, "hard_cut": self.hard_cut}


# ── 参考音频截取 / reference clip ────────────────────────────────────────────


@dataclass
class ClipResult:
    """一段候选参考音频 / One candidate reference clip.

    `text` 必须来自**这一片段自己的转写**，不能拿整段音频的文本 ——
    两者必须严格对应，否则下游音色克隆会错位。
    `text` must come from this very clip's own transcription.
    """

    index: int
    wav_path: str
    start: float
    end: float
    text: str
    score: float
    score_detail: dict[str, float] = field(default_factory=dict)
    srt_path: str | None = None
    txt_path: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "wav_path": self.wav_path, "start": self.start,
                "end": self.end, "duration": self.duration, "text": self.text,
                "score": self.score, "score_detail": self.score_detail,
                "srt_path": self.srt_path, "txt_path": self.txt_path}


__all__ = [
    "AudioChunk",
    "AudioInfo",
    "Capability",
    "ClipResult",
    "Emotion",
    "Segment",
    "SegmentPiece",
    "Transcript",
    "Word",
]
