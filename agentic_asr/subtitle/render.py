"""字幕渲染 / Subtitle rendering.

`Transcript` → **SRT / VTT / ASS**。三条硬规则 / three hard rules:

1. **不依赖厂商导出**。MiniMax 能直出 SRT，但 OpenAI 系不能；为了「同一套抽象」
   一律**本地渲染**，厂商的 `srt` 参数不用。Rendering is always local.
2. **断句先行**（见 `subtitle/segment.py`）—— 渲染器只负责格式化，不负责切句。
3. **时间轴必须单调**。渲染前校验 `start` 递增、非负、不越界；越界即报错，
   因为一个负时间码会让整个字幕文件在某些播放器里直接不加载。
"""

from __future__ import annotations

from pathlib import Path

from agentic_asr.core.errors import ASRError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import Segment, Transcript
from agentic_asr.subtitle.segment import resegment, wrap_text

log = get_logger("subtitle.render")

_FORMATS = ("srt", "vtt", "ass")


def _check_timeline(segments: list[Segment], duration: float | None) -> None:
    """时间轴校验 / timeline validation（失败即抛，不静默出坏文件）."""
    prev = -1.0
    for s in segments:
        if s.start < -1e-3:
            raise ASRError(f"字幕时间轴为负 / negative timestamp: {s.start}")
        if s.end < s.start:
            raise ASRError(f"字幕结束早于开始 / end < start at cue {s.id}")
        if s.start < prev - 1e-3:
            raise ASRError(f"字幕时间轴逆序 / non-monotonic at cue {s.id}")
        prev = s.end
    if duration is not None and segments and segments[-1].end > duration + 0.5:
        raise ASRError(
            f"字幕超出音频时长 / cue beyond audio: {segments[-1].end:.2f} > {duration:.2f}")


def _ts(seconds: float, sep: str = ",") -> str:
    """`HH:MM:SS,mmm` / `HH:MM:SS.mmm`."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _ts_ass(seconds: float) -> str:
    """ASS 用 `H:MM:SS.cc`（百分秒）/ ASS uses centiseconds."""
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def to_srt(segments: list[Segment], max_line_chars: int = 20,
           duration: float | None = None, speakers: bool = False) -> str:
    """渲染 SRT / render SubRip."""
    _check_timeline(segments, duration)
    blocks: list[str] = []
    for i, s in enumerate(segments, start=1):
        text = wrap_text(s.text.strip(), max_line_chars)
        if speakers and s.speaker:
            text = f"[{s.speaker}] {text}"
        blocks.append(f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[Segment], max_line_chars: int = 20,
           duration: float | None = None, speakers: bool = False) -> str:
    """渲染 WebVTT / render WebVTT."""
    _check_timeline(segments, duration)
    blocks: list[str] = ["WEBVTT\n"]
    for i, s in enumerate(segments, start=1):
        text = wrap_text(s.text.strip(), max_line_chars)
        if speakers and s.speaker:
            text = f"<v {s.speaker}>{text}"
        blocks.append(f"{i}\n{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{text}\n")
    return "\n".join(blocks)


def to_ass(segments: list[Segment], max_line_chars: int = 20,
           duration: float | None = None, karaoke: bool = False,
           title: str = "Agentic ASR") -> str:
    """渲染 ASS / render ASS.

    `karaoke=True` 时用 `\\k` 标签输出**词级卡拉 OK**效果（有 `words` 才生效）。
    Word-level karaoke via ASS `\\k` tags when word timings exist.
    """
    _check_timeline(segments, duration)
    head = f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,54,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    for s in segments:
        text = _ass_karaoke(s) if (karaoke and s.words) else wrap_text(
            s.text.strip(), max_line_chars).replace("\n", "\\N")
        name = s.speaker or ""
        lines.append(f"Dialogue: 0,{_ts_ass(s.start)},{_ts_ass(s.end)},Default,"
                     f"{name},0,0,0,,{text}")
    return head + "\n".join(lines) + "\n"


def _ass_karaoke(seg: Segment) -> str:
    """把词级时间戳转成 `\\k` 标签 / turn word timings into `\\k` tags."""
    parts: list[str] = []
    for w in seg.words or []:
        cs = max(1, int(round((w.end - w.start) * 100)))
        parts.append(f"{{\\k{cs}}}{w.text.strip()}")
    return "".join(parts)


def render(transcript: Transcript, fmt: str = "srt", cfg: object = None,
           resegment_first: bool = True, karaoke: bool = False) -> str:
    """按格式渲染 / render in the requested format."""
    fmt = (fmt or "srt").lower().lstrip(".")
    if fmt not in _FORMATS:
        raise ASRError(f"不支持的字幕格式 / unsupported subtitle format: {fmt}"
                       f"　可用 / available: {', '.join(_FORMATS)}")
    segments = list(transcript.segments)
    if resegment_first:
        segments = resegment(segments, cfg)
    max_chars = int(getattr(cfg, "max_line_chars", 20) if cfg is not None else 20)
    duration = transcript.duration or None
    if fmt == "srt":
        return to_srt(segments, max_chars, duration)
    if fmt == "vtt":
        return to_vtt(segments, max_chars, duration)
    return to_ass(segments, max_chars, duration, karaoke=karaoke)


def write_subtitle(transcript: Transcript, path: str | Path, fmt: str | None = None,
                   cfg: object = None, resegment_first: bool = True,
                   karaoke: bool = False) -> Path:
    """渲染并落盘 / render and write to disk."""
    p = Path(path)
    fmt = fmt or p.suffix.lstrip(".") or "srt"
    content = render(transcript, fmt, cfg, resegment_first, karaoke)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    log.info("字幕已写出 / subtitle written: %s (%s)", p, fmt)
    return p


__all__ = ["render", "to_ass", "to_srt", "to_vtt", "write_subtitle"]
