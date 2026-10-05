"""字幕断句 / Subtitle re-segmentation.

引擎给出的 segment **常常过长**（实测 185 s 素材里最长一段超过 12 s；
TTS 侧的段落级字幕更是动辄十几秒），直接渲染会得到一行撑满屏幕、
停留很久的字幕 —— 观感上就是「节奏拖沓」。

**切分规则按优先级 / split rules in priority order**：

1. **强停顿**：字间静音 ≥ `pause_gap`（默认 0.28 s）→ 必切
2. **句末标点**：`。！？…；` → 必切
3. **字数上限**：加权字数 ≥ `max_chars`（默认 20）→ 必切
4. **时长上限**：本条已 ≥ `max_seconds`（默认 6 s）→ 必切
5. **弱停顿**：字间静音 ≥ `soft_gap`（默认 0.12 s）**且**已达 `min_chars` → 切
6. **次级标点**：`，、,` **且**已达 `min_chars` → 切

外加三处收尾：**数字保护**（`零点三|四四米` 这种切法会把数字读坏）、
**合并过短的尾条**（避免一闪而过的字幕）、**去悬挂标点**。

本规则**与引擎无关、与音频无关** —— 输入只要有 `(start, end, text)` 序列即可，
所以 ASR 侧用引擎给的词级时间戳，TTS 侧用「段落时间轴 + 段内摊分」造出的
伪字级时间戳，都能喂同一个函数。详见 `docs/01_design.md §3.5`。
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_asr.core.types import Segment, Word

# 强句末标点 / strong sentence terminators
SENTENCE_END = "。！？…；!?;"
# 弱停顿标点 / weak pause punctuation
CLAUSE_END = "，、,:："
_ALL_PUNCT = SENTENCE_END + CLAUSE_END

# 切分后不要留在条目开头的字符 / characters trimmed from a cue's head
_LEADING_TRIM = "，。、！？…；：,.;!?: "

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")

# 数字粘连字符：切在这里会把数字读坏（"零点三|四四米"）
# Digits and numeric connectives that must not be split across cues.
_NUMERIC = set("0123456789零一二三四五六七八九十百千万亿点．.%％")


@dataclass
class SegmentParams:
    """切分参数 / Segmentation parameters —— 这些就是按观感调的旋钮。"""

    max_chars: int = 20
    min_chars: int = 12
    pause_gap: float = 0.28       # 强停顿阈值（秒），必切
    soft_gap: float = 0.12        # 弱停顿阈值（秒），达 min_chars 才切
    merge_below_seconds: float = 0.9   # 短于此的尾条并入上一条
    max_seconds: float = 6.0      # 单条最长时长，超过强切
    chars_per_word: float = 4.0   # 纯英文时每词折算的字符数
    # 硬上限的倍数：达到 `max_chars` 只是「该切了」，超过这个倍数才**无条件切**
    #
    # 这个区分是为了不再切出「函 / 数」「基 / 本」这种半个词的字幕：中文没有词边界，
    # 一到字数就切必然落在某个词中间。达到软上限之后继续往前走到下一个标点，
    # 切口才落在词与词之间。
    # A soft ceiling that waits for punctuation reads far better than a hard cut at an
    # arbitrary character, which splits Chinese words in half.
    hard_chars_ratio: float = 1.6
    # 单元语速上限（字/秒），用来识别**异常长的单元**。
    # ASR 常把后面的静音归给某个字（实测 faster-whisper 把 11.68 s 静音挂在
    # 一个「上」字上），先按这个上限截断，多出来的部分才会变成「间隙」，
    # 停顿规则才看得见它。详见 docs/issues/007。
    chars_per_second: float = 4.0
    unit_cap_slack: float = 0.4   # 单元时长上限的余量（秒）

    @classmethod
    def from_config(cls, cfg: Any) -> SegmentParams:
        """从 `SubtitleConfig` 取参数 / build from the config object."""
        if cfg is None:
            return cls()

        def get(name: str, default: Any) -> Any:
            return getattr(cfg, name, default)

        return cls(
            max_chars=int(get("max_chars", 42)),
            min_chars=int(get("min_chars", 12)),
            pause_gap=float(get("pause_gap", 0.28)),
            soft_gap=float(get("soft_gap", 0.12)),
            merge_below_seconds=float(get("merge_below_seconds", 0.9)),
            max_seconds=float(get("max_seconds", 6.0)),
            chars_per_word=float(get("chars_per_word", 4.0)),
            chars_per_second=float(get("chars_per_second", 4.0)),
            hard_chars_ratio=float(get("hard_chars_ratio", 1.6)),
        )


@dataclass
class Unit:
    """切分的最小单位 / the atomic unit fed to the segmenter.

    ASR 侧：一个词或一个字（引擎给的词级时间戳）。
    TTS 侧：把段内文本按时间摊分后得到同样的结构 —— **两边共用同一套规则**。
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    emotion: str | None = None


@dataclass
class Cue:
    """一条切好的字幕 / one segmented cue."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    emotion: str | None = None
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── 对外主函数 / public entry points ─────────────────────────────────────────


def resegment(segments: list[Segment], cfg: Any = None) -> list[Segment]:
    """把过长的段切成适合上屏的字幕条 / split segments into subtitle-sized cues.

    参数 / Args:
        segments: 引擎产出的句级分段；带 `words` 时走**字级规则**（精确），
            不带时退化为**按文本长度比例摊分**（估算，精度较低）。
        cfg: `SubtitleConfig`；为空用内置默认。

    返回 / Returns:
        新的 `Segment` 列表，`id` 重新编号，`speaker` / `emotion` 继承自原段。

    说明 / note: 时间轴始终单调且不重叠 —— 越界会让某些播放器整份字幕都不加载。
    """
    params = SegmentParams.from_config(cfg)
    out: list[Cue] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        if seg.words:
            out.extend(segment_units(_units_from_segment(seg, params), params,
                                     fallback_span=(seg.start, seg.end)))
        else:
            out.extend(_segments_from_text(seg, params))
    cues = _postprocess(out, params)
    return [_to_segment(c, i) for i, c in enumerate(cues)]


def segment_units(units: list[Unit], params: SegmentParams | None = None,
                  fallback_span: tuple[float, float] | None = None) -> list[Cue]:
    """把字级单元切成字幕条 / turn character-level units into cues.

    这是**纯规则的核心**，不碰音频也不碰引擎 —— 只要有 `(start, end, text)` 就能用。
    The engine-agnostic core: give it timestamps and text, nothing else.
    """
    p = params or SegmentParams()
    if not units:
        if fallback_span:
            return [Cue(start=fallback_span[0], end=fallback_span[1], text="")]
        return []

    ordered = sorted(units, key=lambda u: (u.start, u.end))
    cues: list[Cue] = []
    buf: list[Unit] = []
    buf_len = 0.0

    def flush(reset: bool = True) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = _join([u.text for u in buf])
        if text:
            cues.append(Cue(start=buf[0].start, end=buf[-1].end, text=text,
                            speaker=buf[0].speaker, emotion=buf[0].emotion,
                            words=[Word(text=u.text, start=u.start, end=u.end)
                                   for u in buf if u.text.strip()]))
        if reset:
            buf, buf_len = [], 0.0

    for i, u in enumerate(ordered):
        # **先预判再入 buf**：如果把当前单元放进去会让这条超时长上限，就先把
        # 已经攒好的切出去。只在「加进来之后」才检查的话，最后那个长单元会把
        # 整条顶穿 —— 实测能顶到 max_seconds + 单元时长。
        # Pre-check before appending, otherwise the last unit overshoots the cap.
        if buf and (u.end - buf[0].start) > p.max_seconds:
            flush()

        buf.append(u)
        buf_len += weight(u.text, p.chars_per_word)

        nxt_gap = ordered[i + 1].start - u.end if i + 1 < len(ordered) else 99.0
        char = u.text.strip()
        last = char[-1] if char else ""
        nxt_first = ordered[i + 1].text.strip()[:1] if i + 1 < len(ordered) else ""

        # 数字内部不切：否则「零点三|四四米」会被读坏
        splits_number = last in _NUMERIC and nxt_first in _NUMERIC

        strong_end = last in SENTENCE_END
        # **软上限只是「该切了」，硬上限才是「必须切」**：中文没有词边界，
        # 一到字数就切必然落在某个词中间（实测切出过「函 / 数」「基 / 本」）。
        # 到软上限后继续往前走到下一个次级标点，切口才落在词与词之间。
        # The soft ceiling waits for punctuation; only the hard ceiling cuts blind.
        soft_over = buf_len >= p.max_chars
        hard_over = buf_len >= p.max_chars * p.hard_chars_ratio
        too_long = (buf[-1].end - buf[0].start) >= p.max_seconds
        hard_pause = nxt_gap >= p.pause_gap
        soft_pause = nxt_gap >= p.soft_gap and buf_len >= p.min_chars
        weak_end = last in CLAUSE_END and buf_len >= p.min_chars

        must_break = strong_end or hard_over or too_long
        may_break = (
            (soft_over and last in CLAUSE_END)
            or hard_pause or soft_pause or weak_end
        ) and not splits_number
        if must_break or may_break:
            flush()

    flush()
    return cues


def weight(text: str, chars_per_word: float = 4.0) -> float:
    """估算「字数额度」/ estimate the character budget of a piece of text.

    中文按字计；英文按**词** × `chars_per_word` 折算，这样中英混排时上限一致 ——
    单纯用 `len()` 会让一行英文比一行中文长得多。
    """
    cjk = len(_CJK.findall(text))
    rest = _CJK.sub(" ", text)
    words = len([w for w in re.split(r"\s+", rest) if w.strip()])
    other = len(re.sub(r"[\s\w]", "", rest))
    return cjk + words * chars_per_word + other


def wrap_text(text: str, max_line_chars: int) -> str:
    """按字符宽度折行 / wrap into lines by character width."""
    if max_line_chars <= 0 or len(text) <= max_line_chars:
        return text
    lines: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if len(cur) >= max_line_chars and ch in _ALL_PUNCT + " ":
            lines.append(cur.strip())
            cur = ""
        elif len(cur) >= max_line_chars * 1.5:
            lines.append(cur.strip())
            cur = ""
    if cur.strip():
        lines.append(cur.strip())
    return "\n".join(lines) if lines else text


# ── 内部 / internals ─────────────────────────────────────────────────────────


def unit_cap_seconds(text: str, p: SegmentParams) -> float:
    """一个单元的**合理最长时长** / plausible upper bound for one unit.

    按语速估算：`字数 / chars_per_second + 余量`。
    超出的部分几乎一定是 ASR 把后面的静音算进来了 —— 见 `docs/issues/007`。
    """
    n = max(1, len(text.strip()))
    return n / max(0.5, p.chars_per_second) + p.unit_cap_slack


def _units_from_segment(seg: Segment, p: SegmentParams) -> list[Unit]:
    """把段的词级时间戳摊成切分单元 / turn a segment's words into units.

    **关键的一步是「异常单元钳制」**：ASR 引擎（实测 faster-whisper）会把一段
    静音归到它前面那个字上 —— 有一个「上」字被标成 11.68 秒，而它后面紧邻的
    字起点正好等于它的终点。于是**字间 gap 仍然是 0**，基于停顿的切分规则
    完全看不见那段静音，切出来的字幕依旧是十几秒一条。

    这里把超长单元截到合理上限，**多出来的时间就成了间隙**，停顿规则才生效。
    Clamping overlength units turns hidden silence into a real gap.
    """
    words = [w for w in (seg.words or []) if w.text.strip()]
    if not words:
        return []
    out: list[Unit] = []
    for w in words:
        text = w.text.strip()
        start, end = float(w.start), float(w.end)
        if end <= start:
            end = start + 0.02
        cap = unit_cap_seconds(text, p)
        if end - start > cap:
            end = start + cap
        out.append(Unit(start=start, end=end, text=text,
                        speaker=seg.speaker, emotion=seg.emotion))
    return out


def _segments_from_text(seg: Segment, p: SegmentParams) -> list[Cue]:
    """没有词级时间戳时的回退：按标点切 + 按**加权字数比例**摊分时间。

    精度不如字级对齐（停顿长度是估的），但对**匀速朗读的 TTS** 足够用。
    走这条路的段会在调用方可见（`Transcript.warnings` 里有一条说明）。
    """
    chunks = _split_text_chunks(seg.text, p)
    if not chunks:
        return [Cue(start=seg.start, end=seg.end, text=seg.text, speaker=seg.speaker,
                    emotion=seg.emotion)]
    weights = [weight(c, p.chars_per_word) for c in chunks]
    total = sum(weights) or 1.0
    span = max(0.0, seg.end - seg.start)
    out: list[Cue] = []
    cursor = seg.start
    for chunk, w in zip(chunks, weights, strict=True):
        share = span * (w / total)
        start, end = cursor, min(seg.end, cursor + share)
        cursor = end
        out.append(Cue(start=round(start, 3), end=round(end, 3), text=chunk.strip(),
                       speaker=seg.speaker, emotion=seg.emotion))
    return out


def _split_text_chunks(text: str, p: SegmentParams) -> list[str]:
    """按标点与字数切文本（不看时间）/ split text by punctuation and length only.

    **两级上限**：到 `max_chars` 后继续往前走，只有超过
    `max_chars * hard_chars_ratio` 才无条件切 —— 与 `segment_units` 同一个道理，
    这里没有时间戳可用，切口更依赖标点。
    """
    hard = p.max_chars * p.hard_chars_ratio
    chunks: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        current = weight(buf, p.chars_per_word)
        if ch in SENTENCE_END:
            chunks.append(buf)
            buf = ""
        elif ch in CLAUSE_END and current >= p.min_chars:
            chunks.append(buf)
            buf = ""
        elif current >= hard:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _join(parts: list[str]) -> str:
    """拼接字级文本 / join character-level text.

    中文不加空格；**两侧都是 ASCII 字母数字时补一个空格** ——
    否则 `Hello` + `world` 会粘成 `Helloworld`。
    """
    out = ""
    for raw in parts:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        prev = out[-1]
        if prev.isascii() and prev.isalnum() and piece[0].isascii() and piece[0].isalnum():
            out += " " + piece
        else:
            out += piece
    return out.strip()


def _postprocess(cues: list[Cue], p: SegmentParams) -> list[Cue]:
    """收尾清理 / cleanup pass.

    1. 去掉条目首尾的悬挂标点；
    2. 合并过短的条目（否则会出现一闪而过的字幕）；
    3. 保证时间单调不重叠。
    """
    cleaned: list[Cue] = []
    for c in cues:
        # **只去开头的悬挂标点，保留句末标点**。
        # 上游那份实现用的是 `strip()`（首尾都去），会把「短句。」改成「短句」——
        # 转写场景下句末标点是原文的一部分，吃掉它等于丢信息；而以逗号起头确实难看。
        # Only trim the head: a trailing period is part of the transcript.
        text = c.text.lstrip(_LEADING_TRIM) or c.text.strip()
        if not text:
            continue
        cleaned.append(dataclasses.replace(c, text=text.rstrip()))

    merged: list[Cue] = []
    for c in cleaned:
        if merged and c.duration < p.merge_below_seconds:
            prev = merged[-1]
            new_end = max(prev.end, c.end)
            # **合并不得突破时长上限** —— 否则「合并过短条目」会把上一条顶穿，
            # 让 max_seconds 形同虚设（实测能到 6.65s 而上限是 4.0s）。
            if new_end - prev.start <= p.max_seconds:
                merged[-1] = dataclasses.replace(
                    prev, end=new_end, text=f"{prev.text}{c.text}",
                    words=[*prev.words, *c.words])
                continue
        merged.append(c)

    for i in range(1, len(merged)):
        if merged[i].start < merged[i - 1].end:
            merged[i] = dataclasses.replace(
                merged[i], start=merged[i - 1].end,
                end=max(merged[i].end, merged[i - 1].end))
    return merged


def _to_segment(cue: Cue, index: int) -> Segment:
    return Segment(id=index, start=round(cue.start, 3), end=round(cue.end, 3),
                   text=cue.text, speaker=cue.speaker, emotion=cue.emotion,
                   words=cue.words or None)


__all__ = [
    "CLAUSE_END",
    "Cue",
    "SENTENCE_END",
    "SegmentParams",
    "Unit",
    "resegment",
    "segment_units",
    "unit_cap_seconds",
    "weight",
    "wrap_text",
]
