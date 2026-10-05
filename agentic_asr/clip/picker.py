"""参考音频截取 / Reference clip picking.

产出 `ref_audio` + `ref_text`，直接喂 Agentic_TTS_Module 的音色克隆
（`SynthRequest(ref_audio=..., ref_text=...)`）。

**为什么要自动挑**：从一小时素材里手工找一段干净的 10~15 s 人声很费时间，
而「干净」其实有客观判据 —— 语音占比高、底噪低、不跨说话人、不切在词中间。
本模块把这些判据量化成五维打分。

**一条不可妥协的约束**：`ref_text` 必须来自**该片段自己的转写**，
不能拿整段音频的文本。两者一旦错位，音色克隆就会学到错误的文本-音频对应。
`ref_text` must come from this very clip's own transcription.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from agentic_asr.audio import features as feat
from agentic_asr.audio.io import slice_chunk, write_wav
from agentic_asr.audio.vad import VadDetector, merge_intervals
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import AudioChunk, ClipResult, Segment, Transcript, Word
from agentic_asr.subtitle.segment import _join

log = get_logger("clip.picker")

_DEFAULT_WEIGHTS = {"speech_ratio": 0.35, "duration": 0.25, "snr": 0.25, "boundary": 0.15}


@dataclasses.dataclass
class _Candidate:
    start: float
    end: float
    speech_ratio: float
    snr_db: float | None
    speaker: str | None
    text: str
    words: list[Word] | None
    detail: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _candidates_from_intervals(intervals: list[tuple[float, float]],
                               target: float, minimum: float, maximum: float,
                               duration: float, step: float = 1.0
                               ) -> list[tuple[float, float]]:
    """生成候选窗口 / build candidate windows.

    两个来源 / two sources:

    1. **语音区间内**的窗口 —— 质量最高，首选；
    2. **跨段滑窗** —— 当素材里每段语音都短于 `minimum` 时（比如每句只有 4~5 秒），
       仍然能凑出目标长度。参考音频中间带一点静音**不影响音色克隆**，
       所以这比「返回空」或「只给一段过短的」更有用。
       A reference clip may contain short silences; it only needs clean speech.
    """
    out: list[tuple[float, float]] = []
    for s, e in intervals:
        span = e - s
        if span < minimum:
            continue
        if span <= maximum:
            out.append((s, e))
            continue
        t = s
        while t + target <= e + 1e-6:
            out.append((t, t + target))
            t += step
        out.append((max(s, e - target), e))

    if duration >= minimum:
        span = min(target, duration)
        fine = max(0.5, span / 4.0)
        t = 0.0
        while t + span <= duration + 1e-6:
            out.append((t, t + span))
            t += fine
        out.append((max(0.0, duration - span), duration))
    return out


def _boundary_score(start: float, end: float, words: list[Word] | None) -> float:
    """边界是否落在词与词之间 / how cleanly the cut lands between words.

    起点贴着某个词的起始、终点贴着某个词的结束 = 1.0；
    切在词中间 = 0.0。没有词级时间戳时给中性值 0.5，**不假装知道**。
    """
    if not words:
        return 0.5
    starts = np.array([w.start for w in words])
    ends = np.array([w.end for w in words])
    if starts.size == 0:
        return 0.5
    tol = 0.08
    head_ok = bool(np.min(np.abs(starts - start)) <= tol)
    tail_ok = bool(np.min(np.abs(ends - end)) <= tol)
    return (0.5 if head_ok else 0.0) + (0.5 if tail_ok else 0.0)


def _snap_to_words(start: float, end: float,
                   words: list[Word]) -> tuple[float, float]:
    """把边界吸附到最近的词边界 / snap boundaries onto word edges."""
    if not words:
        return start, end
    starts = np.array([w.start for w in words])
    ends = np.array([w.end for w in words])
    new_start = float(starts[int(np.argmin(np.abs(starts - start)))])
    new_end = float(ends[int(np.argmin(np.abs(ends - end)))])
    if new_end <= new_start:
        return start, end
    return new_start, new_end


def _text_of(words: list[Word] | None, segments: list[Segment] | None,
             start: float, end: float) -> str:
    """取候选片段对应的文本 / text belonging to this very window."""
    if words:
        picked = [w.text for w in words if w.end > start and w.start < end]
        if picked:
            return _join(picked)
    if segments:
        parts = [s.text.strip() for s in segments if s.end > start and s.start < end]
        if parts:
            return "".join(parts)
    return ""


def pick_reference(chunk: AudioChunk, transcript: Transcript | None = None,
                   cfg: Any = None, vad: VadDetector | None = None,
                   out_dir: str | Path | None = None, top: int | None = None,
                   stem: str = "ref", vad_model_path: str | None = None
                   ) -> list[ClipResult]:
    """挑出 top-N 段参考音频并落盘 / pick and export the top-N reference clips.

    参数 / Args:
        chunk: 已解码的音频（建议 16 kHz 单声道）。
        transcript: 有则用它的词级时间戳吸附边界、并给出 `ref_text`。
        out_dir: 输出目录；None 时只返回结果不落盘。
        top: 取前 N 个；默认用 `cfg.top`。

    返回 / Returns:
        按分数降序的 `ClipResult` 列表；`text` 与该片段严格对应。
    """
    target = float(_cfg(cfg, "target_seconds", 12.0))
    minimum = float(_cfg(cfg, "min_seconds", 10.0))
    maximum = float(_cfg(cfg, "max_seconds", 15.0))
    top_n = int(top if top is not None else _cfg(cfg, "top", 3))
    weights = dict(_cfg(cfg, "weights", None) or _DEFAULT_WEIGHTS)
    out_sr = int(_cfg(cfg, "sample_rate", 16000))

    detector = vad or VadDetector(model_path=vad_model_path)
    intervals = merge_intervals(detector.intervals(chunk.samples, chunk.sample_rate),
                                gap=0.30)
    if not intervals:
        log.warning("没有检出语音，无法截取参考音频 / no speech detected")
        return []

    # 全局掩码算一次，供所有候选窗口切片使用 / compute the mask once
    full_mask = feat.mask_from_intervals(intervals, chunk.samples.size,
                                         chunk.sample_rate)

    all_words = [w for s in (transcript.segments if transcript else []) for w in (s.words or [])]
    all_words.sort(key=lambda w: w.start)
    segments = transcript.segments if transcript else None

    raw = _candidates_from_intervals(intervals, target, minimum, maximum,
                                     chunk.duration)
    if not raw:
        return []

    candidates: list[_Candidate] = []
    for start, end in raw:
        s2, e2 = (start, end)
        if all_words:
            s2, e2 = _snap_to_words(start, end, all_words)
        if e2 - s2 < minimum * 0.8:      # 吸附过度导致过短就回退原窗口
            s2, e2 = start, end
        cov = _coverage(intervals, s2, e2)
        speaker = _dominant_speaker(segments, s2, e2)
        candidates.append(_Candidate(
            start=round(s2, 3), end=round(e2, 3), speech_ratio=cov,
            snr_db=_local_snr(chunk, s2, e2, full_mask), speaker=speaker,
            text=_text_of(all_words or None, segments, s2, e2),
            words=[w for w in all_words if w.end > s2 and w.start < e2] or None))

    scored = [_apply_score(c, target, minimum, maximum, weights) for c in candidates]
    scored.sort(key=lambda c: c.detail.get("score", 0.0), reverse=True)
    picked = _dedupe(scored)[:top_n]

    results: list[ClipResult] = []
    for i, cand in enumerate(picked):
        result = ClipResult(index=i, wav_path="", start=cand.start, end=cand.end,
                            text=cand.text, score=cand.detail.get("score", 0.0),
                            score_detail=dict(cand.detail))
        if out_dir:
            result = _export(chunk, result, Path(out_dir), stem, i, out_sr)
        results.append(result)
    return results


# ── 打分 / scoring ───────────────────────────────────────────────────────────


def _apply_score(cand: _Candidate, target: float, minimum: float, maximum: float,
                 weights: dict[str, float]) -> _Candidate:
    """五维打分 / five-dimension scoring.

    时长项用**梯形**而不是单点：区间 `[minimum, maximum]` 内满分，
    越靠近 `target` 越好，超出范围线性衰减 —— 直接拿 `|dur - target|`
    会让 14.9 s 和 15.1 s 的分数断崖式不同，没道理。
    """
    dur = cand.duration
    if minimum <= dur <= maximum:
        span = max(target - minimum, maximum - target, 1e-6)
        duration_score = 1.0 - 0.3 * abs(dur - target) / span
    else:
        over = (minimum - dur) if dur < minimum else (dur - maximum)
        duration_score = max(0.0, 1.0 - over / max(target, 1e-6))

    snr = cand.snr_db
    # 20 dB 及以上视为满分，0 dB 及以下视为 0 分
    snr_score = 0.5 if snr is None else float(min(1.0, max(0.0, snr / 20.0)))
    boundary = _boundary_score(cand.start, cand.end, cand.words)

    detail = {
        "speech_ratio": round(cand.speech_ratio, 4),
        "duration": round(duration_score, 4),
        "snr": round(snr_score, 4),
        "boundary": round(boundary, 4),
    }
    score = (weights.get("speech_ratio", 0.0) * cand.speech_ratio
             + weights.get("duration", 0.0) * duration_score
             + weights.get("snr", 0.0) * snr_score
             + weights.get("boundary", 0.0) * boundary)
    total_w = sum(weights.values()) or 1.0
    detail["score"] = round(score / total_w, 4)
    detail["snr_db"] = round(cand.snr_db, 2) if cand.snr_db is not None else -1.0
    cand.detail = detail
    return cand


def _coverage(intervals: list[tuple[float, float]], start: float,
              end: float) -> float:
    """窗口内语音占比（已合并区间，直接求交）/ speech coverage inside the window."""
    span = end - start
    if span <= 0:
        return 0.0
    covered = 0.0
    for s, e in intervals:
        covered += max(0.0, min(end, e) - max(start, s))
    return float(min(1.0, covered / span))


def _local_snr(chunk: AudioChunk, start: float, end: float,
               full_mask: np.ndarray) -> float | None:
    """窗口内的语音/非语音 RMS 差 / SNR within the window.

    用**预先算好**的全局掩码切片，而不是对每个候选再跑一遍 VAD ——
    候选有几十个，逐个跑 VAD 会让参考音频截取从毫秒级变成秒级。
    Slice a precomputed mask instead of re-running the VAD per candidate.
    """
    from agentic_asr.audio.features import estimate_snr_db

    sr = chunk.sample_rate
    i0 = max(0, int(round(start * sr)))
    i1 = min(chunk.samples.size, int(round(end * sr)))
    if i1 - i0 < 800:
        return None
    return estimate_snr_db(chunk.samples[i0:i1], full_mask[i0:i1])


def _dominant_speaker(segments: list[Segment] | None,
                      start: float, end: float) -> str | None:
    """窗口内占主导的说话人 / dominant speaker in the window."""
    if not segments:
        return None
    tally: dict[str, float] = {}
    for s in segments:
        if not s.speaker or s.end <= start or s.start >= end:
            continue
        overlap = min(end, s.end) - max(start, s.start)
        if overlap > 0:
            tally[s.speaker] = tally.get(s.speaker, 0.0) + overlap
    if not tally:
        return None
    return max(tally.items(), key=lambda kv: kv[1])[0]


def _dedupe(cands: list[_Candidate], min_gap: float = 1.0) -> list[_Candidate]:
    """去掉时间上高度重叠的候选 / drop candidates that overlap heavily."""
    kept: list[_Candidate] = []
    for c in cands:
        if any(not (c.end <= k.start + min_gap or c.start >= k.end - min_gap) for k in kept):
            continue
        kept.append(c)
    return kept


def _export(chunk: AudioChunk, result: ClipResult, out_dir: Path, stem: str,
            index: int, target_sr: int) -> ClipResult:
    """落盘 wav + txt + srt / write the clip and its subtitle sidecars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}_{index + 1:02d}"
    sub = slice_chunk(chunk, result.start, result.end, target_sr=target_sr)
    wav_path = write_wav(base.with_suffix(".wav"), sub.samples, sub.sample_rate)

    text_path = base.with_suffix(".txt")
    text_path.write_text(result.text or "", encoding="utf-8")

    srt_path = None
    if result.text:
        srt_path = base.with_suffix(".srt")
        # 片段内的字幕时间轴从 0 起算 / the clip's own timeline starts at zero
        srt_path.write_text(
            f"1\n00:00:00,000 --> {_srt_ts(sub.duration)}\n{result.text}\n",
            encoding="utf-8")

    return dataclasses.replace(
        result, wav_path=str(wav_path), txt_path=str(text_path),
        srt_path=str(srt_path) if srt_path else None)


def _srt_ts(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


__all__ = ["pick_reference"]
