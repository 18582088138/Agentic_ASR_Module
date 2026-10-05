"""音频分段 / Audio segmentation.

**目的**：避免过长音频导致 OOM，并为下游 ASR 提供规整的输入单元。
用户确认的策略是「**静音点优先，超长时回退固定时长**」。

为什么每段都必须带时间轴 / why every piece carries its timestamps：
下游要把各段的转写结果**合并回一条完整时间轴**，靠的就是每段的 `start`。
没有它，每段都从 0 开始重新计数，长音频的字幕必然整体错位 ——
这是本模块最容易出、也最难查的一类静默 bug。
"""

from __future__ import annotations

from typing import Any

from agentic_asr.audio.vad import VadDetector, merge_intervals
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import AudioChunk, SegmentPiece

log = get_logger("segment.splitter")

_EPS = 1e-6


def _fixed_split(start: float, end: float, max_seconds: float,
                 hard: bool = True) -> list[tuple[float, float, bool]]:
    """按固定时长硬切 / hard-split into fixed-length pieces."""
    out: list[tuple[float, float, bool]] = []
    t = start
    guard = 0
    while t < end - _EPS and guard < 100000:
        e = min(end, t + max_seconds)
        out.append((t, e, hard))
        t = e
        guard += 1
    return out


def split_audio(chunk: AudioChunk, cfg: Any = None,
                vad: VadDetector | None = None,
                max_seconds: float | None = None,
                min_seconds: float | None = None,
                merge_gap: float | None = None,
                split_gap: float | None = None,
                pad: float | None = None,
                cover_full: bool = True,
                vad_model_path: str | None = None) -> list[SegmentPiece]:
    """把一段音频切成若干片 / split a chunk into pieces.

    参数 / Args:
        cfg: `SegmentConfig`；显式参数优先于它。
        vad: 复用的 VAD 实例。
        max_seconds: 单片上限（秒）。**同时也决定门面切片送引擎的大小。**
        min_seconds: 短于此的片并入相邻片（合并后不得超 `max_seconds`）。
        merge_gap: 小于此间隔的语音区间先合并，避免碎段。
        pad: 片两侧留白，防止把辅音切掉。

    返回 / Returns:
        按时间排序的 `SegmentPiece` 列表，**覆盖 [0, duration] 全片**。

    说明 / note:
        当 VAD 没有给出任何语音区间（纯静音 / 纯音乐 / VAD 失效）时，回退为
        固定时长切分并把 `hard_cut` 标为 False —— 「有没有语音」由门面的
        幻觉闸门判定，分段器不替它下结论。
    """
    max_seconds = float(max_seconds or getattr(cfg, "max_seconds", 30.0) or 30.0)
    min_seconds = float(min_seconds if min_seconds is not None
                        else getattr(cfg, "min_seconds", 0.5) or 0.5)
    merge_gap = float(merge_gap if merge_gap is not None
                      else getattr(cfg, "merge_gap", 0.30) or 0.30)
    split_gap = float(split_gap if split_gap is not None
                      else getattr(cfg, "split_gap", 1.00) or 1.00)
    pad = float(pad if pad is not None else getattr(cfg, "pad", 0.10) or 0.10)

    total = chunk.duration
    if total <= 0 or chunk.samples.size == 0:
        return []
    max_seconds = max(0.1, min(max_seconds, total))

    detector = vad or VadDetector(model_path=vad_model_path)
    try:
        intervals = detector.intervals(chunk.samples, chunk.sample_rate)
    except Exception as exc:  # noqa: BLE001 - VAD 失败退固定切分，不阻断主流程
        log.warning("VAD 失败，改用固定时长切分 / VAD failed, using fixed split: %s", exc)
        intervals = []

    if not intervals:
        raw = _fixed_split(0.0, total, max_seconds, hard=False)
    else:
        padded = merge_intervals(
            [(max(0.0, s - pad), min(total, e + pad)) for s, e in intervals], 0.0)
        raw = []
        cur_s: float | None = None
        cur_e = 0.0
        for s, e in padded:
            if (e - s) > max_seconds:
                # 连续说话、没有静音点可切 → 只能硬切，并明确标记
                if cur_s is not None:
                    raw.append((cur_s, cur_e, False))
                    cur_s = None
                raw.extend(_fixed_split(s, e, max_seconds, hard=True))
                continue
            if cur_s is None:
                cur_s, cur_e = s, e
                continue
            # 「静音点优先」：停顿超过 split_gap 就一定切，不再贪心填满 max_seconds
            pause = s - cur_e
            if pause > split_gap or (e - cur_s) > max_seconds:
                raw.append((cur_s, cur_e, False))
                cur_s, cur_e = s, e
            else:
                cur_e = e
        if cur_s is not None:
            raw.append((cur_s, cur_e, False))

    # 合并过短的片（仅在不违反 max_seconds 时）/ merge short pieces when allowed
    merged: list[tuple[float, float, bool]] = []
    for s, e, hard in raw:
        if merged and (e - s) < min_seconds:
            ps, pe, ph = merged[-1]
            if e - ps <= max_seconds:
                merged[-1] = (ps, e, ph)
                continue
        if merged and (merged[-1][1] - merged[-1][0]) < min_seconds:
            ps, _, ph = merged.pop()
            s, hard = ps, hard or ph
        merged.append((s, e, hard))

    pieces = [SegmentPiece(index=i, start=round(s, 3), end=round(e, 3), hard_cut=h)
              for i, (s, e, h) in enumerate(merged) if e - s > _EPS]

    # 补首尾静音段，使分段**覆盖全片** / pad the head and tail so pieces cover it all.
    # 这一步是为了通用性：调用方可以按段索引对齐原音频。静音段本身不会浪费
    # ASR 调用 —— 门面的幻觉闸门会把它们跳过（见 asr.py 的分段级闸门）。
    if cover_full and pieces:
        if pieces[0].start > 0.01:
            pieces.insert(0, SegmentPiece(index=0, start=0.0, end=pieces[0].start))
        if pieces[-1].end < total - 0.01:
            pieces.append(SegmentPiece(index=0, start=pieces[-1].end,
                                       end=round(total, 3)))
        for i, p in enumerate(pieces):
            p.index = i

    problems = validate_pieces(pieces, total)
    if problems:
        log.warning("分段校验发现问题 / segmentation issues: %s", "; ".join(problems))
    return pieces


def validate_pieces(pieces: list[SegmentPiece], duration: float,
                    tolerance: float = 0.05) -> list[str]:
    """校验时间轴 / validate the timeline.

    返回问题描述列表（空列表 = 通过）。这是**防静默 bug 的闸门**：
    时间轴不单调或越界时，宁可报出来也不要让它流到字幕里。
    """
    issues: list[str] = []
    if not pieces:
        return ["分段结果为空 / no pieces"]
    prev_end = -1.0
    for p in pieces:
        if p.start < -tolerance:
            issues.append(f"段 {p.index} 起点为负 / negative start: {p.start}")
        if p.end <= p.start:
            issues.append(f"段 {p.index} 时长非正 / non-positive duration")
        if p.start < prev_end - tolerance:
            issues.append(f"段 {p.index} 与上一段重叠或逆序 / overlaps previous")
        prev_end = max(prev_end, p.end)
    if pieces[-1].end > duration + tolerance:
        issues.append(f"末段超出音频时长 / last piece ends at {pieces[-1].end:.3f} "
                      f"> duration {duration:.3f}")
    if pieces[0].start > tolerance:
        issues.append(f"首段未从 0 开始 / first piece starts at {pieces[0].start:.3f}")
    return issues


def merge_pieces(pieces: list[SegmentPiece], duration: float | None = None
                 ) -> list[SegmentPiece]:
    """把相邻的碎片合成一整段（用于「不要分段」的场景）."""
    if not pieces:
        return []
    start = pieces[0].start
    end = pieces[-1].end
    return [SegmentPiece(index=0, start=start, end=end, hard_cut=False)]


__all__ = ["merge_pieces", "split_audio", "validate_pieces"]
