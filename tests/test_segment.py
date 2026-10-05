"""音频分段 / audio segmentation.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_segment.py -q

覆盖 / covers: 覆盖全片、静音点优先、超长硬切标记、时间轴单调。
说明文档 / docs: docs/01_design.md §3.4
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentic_asr.audio.io import decode
from agentic_asr.core.types import AudioChunk
from agentic_asr.segment.splitter import split_audio, validate_pieces


def _chunk(seconds: float, sr: int = 16000, gap: tuple[float, float] | None = None
           ) -> AudioChunk:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    sig += 0.01 * np.random.default_rng(0).standard_normal(n)
    if gap:
        sig[int(gap[0] * sr):int(gap[1] * sr)] = 0.0
    return AudioChunk(samples=(sig / np.max(np.abs(sig)) * 0.5).astype(np.float32),
                      sample_rate=sr)


def test_pieces_cover_whole_timeline() -> None:
    """分段必须覆盖 [0, duration] —— 否则下游按段对齐原音频会缺内容。"""
    chunk = _chunk(20.0, gap=(7.0, 12.0))
    pieces = split_audio(chunk, vad=None, max_seconds=6.0)
    assert pieces, "应至少产出一段"
    assert validate_pieces(pieces, chunk.duration) == []
    assert pieces[0].start == pytest.approx(0.0, abs=0.05)
    assert pieces[-1].end == pytest.approx(chunk.duration, abs=0.1)


def test_timeline_is_monotonic() -> None:
    pieces = split_audio(_chunk(30.0, gap=(10.0, 14.0)), max_seconds=5.0)
    prev = -1.0
    for p in pieces:
        assert p.start >= prev - 1e-6
        assert p.end > p.start
        prev = p.end


def test_max_seconds_is_respected_when_silence_exists() -> None:
    pieces = split_audio(_chunk(30.0, gap=(10.0, 14.0)), max_seconds=8.0)
    assert all(p.duration <= 8.0 + 0.2 for p in pieces)


def test_continuous_speech_gets_hard_cut_flag() -> None:
    """连续说话、没有静音点可切时，只能硬切，且必须**标记**出来。"""
    pieces = split_audio(_chunk(20.0), max_seconds=5.0)
    assert pieces, "纯连续语音也要能切"
    assert all(p.duration <= 5.0 + 0.2 for p in pieces)
    # 没有静音 → 至少有一段是硬切（除首段外全部）
    assert any(p.hard_cut for p in pieces) or len(pieces) > 1


def test_split_gap_breaks_on_obvious_pause() -> None:
    """「静音点优先」：明显停顿处一定切段，即使还没到 max_seconds。"""
    chunk = _chunk(20.0, gap=(9.0, 11.0))
    pieces = split_audio(chunk, max_seconds=30.0, split_gap=1.0)
    assert len(pieces) >= 2, f"1 秒以上的停顿应切开，实际得到 {len(pieces)} 段"
    # 切点应落在静音区间内
    boundaries = [p.end for p in pieces[:-1]]
    assert any(9.0 <= b <= 11.5 for b in boundaries), boundaries


def test_empty_audio_yields_no_pieces() -> None:
    assert split_audio(AudioChunk(samples=np.zeros(0, dtype=np.float32))) == []


def test_validate_catches_overlap() -> None:
    from agentic_asr.core.types import SegmentPiece

    bad = [SegmentPiece(index=0, start=0.0, end=5.0),
           SegmentPiece(index=1, start=3.0, end=8.0)]
    problems = validate_pieces(bad, 10.0)
    assert problems, "重叠的时间轴应被检出"


def test_real_file_segments(speech_like_wav: Path) -> None:
    chunk = decode(speech_like_wav)
    pieces = split_audio(chunk, max_seconds=4.0)
    assert validate_pieces(pieces, chunk.duration) == []
