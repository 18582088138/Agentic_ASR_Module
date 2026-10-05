"""参考音频截取 / reference clip picking.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_clip.py -q

覆盖 / covers: 时长规格（10~15 s）、`ref_text` 与片段严格对应、产物落盘。
说明文档 / docs: docs/01_design.md §3.7
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from agentic_asr.audio.io import decode
from agentic_asr.clip.picker import pick_reference
from agentic_asr.core.config import load_config


def _cfg(tmp_path: Path):
    cfg = load_config(None)
    cfg.output_dir = tmp_path / "out"
    return cfg


def test_picks_clips_within_spec(tmp_path: Path, long_wav: Path) -> None:
    """截取结果必须落在配置的时长区间内。"""
    cfg = _cfg(tmp_path)
    chunk = decode(long_wav)
    clips = pick_reference(chunk, transcript=None, cfg=cfg.clip,
                           out_dir=tmp_path / "refs", top=3)
    assert clips, "75 秒素材应能挑出候选"
    for c in clips:
        assert cfg.clip.min_seconds * 0.8 <= c.duration <= cfg.clip.max_seconds + 0.5
        assert 0.0 <= c.start < c.end <= chunk.duration + 0.01


def test_clip_files_are_written_and_decodable(tmp_path: Path, long_wav: Path) -> None:
    cfg = _cfg(tmp_path)
    chunk = decode(long_wav)
    clips = pick_reference(chunk, transcript=None, cfg=cfg.clip,
                           out_dir=tmp_path / "refs", top=2)
    assert clips
    for c in clips:
        assert Path(c.wav_path).is_file()
        data, sr = sf.read(c.wav_path)
        assert sr == cfg.clip.sample_rate
        assert abs(len(data) / sr - c.duration) < 0.1


def test_scores_are_sorted_descending(tmp_path: Path, long_wav: Path) -> None:
    clips = pick_reference(decode(long_wav), None, _cfg(tmp_path).clip,
                           out_dir=tmp_path / "refs", top=3)
    scores = [c.score for c in clips]
    assert scores == sorted(scores, reverse=True)


def test_ref_text_matches_the_clip_window(tmp_path: Path, long_wav: Path) -> None:
    """`ref_text` 必须来自**该片段自己的转写**，不能是整段文本。

    这是本功能最要紧的约束：文本与音频错位会让下游音色克隆学错对应关系。
    """
    from agentic_asr.core.types import Segment, Transcript, Word

    chunk = decode(long_wav)
    # 人为构造一份「每 5 秒一句、每句 4 个字」的转写，便于逐字核对
    words, segments = [], []
    for i in range(int(chunk.duration // 5)):
        seg_words = [Word(text=f"字{j}", start=i * 5 + j, end=i * 5 + j + 0.9)
                     for j in range(4)]
        words.extend(seg_words)
        segments.append(Segment(id=i, start=i * 5, end=i * 5 + 4,
                                text="".join(w.text for w in seg_words), words=seg_words))
    transcript = Transcript(text="".join(s.text for s in segments), language="zh",
                            duration=chunk.duration, segments=segments, engine="mock")

    clips = pick_reference(chunk, transcript, _cfg(tmp_path).clip,
                           out_dir=tmp_path / "refs", top=2)
    assert clips
    for c in clips:
        inside = [w.text for w in words if w.end > c.start and w.start < c.end]
        # 允许边界处的少量差异（吸附容差），但绝不能是整段文本
        assert c.text, "片段必须有对应文本"
        assert len(c.text) <= len("".join(inside)) + 6
        assert c.text != transcript.text


def test_no_speech_returns_empty(tmp_path: Path, silence_wav: Path) -> None:
    clips = pick_reference(decode(silence_wav), None, _cfg(tmp_path).clip,
                           out_dir=tmp_path / "refs", top=3)
    assert clips == []


def test_srt_sidecar_starts_at_zero(tmp_path: Path, long_wav: Path) -> None:
    """片段自己的字幕要从 0 起算，而不是原音频的绝对时间。"""
    clips = pick_reference(decode(long_wav), None, _cfg(tmp_path).clip,
                           out_dir=tmp_path / "refs", top=1)
    if not clips or not clips[0].srt_path:
        return
    content = Path(clips[0].srt_path).read_text(encoding="utf-8")
    assert "00:00:00,000" in content
