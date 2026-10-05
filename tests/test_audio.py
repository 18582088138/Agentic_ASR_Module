"""音频解码与归一 / decoding, normalisation, slicing.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_audio.py -q

覆盖 / covers: 16 kHz 单声道归一（`issues/001` 的回归）、切片时间轴、重采样。
不含 / excludes: 真实引擎推理（见 `test_real.py`）。
说明文档 / docs: docs/issues/001-faster-whisper-incompatible-with-av19.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from agentic_asr.audio.io import decode, decode_bytes, resample, slice_chunk
from agentic_asr.core.errors import DecodeError


def test_decode_to_16k_mono(speech_like_wav: Path) -> None:
    """任意输入都必须归一成 16 kHz 单声道 float32（issues/001 的回归）。"""
    chunk = decode(speech_like_wav)
    assert chunk.sample_rate == 16000
    assert chunk.samples.ndim == 1
    assert chunk.samples.dtype == np.float32
    assert chunk.duration == pytest.approx(9.0, abs=0.05)


def test_decode_stereo_is_downmixed(tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    # 8000 样本 @ 8 kHz = 1.0 s；重采样到 16 kHz 后时长不变
    data = np.stack([np.linspace(-0.5, 0.5, 8000, dtype=np.float32)] * 2, axis=1)
    sf.write(str(stereo), data, 8000, subtype="PCM_16")
    chunk = decode(stereo)
    assert chunk.samples.ndim == 1
    assert chunk.sample_rate == 16000
    assert chunk.duration == pytest.approx(1.0, abs=0.05)


def test_decode_missing_file_raises() -> None:
    with pytest.raises(DecodeError):
        decode("Z:/definitely/not/here.wav")


def test_decode_bytes_roundtrip(speech_like_wav: Path) -> None:
    payload = speech_like_wav.read_bytes()
    chunk = decode_bytes(payload, suffix=".wav")
    assert chunk.duration == pytest.approx(9.0, abs=0.05)
    assert chunk.source.startswith("<bytes:")


def test_slice_keeps_timeline_offset(speech_like_wav: Path) -> None:
    """切片必须记住自己在原音频里的位置，否则合并时会整体漂移。"""
    chunk = decode(speech_like_wav)
    part = slice_chunk(chunk, 2.0, 5.0)
    assert part.duration == pytest.approx(3.0, abs=0.02)
    assert part.offset == pytest.approx(2.0, abs=0.02)


def test_slice_out_of_range_is_clamped(speech_like_wav: Path) -> None:
    chunk = decode(speech_like_wav)
    part = slice_chunk(chunk, -5.0, 999.0)
    assert part.duration == pytest.approx(chunk.duration, abs=0.05)


def test_resample_preserves_duration() -> None:
    samples = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000).astype(np.float32)
    out = resample(samples, 8000, 16000)
    assert abs(len(out) / 16000 - len(samples) / 8000) < 0.05
