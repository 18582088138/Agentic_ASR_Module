"""VAD 行为 / voice activity detection.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_vad.py -q

覆盖 / covers: 能量回退路径（离线）、Silero 分块喂入的正确性（`issues/003` 的回归）。
说明文档 / docs: docs/issues/003-sherpa-onnx-vad-must-be-fed-in-chunks.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from agentic_asr.audio.vad import VadDetector, merge_intervals


def test_energy_fallback_detects_gap(speech_like_wav: Path) -> None:
    """没有模型文件时也必须能用（回退到能量 VAD）。"""
    chunk_samples, sr = sf.read(str(speech_like_wav), dtype="float32")
    detector = VadDetector(model_path=None)
    assert detector.backend == "energy"
    intervals = detector.intervals(chunk_samples, sr)
    assert intervals, "能量 VAD 应至少检出一段语音"
    ratio = detector.speech_ratio(chunk_samples, sr)
    # 素材里 1/3 是静音，所以占比应明显小于 1 但大于 0
    assert 0.2 < ratio < 0.95


@pytest.mark.real
def test_silero_detects_speech_in_clean_clip(speech_like_wav: Path, vad_model: Path | None) -> None:
    """Silero 路径：占比必须够高。

    **这条用例是 issues/003 的回归**。当时的 bug 是「整段一次性喂入」，
    会让占比掉到 0.06 左右；阈值故意写死在 0.5，谁改回去它立刻变红。
    """
    if vad_model is None:
        pytest.skip("本机没有 Silero VAD 模型")
    samples, sr = sf.read(str(speech_like_wav), dtype="float32")
    detector = VadDetector(model_path=vad_model)
    assert detector.backend == "silero"
    ratio = detector.speech_ratio(samples, sr)
    assert ratio > 0.5, f"Silero VAD 占比异常偏低（{ratio:.3f}），检查是否整段喂入"


@pytest.mark.real
def test_silero_long_audio_does_not_drop_intervals(long_wav: Path, vad_model: Path | None) -> None:
    """长音频必须检出**多个**区间（issues/003 的第二个坑：环形缓冲挤掉早先的段）。

    合成素材是「有声 25 s + 静音 + 有声 25 s」，所以正确结果是 2 段。
    曾经的 bug（先喂完再取）会把结果压成 1 段且只剩尾部。
    """
    if vad_model is None:
        pytest.skip("本机没有 Silero VAD 模型")
    samples, sr = sf.read(str(long_wav), dtype="float32")
    detector = VadDetector(model_path=vad_model)
    intervals = detector.intervals(samples, sr)
    assert len(intervals) >= 2, f"区间数偏少（{len(intervals)}），可能边喂边取的逻辑坏了"
    # 关键证据：首段必须从**开头附近**开始，而不是只剩尾部
    assert intervals[0][0] < 5.0, f"首段起点异常靠后：{intervals[0]}"
    covered = sum(e - s for s, e in intervals)
    assert covered > 20.0


def test_merge_intervals() -> None:
    merged = merge_intervals([(0.0, 1.0), (1.2, 2.0), (5.0, 6.0)], gap=0.5)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_empty_input_returns_nothing() -> None:
    detector = VadDetector(model_path=None)
    assert detector.intervals(np.zeros(0, dtype=np.float32), 16000) == []
    assert detector.speech_ratio(np.zeros(0, dtype=np.float32), 16000) == 0.0
