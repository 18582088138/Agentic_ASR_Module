"""测试共用夹具 / Shared pytest fixtures.

**离线优先**：默认的 `addopts = "-m 'not real and not live'"` 会排除真实模型与
真实 API 用例，所以这里提供的素材必须是**合成**的，不能依赖仓库里的大文件。

`speech_like_wav` 造的是「正弦调制 + 静音段」的信号：它的能量包络足够像语音，
VAD（尤其是能量回退路径）能把中间的静音识别成断点，从而让分段与参考音频的
用例有可断言的结构。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 16000


def _speech_like(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """合成「像语音」的信号：两段有声 + 中间静音。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    # 用低频调制 + 谐波堆叠，模拟浊音的音高结构
    carrier = (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t)
               + 0.25 * np.sin(2 * np.pi * 540 * t))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
    signal = carrier * envelope
    signal += 0.01 * rng.standard_normal(n)
    # 中间挖一段静音，保证 VAD 至少能找到一个断点
    third = n // 3
    signal[third: 2 * third] *= 0.0
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


@pytest.fixture(scope="session")
def speech_like_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """9 秒、16 kHz 单声道、中间有静音的合成音频 / a 9 s clip with a silent gap."""
    path = tmp_path_factory.mktemp("audio") / "speech_like.wav"
    sf.write(str(path), _speech_like(9.0), SR, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def silence_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """纯静音（幻觉闸门用）/ pure silence, for the hallucination guard."""
    path = tmp_path_factory.mktemp("audio") / "silence.wav"
    sf.write(str(path), np.zeros(SR * 4, dtype=np.float32), SR, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def long_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """75 秒长素材（切分与时间轴用例）/ 75 s material for splitting tests."""
    path = tmp_path_factory.mktemp("audio") / "long.wav"
    sf.write(str(path), _speech_like(75.0, seed=1), SR, subtype="PCM_16")
    return path


@pytest.fixture()
def module(tmp_path: Path):
    """用 mock 引擎、输出到临时目录的门面 / a facade on the mock engine."""
    from agentic_asr import ASRModule
    from agentic_asr.core.config import load_config

    cfg = load_config(None)
    cfg.output_dir = tmp_path / "outputs"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return ASRModule(cfg, engine="mock")


@pytest.fixture(scope="session")
def vad_model() -> Path | None:
    """本机 Silero VAD 模型；不存在时返回 None（用例自行跳过）."""
    from agentic_asr.core.config import ROOT

    path = ROOT / "models" / "sherpa-vad" / "silero_vad.onnx"
    return path if path.is_file() else None
