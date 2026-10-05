"""真实模型用例 / real-model tests（默认被 addopts 排除）。

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_real.py -q -m real

覆盖 / covers: 真实 faster-whisper 转写（中/英/日）、真实 SenseVoice 情绪、真实的
「不预先 import torch 也能 GPU 推理」（`issues/004` 的回归）。
不含 / excludes: 云 API（见 `test_live.py`，会花钱）。
说明文档 / docs: docs/00_research.md §4、docs/issues/004
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agentic_asr.core.config import ROOT

pytestmark = pytest.mark.real

FW_DIR = ROOT / "models" / "faster-whisper-large-v3-turbo"
SV_DIR = ROOT / "models" / "sherpa-sensevoice"
SAMPLES = ROOT / "models" / "SenseVoiceSmall" / "example"


def _need(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"缺少真实模型 / missing weights: {path}")


@pytest.fixture(scope="module")
def asr():
    from agentic_asr import ASRModule

    _need(FW_DIR)
    module = ASRModule()
    yield module
    module.release()


@pytest.mark.parametrize(("sample", "keywords"), [
    ("zh.mp3", ("开放时间", "九点", "五点")),
    ("en.mp3", ("tribal", "chieftain", "gold")),
    ("ja.mp3", ("弁当", "中学", "50")),
])
def test_transcribes_three_languages(asr, sample: str, keywords: tuple[str, ...]) -> None:
    """中/英/日三语关键词命中 —— 这是选型时实测过的原始结论。"""
    path = SAMPLES / sample
    if not path.is_file():
        pytest.skip(f"缺少样例 / missing sample: {path}")
    result = asr.transcribe(path, emotion=False)
    lowered = result.text.lower()
    assert any(k.lower() in lowered for k in keywords), \
        f"{sample} 未命中任何关键词：{result.text!r}"
    assert result.segments, "应有句级分段"


def test_word_timestamps_are_monotonic(asr) -> None:
    """V3：词级时间戳可用且单调。"""
    path = SAMPLES / "zh.mp3"
    if not path.is_file():
        pytest.skip("缺少中文样例")
    result = asr.transcribe(path, words=True, emotion=False)
    words = [w for s in result.segments for w in (s.words or [])]
    assert words, "应产出词级时间戳"
    prev_end = -1.0
    for w in words:
        assert w.start < w.end
        assert w.start >= prev_end - 0.05
        prev_end = w.end


def test_emotion_engine_returns_label(asr) -> None:
    """V5：真实 SenseVoice 情绪必须是 7 类之一，且 `<|TAG|>` 被正确解析。"""
    _need(SV_DIR / "model.int8.onnx")
    if not (SAMPLES / "zh.mp3").is_file():
        pytest.skip("缺少样例")
    from agentic_asr.audio.io import decode

    engine = asr.engine("sensevoice")
    result = engine.transcribe(decode(SAMPLES / "zh.mp3"))
    assert result.segments
    emotion = result.segments[0].emotion
    assert emotion in {"neutral", "happy", "sad", "angry",
                       "fearful", "disgusted", "surprised"}
    assert "<|" not in str(emotion), "标记没有被解析掉"


def test_emotion_backfill_path(asr) -> None:
    """主力引擎没有情绪能力时，门面必须按段补齐。"""
    _need(SV_DIR / "model.int8.onnx")
    if not (SAMPLES / "zh.mp3").is_file():
        pytest.skip("缺少样例")
    result = asr.transcribe(SAMPLES / "zh.mp3", emotion=True)
    assert result.emotion_summary
    assert any(s.emotion for s in result.segments)


def test_gpu_inference_without_torch_preimport(tmp_path: Path) -> None:
    """`issues/004` 的回归：**不先 import torch** 也要能 GPU 推理。

    这个 bug 的表现是「另一个脚本能用、这个不能用」，根因是 `cublas64_12.dll`
    靠 torch 的 `__init__` 注册进 DLL 搜索路径。本用例在**独立子进程**里
    断言 ASR 可以自行完成注册。
    """
    _need(FW_DIR)
    sample = SAMPLES / "zh.mp3"
    if not sample.is_file():
        pytest.skip("缺少样例")
    code = textwrap.dedent(f"""
        import sys
        assert 'torch' not in sys.modules, '子进程里不应预先导入 torch'
        from agentic_asr import ASRModule
        asr = ASRModule()
        result = asr.transcribe(r"{sample}", emotion=False)
        asr.release()
        assert result.text.strip(), 'GPU 转写返回空，CUDA 运行库可能没注册上'
        print('OK', len(result.text))
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, f"子进程失败：{proc.stdout}\n{proc.stderr}"


def test_silence_does_not_hallucinate(asr) -> None:
    """V10 的真实模型版本：真实引擎对静音也不得编出字幕。

    **故意不写死幻觉文本**（不同模型/版本内容不同），只断言结果为空。
    """
    import numpy as np

    from agentic_asr.core.types import AudioChunk

    chunk = AudioChunk(samples=np.zeros(16000 * 5, dtype=np.float32), sample_rate=16000)
    result = asr.transcribe(chunk, emotion=False)
    assert not result.text.strip(), f"静音被编出了内容：{result.text!r}"


def test_long_audio_timeline(asr) -> None:
    """V8：长音频切片后，时间轴仍正确（末段不超时长）。"""
    import numpy as np
    import soundfile as sf

    from agentic_asr.core.types import AudioChunk

    if not (SAMPLES / "zh.mp3").is_file():
        pytest.skip("缺少样例")
    base, sr = sf.read(str(SAMPLES / "zh.mp3"), dtype="float32")
    reps = int(120 * sr / len(base)) + 1
    long = np.tile(base, reps)
    chunk = AudioChunk(samples=long.astype(np.float32), sample_rate=sr)
    result = asr.transcribe(chunk, emotion=False)
    assert result.segments
    assert result.segments[-1].end <= chunk.duration + 0.6
    prev = -1.0
    for s in result.segments:
        assert s.start >= prev - 0.05
        prev = s.start
