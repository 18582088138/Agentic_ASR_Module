"""幻觉闸门 / hallucination guard.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_guard.py -q

覆盖 / covers: 纯静音不进引擎、压缩比异常被标记（`issues/002` 的回归）。
说明文档 / docs: docs/issues/002-music-only-hallucination.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentic_asr.core.types import AudioChunk, Segment, Transcript


def test_silence_returns_empty_without_calling_engine(module, silence_wav: Path) -> None:
    """纯静音必须返回空，且**不调用引擎** —— 实测 Whisper 对无语音输入会编内容。"""
    result = module.transcribe(silence_wav)
    assert result.text == ""
    assert result.segments == []
    assert result.suspicious is True
    assert result.warnings, "必须留下说明，不能静默返回空"


def test_silence_guard_blocks_before_engine(module) -> None:
    """把引擎换成一个「一旦被调用就爆炸」的假引擎，验证闸门真的在前置拦截。"""
    from agentic_asr.core.types import Capability
    from agentic_asr.engines.base import ASREngine

    class Exploding(ASREngine):
        name = "boom"
        implemented = True
        max_audio_seconds = None

        @classmethod
        def declared_capabilities(cls) -> set[Capability]:
            return {Capability.TIMESTAMPS, Capability.WORD_TIMESTAMPS}

        def transcribe(self, chunk, **kwargs):  # pragma: no cover - 不该被调用
            raise AssertionError("闸门没有拦住静音输入 / guard failed to block silence")

    module._engines["boom"] = Exploding(module.config)
    chunk = AudioChunk(samples=np.zeros(16000 * 3, dtype=np.float32), sample_rate=16000)
    result = module.transcribe(chunk, engine="boom")
    assert result.text == ""
    assert result.suspicious is True


def test_speech_passes_the_guard(module, speech_like_wav: Path) -> None:
    result = module.transcribe(speech_like_wav)
    assert result.text, "有语音的素材不应被闸门拦掉"
    assert not result.suspicious


def test_post_guard_flags_dense_output() -> None:
    """压缩比异常 → 标记可疑（只标记，不丢弃）。"""
    from agentic_asr.asr import _post_guard

    tr = Transcript(text="字" * 5000, language="zh", duration=10.0,
                    segments=[Segment(id=0, start=0.0, end=10.0, text="字" * 5000)],
                    engine="mock")
    _post_guard(tr, max_ratio=12.0)
    assert tr.suspicious is True
    assert any("压缩比" in w for w in tr.warnings)


def test_post_guard_ignores_normal_output() -> None:
    from agentic_asr.asr import _post_guard

    tr = Transcript(text="正常的一句话", language="zh", duration=10.0,
                    segments=[Segment(id=0, start=0.0, end=2.0, text="正常的一句话")],
                    engine="mock")
    _post_guard(tr, max_ratio=12.0)
    assert tr.suspicious is False


def test_empty_audio_returns_empty(module) -> None:
    chunk = AudioChunk(samples=np.zeros(0, dtype=np.float32), sample_rate=16000)
    result = module.transcribe(chunk)
    assert result.text == ""


def test_real_engine_does_not_hallucinate_on_silence(silence_wav: Path) -> None:
    """`-m real`：真实 fast-whisper 对静音也不得产出「优优独播剧场」类内容。

    这条**故意不写死幻觉文本**（不同模型/版本幻觉内容不同），
    只断言「结果为空或不含字幕组/剧场类典型幻觉词」。
    """
    pytest.skip("需要真实权重，见 tests/test_real.py")
