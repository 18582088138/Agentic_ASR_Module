"""门面与配置 / facade and configuration.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_module.py tests/test_config.py -q

覆盖 / covers: mock 引擎全链路（V1）、引擎清单与能力查询不加载权重、配置加载与
YAML 布尔陷阱、时间轴平移合并的正确性。
说明文档 / docs: docs/01_design.md §7
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_asr.asr import _merge, _shift
from agentic_asr.core.config import ASRConfig, Secrets, load_config
from agentic_asr.core.errors import ConfigError
from agentic_asr.core.registry import available, resolve
from agentic_asr.core.types import Capability, Segment, Transcript

# ── 配置 / configuration ─────────────────────────────────────────────────────


def test_default_config_is_runnable() -> None:
    cfg = load_config(None)
    assert cfg.engine.default
    assert cfg.output_dir.exists()


def test_yaml_bare_off_is_not_boolean(tmp_path: Path) -> None:
    """YAML 1.1 会把裸写的 `off` 解析成布尔 False —— 必须被规范化回字符串。

    这条回归曾经真的挡住过一次启动：`preprocess.denoise: off` 触发
    pydantic 的 `string_type` 校验错误。
    """
    p = tmp_path / "c.yaml"
    p.write_text("preprocess:\n  denoise: off\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.preprocess.denoise == "off"
    assert isinstance(cfg.preprocess.denoise, str)


def test_quoted_on_stays_string(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("preprocess:\n  denoise: 'on'\n", encoding="utf-8")
    assert load_config(p).preprocess.denoise == "on"


def test_invalid_field_raises_config_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("subtitle:\n  max_chars: not-a-number\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_ENGINE", "mock")
    cfg = load_config(None)
    assert cfg.engine.default == "mock"


def test_relative_model_paths_resolve_to_root() -> None:
    cfg = ASRConfig()
    assert cfg.faster_whisper_model_path().is_absolute()
    model, tokens = cfg.sensevoice_paths()
    assert model.is_absolute() and tokens.is_absolute()


def test_api_key_returns_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """取密钥必须返回字符串（缺配置时为空串），不能是 None 或抛异常。

    注意不能断言「一定为空」—— 仓库里存在 `.env` 时它会读到真实 key，
    所以这里只校验类型契约，避免把环境状态写进断言。
    """
    assert isinstance(ASRConfig().api_key("minimax"), str)
    assert isinstance(ASRConfig().api_key("nonexistent-engine"), str)
    assert ASRConfig().api_key("nonexistent-engine") == ""


# ── 注册表 / registry ────────────────────────────────────────────────────────


def test_registry_lists_all_engines() -> None:
    names = available()
    for expected in ("faster_whisper", "sensevoice", "minimax", "openai_compat", "mock"):
        assert expected in names


def test_capabilities_answerable_without_weights() -> None:
    """能力查询**不得**加载权重 —— GUI 开机就要用它把开关置灰。"""
    for name in available():
        caps = resolve(name).declared_capabilities()
        assert isinstance(caps, set)
        assert all(isinstance(c, Capability) for c in caps)


def test_unknown_engine_raises() -> None:
    with pytest.raises(ConfigError):
        resolve("nope")


# ── 门面 / facade ────────────────────────────────────────────────────────────


def test_mock_end_to_end(module, speech_like_wav: Path) -> None:
    """V1：不加载任何权重跑通转写、分段、信息提取。"""
    info = module.probe(speech_like_wav)
    assert info.duration > 0
    pieces = module.segment(speech_like_wav)
    assert pieces
    result = module.transcribe(speech_like_wav)
    assert result.text.startswith("mock segment")
    assert result.segments
    assert result.emotion_summary


def test_engines_endpoint_like_listing(module) -> None:
    items = module.engines()
    names = {i["name"] for i in items}
    assert "mock" in names and "faster_whisper" in names
    default = [i for i in items if i.get("default")]
    assert len(default) == 1


def test_doctor_reports_models(module) -> None:
    report = module.doctor()
    assert "models" in report and "checks" in report
    assert "cuda_runtime" in report
    assert isinstance(report["checks"], list)


def test_transcribe_writes_srt(module, speech_like_wav: Path, tmp_path: Path) -> None:
    """V4：SRT 可被外部工具导入（这里校验时间码格式与序号）。"""
    out = tmp_path / "a.srt"
    module.transcribe(speech_like_wav, srt=out)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "-->" in text
    assert text.startswith("1\n")


def test_capability_error_is_raised_not_silently_ignored(module, speech_like_wav: Path) -> None:
    """请求引擎不具备的能力时必须报错，而不是悄悄降级。"""
    from agentic_asr.core.errors import CapabilityError
    from agentic_asr.engines.base import ASREngine

    class NoWords(ASREngine):
        name = "nowords"
        implemented = True
        max_audio_seconds = None

        @classmethod
        def declared_capabilities(cls):
            return {Capability.TIMESTAMPS}

        def transcribe(self, chunk, *, language="auto", word_timestamps=True, **kw):
            self.require(Capability.WORD_TIMESTAMPS)
            raise AssertionError("不该走到这里")

    class Cfg:
        engine = None
        segment = None

        class transcribe:  # noqa: N801
            language = "auto"
            word_timestamps = True
            emotion = "off"
            min_speech_ratio = 0.0
            max_compression_ratio = 1e9
            chunk_seconds = 480.0

    eng = NoWords(Cfg())
    with pytest.raises(CapabilityError):
        eng.transcribe(None)  # type: ignore[arg-type]


# ── 时间轴平移与合并 / shifting and merging ──────────────────────────────────


def test_shift_moves_segments_and_words() -> None:
    from agentic_asr.core.types import Word

    tr = Transcript(text="a", language="zh", duration=1.0, engine="mock",
                    segments=[Segment(id=0, start=0.0, end=1.0, text="a",
                                      words=[Word(text="a", start=0.0, end=1.0)])])
    _shift(tr, 10.0)
    assert tr.segments[0].start == pytest.approx(10.0)
    assert tr.segments[0].words[0].start == pytest.approx(10.0)


def test_clamp_keeps_output_inside_the_slice() -> None:
    """Whisper 会给出超出实际音频的尾段时间戳，必须裁剪回该片区间。

    实测一次 123.55 s 的素材，末段被标到 131.51 s —— 字幕比音频长 8 秒。
    """
    from agentic_asr.asr import _clamp
    from agentic_asr.core.types import Word

    tr = Transcript(text="x", language="zh", duration=30.0, engine="mock",
                    segments=[Segment(id=0, start=8.0, end=25.0, text="x",
                                      words=[Word(text="x", start=8.0, end=25.0)]),
                              Segment(id=1, start=40.0, end=60.0, text="y")])
    _clamp(tr, 10.0)
    assert tr.segments[0].end == pytest.approx(10.0)
    assert tr.segments[0].words[0].end == pytest.approx(10.0)
    # 起点就超出区间的段必须被丢掉，而不是留下一个 start > end 的怪物
    assert [s.text for s in tr.segments] == ["x"]


def test_merge_renumbers_and_sorts() -> None:
    a = Transcript(text="A", language="zh", duration=5.0, engine="mock",
                   segments=[Segment(id=7, start=3.0, end=5.0, text="A")])
    b = Transcript(text="B", language="zh", duration=5.0, engine="mock",
                   segments=[Segment(id=9, start=0.0, end=2.0, text="B")])
    merged = _merge([a, b], 5.0, "mock")
    assert [s.id for s in merged.segments] == [0, 1]
    assert [s.text for s in merged.segments] == ["B", "A"]
    assert merged.text == "AB"


def test_release_clears_engines(module) -> None:
    module.engine("mock")
    module.release()
    assert not module._engines


def test_secrets_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "dummy-not-a-real-key")
    assert Secrets().minimax_api_key == "dummy-not-a-real-key"
