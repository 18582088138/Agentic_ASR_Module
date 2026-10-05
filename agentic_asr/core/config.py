"""配置 / Configuration.

单一 YAML（`configs/asr.yaml`）+ 环境变量覆盖。优先级：
**进程环境变量 > `.env` > YAML > 内置默认值**。

设计取舍 / design trade-off：
- 内置默认值必须**能直接跑**（指向 `models/` 下已下载的权重），
  这样 `ASRModule()` 零参数即可工作；
- 密钥**只**从本项目自己的 `.env` 读，不跨目录读 Agentic_TTS_Module 的 key
  —— 那会让两个项目耦合，TTS 一改变量名就连带弄坏 ASR。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_asr.core.errors import ConfigError

# 项目根目录 / project root：agentic_asr/core/config.py -> 上溯三层
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "asr.yaml"


class Secrets(BaseSettings):
    """从环境变量 / `.env` 读到的凭证与运行时覆盖 / credentials & runtime overrides."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    minimax_api_key: str = ""
    asr_api_endpoint: str = ""
    openai_compat_api_key: str = ""
    openai_compat_base_url: str = ""
    openai_compat_model: str = ""

    asr_engine: str = ""
    asr_device: str = ""
    asr_models_dir: str = ""
    asr_output_dir: str = ""
    asr_server_host: str = "127.0.0.1"
    asr_server_port: int = 8301


# ── 各段配置 / config sections ───────────────────────────────────────────────


class FasterWhisperConfig(BaseModel):
    model_dir: str = "models/faster-whisper-large-v3-turbo"
    device: str = "cuda"            # cuda | cpu | auto
    compute_type: str = "float16"   # float16 | int8_float16 | int8 | float32
    beam_size: int = 5
    vad_filter: bool = True
    # 单次送引擎的最长音频（秒）；门面据此切片，避免长音频峰值显存失控
    max_chunk_seconds: float = 300.0
    cpu_fallback: bool = True       # CUDA 不可用时自动退 CPU


class SenseVoiceConfig(BaseModel):
    """sherpa-onnx 的 SenseVoice（CPU）/ SenseVoice via sherpa-onnx on CPU.

    实测：`pip install sherpa-onnx` 只新增 2 个包，且 `from_sense_voice()`
    返回对象原生带 `emotion` / `event` / `lang` 字段 —— 比 funasr 少 16 个包，
    还跑在 CPU 上，不与 faster-whisper 抢显存。
    """

    model_dir: str = "models/sherpa-sensevoice"
    model_file: str = "model.int8.onnx"
    tokens_file: str = "tokens.txt"
    num_threads: int = 4
    provider: str = "cpu"
    use_itn: bool = True
    language: str = "auto"


class MiniMaxConfig(BaseModel):
    # 【实测】本 key 只在国内端点有效；海外 api.minimax.io 返回 401
    endpoint: str = "https://api.minimaxi.com"
    model: str = "asr-1.0"
    max_seconds: float = 480.0      # 官方上限 500 s，留余量
    max_bytes: int = 48 * 1024 * 1024   # 官方上限 50 MB，留余量
    timeout: float = 300.0


class OpenAICompatConfig(BaseModel):
    """通用 OpenAI 兼容适配器 / generic OpenAI-compatible adapter.

    一个适配器覆盖 OpenAI / Groq / 硅基流动 / OpenRouter。
    能力**按配置声明**而非按厂商硬编码 —— 同一厂商内部差异就极大
    （OpenAI `whisper-1` 有词级时间戳，`gpt-4o-transcribe` 完全没有）。
    """

    base_url: str = ""
    model: str = "whisper-1"
    # None = 未知，按「请求了词级就带上参数」处理并在结果里标注
    supports_word_timestamps: bool | None = None
    supports_srt: bool = False
    max_bytes: int = 24 * 1024 * 1024
    timeout: float = 300.0


class EngineConfig(BaseModel):
    default: str = "faster_whisper"
    max_resident: int = 1
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)
    sensevoice: SenseVoiceConfig = Field(default_factory=SenseVoiceConfig)
    minimax: MiniMaxConfig = Field(default_factory=MiniMaxConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)


class TranscribeConfig(BaseModel):
    language: str = "auto"
    word_timestamps: bool = True
    emotion: str = "auto"           # auto | off | always
    diarization: bool = False       # 能力位：第一版未实现，见 01_design.md §10.4
    # 幻觉闸门 / hallucination guard：语音占比低于此值直接返回空，不调用引擎
    min_speech_ratio: float = 0.02
    max_compression_ratio: float = 12.0
    chunk_seconds: float = 480.0    # 云 API 上限留余量
    # 注意 / note：这里**没有**「自动调用分离模组」的开关。
    # 两个模组相互独立、谁也不 import 谁，协作由**应用层**组合完成
    # （跨模组的脚本放 `F:\2026年\Agentic_Pipelines\`，见 01_design.md §4）。
    # 放一个本模组不读的配置项，只会让人以为设了它就自动分离。
    # No auto-separation switch here on purpose: the modules are independent and are
    # composed by the application layer, never by importing one another.


class SegmentConfig(BaseModel):
    """音频分段 / audio segmentation（避免过长 OOM）。"""

    max_seconds: float = 30.0
    min_seconds: float = 0.5
    merge_gap: float = 0.30
    # 停顿超过它就**一定**切段，即使还没到 max_seconds。
    # 体现用户要的「静音点优先」：明显的句子/段落边界不该被贪心合并掉。
    # A pause longer than this always splits, honouring "silence first".
    split_gap: float = 1.00
    pad: float = 0.10
    export_audio: bool = False


class SubtitleConfig(BaseModel):
    """字幕断句与渲染 / subtitle segmentation and rendering.

    断句参数就是**观感旋钮**：把 `max_chars` 调小 → 字幕更短更碎；
    调大 → 更接近整句。规则本身与引擎、音频无关，见 `subtitle/segment.py`。
    These knobs decide how "choppy" the subtitles look.
    """

    formats: list[str] = Field(default_factory=lambda: ["srt"])
    max_line_chars: int = 20        # 每行折行宽度（中日文按字符宽度）

    # ── 断句 / segmentation ──────────────────────────────────────────────
    # 加权字数上限：中文按字计，英文按词 × chars_per_word
    max_chars: int = 20
    # 达到它才允许在「弱边界」（弱停顿 / 次级标点）切，避免切出两三个字的碎片
    min_chars: int = 12
    # 字间静音达到它 → 必切（说话人换气处）
    pause_gap: float = 0.28
    # 字间静音达到它、且已达 min_chars → 可切
    soft_gap: float = 0.12
    # 短于此的尾条并入上一条（否则会一闪而过）
    merge_below_seconds: float = 0.9
    # 单条最长时长，超过则强切（防止一条字幕挂很久）
    max_seconds: float = 6.0
    # 纯英文时每词折算的字符数
    chars_per_word: float = 4.0


class ClipConfig(BaseModel):
    target_seconds: float = 12.0
    min_seconds: float = 10.0
    max_seconds: float = 15.0
    top: int = 3
    sample_rate: int = 16000
    weights: dict[str, float] = Field(default_factory=lambda: {
        "speech_ratio": 0.35, "duration": 0.25, "snr": 0.25, "boundary": 0.15})


class PreprocessConfig(BaseModel):
    """前处理（第一版只留配置位）/ preprocessing (config slot only for v1)."""

    denoise: str = "off"            # off | auto | on
    snr_threshold_db: float = 15.0

    @field_validator("denoise", mode="before")
    @classmethod
    def _normalise_denoise(cls, value: Any) -> Any:
        """YAML 1.1 会把裸写的 `off`/`on`/`yes`/`no` 解析成布尔值。

        这是配置里最容易踩的一个坑：`denoise: off` 会变成 `False`。
        这里统一规范化，免得用户被一个类型错误挡住。
        YAML 1.1 parses bare `off`/`on` as booleans; normalise them back.
        """
        if isinstance(value, bool):
            return "on" if value else "off"
        return value


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8301


class ASRConfig(BaseModel):
    """顶层配置 / top-level configuration."""

    engine: EngineConfig = Field(default_factory=EngineConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    models_dir: Path = ROOT / "models"
    output_dir: Path = ROOT / "outputs"

    # ── 便捷访问器 / convenience accessors ──────────────────────────────────

    def resolve_model_dir(self, value: str | Path) -> Path:
        """把配置里的相对路径按项目根解析 / resolve a relative model path."""
        p = Path(value)
        return p if p.is_absolute() else (ROOT / p)

    def faster_whisper_model_path(self) -> Path:
        return self.resolve_model_dir(self.engine.faster_whisper.model_dir)

    def sensevoice_paths(self) -> tuple[Path, Path]:
        d = self.resolve_model_dir(self.engine.sensevoice.model_dir)
        return d / self.engine.sensevoice.model_file, d / self.engine.sensevoice.tokens_file

    def api_key(self, engine: str) -> str:
        """按引擎取密钥 / fetch the credential for an engine."""
        secrets = Secrets()
        if engine == "minimax":
            return secrets.minimax_api_key
        if engine == "openai_compat":
            return secrets.openai_compat_api_key
        return ""


# ── 加载 / loading ───────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> ASRConfig:
    """加载配置 / load configuration.

    参数 / Args:
        path: YAML 路径；None 表示用 `configs/asr.yaml`（不存在则纯默认值）。
        **overrides: 直接覆盖顶层字段（测试与 CLI 用）。

    抛出 / Raises:
        ConfigError: YAML 非法或字段类型不对。
    """
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if cfg_path.is_file():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - 仅坏 YAML 时触发
            raise ConfigError(f"YAML 解析失败 / cannot parse {cfg_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"配置根节点必须是映射 / root must be a mapping: {cfg_path}")
        data = loaded

    secrets = Secrets()
    # 环境变量优先级最高 / process env wins over YAML
    if secrets.asr_engine:
        data = _deep_merge(data, {"engine": {"default": secrets.asr_engine}})
    if secrets.asr_device:
        data = _deep_merge(data, {"engine": {"faster_whisper": {"device": secrets.asr_device}}})
    if secrets.asr_server_host:
        data = _deep_merge(data, {"server": {"host": secrets.asr_server_host}})
    if secrets.asr_server_port:
        data = _deep_merge(data, {"server": {"port": int(secrets.asr_server_port)}})
    if secrets.asr_models_dir:
        data = _deep_merge(data, {"models_dir": secrets.asr_models_dir})
    if secrets.asr_output_dir:
        data = _deep_merge(data, {"output_dir": secrets.asr_output_dir})
    if secrets.asr_api_endpoint:
        data = _deep_merge(data, {"engine": {"minimax": {"endpoint": secrets.asr_api_endpoint}}})
    if secrets.openai_compat_base_url:
        data = _deep_merge(data, {"engine": {"openai_compat": {
            "base_url": secrets.openai_compat_base_url}}})
    if secrets.openai_compat_model:
        data = _deep_merge(data, {"engine": {"openai_compat": {
            "model": secrets.openai_compat_model}}})
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        cfg = ASRConfig(**data)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"配置字段非法 / invalid config: {exc}") from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def env_flag(name: str, default: bool = False) -> bool:
    """读布尔环境变量 / read a boolean env var."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "ASRConfig",
    "ROOT",
    "Secrets",
    "load_config",
]
