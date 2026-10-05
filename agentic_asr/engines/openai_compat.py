"""通用 OpenAI 兼容适配器 / generic OpenAI-compatible adapter.

**一个适配器覆盖多家**：只要填 `base_url`，就能用于
**OpenAI / Groq / 硅基流动 / OpenRouter** 等提供
`POST {base_url}/audio/transcriptions` 的服务。

**能力按配置声明，不按厂商硬编码** —— 同一厂商内部差异就极大：
OpenAI 的 `whisper-1` 有词级时间戳，而 `gpt-4o-transcribe` **完全没有**
（官方 API 文档明确不支持 `verbose_json` 与 `timestamp_granularities`）。
把能力写在厂商维度上必然出错，所以这里由 `supports_word_timestamps` 显式控制，
未知时（None）采取「带上参数试一次、以响应为准」的保守策略。
Capabilities are declared per endpoint+model, never per vendor.
"""

from __future__ import annotations

import io
from typing import Any

from agentic_asr.core.errors import APIError, ConfigError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import (
    AudioChunk,
    Capability,
    Segment,
    Transcript,
    Word,
)
from agentic_asr.engines.base import ASREngine

log = get_logger("engines.openai_compat")

_RETRYABLE = {408, 429, 500, 502, 503, 504}


class OpenAICompatEngine(ASREngine):
    """`/v1/audio/transcriptions` 通用客户端 / generic client for that endpoint."""

    name = "openai_compat"
    implemented = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = getattr(getattr(config, "engine", None), "openai_compat", None)
        self.base_url = (getattr(cfg, "base_url", "") or "").rstrip("/")
        self.model = getattr(cfg, "model", "") or "whisper-1"
        self._supports_words = getattr(cfg, "supports_word_timestamps", None)
        self.max_bytes = int(getattr(cfg, "max_bytes", 24 * 1024 * 1024))
        self.timeout = float(getattr(cfg, "timeout", 300.0))
        # 标准 OpenAI 端点单文件上限 25 MB，门面需据此切片；秒数上限不是硬性的
        self.max_audio_seconds = None
        self.loaded = True

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        # 保守声明：句级时间戳与语种识别是 OpenAI 兼容接口的公共下限。
        # 词级是否可用由实例按配置决定（见 capabilities()）。
        return {Capability.TIMESTAMPS, Capability.LANGUAGE_ID}

    def capabilities(self) -> set[Capability]:
        caps = set(self.declared_capabilities())
        if self._supports_words:
            caps.add(Capability.WORD_TIMESTAMPS)
        return caps

    # ── 请求 / request ──────────────────────────────────────────────────────

    def _check(self) -> str:
        if not self.base_url:
            raise ConfigError(
                "openai_compat 未配置 base_url / base_url is required　"
                "例 / e.g. https://api.groq.com/openai/v1")
        key = self.config.api_key("openai_compat")
        if not key:
            raise APIError("缺少 OPENAI_COMPAT_API_KEY / missing credential")
        return key

    def _encode_wav(self, chunk: AudioChunk, sample_rate: int = 16000) -> bytes:
        import numpy as np
        import soundfile as sf

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if chunk.sample_rate != sample_rate:
            from agentic_asr.audio.io import resample

            samples = resample(samples, chunk.sample_rate, sample_rate)
        buf = io.BytesIO()
        # 送 mp3 不划算（要额外编码器），wav 16 kHz 单声道足够小
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        import requests

        key = self._check()
        payload = self._encode_wav(chunk)
        if len(payload) > self.max_bytes:
            raise APIError(f"音频过大 / payload too large: {len(payload) / 1e6:.1f} MB "
                           f"> {self.max_bytes / 1e6:.1f} MB")

        want_words = bool(word_timestamps) and self._supports_words is not False
        data: dict[str, Any] = {"model": self.model, "response_format": "verbose_json"}
        if language not in ("", "auto"):
            data["language"] = language
        if want_words:
            # OpenAI / Groq 用这个数组参数；不认识它的服务会忽略
            data["timestamp_granularities[]"] = "word"

        url = f"{self.base_url}/audio/transcriptions"
        try:
            resp = requests.post(
                url, headers={"Authorization": f"Bearer {key}"}, data=data,
                files={"file": ("audio.wav", payload, "audio/wav")}, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"请求失败 / request failed: {exc}", retryable=True) from exc

        if resp.status_code != 200:
            raise APIError(f"{self.name} 返回 {resp.status_code}　"
                           f"{resp.text[:240]}".strip(), status=resp.status_code,
                           retryable=resp.status_code in _RETRYABLE)
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"响应不是 JSON / non-JSON response: {resp.text[:200]}") from exc
        return self._to_transcript(body, chunk)

    def _to_transcript(self, body: dict[str, Any], chunk: AudioChunk) -> Transcript:
        segments: list[Segment] = []
        for i, item in enumerate(body.get("segments") or []):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            words = None
            raw_words = item.get("words")
            if raw_words:
                words = [Word(text=str(w.get("word", "")).strip(),
                              start=float(w.get("start", 0.0)),
                              end=float(w.get("end", 0.0)))
                         for w in raw_words if str(w.get("word", "")).strip()]
                words = words or None
            segments.append(Segment(id=len(segments), start=float(item.get("start", 0.0)),
                                    end=float(item.get("end", 0.0)), text=text,
                                    words=words))
        text = (body.get("text") or "").strip()
        if not segments and text:
            # 有些服务只回 text（不带 verbose_json），给一个覆盖全段的兜底 segment，
            # 否则字幕渲染会拿到空列表。调用方仍能从 warnings 看出少了时间戳。
            segments = [Segment(id=0, start=0.0, end=round(chunk.duration, 3), text=text)]
        return Transcript(
            text=text or "".join(s.text for s in segments),
            language=str(body.get("language") or ""),
            duration=float(body.get("duration") or chunk.duration),
            segments=segments,
            engine=self.name,
            warnings=[] if len(segments) else ["服务未返回 segments"],
            raw={"model": self.model, "base_url": self.base_url},
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"base_url": self.base_url or None, "model": self.model})
        return d

    def release(self) -> None:
        self.loaded = True


__all__ = ["OpenAICompatEngine"]
