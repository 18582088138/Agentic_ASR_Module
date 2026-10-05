"""MiniMax Speech-to-Text 适配器 / MiniMax ASR adapter.

**能力**（官方 OpenAPI 规格 + 本机真实调用实测）：

`POST {endpoint}/v1/speech_to_text`，`multipart/form-data`，
`Authorization: Bearer <API_KEY>`；`model=asr-1.0`。

| 项 | 事实 |
|---|---|
| 语种 | header `language` 支持 zh/yue/en/ja/ko/th/vi/id/ms/fil/ar/tr/fr/de/es/it/pt/pl/ru/uk；**留空 = 混合语种识别** |
| `response_format` | `json` / **`verbose_json`**（含 segments + n_speakers）/ `srt` / `vtt` |
| `timestamp_level` | `sentence`（默认）/ **`word`**（中文按字、英文按词）；仅 verbose_json/srt/vtt 生效 |
| 说话人分离 | ✅ `verbose_json` 返回 `n_speakers` 与每段 `speaker`（`S1`/`S2`） |
| 情绪 | ❌ 无 |
| 限制 | **≤500 秒**、**≤50 MB**（超时返 400、超大小返 413，**不静默截断**） |

**端点归属**：实测本项目的 key 在 `api.minimaxi.com`（国内）通过鉴权、
在 `api.minimax.io`（海外）返回 401 `invalid api key`。所以默认 endpoint 是国内。

**日语质量提醒**：真实调用实测 `ja` 样例漏字（「持って**いけない**」→「持ってい**き**ない」），
而本地 faster-whisper 是对的 —— 云 API 不是无条件更强，别默认把它当质量上限。
"""

from __future__ import annotations

import io
from typing import Any

from agentic_asr.core.errors import APIError, CapabilityError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import (
    AudioChunk,
    Capability,
    Segment,
    Transcript,
    Word,
)
from agentic_asr.engines.base import ASREngine

log = get_logger("engines.minimax")

# HTTP 状态码 -> 是否值得重试 / which status codes are worth retrying
_RETRYABLE = {408, 429, 500, 502, 503, 504}
_STATUS_HINT = {
    400: "请求参数非法（常见：音频超过 500 秒）",
    401: "鉴权失败，检查 MINIMAX_API_KEY 与 endpoint 归属",
    402: "余额或配额不足",
    413: "请求体过大（上限 50 MB）",
    422: "内容被判定为敏感",
    429: "触发限流",
}


class MiniMaxEngine(ASREngine):
    """MiniMax 云端 ASR / MiniMax cloud ASR."""

    name = "minimax"
    implemented = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = getattr(getattr(config, "engine", None), "minimax", None)
        self.max_audio_seconds = float(getattr(cfg, "max_seconds", 480.0))
        self.max_bytes = int(getattr(cfg, "max_bytes", 48 * 1024 * 1024))
        # 云端常驻无需权重 / nothing resident locally
        self.loaded = True

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.TIMESTAMPS,
            Capability.WORD_TIMESTAMPS,
            Capability.LANGUAGE_ID,
            Capability.DIARIZATION,
        }

    def languages(self) -> list[str]:
        return ["zh", "yue", "en", "ja", "ko", "th", "vi", "id", "ms", "fil",
                "ar", "tr", "fr", "de", "es", "it", "pt", "pl", "ru", "uk", "auto"]

    # ── 请求 / request ──────────────────────────────────────────────────────

    def _endpoint(self) -> str:
        return (self.config.engine.minimax.endpoint or "").rstrip("/")

    def _encode_wav(self, chunk: AudioChunk) -> bytes:
        """把 `AudioChunk` 编码成内存 wav / encode the chunk into a wav payload.

        官方建议送**单声道 16 kHz**（识别质量不受高采样率/立体声益处，
        而未压缩的高规格音频很容易超 50 MB）。
        """
        import numpy as np
        import soundfile as sf

        buf = io.BytesIO()
        data = np.asarray(chunk.samples, dtype=np.float32)
        sf.write(buf, data, chunk.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        import requests

        cfg = self.config.engine.minimax
        key = self.config.api_key("minimax")
        if not key:
            raise APIError("缺少 MINIMAX_API_KEY / missing credential（写进项目 .env）")

        payload = self._encode_wav(chunk)
        if len(payload) > self.max_bytes:
            raise APIError(
                f"音频过大 / payload too large: {len(payload) / 1e6:.1f} MB "
                f"> {self.max_bytes / 1e6:.1f} MB；门面应先切片")

        headers = {"Authorization": f"Bearer {key}"}
        # 留空 = 混合语种识别；显式指定才带 header
        if language not in ("", "auto"):
            headers["language"] = language

        data: dict[str, str] = {
            "model": cfg.model,
            "response_format": "verbose_json",
            "timestamp_level": "word" if word_timestamps else "sentence",
        }
        url = f"{self._endpoint()}/v1/speech_to_text"
        try:
            resp = requests.post(url, headers=headers, data=data,
                                 files={"file": ("audio.wav", payload, "audio/wav")},
                                 timeout=cfg.timeout)
        except Exception as exc:  # noqa: BLE001 - 网络层异常统一成 APIError
            raise APIError(f"请求失败 / request failed: {exc}", retryable=True) from exc

        if resp.status_code != 200:
            raise self._to_error(resp.status_code, resp.text)
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise APIError(f"响应不是 JSON / non-JSON response: {resp.text[:200]}") from exc

        return self._to_transcript(body, chunk, word_timestamps)

    @staticmethod
    def _to_error(status: int, text: str) -> APIError:
        hint = _STATUS_HINT.get(status, "")
        detail = text[:240].replace("\n", " ")
        return APIError(f"MiniMax 返回 {status}　{hint}　{detail}".strip(),
                        status=status, retryable=status in _RETRYABLE)

    def _to_transcript(self, body: dict[str, Any], chunk: AudioChunk,
                       word_timestamps: bool) -> Transcript:
        """把厂商响应归一成 `Transcript` / normalise the vendor response.

        `timestamp_level=word` 时厂商返回的是**字/词级**单元，这里按
        「说话人变化 + 时间间隙」聚合成句级 `Segment`，同时把细粒度单元留在
        `words` 里 —— 门面的断句与字幕渲染都依赖句级结构。
        """
        text = (body.get("text") or "").strip()
        raw_segments = body.get("segments") or []
        n_speakers = body.get("n_speakers")

        units: list[Segment] = []
        for i, item in enumerate(raw_segments):
            t = (item.get("text") or "").strip()
            if not t:
                continue
            units.append(Segment(id=i, start=float(item.get("start", 0.0)),
                                 end=float(item.get("end", 0.0)), text=t,
                                 speaker=item.get("speaker")))

        segments = (_group_units(units) if word_timestamps else units)
        return Transcript(
            text=text,
            language="",                     # 该接口不返回语种码
            duration=float(body.get("duration") or chunk.duration),
            segments=segments,
            engine=self.name,
            raw={"n_speakers": n_speakers, "trace_id": body.get("trace_id")},
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"endpoint": self._endpoint(), "model": self.config.engine.minimax.model,
                  "max_audio_seconds": self.max_audio_seconds})
        return d

    def release(self) -> None:
        self.loaded = True   # 云端无本地资源 / nothing to release


def _group_units(units: list[Segment], max_gap: float = 0.6,
                 max_chars: int = 60) -> list[Segment]:
    """把字/词级单元聚合成句 / merge word-level units into sentences.

    切分依据：**说话人变化**、**时间间隙 > `max_gap`**、或**累计字数超限**。
    Splits on speaker change, a time gap, or accumulated length.
    """
    if not units:
        return []
    out: list[Segment] = []
    cur_words: list[Word] = [Word(text=units[0].text, start=units[0].start, end=units[0].end)]
    cur_speaker = units[0].speaker
    cur_start, cur_end = units[0].start, units[0].end

    def flush() -> None:
        if not cur_words:
            return
        text = "".join(w.text for w in cur_words) if _is_cjk(cur_words[0].text) \
            else " ".join(w.text for w in cur_words)
        out.append(Segment(id=len(out), start=cur_start, end=cur_end, text=text,
                           speaker=cur_speaker, words=list(cur_words)))

    for u in units[1:]:
        gap = u.start - cur_end
        too_long = sum(len(w.text) for w in cur_words) + len(u.text) > max_chars
        if u.speaker != cur_speaker or gap > max_gap or too_long:
            flush()
            cur_words = [Word(text=u.text, start=u.start, end=u.end)]
            cur_speaker, cur_start, cur_end = u.speaker, u.start, u.end
        else:
            cur_words.append(Word(text=u.text, start=u.start, end=u.end))
            cur_end = u.end
    flush()
    return out


def _is_cjk(sample: str) -> bool:
    """粗判是否中日韩文本（决定拼接时要不要空格）/ rough CJK detection."""
    return any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in sample)


__all__ = ["MiniMaxEngine"]
