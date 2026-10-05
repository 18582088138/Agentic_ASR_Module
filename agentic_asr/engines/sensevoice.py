"""SenseVoice 引擎（情绪 / 事件 / 语种，CPU）/ SenseVoice via sherpa-onnx on CPU.

**为什么用 sherpa-onnx 而不是 funasr**（这是实测结论，不是偏好）：

- 官方 `funasr` 会新增 **18 个包**（hydra-core / umap-learn / jieba / tensorboardX /
  kaldiio / oss2 …）；`sherpa-onnx` 只新增 **2 个包**。
- 实测 `sherpa_onnx.OfflineRecognizer.from_sense_voice()` 的返回对象**原生带**
  `emotion` / `event` / `lang` 字段与字级 `timestamps`：

  ```
  text: 开放时间早上9点至下午5点。      lang: <|zh|>
  emotion: <|NEUTRAL|>                 event: <|Speech|>
  ```

- 它跑在 **CPU** 上，不与 faster-whisper 争显存（8 GB 卡的余量全部留给 ASR）。
- 顺带白拿了**语种识别**与**音频事件检测**。

字段值是 `<|NEUTRAL|>` 这种带标记的形式，本模块负责**解析**并保留原值以便回溯。
Tag values like `<|NEUTRAL|>` are parsed here; the raw tag is kept for traceability.
"""

from __future__ import annotations

import threading
from typing import Any

from agentic_asr.core.errors import EngineError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import (
    AudioChunk,
    Capability,
    Emotion,
    Segment,
    Transcript,
    Word,
)
from agentic_asr.engines.base import ASREngine

log = get_logger("engines.sensevoice")


def clean_tag(value: str | None) -> str:
    """把 `<|NEUTRAL|>` 解析成 `neutral` / strip the `<|...|>` wrapper.

    非标记形式的字符串原样返回小写（不同版本 sherpa-onnx 的返回格式不完全一致）。
    """
    if not value:
        return ""
    return value.strip().strip("<|>").strip().lower()


def to_emotion(value: str | None) -> str | None:
    """映射到 7 类情绪；无法识别时返回 None 而**不编造** / map to the 7 labels.

    返回 None 而不是 `unknown`：调用方据此知道「这段没有可信情绪」，
    而不是拿到一个看起来像结论的字符串。
    """
    label = clean_tag(value)
    if not label:
        return None
    try:
        return Emotion(label).value
    except ValueError:
        return label if label in {e.value for e in Emotion} else None


class SenseVoiceEngine(ASREngine):
    """sherpa-onnx 的 SenseVoice-Small / SenseVoice-Small on sherpa-onnx."""

    name = "sensevoice"
    implemented = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._rec = None
        self._lock = threading.Lock()
        # 一次只识别 30 s：情绪是 utterance 级的，切段才有逐段情绪；
        # 顺带也把峰值内存压住 / segment so that emotion is per-utterance.
        seg_cfg = getattr(config, "segment", None)
        self.max_audio_seconds = float(getattr(seg_cfg, "max_seconds", 30.0) or 30.0)

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        return {
            Capability.TIMESTAMPS,
            Capability.WORD_TIMESTAMPS,
            Capability.LANGUAGE_ID,
            Capability.EMOTION,
            Capability.EVENTS,
        }

    def languages(self) -> list[str]:
        return ["zh", "en", "ja", "ko", "yue", "auto"]

    # ── 加载 / loading ──────────────────────────────────────────────────────

    @property
    def paths(self) -> tuple[Any, Any]:
        return self.config.sensevoice_paths()

    def ensure_loaded(self) -> None:
        """懒加载 / lazily build the recognizer (CPU, so no VRAM pre-flight)."""
        if self._rec is not None:
            return
        model_path, tokens_path = self.paths
        if not model_path.is_file() or not tokens_path.is_file():
            raise EngineError(
                f"SenseVoice 模型缺失 / model files missing: {model_path} / {tokens_path}　"
                "下载见 docs/05_deployment.md")
        cfg = self.config.engine.sensevoice
        try:
            import sherpa_onnx

            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=int(cfg.num_threads),
                use_itn=bool(cfg.use_itn),
                language=cfg.language or "auto",
                provider=cfg.provider or "cpu",
                debug=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"加载 SenseVoice 失败 / failed to load: {exc}") from exc
        self.loaded = True
        log.info("SenseVoice 已加载 / loaded: %s (%s)", model_path.name, cfg.provider)

    # ── 推理 / inference ────────────────────────────────────────────────────

    def _decode(self, chunk: AudioChunk) -> Any:
        """跑一次识别并返回原始结果对象 / run once and return the raw result."""
        self.ensure_loaded()
        samples = chunk.samples
        if chunk.sample_rate != 16000:
            from agentic_asr.audio.io import resample

            samples = resample(samples, chunk.sample_rate, 16000)
        # sherpa-onnx 的识别器不保证线程安全，串行化 / not guaranteed thread-safe
        with self._lock:
            stream = self._rec.create_stream()
            stream.accept_waveform(16000, samples)
            self._rec.decode_stream(stream)
            return stream.result

    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        try:
            result = self._decode(chunk)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"SenseVoice 推理失败 / inference failed: {exc}") from exc

        text = (getattr(result, "text", "") or "").strip()
        emotion = to_emotion(getattr(result, "emotion", None))
        event = clean_tag(getattr(result, "event", None))
        lang = clean_tag(getattr(result, "lang", None))
        if language not in ("", "auto") and not lang:
            lang = language

        segments: list[Segment] = []
        if text:
            segments.append(Segment(
                id=0, start=0.0, end=round(chunk.duration, 3), text=text,
                emotion=emotion,
                words=self._words(result, text, chunk) if word_timestamps else None))

        return Transcript(
            text=text,
            language=lang,
            duration=round(chunk.duration, 3),
            segments=segments,
            engine=self.name,
            emotion_summary=emotion,
            events=[event] if event else [],
            raw={"emotion_raw": getattr(result, "emotion", None),
                 "event_raw": getattr(result, "event", None),
                 "lang_raw": getattr(result, "lang", None)},
        )

    def detect_emotion(self, chunk: AudioChunk) -> str | None:
        """只取情绪，供门面逐段补齐 / emotion only, used by the facade to backfill."""
        try:
            result = self._decode(chunk)
        except Exception as exc:  # noqa: BLE001 - 补齐失败不该让整次请求失败
            log.warning("情绪补齐失败 / emotion backfill failed: %s", exc)
            return None
        return to_emotion(getattr(result, "emotion", None))

    # ── 辅助 / helpers ──────────────────────────────────────────────────────

    def _words(self, result: Any, text: str, chunk: AudioChunk) -> list[Word] | None:
        """用字级时间戳构造 words / build `Word`s from character-level timestamps.

        SenseVoice 给出的是「每个 token 一个时间戳」。token 与最终文本不一定逐字
        对齐，所以只在数量接近时才构造 —— 对不上就返回 None，**不硬凑**。
        Only build words when the counts roughly line up; otherwise return None.
        """
        stamps = list(getattr(result, "timestamps", None) or [])
        tokens = [t for t in (getattr(result, "tokens", None) or []) if t and t.strip()]
        if not stamps:
            return None
        n = len(stamps)
        if tokens and len(tokens) == n:
            texts = [t.strip() for t in tokens]
        else:
            stripped = text.replace(" ", "")
            if len(stripped) != n:
                return None
            texts = list(stripped)
        words: list[Word] = []
        for i, start in enumerate(stamps):
            end = float(stamps[i + 1]) if i + 1 < n else chunk.duration
            if end <= start:
                end = start + 0.02
            words.append(Word(text=texts[i], start=round(float(start), 3),
                              end=round(float(end), 3)))
        return words or None

    def info(self) -> dict[str, Any]:
        d = super().info()
        model_path, _ = self.paths
        d.update({"model_dir": str(model_path.parent), "provider":
                  self.config.engine.sensevoice.provider})
        return d

    def release(self) -> None:
        if self._rec is not None:
            log.info("卸载 SenseVoice / releasing model")
        self._rec = None
        self.loaded = False
        import gc

        gc.collect()


__all__ = ["SenseVoiceEngine", "clean_tag", "to_emotion"]
