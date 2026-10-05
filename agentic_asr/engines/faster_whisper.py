"""faster-whisper 引擎（本地主力）/ the faster-whisper engine — local default.

本机实测（RTX 4060 8 GB，`large-v3-turbo` + `float16` + `beam_size=5`）：

| 素材 | 语种识别 | RTF | 显存 |
|---|---|---|---|
| zh 5.62 s | zh `p=0.989` | 0.131 | +2.17 GB |
| en 7.18 s | en `p=1.000` | 0.073 | 同上 |
| ja 7.22 s | ja `p=1.000` | 0.080 | 同上 |
| zh 185.3 s | — | **0.037** | 同上 |

⇒ 1 小时音频约 2.2 分钟转完，词级时间戳开箱可用（中文按字、英文按词）。

**两条硬性注意事项 / two hard-won rules：**

1. **只喂 `np.ndarray`，永不喂文件路径。** faster-whisper 1.2.1 与 `av` 19 不兼容
   （`av.open()` 已移除 `metadata_errors`），传路径必崩。见 `docs/issues/001`。
   Never hand it a file path.
2. **显存只能用 `nvidia-smi` 读。** CTranslate2 不走 torch 的 CUDA 分配器，
   `torch.cuda.max_memory_allocated()` 全程返回 0（实测）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_asr.core.errors import EngineError
from agentic_asr.core.gpu import check_headroom, cuda_available
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import (
    AudioChunk,
    Capability,
    Segment,
    Transcript,
    Word,
)
from agentic_asr.engines.base import ASREngine

log = get_logger("engines.faster_whisper")

# 权重体积的经验系数：model.bin 之外还有解码缓存等运行时占用
# Empirical multiplier over the on-disk weight size to cover runtime buffers.
_VRAM_FACTOR = 1.35


class FasterWhisperEngine(ASREngine):
    """CTranslate2 后端的多语种 Whisper / multilingual Whisper on CTranslate2."""

    name = "faster_whisper"
    implemented = True

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._model = None
        self._device_used = ""
        self._compute_type_used = ""
        cfg = getattr(config, "engine", None)
        self.max_audio_seconds = float(
            getattr(getattr(cfg, "faster_whisper", None), "max_chunk_seconds", 300.0))

    # ── 能力 / capabilities ─────────────────────────────────────────────────

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        # 无可用的说话人分离与情绪：这两项由门面用别的引擎补齐或留空
        return {Capability.TIMESTAMPS, Capability.WORD_TIMESTAMPS, Capability.LANGUAGE_ID}

    def languages(self) -> list[str]:
        return ["zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru", "auto"]

    # ── 加载 / loading ──────────────────────────────────────────────────────

    @property
    def model_dir(self) -> Path:
        return self.config.faster_whisper_model_path()

    def _resolve_device(self) -> str:
        cfg = self.config.engine.faster_whisper
        want = (cfg.device or "auto").lower()
        have_cuda = cuda_available()
        if want == "auto":
            want = "cuda" if have_cuda else "cpu"
        if want == "cuda" and not have_cuda:
            if cfg.cpu_fallback:
                log.warning("CUDA 不可用，退回 CPU / CUDA unavailable, falling back to CPU")
                want = "cpu"
            else:
                raise EngineError(
                    "配置要求 cuda 但当前不可用 / configured cuda but unavailable；"
                    "可设 engine.faster_whisper.cpu_fallback=true 或 device=cpu")
        return want

    def _resolve_compute_type(self, device: str) -> str:
        ct = self.config.engine.faster_whisper.compute_type or "float16"
        # float16 在 CPU 上不可用，静默降级会误导，这里显式改写并记日志
        if device == "cpu" and ct.startswith("float16"):
            log.info("CPU 上不支持 %s，改用 int8 / switching to int8 on CPU", ct)
            return "int8"
        return ct

    def ensure_loaded(self) -> None:
        """懒加载权重 / lazily load the weights (with a VRAM pre-flight)."""
        if self._model is not None:
            return
        model_dir = self.model_dir
        if not model_dir.is_dir():
            raise EngineError(
                f"模型目录不存在 / model dir missing: {model_dir}　"
                "下载见 docs/05_deployment.md")

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)

        if device == "cuda":
            bin_path = model_dir / "model.bin"
            size_mb = (bin_path.stat().st_size / 1e6) if bin_path.is_file() else 1600.0
            ok, msg = check_headroom(size_mb * _VRAM_FACTOR)
            if not ok:
                raise EngineError(
                    f"显存可能不足 / not enough VRAM: {msg}　"
                    "可改 engine.faster_whisper.device=cpu 或用更小的模型")
            # CTranslate2 自己不带 cublas/cudnn，必须先把 CUDA 运行库目录注册进
            # DLL 搜索路径，否则报 "Library cublas64_12.dll is not found"。
            # 详见 docs/issues/004 与 core/cuda.py。
            from agentic_asr.core.cuda import register_cuda_dlls

            dirs = register_cuda_dlls()
            if not dirs:
                log.warning(
                    "未找到 CUDA 运行库目录，GPU 推理可能失败 / no CUDA runtime dirs found；"
                    "装 torch 或 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(str(model_dir), device=device,
                                       compute_type=compute_type)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"加载 faster-whisper 失败 / failed to load: {exc}") from exc

        self._device_used = device
        self._compute_type_used = compute_type
        self.loaded = True
        log.info("faster-whisper 已加载 / loaded: %s device=%s compute=%s",
                 model_dir.name, device, compute_type)

    # ── 转写 / transcription ────────────────────────────────────────────────

    def transcribe(self, chunk: AudioChunk, *, language: str = "auto",
                   word_timestamps: bool = True, **kwargs: Any) -> Transcript:
        self.ensure_loaded()
        cfg = self.config.engine.faster_whisper
        want_words = bool(word_timestamps)
        if want_words:
            self.require(Capability.WORD_TIMESTAMPS)

        lang = None if language in ("", "auto") else language
        try:
            raw_segments, info = self._model.transcribe(
                chunk.samples,
                language=lang,
                beam_size=int(cfg.beam_size),
                word_timestamps=want_words,
                # 门面已做 VAD 与切片；内置 VAD 会改写时间轴，故关闭
                vad_filter=False,
                # 关闭上文条件化可显著减少长音频上的幻觉滚雪球（实测有效）
                condition_on_previous_text=False,
            )
            raw_segments = list(raw_segments)   # 生成器，必须消费完才真正计算
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"转写失败 / transcription failed: {exc}") from exc

        segments: list[Segment] = []
        for i, seg in enumerate(raw_segments):
            words = None
            if want_words and getattr(seg, "words", None):
                words = [Word(text=w.word.strip(), start=float(w.start),
                              end=float(w.end),
                              confidence=(float(w.probability)
                                          if getattr(w, "probability", None) is not None
                                          else None))
                         for w in seg.words if w.word.strip()]
                words = words or None
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append(Segment(id=len(segments), start=float(seg.start),
                                    end=float(seg.end), text=text, words=words))

        return Transcript(
            text="".join(s.text for s in segments),
            language=getattr(info, "language", "") or "",
            duration=float(getattr(info, "duration", chunk.duration)),
            segments=segments,
            engine=self.name,
            raw={"language_probability": float(getattr(info, "language_probability", 0.0)),
                 "device": self._device_used, "compute_type": self._compute_type_used},
        )

    # ── 生命周期 / lifecycle ────────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update({"model_dir": str(self.model_dir), "device": self._device_used or None,
                  "compute_type": self._compute_type_used or None})
        return d

    def release(self) -> None:
        """卸权重、清显存 / unload weights and free VRAM."""
        if self._model is not None:
            log.info("卸载 faster-whisper / releasing model")
        self._model = None
        self.loaded = False
        import gc

        gc.collect()
        try:  # torch 可能没装（本项目不依赖它），有则顺手清缓存
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["FasterWhisperEngine"]
