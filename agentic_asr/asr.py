"""门面 / Facade —— `ASRModule` 是唯一必须记住的类。

它负责**所有与引擎无关**的事 / everything engine-independent：

- 解码归一（任意媒体 → 16 kHz 单声道 `AudioChunk`）
- 切片与**时间轴平移合并**（云 API 有 500 s/25 MB 上限，长音频也必须分块）
- 引擎路由、能力检查、加载前显存预检
- **幻觉闸门**（纯音乐/静音不得送去转写，见 `docs/issues/002`）
- 情绪补齐（主力引擎不支持情绪时，按段调情绪引擎）
- 字幕渲染、参考音频截取、媒体信息提取

引擎只做一件事：一段音频 → 一份 `Transcript`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_asr.audio.io import decode, slice_chunk
from agentic_asr.audio.vad import VadDetector
from agentic_asr.clip.picker import pick_reference
from agentic_asr.core.config import ASRConfig, load_config
from agentic_asr.core.errors import ASRError, CapabilityError
from agentic_asr.core.gpu import cuda_available, gpu_name, vram_free_mb, vram_total_mb
from agentic_asr.core.logging import get_logger
from agentic_asr.core.registry import available as available_engines
from agentic_asr.core.registry import resolve
from agentic_asr.core.types import (
    AudioChunk,
    AudioInfo,
    Capability,
    ClipResult,
    Segment,
    SegmentPiece,
    Transcript,
)
from agentic_asr.engines.base import ASREngine
from agentic_asr.media.probe import probe as probe_media
from agentic_asr.segment.splitter import split_audio
from agentic_asr.subtitle.render import write_subtitle

log = get_logger("asr")


def _summarise_emotions(segments: list[Segment]) -> str | None:
    """情绪汇总：主导情绪 + 占比 / dominant emotion with its share."""
    labels = [s.emotion for s in segments if s.emotion]
    if not labels:
        return None
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    top, n = max(counts.items(), key=lambda kv: kv[1])
    return f"{top} ({n}/{len(labels)})"


class ASRModule:
    """ASR 门面 / the ASR facade. 线程内单例式使用即可，内部有锁保护模型加载。"""

    def __init__(self, config: ASRConfig | None = None,
                 engine: str | ASREngine | None = None) -> None:
        self.config = config or load_config()
        self._engines: dict[str, ASREngine] = {}
        self._vad: VadDetector | None = None
        self._default_engine_name = (
            engine if isinstance(engine, str) else self.config.engine.default)
        if isinstance(engine, ASREngine):
            self._engines[engine.name] = engine
            self._default_engine_name = engine.name

    # ── 引擎管理 / engine management ────────────────────────────────────────

    def _vad_detector(self) -> VadDetector:
        """共享一个 VAD 实例 / lazily build and reuse one VAD detector."""
        if self._vad is None:
            model = self.config.resolve_model_dir("models/sherpa-vad/silero_vad.onnx")
            self._vad = VadDetector(
                model_path=model,
                max_speech_duration=float(self.config.segment.max_seconds),
                min_silence_duration=float(self.config.segment.merge_gap))
        return self._vad

    def engine(self, name: str | None = None) -> ASREngine:
        """取（并缓存）一个引擎实例 / get (and cache) an engine instance."""
        target = name or self._default_engine_name
        if target in self._engines:
            return self._engines[target]
        if self.config.engine.max_resident == 1:
            self._release_all(keep=target)
        cls = resolve(target)
        self._engines[target] = cls(self.config)
        return self._engines[target]

    def _release_all(self, keep: str | None = None) -> None:
        for name, eng in list(self._engines.items()):
            if name == keep:
                continue
            try:
                eng.release()
            except Exception as exc:  # noqa: BLE001
                log.warning("卸载引擎 %s 失败 / release failed: %s", name, exc)
            self._engines.pop(name, None)

    def engines(self) -> list[dict[str, Any]]:
        """可用引擎清单（**不加载权重**）/ engine catalogue, weight-free."""
        out: list[dict[str, Any]] = []
        for name in available_engines():
            item: dict[str, Any] = {"name": name, "default": name == self._default_engine_name}
            try:
                cls = resolve(name)
                item["capabilities"] = sorted(c.value for c in cls.declared_capabilities())
                item["implemented"] = getattr(cls, "implemented", True)
            except ASRError as exc:
                item["error"] = str(exc)
                item["capabilities"] = []
            out.append(item)
        return out

    def doctor(self) -> dict[str, Any]:
        """体检 / environment self-check（不加载权重）."""
        from agentic_asr.core import cuda as cuda_info

        report: dict[str, Any] = {
            "engines": self.engines(),
            "default_engine": self._default_engine_name,
            "cuda_available": cuda_available(),
            "gpu": gpu_name() or None,
            "vram_total_mb": vram_total_mb(),
            "vram_free_mb": vram_free_mb(),
            "cuda_runtime": cuda_info.describe(),
            "models": {},
            "checks": [],
        }
        fw = self.config.faster_whisper_model_path()
        report["models"]["faster_whisper"] = {"path": str(fw), "exists": fw.is_dir()}
        sv_model, sv_tokens = self.config.sensevoice_paths()
        report["models"]["sensevoice"] = {
            "model": str(sv_model), "exists": sv_model.is_file(),
            "tokens_exists": sv_tokens.is_file()}
        vad_model = self.config.resolve_model_dir("models/sherpa-vad/silero_vad.onnx")
        report["models"]["vad"] = {"path": str(vad_model), "exists": vad_model.is_file()}

        checks: list[str] = report["checks"]  # type: ignore[assignment]
        if not fw.is_dir():
            checks.append("faster-whisper 权重缺失：本地转写不可用（可先用 --engine mock）")
        if not sv_model.is_file():
            checks.append("SenseVoice 模型缺失：情绪识别不可用")
        if not vad_model.is_file():
            checks.append("Silero VAD 缺失：将回退到能量 VAD（精度略低）")
        if cuda_available() and not report["cuda_runtime"]["cublas_available"]:  # type: ignore[index]
            # 这条不报出来，用户只会看到 "cublas64_12.dll is not found" 这种底层错误
            checks.append("未找到 cublas64_12.dll：GPU 推理会失败 —— "
                          "装 torch，或 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        try:
            import shutil as _sh

            if _sh.which("ffmpeg") is None:
                checks.append("ffmpeg 不在 PATH：非 wav 输入与切片将不可用")
        except Exception:  # noqa: BLE001
            pass
        report["ok"] = not checks
        return report

    def release(self) -> None:
        """卸载全部引擎、释放资源 / release every loaded engine."""
        self._release_all()
        self._vad = None

    def __enter__(self) -> ASRModule:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    # ── 输入 / input ────────────────────────────────────────────────────────

    def _load(self, media: str | Path | AudioChunk) -> AudioChunk:
        if isinstance(media, AudioChunk):
            return media
        return decode(media, target_sr=16000, mono=True)

    # ── ① 音频信息提取 / probing ────────────────────────────────────────────

    def probe(self, media: str | Path, deep: bool = True,
              max_deep_seconds: float = 300.0) -> AudioInfo:
        """音频/视频信息提取 / probe media metadata and acoustic stats."""
        return probe_media(media, vad=self._vad_detector(), deep=deep,
                           max_deep_seconds=max_deep_seconds)

    # ── ② 音频分段 / segmentation ───────────────────────────────────────────

    def segment(self, media: str | Path | AudioChunk,
                max_seconds: float | None = None,
                export_audio: bool | None = None,
                out_dir: str | Path | None = None) -> list[SegmentPiece]:
        """音频分段 / split audio into well-formed pieces.

        返回带时间轴的段清单；`export_audio=True` 时同时落盘每段 wav。
        """
        chunk = self._load(media)
        pieces = split_audio(chunk, self.config.segment, vad=self._vad_detector(),
                             max_seconds=max_seconds)
        want_export = (self.config.segment.export_audio if export_audio is None
                       else export_audio)
        if want_export:
            from agentic_asr.audio.io import write_wav

            base = Path(out_dir or self.config.output_dir) / "segments"
            stem = Path(chunk.source).stem or "audio"
            for p in pieces:
                sub = slice_chunk(chunk, p.start, p.end)
                write_wav(base / f"{stem}_{p.index + 1:04d}.wav", sub.samples,
                          sub.sample_rate)
            log.info("已导出 %d 段音频 / exported %d pieces", len(pieces), len(pieces))
        return pieces

    # ── ③④ 转写 + 情绪 / transcription and emotion ──────────────────────────

    def transcribe(self, media: str | Path | AudioChunk, *, engine: str | None = None,
                   language: str | None = None, words: bool | None = None,
                   emotion: bool | None = None, srt: str | Path | None = None,
                   srt_format: str = "srt") -> Transcript:
        """转写一段媒体 / transcribe one piece of media.

        参数 / Args:
            engine: 引擎名；None 用配置默认。
            language: `auto` 或语种码；None 用配置。
            words: 是否要词级时间戳；None 用配置。
            emotion: 是否要情绪；None 用配置的 `auto`（能出就出）。
            srt: 给定路径则顺带写出字幕文件。

        返回 / Returns:
            合并后的 `Transcript`，时间轴已平移回**原始媒体**。
        """
        cfg = self.config
        lang = language or cfg.transcribe.language or "auto"
        want_words = cfg.transcribe.word_timestamps if words is None else bool(words)
        want_emotion = (cfg.transcribe.emotion != "off") if emotion is None else bool(emotion)

        eng = self.engine(engine)
        if want_words:
            eng.require(Capability.WORD_TIMESTAMPS,
                        "可设 transcribe.word_timestamps=false 退到句级")

        chunk = self._load(media)
        if chunk.duration <= 0:
            return Transcript.empty(eng.name, 0.0, "空音频 / empty audio")

        # ── 前置幻觉闸门：整段几乎没有语音就不进引擎 ──
        speech_ratio = self._speech_ratio(chunk)
        if speech_ratio is not None and speech_ratio < cfg.transcribe.min_speech_ratio:
            log.info("语音占比 %.4f 低于阈值，跳过引擎 / below speech threshold",
                     speech_ratio)
            return Transcript(
                text="", language="", duration=round(chunk.duration, 3), engine=eng.name,
                suspicious=True,
                warnings=[f"语音占比 {speech_ratio:.4f} 低于 "
                          f"{cfg.transcribe.min_speech_ratio}，判定为无语音内容"])

        # ── 切片：按**引擎单次上限**切，而不是按用户的分段配置 ──
        # 引擎没有上限（如 mock）就整段送一次；有上限（云 API 500 s、
        # SenseVoice 30 s 逐段情绪）才切。这样本地长音频不会被无谓地切碎。
        limit = eng.max_audio_seconds
        if limit:
            pieces = split_audio(chunk, cfg.segment, vad=self._vad_detector(),
                                 max_seconds=min(float(limit), chunk.duration))
        else:
            pieces = [SegmentPiece(index=0, start=0.0, end=round(chunk.duration, 3))]
        if not pieces:
            pieces = [SegmentPiece(index=0, start=0.0, end=round(chunk.duration, 3))]

        results: list[Transcript] = []
        for piece in pieces:
            sub = slice_chunk(chunk, piece.start, piece.end)
            if sub.duration <= 0:
                continue
            # ── 分段级闸门：该片没有语音就跳过，别送去编字幕 ──
            piece_ratio = self._speech_ratio(sub)
            if piece_ratio is not None and piece_ratio < cfg.transcribe.min_speech_ratio:
                log.debug("段 %d 无语音，跳过 / skipping silent piece", piece.index)
                continue
            try:
                part = eng.transcribe(sub, language=lang, word_timestamps=want_words)
            except CapabilityError:
                raise
            except ASRError as exc:
                results.append(Transcript.empty(eng.name, sub.duration,
                                                f"段 {piece.index} 失败: {exc}"))
                continue
            _shift(part, piece.start)
            _clamp(part, piece.end)
            results.append(part)

        merged = _merge(results, chunk.duration, eng.name)
        if want_emotion:
            self._backfill_emotion(merged, chunk, main=eng)
        _post_guard(merged, cfg.transcribe.max_compression_ratio)

        if srt:
            write_subtitle(merged, srt, srt_format, cfg.subtitle)
        return merged

    def _speech_ratio(self, chunk: AudioChunk) -> float | None:
        """语音占比；VAD 失败返回 None（表示「不知道」，不当作 0）."""
        try:
            return self._vad_detector().speech_ratio(chunk.samples, chunk.sample_rate)
        except Exception as exc:  # noqa: BLE001
            log.warning("VAD 失败，跳过闸门 / VAD failed, skipping guard: %s", exc)
            return None

    def _backfill_emotion(self, transcript: Transcript, chunk: AudioChunk,
                          main: ASREngine) -> None:
        """给缺情绪的段补齐 / fill in missing emotions segment by segment.

        主力引擎（如 faster-whisper）没有情绪能力，就对每个 segment 的音频区间
        调用情绪引擎 —— 这就是「组合拳」，用户不必关心谁是主力。
        """
        if main.supports(Capability.EMOTION) and any(
                s.emotion for s in transcript.segments):
            transcript.emotion_summary = _summarise_emotions(transcript.segments)
            return
        try:
            emo_engine = self.engine("sensevoice")
        except ASRError as exc:
            transcript.warnings.append(f"情绪引擎不可用 / emotion engine unavailable: {exc}")
            return
        for seg in transcript.segments:
            if seg.emotion:
                continue
            # 过短的段情绪不稳定，宁可留空也不硬给一个假标签
            if seg.duration < 0.3:
                continue
            sub = slice_chunk(chunk, seg.start, seg.end)
            try:
                seg.emotion = emo_engine.detect_emotion(sub)
            except Exception as exc:  # noqa: BLE001
                transcript.warnings.append(f"段 {seg.id} 情绪识别失败: {exc}")
        transcript.emotion_summary = _summarise_emotions(transcript.segments)

    # ── ⑤ 参考音频截取 / reference clips ────────────────────────────────────

    def clip_reference(self, media: str | Path | AudioChunk,
                       transcript: Transcript | None = None,
                       top: int | None = None, out_dir: str | Path | None = None,
                       stem: str = "ref", engine: str | None = None
                       ) -> list[ClipResult]:
        """挑出 10~15 s 的参考音频（喂 TTS 音色克隆）/ pick reference clips.

        `transcript=None` 时会先转写一次 —— 因为 `ref_text` 必须来自
        **该片段自己的转写**，没有它就只能给出音频而没有对应文本。
        """
        chunk = self._load(media)
        if transcript is None:
            transcript = self.transcribe(chunk, engine=engine)
        out = Path(out_dir) if out_dir else (self.config.output_dir / "refs")
        return pick_reference(chunk, transcript, self.config.clip, self._vad_detector(),
                             out_dir=out, top=top, stem=stem)


# ── 模块级辅助 / module helpers ──────────────────────────────────────────────


def _shift(transcript: Transcript, offset: float) -> None:
    """把一个片段的时间轴平移回原始时间轴 / shift onto the original timeline."""
    if not offset:
        return
    for seg in transcript.segments:
        seg.start = round(seg.start + offset, 3)
        seg.end = round(seg.end + offset, 3)
        for w in seg.words or []:
            w.start = round(w.start + offset, 3)
            w.end = round(w.end + offset, 3)


def _clamp(transcript: Transcript, limit: float) -> None:
    """把片段输出的时间轴**裁剪回该片的区间内** / clamp onto the slice.

    为什么必须做：Whisper 在 30 秒解码窗口里工作，给短片的尾段时间戳可能
    **超出实际音频长度**。实测一次 123.55 s 的素材，末段被标到 131.51 s ——
    字幕直接比音频长出 8 秒。这类错误不会报异常，只会让字幕在末尾空转。

    Whisper decodes inside a 30 s window and can emit end timestamps beyond
    the real audio, so each slice's output must be clamped back onto it.
    """
    kept: list[Segment] = []
    for seg in transcript.segments:
        if seg.start >= limit:
            # 整段都落在本片之外：多半是解码窗口的越界输出，丢弃而不是硬塞到边界
            continue
        seg.end = round(min(seg.end, limit), 3)
        if seg.end <= seg.start:
            seg.end = round(seg.start + 0.01, 3)
        for w in seg.words or []:
            w.end = round(min(w.end, limit), 3)
            if w.end <= w.start:
                w.end = round(w.start + 0.01, 3)
        seg.words = [w for w in (seg.words or []) if w.start < limit] or None
        kept.append(seg)
    transcript.segments = kept


def _merge(parts: list[Transcript], duration: float, engine: str) -> Transcript:
    """合并各片结果并重新编号 / merge piece results and renumber."""
    segments: list[Segment] = []
    texts: list[str] = []
    warnings: list[str] = []
    language = ""
    for part in parts:
        language = language or part.language
        warnings.extend(part.warnings)
        if part.text.strip():
            texts.append(part.text.strip())
        for seg in part.segments:
            seg.id = len(segments)
            segments.append(seg)
    segments.sort(key=lambda s: s.start)
    for i, seg in enumerate(segments):
        seg.id = i
    return Transcript(
        text="".join(texts),
        language=language,
        duration=round(duration, 3),
        segments=segments,
        engine=engine,
        emotion_summary=_summarise_emotions(segments),
        warnings=warnings,
    )


def _post_guard(transcript: Transcript, max_ratio: float) -> None:
    """后置闸门：压缩比异常 → 标记可疑 / flag suspiciously dense output.

    压缩比 = 文本长度 / 音频时长。值异常高通常意味着模型在「编」。
    这里**只标记不丢弃** —— 丢不丢由调用方决定，静默丢会更难查。
    """
    dur = transcript.duration or 0.0
    if dur <= 0 or not transcript.text:
        return
    ratio = len(transcript.text) / dur
    if ratio > max_ratio:
        transcript.suspicious = True
        transcript.warnings.append(
            f"文本/时长压缩比 {ratio:.1f} 超过 {max_ratio}，结果可能包含重复或幻觉")


__all__ = ["ASRModule"]
