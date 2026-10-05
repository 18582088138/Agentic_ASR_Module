"""语音活动检测 / Voice activity detection.

VAD 在本插件里承担三件事，全部是**决策输入**：

1. **音频分段** —— 在静音点切，避免把词切成两半（用户明确要求「静音点优先」）；
2. **幻觉闸门** —— 语音占比过低就不送引擎（见 `docs/issues/002`，
   实测纯背景音会被 Whisper 编出「优优独播剧场」这种假字幕）；
3. **参考音频截取与 SNR 估计**的打分。

主路径是 sherpa-onnx 的 Silero VAD（模型 0.64 MB，CPU）。模型缺失时**回退到
零依赖的能量 VAD** —— 精度低一些，但保证整条链路不会因为少一个模型文件就瘫掉。
Falls back to a dependency-free energy VAD when the model file is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_asr.core.logging import get_logger

log = get_logger("audio.vad")


class VadDetector:
    """语音区间检测器 / Speech interval detector.

    参数 / Args:
        model_path: Silero VAD 的 onnx 路径；None 或不存在则用能量 VAD。
        sample_rate: 输入采样率，默认 16000（Silero 只支持 8k/16k）。
        threshold: 语音判定阈值，越大越保守。
        min_speech_duration: 短于此的语音片被丢弃（秒）。
        min_silence_duration: 长于此的静音才判定为断点（秒）。
        max_speech_duration: 单段最长时长（秒），超长会被强制切开。
    """

    def __init__(self, model_path: str | Path | None = None,
                 sample_rate: int = 16000, threshold: float = 0.5,
                 min_speech_duration: float = 0.25,
                 min_silence_duration: float = 0.25,
                 max_speech_duration: float = 30.0,
                 num_threads: int = 1) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.max_speech_duration = max_speech_duration
        self.num_threads = num_threads
        self.model_path = Path(model_path) if model_path else None
        self._vad = None
        self._tried = False

    # ── 后端选择 / backend selection ────────────────────────────────────────

    @property
    def backend(self) -> str:
        """当前实际使用的后端 / which backend is actually in use."""
        if self.model_path and self.model_path.is_file():
            return "silero"
        return "energy"

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not (self.model_path and self.model_path.is_file()):
            log.warning("VAD 模型不存在，回退到能量 VAD / falling back to energy VAD: %s",
                        self.model_path)
            return
        try:
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(self.model_path)
            config.silero_vad.threshold = float(self.threshold)
            config.silero_vad.min_silence_duration = float(self.min_silence_duration)
            config.silero_vad.min_speech_duration = float(self.min_speech_duration)
            config.silero_vad.max_speech_duration = float(self.max_speech_duration)
            config.sample_rate = int(self.sample_rate)
            config.num_threads = int(self.num_threads)
            config.provider = "cpu"
            if not config.validate():
                raise ValueError("VadModelConfig.validate() 返回 False")
            self._vad = sherpa_onnx.VoiceActivityDetector(
                config, buffer_size_in_seconds=max(300.0, self.max_speech_duration * 4))
        except Exception as exc:  # noqa: BLE001 - 任何失败都退能量 VAD
            log.warning("Silero VAD 初始化失败，回退能量 VAD / init failed: %s", exc)
            self._vad = None

    # ── 主入口 / main entry ─────────────────────────────────────────────────

    def intervals(self, samples: np.ndarray,
                  sample_rate: int | None = None) -> list[tuple[float, float]]:
        """返回语音区间（秒）/ speech intervals in seconds.

        时间轴以传入数组的起点为 0 —— 调用方负责加上 `AudioChunk.offset`。
        """
        sr = sample_rate or self.sample_rate
        if samples.size == 0:
            return []
        if sr != self.sample_rate:
            # Silero 只认 8k/16k；不一致时先重采样，避免静默给出错误的区间
            from agentic_asr.audio.io import resample

            samples = resample(samples, sr, self.sample_rate)
            sr = self.sample_rate

        self._ensure()
        if self._vad is not None:
            try:
                return self._silero_intervals(samples)
            except Exception as exc:  # noqa: BLE001
                log.warning("Silero VAD 推理失败，改用能量 VAD / runtime failed: %s", exc)
        return self._energy_intervals(samples)

    def _silero_intervals(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """分块喂入 **并边喂边取** / feed in windows and drain as we go.

        ⚠️ 两个坑都在这里，而且都是**静默**的：

        1. **必须分块喂**。整段一次性传给 `accept_waveform` 会严重漏检 ——
           同一段 5.6 s 人声素材，整段喂只报 `(5.29, 5.60)`，
           按 512 样本分块喂才报出 `(0.74, 5.13)`。
        2. **必须边喂边 `pop()`**。`VoiceActivityDetector` 内部是环形缓冲，
           喂满 `buffer_size_in_seconds` 后早先的语音段会被挤掉。实测 60 s 素材
           先喂完再取结果时，只剩最后一个 4.5 s 的区间 —— 表现为参考音频截取
           误判「素材太短」。

        Both traps are silent: no exception, just useless output.
        """
        self._vad.reset()
        window = 512     # Silero 的 window_size
        sr = self.sample_rate
        out: list[tuple[float, float]] = []

        def drain() -> None:
            while not self._vad.empty():
                seg = self._vad.front
                start = float(seg.start) / sr
                out.append((start, start + len(seg.samples) / sr))
                self._vad.pop()

        for i in range(0, samples.size, window):
            self._vad.accept_waveform(samples[i:i + window])
            drain()
        self._vad.flush()
        drain()
        return _merge(out, self.min_silence_duration)

    def _energy_intervals(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """零依赖回退 / dependency-free fallback.

        帧长 25 ms、跳步 10 ms，阈值取「噪声底 + 10 dB」；噪声底用 10 分位数估计，
        因此对稳态背景噪声不敏感。精度不如 Silero，但足以支撑分段与占比闸门。
        """
        sr = self.sample_rate
        frame = max(1, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        if samples.size < frame:
            return []
        n_frames = 1 + (samples.size - frame) // hop
        idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = samples[idx]
        rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + 1e-12)
        noise_floor = float(np.percentile(rms, 10))
        thresh = max(noise_floor * 3.162, 1e-4)      # +10 dB
        voiced = rms > thresh

        out: list[tuple[float, float]] = []
        start_i: int | None = None
        for i, v in enumerate(voiced):
            if v and start_i is None:
                start_i = i
            elif not v and start_i is not None:
                out.append((start_i * hop / sr, (i * hop + frame) / sr))
                start_i = None
        if start_i is not None:
            out.append((start_i * hop / sr, samples.size / sr))
        return _merge(_filter_short(out, self.min_speech_duration),
                      self.min_silence_duration)

    # ── 派生量 / derived values ─────────────────────────────────────────────

    def speech_ratio(self, samples: np.ndarray,
                     sample_rate: int | None = None) -> float:
        """语音占比（0~1）/ fraction of the audio that is speech."""
        if samples.size == 0:
            return 0.0
        sr = sample_rate or self.sample_rate
        total = samples.size / float(sr)
        covered = sum(max(0.0, e - s) for s, e in self.intervals(samples, sr))
        return float(min(1.0, covered / total)) if total > 0 else 0.0

    def mask(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """逐样本布尔掩码 / per-sample boolean mask."""
        from agentic_asr.audio.features import mask_from_intervals

        sr = sample_rate or self.sample_rate
        return mask_from_intervals(self.intervals(samples, sr), samples.size, sr)


def _merge(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """合并间隔小于 `gap` 的区间 / merge intervals separated by less than `gap`."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(float(s), float(e)) for s, e in out]


def _filter_short(intervals: list[tuple[float, float]],
                  minimum: float) -> list[tuple[float, float]]:
    return [(s, e) for s, e in intervals if (e - s) >= minimum]


def merge_intervals(intervals: list[tuple[float, float]],
                    gap: float = 0.0) -> list[tuple[float, float]]:
    """公开的区间合并工具 / public interval merger."""
    return _merge(intervals, gap)


__all__ = ["VadDetector", "merge_intervals"]
