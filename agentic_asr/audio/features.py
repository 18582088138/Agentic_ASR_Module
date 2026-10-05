"""声学统计 / Acoustic features.

只做「不加载模型就能算」的统计分析。它们的用途是**决策**，不是展示：

- `estimated_snr_db` → 前处理（降噪）开关的判据；
- `speech_ratio`    → 幻觉闸门的判据（纯音乐/静音不能送去转写，见 issues/002）；
- `peak_dbfs`/`clipping` → 提醒素材已经削波，转写结果不可全信。

统计只用于决策，不追求仪器级精确 —— 这里不做 ITU-R BS.1770 那种响度计权。
These are decision inputs, not metrology.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, _EPS)))


def peak_dbfs(samples: np.ndarray) -> float:
    """峰值电平（dBFS）/ peak level in dBFS. 0 dBFS = 满刻度。"""
    if samples.size == 0:
        return -np.inf
    return _db(float(np.max(np.abs(samples))))


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS 电平（dBFS）/ RMS level in dBFS."""
    if samples.size == 0:
        return -np.inf
    return _db(float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))))


def is_clipping(samples: np.ndarray, threshold: float = 0.999) -> bool:
    """是否已削波 / whether the material is clipped.

    连续多个样本贴顶即判定削波 —— 单点贴顶可能只是运气。
    """
    if samples.size < 8:
        return False
    at_top = np.abs(samples) >= threshold
    if not at_top.any():
        return False
    # 找最长连续贴顶长度 / longest run of clipped samples
    idx = np.flatnonzero(at_top)
    if idx.size < 2:
        return False
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return max(len(r) for r in runs) >= 3


def estimate_snr_db(samples: np.ndarray, speech_mask: np.ndarray | None) -> float | None:
    """估计信噪比 / estimate SNR in dB from a speech mask.

    做法：把样本按 VAD 掩码分成「语音」与「非语音」两堆，各自算 RMS，取差值。
    非语音堆太短（<50 ms）时无法估计，返回 None —— **不硬给一个数**。
    Speech vs non-speech RMS difference; returns None when there is too little
    non-speech material to be meaningful.
    """
    if speech_mask is None or samples.size != speech_mask.size:
        return None
    mask = speech_mask.astype(bool)
    n_speech = int(mask.sum())
    n_other = int((~mask).sum())
    min_needed = 800  # 50 ms @ 16 kHz
    if n_speech < min_needed or n_other < min_needed:
        return None
    speech_rms = float(np.sqrt(np.mean(np.square(samples[mask].astype(np.float64)))))
    noise_rms = float(np.sqrt(np.mean(np.square(samples[~mask].astype(np.float64)))))
    if noise_rms <= _EPS:
        return None
    return float(_db(speech_rms) - _db(noise_rms))


def speech_ratio(speech_mask: np.ndarray | None, total: int) -> float | None:
    """语音占比 / fraction of samples flagged as speech."""
    if speech_mask is None or total <= 0:
        return None
    return float(min(1.0, max(0.0, speech_mask.sum() / float(total))))


def mask_from_intervals(intervals: list[tuple[float, float]], total: int,
                        sample_rate: int) -> np.ndarray:
    """把秒级区间转成逐样本布尔掩码 / build a per-sample mask from intervals."""
    mask = np.zeros(total, dtype=bool)
    for start, end in intervals:
        i0 = max(0, int(round(start * sample_rate)))
        i1 = min(total, int(round(end * sample_rate)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def analyse(samples: np.ndarray, sample_rate: int,
            intervals: list[tuple[float, float]] | None) -> dict[str, object]:
    """一次算完所有统计 / compute every statistic in one call."""
    mask = (mask_from_intervals(intervals, samples.size, sample_rate)
            if intervals is not None else None)
    return {
        "peak_dbfs": round(peak_dbfs(samples), 2),
        "rms_dbfs": round(rms_dbfs(samples), 2),
        "estimated_snr_db": (None if (snr := estimate_snr_db(samples, mask)) is None
                             else round(snr, 2)),
        "speech_ratio": (None if (r := speech_ratio(mask, samples.size)) is None
                         else round(r, 4)),
        "clipping": is_clipping(samples),
    }


__all__ = [
    "analyse",
    "estimate_snr_db",
    "is_clipping",
    "mask_from_intervals",
    "peak_dbfs",
    "rms_dbfs",
    "speech_ratio",
]
