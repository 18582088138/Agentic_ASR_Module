"""音频层 / Audio layer —— 解码归一、VAD、声学统计。

对外只暴露函数与一个 VAD 类；引擎**永远**只看到 `AudioChunk`。
"""

from agentic_asr.audio.features import (
    analyse,
    estimate_snr_db,
    is_clipping,
    mask_from_intervals,
    peak_dbfs,
    rms_dbfs,
    speech_ratio,
)
from agentic_asr.audio.io import (
    DEFAULT_SR,
    decode,
    decode_bytes,
    ffmpeg_available,
    resample,
    slice_chunk,
    to_ffmpeg_wav,
    write_wav,
)
from agentic_asr.audio.vad import VadDetector, merge_intervals

__all__ = [
    "DEFAULT_SR",
    "VadDetector",
    "analyse",
    "decode",
    "decode_bytes",
    "estimate_snr_db",
    "ffmpeg_available",
    "is_clipping",
    "mask_from_intervals",
    "merge_intervals",
    "peak_dbfs",
    "resample",
    "rms_dbfs",
    "slice_chunk",
    "speech_ratio",
    "to_ffmpeg_wav",
    "write_wav",
]
