"""音频解码与归一 / Audio decoding and normalisation.

**所有引擎只接受 `AudioChunk`（`sample_rate` Hz 单声道 float32），永不接受文件路径。**
引擎永远拿到 `AudioChunk`, never a file path.

这条规定的直接起因：faster-whisper 1.2.1 与 `av` 19 不兼容
（`av.open()` 已移除 `metadata_errors`，见 `docs/issues/001`）。
根本原因则是四条输入通路（文件 / 视频 / 字节流 / 麦克风）本就该归一成同一份表示，
否则每条通路都要各自处理重采样、声道降混与容器差异。

Why we own decoding:
1. `av` 19 removed `metadata_errors`, which breaks faster-whisper 1.2.1's
   internal decoder (`issues/001`).
2. Every input path must collapse to one canonical representation anyway.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from agentic_asr.core.errors import DecodeError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import AudioChunk

log = get_logger("audio.io")

DEFAULT_SR = 16000
_FFMPEG = shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    """ffmpeg 是否可用 / whether ffmpeg is on PATH."""
    return _FFMPEG is not None


def _read_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """soundfile 快路径（wav/flac/mp3/ogg 等）/ fast path."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def _ffmpeg_decode(path: Path, target_sr: int, mono: bool,
                   offset: float | None = None,
                   duration: float | None = None) -> np.ndarray:
    """用 ffmpeg 解码为 float32 PCM / decode to raw float32 PCM via ffmpeg.

    一步完成「解码 + 降混 + 重采样」，因此不需要额外的重采样库。
    Decoding, downmixing and resampling happen in one pass.
    """
    if _FFMPEG is None:
        raise DecodeError(
            "ffmpeg 不在 PATH 上 / ffmpeg not found on PATH；"
            "读非 wav 或需要重采样时必须安装 ffmpeg")

    cmd = [_FFMPEG, "-v", "error", "-nostdin"]
    if offset is not None:
        cmd += ["-ss", f"{max(0.0, offset):.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.6f}"]
    cmd += ["-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", "1" if mono else "2",
            "-ar", str(target_sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-4:]
        raise DecodeError(f"ffmpeg 解码失败 / ffmpeg failed for {path.name}: "
                          + " / ".join(tail))
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise DecodeError(f"解码结果为空 / decoded nothing: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode(path: str | Path, target_sr: int = DEFAULT_SR, mono: bool = True,
           offset: float | None = None, duration: float | None = None) -> AudioChunk:
    """把任意音频/视频解码为 `AudioChunk` / decode any media into an `AudioChunk`.

    参数 / Args:
        path: 输入文件；支持 ffmpeg 能解的一切容器（含视频，自动抽音轨）。
        target_sr: 目标采样率，默认 16000（ASR 引擎的通用约定）。
        mono: 是否降混为单声道。
        offset / duration: 只解码一段（秒），用于切片与参考音频截取。

    返回 / Returns:
        `AudioChunk`，`offset` 字段记录这段在原始音频中的起始秒数。

    抛出 / Raises:
        DecodeError: 文件不存在或解码失败。
    """
    p = Path(path)
    if not p.is_file():
        raise DecodeError(f"文件不存在 / no such file: {p}")

    # 快路径：只要不需要切段/重采样，且 soundfile 能读，就直接用
    if offset is None and duration is None:
        try:
            samples, sr = _read_with_soundfile(p)
            if sr == target_sr and samples.ndim == 1:
                return AudioChunk(samples=np.ascontiguousarray(samples, dtype=np.float32),
                                  sample_rate=sr, offset=0.0, source=str(p))
        except Exception:  # noqa: BLE001 - 交给 ffmpeg 兜底
            pass

    samples = _ffmpeg_decode(p, target_sr, mono, offset, duration)
    return AudioChunk(samples=samples, sample_rate=target_sr,
                      offset=float(offset or 0.0), source=str(p))


def decode_bytes(data: bytes, suffix: str = ".bin", target_sr: int = DEFAULT_SR,
                 mono: bool = True) -> AudioChunk:
    """从字节流解码（HTTP 上传用）/ decode from bytes (HTTP uploads)."""
    if not data:
        raise DecodeError("空字节流 / empty payload")
    tmp = Path(tempfile.mkdtemp(prefix="agentic_asr_")) / f"input{suffix}"
    try:
        tmp.write_bytes(data)
        chunk = decode(tmp, target_sr=target_sr, mono=mono)
        chunk.source = f"<bytes:{len(data)}>"
        return chunk
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def to_ffmpeg_wav(src: str | Path, dst: str | Path, sample_rate: int = 44100,
                  channels: int = 2) -> Path:
    """转成「44.1 kHz 立体声 16-bit wav」/ convert to the canonical separation input.

    sherpa-onnx 的 UVR 分离模型**只接受 wav、且要求 44.1 kHz 立体声**
    （官方：`ffmpeg -i in.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav`）。
    sherpa-onnx's UVR models only accept 44.1 kHz stereo wav.
    """
    src_p, dst_p = Path(src), Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_available():
        raise DecodeError("转 wav 需要 ffmpeg / ffmpeg is required to convert to wav")
    cmd = [_FFMPEG, "-v", "error", "-nostdin", "-y", "-i", str(src_p), "-vn",
           "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
           str(dst_p)]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not dst_p.is_file():
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-4:]
        raise DecodeError(f"转 wav 失败 / conversion failed: " + " / ".join(tail))
    return dst_p


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = DEFAULT_SR) -> Path:
    """写 wav（自动建目录）/ write a wav file, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float32)
    sf.write(str(p), data, sample_rate, subtype="PCM_16")
    return p


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """重采样（用 ffmpeg，保证质量一致）/ resample via ffmpeg.

    只在没有文件、只有内存数组时使用（例如参考音频截取后的重采样）。
    """
    if src_sr == dst_sr:
        return np.ascontiguousarray(samples, dtype=np.float32)
    if not ffmpeg_available():
        raise DecodeError("重采样需要 ffmpeg / ffmpeg is required to resample")
    raw = np.asarray(samples, dtype=np.float32).tobytes()
    cmd = [_FFMPEG, "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(src_sr),
           "-ac", "1", "-i", "pipe:0", "-f", "f32le", "-ar", str(dst_sr), "-ac", "1",
           "pipe:1"]
    proc = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DecodeError("重采样失败 / resampling failed")
    return np.ascontiguousarray(np.frombuffer(proc.stdout, dtype=np.float32), dtype=np.float32)


def slice_chunk(chunk: AudioChunk, start: float, end: float,
                target_sr: int | None = None) -> AudioChunk:
    """在内存里切一段 / slice a chunk in memory.

    时间轴以**原始音频**为基准（`chunk.offset + 段内偏移`），
    这样合并转写结果时不会漂移。
    """
    sr = chunk.sample_rate
    i0 = max(0, int(round(start * sr)))
    i1 = min(len(chunk.samples), int(round(end * sr)))
    out = chunk.samples[i0:i1]
    if target_sr and target_sr != sr:
        out = resample(out, sr, target_sr)
        sr = target_sr
    return AudioChunk(samples=np.ascontiguousarray(out, dtype=np.float32),
                      sample_rate=sr, offset=chunk.offset + i0 / chunk.sample_rate,
                      source=chunk.source)


__all__ = [
    "DEFAULT_SR",
    "decode",
    "decode_bytes",
    "ffmpeg_available",
    "resample",
    "slice_chunk",
    "to_ffmpeg_wav",
    "write_wav",
]
