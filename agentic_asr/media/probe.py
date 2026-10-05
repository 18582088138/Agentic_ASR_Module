"""音频/视频信息提取 / Media probing.

两段式 / two stages:

1. **容器与流**（`ffprobe -print_format json`）：容器格式、总时长、总码率、
   每条流的编码/采样率/声道/位深，以及是否存在视频轨；
2. **声学统计**（解码后计算）：峰值 / RMS dBFS、**估计 SNR**、VAD 语音占比、是否削波。

两个统计量是**决策输入**，不只是展示：
- `estimated_snr_db` → 前处理（降噪）开关的判据；
- `speech_ratio`     → 幻觉闸门的判据（见 `docs/issues/002`）。

深度分析默认只解码前 `max_deep_seconds` 秒 —— 对一小时的素材没必要全解码
才能知道它有没有削波。Deep analysis decodes only the head of long media.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agentic_asr.audio import features as feat
from agentic_asr.audio.io import DEFAULT_SR, decode
from agentic_asr.audio.vad import VadDetector
from agentic_asr.core.errors import DecodeError
from agentic_asr.core.logging import get_logger
from agentic_asr.core.types import AudioInfo

log = get_logger("media.probe")

_FFPROBE = shutil.which("ffprobe")


def ffprobe_available() -> bool:
    return _FFPROBE is not None


def probe_container(path: str | Path) -> dict[str, Any]:
    """用 ffprobe 读容器与流信息 / read container & stream metadata.

    抛出 / Raises:
        DecodeError: ffprobe 不存在或文件读不了。
    """
    p = Path(path)
    if not p.is_file():
        raise DecodeError(f"文件不存在 / no such file: {p}")
    if _FFPROBE is None:
        raise DecodeError("ffprobe 不在 PATH 上 / ffprobe not found on PATH")
    cmd = [_FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(p)]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise DecodeError("ffprobe 失败 / ffprobe failed: " + " / ".join(tail))
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise DecodeError(f"ffprobe 输出不是 JSON / bad JSON: {exc}") from exc


def probe(path: str | Path, vad: VadDetector | None = None, deep: bool = True,
          max_deep_seconds: float = 300.0,
          vad_model_path: str | Path | None = None) -> AudioInfo:
    """完整探测 / full probe.

    参数 / Args:
        vad: 复用的 VAD 实例（门面传入可省一次模型加载）。
        deep: 是否做声学统计（要解码，慢）；False 时只读容器元信息。
        max_deep_seconds: 深度分析最多解码多少秒。
        vad_model_path: 未传 `vad` 时用它构造一个 VAD。
    """
    p = Path(path)
    info = AudioInfo(path=str(p))

    # 文件不存在是**明确的用户错误**，直接抛 —— 不要静默返回一份空 info，
    # 否则 CLI/HTTP 会给出「成功」的退出码而结果里什么都没有（实测踩过）。
    # A missing file is an explicit user error, not an empty result.
    if not p.is_file():
        raise DecodeError(f"文件不存在 / no such file: {p}")

    try:
        meta = probe_container(p)
    except DecodeError as exc:
        info.warnings.append(str(exc))
        return info

    fmt = meta.get("format") or {}
    streams = meta.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]

    info.container = str(fmt.get("format_name") or "")
    try:
        info.duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0
    try:
        info.bit_rate = int(fmt.get("bit_rate") or 0) or None
    except (TypeError, ValueError):
        info.bit_rate = None

    if audio_streams:
        a = audio_streams[0]
        info.codec = str(a.get("codec_name") or "")
        try:
            info.sample_rate = int(a.get("sample_rate") or 0) or None
        except (TypeError, ValueError):
            info.sample_rate = None
        info.channels = int(a.get("channels") or 0) or None
        info.bits_per_sample = int(a.get("bits_per_raw_sample") or 0) or None
    else:
        info.warnings.append("没有音频流 / no audio stream found")

    info.has_video = bool(video_streams)
    if info.has_video:
        v = video_streams[0]
        info.video = {"codec": v.get("codec_name"),
                      "width": v.get("width"), "height": v.get("height"),
                      "fps": v.get("r_frame_rate")}

    if not deep or not audio_streams:
        return info

    # ── 声学统计 / acoustic stats ──────────────────────────────────────────
    try:
        duration = min(max_deep_seconds, info.duration) if info.duration else None
        chunk = decode(p, target_sr=DEFAULT_SR, duration=duration)
    except DecodeError as exc:
        info.warnings.append(f"深度分析失败 / deep analysis failed: {exc}")
        return info

    detector = vad or VadDetector(model_path=vad_model_path)
    try:
        intervals = detector.intervals(chunk.samples, chunk.sample_rate)
        stats = feat.analyse(chunk.samples, chunk.sample_rate, intervals)
    except Exception as exc:  # noqa: BLE001 - 统计失败不该让 probe 整体失败
        info.warnings.append(f"统计失败 / stats failed: {exc}")
        return info

    info.peak_dbfs = stats["peak_dbfs"]                  # type: ignore[assignment]
    info.rms_dbfs = stats["rms_dbfs"]                    # type: ignore[assignment]
    info.estimated_snr_db = stats["estimated_snr_db"]    # type: ignore[assignment]
    info.speech_ratio = stats["speech_ratio"]            # type: ignore[assignment]
    info.clipping = bool(stats["clipping"])
    if info.clipping:
        info.warnings.append("素材已削波 / material is clipped，转写结果可能不可靠")
    if chunk.duration + 1e-6 < (info.duration or 0.0):
        info.warnings.append(
            f"深度分析只覆盖前 {chunk.duration:.1f}s / analysed the head only")
    return info


__all__ = ["ffprobe_available", "probe", "probe_container"]
