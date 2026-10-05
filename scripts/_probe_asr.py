"""调研期临时探针：验证 faster-whisper large-v3-turbo 在中/英/日上的实测表现。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_probe_asr.py

产出：语言识别、转写文本、词级时间戳、RTF、显存峰值。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = Path(__file__).resolve().parent.parent / "models"
CT2_DIR = MODELS / "faster-whisper-large-v3-turbo"
SAMPLES = MODELS / "SenseVoiceSmall" / "example"


def vram_used_mb() -> float:
    """CTranslate2 不走 torch 的 CUDA 分配器，只能问驱动 / only the driver knows."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False).stdout.strip().splitlines()
    return float(out[0]) if out else float("nan")


def load_audio(path: Path, target_sr: int = 16000):
    """自控解码：绕开 faster-whisper 内部的 av 解码路径（av 19 已移除 metadata_errors）。"""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = torchaudio.functional.resample(
            torch.from_numpy(mono), sr, target_sr).numpy()
    return np.ascontiguousarray(mono, dtype=np.float32)


def main() -> int:
    import torch
    from faster_whisper import WhisperModel

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    base = vram_used_mb()
    t0 = time.perf_counter()
    model = WhisperModel(str(CT2_DIR), device="cuda", compute_type="float16")
    print(f"load: {time.perf_counter() - t0:.1f}s  vram+{vram_used_mb() - base:.2f}GB")

    for tag in ("zh", "en", "ja", "ko"):
        audio = SAMPLES / f"{tag}.mp3"
        if not audio.exists():
            continue
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            load_audio(audio), word_timestamps=True, vad_filter=True, beam_size=5)
        segments = list(segments)          # 生成器，必须消费完才真正计算
        elapsed = time.perf_counter() - t0

        print(f"\n--- {tag}.mp3  {info.duration:.2f}s audio ---")
        print(f"  lang={info.language} p={info.language_probability:.3f} "
              f"elapsed={elapsed:.2f}s RTF={elapsed / max(info.duration, 1e-6):.3f} "
              f"vram+{vram_used_mb() - base:.2f}GB")
        for seg in segments:
            print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] {seg.text.strip()}")
            words = getattr(seg, "words", None) or []
            if words:
                head = " ".join(f"{w.word.strip()}@{w.start:.2f}" for w in words[:6])
                print(f"      words({len(words)}): {head}")

    # 长音频 RTF：把 zh 样例拼到约 3 分钟，观察吞吐是否线性
    import numpy as np
    import soundfile as sf

    mono = load_audio(SAMPLES / "zh.mp3")
    reps = int(180 * 16000 / len(mono)) + 1
    long_path = Path(os.environ.get("TEMP", ".")) / "asr_probe_long.wav"
    sf.write(str(long_path), np.tile(mono, reps), 16000)

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        load_audio(long_path), vad_filter=True, beam_size=5)
    n = sum(1 for _ in segments)
    elapsed = time.perf_counter() - t0
    print(f"\n--- long {info.duration:.1f}s audio, {n} segments ---")
    print(f"  elapsed={elapsed:.1f}s RTF={elapsed / info.duration:.3f} "
          f"vram+{vram_used_mb() - base:.2f}GB")
    long_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
