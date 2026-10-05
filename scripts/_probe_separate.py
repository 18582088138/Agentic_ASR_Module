"""调研期临时探针：验证 demucs htdemucs 在 4060 8 GB 上做「人声/背景音分离」的可行性。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_probe_separate.py

验证四件事：① 权重能否下载；② 8 GB 显存够不够；③ 耗时与 RTF；
④ 输出是否与原音频严格等长（下游要靠同一时间轴对齐字幕）。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_probe"
SR = 44100
SECONDS = 30.0


def vram_mb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False).stdout.strip().splitlines()
    return float(out[0]) if out else float("nan")


def make_mix(dst: Path) -> float:
    """人声（SenseVoice 样例）+ 合成背景音乐 → 44.1 kHz 立体声混音。"""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    voice, sr = sf.read(str(ROOT / "models/SenseVoiceSmall/example/zh.mp3"),
                        dtype="float32", always_2d=True)
    voice = voice.mean(axis=1)
    voice = torchaudio.functional.resample(
        torch.from_numpy(voice), sr, SR).numpy()
    reps = int(SECONDS * SR / len(voice)) + 1
    voice = np.tile(voice, reps)[: int(SECONDS * SR)]
    voice = voice / max(float(np.max(np.abs(voice))), 1e-9) * 0.8

    t = np.arange(len(voice), dtype=np.float64) / SR
    music = (0.15 * np.sin(2 * np.pi * 220 * t)
             + 0.12 * np.sin(2 * np.pi * 330 * t)
             + 0.10 * np.sin(2 * np.pi * 440 * t))
    music += 0.02 * np.random.default_rng(0).standard_normal(len(voice))

    mix = np.stack([voice + music, voice + music], axis=1).astype(np.float32)
    sf.write(str(dst), mix, SR)
    return len(mix) / SR


def peak_watch(stop: threading.Event, box: list[float]) -> None:
    while not stop.is_set():
        box.append(vram_mb())
        stop.wait(0.25)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / "mix_30s.wav"
    dur = make_mix(src)
    print(f"test mix: {dur:.2f}s @ {SR}Hz stereo -> {src}")

    base = vram_mb()
    stop = threading.Event()
    samples: list[float] = []
    watcher = threading.Thread(target=peak_watch, args=(stop, samples), daemon=True)
    watcher.start()

    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "--segment", "7", "-d", "cuda", "-o", str(OUT / "sep"), str(src)]
    print("cmd:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - t0

    stop.set()
    watcher.join(timeout=2)
    print(f"exit={proc.returncode} elapsed={elapsed:.1f}s RTF={elapsed/dur:.3f}")
    print(f"vram base={base:.0f}MB peak={max(samples):.0f}MB (+{max(samples)-base:.0f}MB)")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        print("--- stderr tail ---")
        print("\n".join(tail))

    track = OUT / "sep" / "htdemucs" / src.stem
    import soundfile as sf
    for stem in ("vocals", "no_vocals"):
        f = track / f"{stem}.wav"
        if f.exists():
            data, sr = sf.read(str(f), always_2d=True)
            print(f"  {stem}.wav  {len(data)/sr:.3f}s @ {sr}Hz  "
                  f"ch={data.shape[1]}  equal_len={abs(len(data)/sr - dur) < 0.05}")
        else:
            print(f"  {stem}.wav  MISSING")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
