"""调研期临时探针：「主体人声去除」泛化性实测。

构造一段同时含【主体人声】与【背景人声】的素材（此前只有单一人声，无法检验
KARA 系列是否真能区分 lead / backing），再对比各方案的输出内容。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_probe_leadvocal.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/_probe/leadvocal"
MODELS = ROOT / "models/sherpa-separation"
SR = 44100
SECONDS = 20.0


def resample_to(path: Path, target: int = SR) -> np.ndarray:
    import torch
    import torchaudio

    d, sr = sf.read(str(path), dtype="float32", always_2d=True)
    m = d.mean(axis=1)
    if sr != target:
        m = torchaudio.functional.resample(torch.from_numpy(m), sr, target).numpy()
    return m


def build_mix() -> Path:
    """主体人声（中文，居中）+ 背景人声（英文，偏侧且更轻）→ 44.1 kHz 立体声。"""
    ex = ROOT / "models/SenseVoiceSmall/example"
    lead = resample_to(ex / "zh.mp3")
    back = resample_to(ex / "en.mp3")

    n = int(SECONDS * SR)
    lead = np.tile(lead, int(n / len(lead)) + 1)[:n]
    back = np.tile(back, int(n / len(back)) + 1)[:n]
    lead = lead / max(float(np.abs(lead).max()), 1e-9) * 0.85
    back = back / max(float(np.abs(back).max()), 1e-9) * 0.30   # 背景更轻

    # 主体严格居中；背景给一点左右差，模拟“不在画面中央”的人声
    left = lead + back
    right = lead + back * 0.65
    mix = np.stack([left, right], axis=1).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "mix_lead_back.wav"
    sf.write(str(p), mix, SR)
    print(f"built {p.name}: {SECONDS}s @ {SR}Hz  lead=zh(center) back=en(offset, -10dB)")
    return p


def run_sherpa(model_file: Path, tag: str) -> dict[str, Path]:
    import sherpa_onnx

    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model_file)),
            num_threads=4, debug=False, provider="cpu"))
    if not config.validate():
        raise SystemExit("bad config")
    sp = sherpa_onnx.OfflineSourceSeparation(config)

    samples, sr = sf.read(str(OUT / "mix_lead_back.wav"), dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(np.transpose(samples))
    t0 = time.perf_counter()
    out = sp.process(sample_rate=sr, samples=samples)
    dt = time.perf_counter() - t0
    dur = samples.shape[1] / sr
    print(f"  [{tag}] elapsed={dt:.1f}s RTF={dt / dur:.3f} stems={len(out.stems)}")
    paths = {}
    for i, stem in enumerate(out.stems):
        p = OUT / f"{tag}_stem{i}.wav"
        sf.write(str(p), np.transpose(stem.data), out.sample_rate)
        paths[f"{tag}_stem{i}"] = p
    return paths


def run_demucs() -> dict[str, Path]:
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "--segment", "7",
           "--clip-mode", "none", "-d", "cuda", "-o", str(OUT / "demucs"),
           str(OUT / "mix_lead_back.wav")]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, check=False,
                       encoding="utf-8", errors="replace")
    print(f"  [demucs] exit={r.returncode} elapsed={time.perf_counter() - t0:.1f}s")
    base = OUT / "demucs/htdemucs/mix_lead_back"
    return {"demucs_vocals": base / "vocals.wav",
            "demucs_no_vocals": base / "no_vocals.wav"}


def verdict(paths: dict[str, Path]) -> None:
    import torch
    import torchaudio
    from faster_whisper import WhisperModel

    model = WhisperModel(str(ROOT / "models/faster-whisper-large-v3-turbo"),
                         device="cuda", compute_type="float16")

    def load(p: Path) -> np.ndarray:
        d, sr = sf.read(str(p), dtype="float32", always_2d=True)
        m = d.mean(axis=1)
        if sr != 16000:
            m = torchaudio.functional.resample(torch.from_numpy(m), sr, 16000).numpy()
        return np.ascontiguousarray(m, dtype=np.float32)

    print("\n--- 各轨内容（用 ASR 判读：中文=主体人声，英文=背景人声）---")
    src = OUT / "mix_lead_back.wav"
    items = [("原始混音(zh+en)", src), *sorted(paths.items())]
    for label, p in items:
        if not p.exists():
            print(f"{label:26} MISSING")
            continue
        a = load(p)
        rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
        segs, _ = model.transcribe(a, beam_size=5, vad_filter=False)
        text = "".join(s.text for s in segs).strip().replace(" ", "")
        print(f"{label:26} rms={rms:.4f}  {text[:88]}")


def main() -> int:
    build_mix()
    paths: dict[str, Path] = {}
    for f, tag in ((MODELS / "UVR_MDXNET_KARA_2.onnx", "kara2"),
                   (MODELS / "UVR_MDXNET_KARA.onnx", "kara1")):
        if f.is_file():
            try:
                paths |= run_sherpa(f, tag)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tag}] FAILED {type(e).__name__}: {e}")
    try:
        paths |= run_demucs()
    except Exception as e:  # noqa: BLE001
        print(f"  [demucs] FAILED {type(e).__name__}: {e}")
    verdict(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
