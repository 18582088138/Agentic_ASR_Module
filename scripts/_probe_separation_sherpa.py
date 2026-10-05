"""调研期临时探针：验证 sherpa-onnx 的 OfflineSourceSeparation（UVR 系）实测表现。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_probe_separation_sherpa.py

验证：① UVR KARA 模型能否分离；② 输出是否与输入严格等长；③ RTF；④ 各轨实际内容。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/_probe/mix_30s.wav"
OUT = ROOT / "outputs/_probe/sherpa_sep"
MODELS = ROOT / "models/sherpa-separation"


def run(model_file: Path, tag: str, num_threads: int = 4):
    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model_file)),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        )
    )
    if not config.validate():
        raise SystemExit(f"config invalid: {model_file}")
    sp = sherpa_onnx.OfflineSourceSeparation(config)

    samples, sr = sf.read(str(SRC), dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(np.transpose(samples))   # (channels, n)

    t0 = time.perf_counter()
    out = sp.process(sample_rate=sr, samples=samples)
    dt = time.perf_counter() - t0
    dur = samples.shape[1] / sr

    print(f"\n=== {tag} ===")
    print(f"  in: sr={sr} shape={samples.shape} dur={dur:.3f}s")
    print(f"  out: sr={out.sample_rate} stems={len(out.stems)} "
          f"elapsed={dt:.1f}s RTF={dt / dur:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(out.stems):
        d = stem.data
        same = d.shape[1] == samples.shape[1]
        p = OUT / f"{tag}_stem{i}.wav"
        sf.write(str(p), np.transpose(d), out.sample_rate)
        print(f"  stem{i}: shape={d.shape} equal_len={same} -> {p.name}")
    return out


def main() -> int:
    for f, tag in ((MODELS / "UVR_MDXNET_KARA_2.onnx", "kara2"),
                   (MODELS / "UVR_MDXNET_KARA.onnx", "kara1")):
        if not f.is_file():
            print(f"missing {f}")
            continue
        try:
            run(f, tag)
        except Exception as e:  # noqa: BLE001
            print(f"{tag} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
