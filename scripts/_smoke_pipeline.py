"""跨插件端到端验证 / cross-plugin end-to-end demo.

把两个插件串起来跑，用**真实模型**展示实际效果：

    素材（中文主体 + 英文背景 + 背景音乐）
      ├─ 分离：人声提取      → vocals 轨
      ├─ 分离：主体人声去除  → background 轨（应该不再有中文主体）
      └─ ASR：分别转写原混音 / vocals / background，对照结果

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_smoke_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_pipeline"
SAMPLES = ROOT / "models" / "SenseVoiceSmall" / "example"
SR = 44100


def build_material() -> Path:
    """中文主体（居中）+ 英文背景（偏侧、更轻）+ 背景音乐。"""
    import torch
    import torchaudio

    def load(path: Path) -> np.ndarray:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        return torchaudio.functional.resample(torch.from_numpy(mono), sr, SR).numpy()

    lead, back = load(SAMPLES / "zh.mp3"), load(SAMPLES / "en.mp3")
    n = int(24 * SR)
    lead = np.tile(lead, int(n / len(lead)) + 1)[:n]
    back = np.tile(back, int(n / len(back)) + 1)[:n]
    lead = lead / (np.max(np.abs(lead)) or 1.0) * 0.80
    back = back / (np.max(np.abs(back)) or 1.0) * 0.30
    t = np.arange(n) / SR
    music = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 330 * t)
    mix = np.stack([lead + back + music, lead + back * 0.6 + music], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "material.wav"
    sf.write(str(path), mix.astype(np.float32), SR, subtype="PCM_16")
    return path


def main() -> int:
    from agentic_asr import ASRModule
    from agentic_separate import SeparateModule

    material = build_material()
    print(f"素材 / material: {material.name}  24.0s @ {SR}Hz stereo")
    print("内容 / content: 中文主体（居中）+ 英文背景（偏侧 −10dB）+ 背景音乐\n")

    sep = SeparateModule()
    out_dir = OUT / "stems"

    print("=== 分离插件 / Agentic_VoiceSeparate_Module ===")
    t0 = time.perf_counter()
    v = sep.extract_vocals(material)
    t_vocals = time.perf_counter() - t0
    print(f"① 人声提取  RTF≈{t_vocals / v.duration:.3f}  -> {Path(v.stems['vocals'].path).name}")

    t0 = time.perf_counter()
    n = sep.remove_lead_vocal(material, out_dir=out_dir)
    t_lead = time.perf_counter() - t0
    print(f"③ 主体人声去除  RTF≈{t_lead / n.duration:.3f}  "
          f"lead={n.lead_stem}  依据={n.stem_order_resolved_by}")
    print(f"   background -> {Path(n.stems['background'].path).name}")
    print(f"   lead       -> {Path(n.stems['lead'].path).name}")

    asr = ASRModule()
    print("\n=== ASR 插件 / Agentic_ASR_Module ===")
    targets = [
        ("原始混音 / original", material),
        ("人声轨 / vocals", Path(v.stems["vocals"].path)),
        ("去主体后 / background", Path(n.stems["background"].path)),
        ("被移除的主体 / lead", Path(n.stems["lead"].path)),
    ]
    for label, path in targets:
        t0 = time.perf_counter()
        result = asr.transcribe(path, emotion=False)
        dt = time.perf_counter() - t0
        text = (result.text or "（空 / empty）").strip()
        print(f"{label:<24} RTF={dt / max(result.duration, 1e-6):.3f}  "
              f"segs={len(result.segments):<3} {text[:72]}")

    asr.release()
    sep.release()
    print("\n两个插件协作完成 / both plugins ran end to end")
    print(f"产物 / artefacts: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
