"""端到端冒烟：真实 faster-whisper 引擎跑通五个功能。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_smoke_e2e.py

覆盖：① 信息提取 ② 音频分段 ③ 字幕生成 ④ 情绪识别 ⑤ 参考音频截取。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "_smoke"


def build_demo(seconds: float = 60.0) -> Path:
    """人声循环 + 背景音乐 → 时长可控的演示素材。"""
    import torch
    import torchaudio

    src = ROOT / "models/SenseVoiceSmall/example/zh.mp3"
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    voice = data.mean(axis=1)
    voice = torchaudio.functional.resample(torch.from_numpy(voice), sr, 44100).numpy()
    reps = int(seconds * 44100 / len(voice)) + 1
    voice = np.tile(voice, reps)[: int(seconds * 44100)]
    voice = voice / max(float(np.abs(voice).max()), 1e-9) * 0.75

    t = np.arange(len(voice), dtype=np.float64) / 44100
    music = (0.06 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 330 * t))
    mix = np.stack([voice + music, voice + music], axis=1).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "demo_60s.wav"
    sf.write(str(path), mix, 44100)
    print(f"素材 / demo: {path.name}  {len(mix) / 44100:.1f}s @ 44100Hz stereo")
    return path


def main() -> int:
    from agentic_asr import ASRModule

    demo = build_demo()
    asr = ASRModule()

    print("\n=== ① 音频信息提取 / probe ===")
    t0 = time.perf_counter()
    info = asr.probe(demo)
    d = info.to_dict()
    print(f"  {time.perf_counter() - t0:.2f}s  duration={d['duration']:.2f} "
          f"sr={d['sample_rate']} ch={d['channels']} codec={d['codec']}")
    print(f"  peak={d['peak_dbfs']}dBFS rms={d['rms_dbfs']}dBFS "
          f"snr={d['estimated_snr_db']}dB speech={d['speech_ratio']} clip={d['clipping']}")

    print("\n=== ② 音频分段 / segment ===")
    t0 = time.perf_counter()
    pieces = asr.segment(demo)
    print(f"  {time.perf_counter() - t0:.2f}s  pieces={len(pieces)}")
    for p in pieces[:4]:
        print(f"    {p.index}: {p.start:7.3f}-{p.end:7.3f} ({p.duration:6.2f}s) hard={p.hard_cut}")
    if len(pieces) > 4:
        print(f"    ... 共 {len(pieces)} 段，末段 {pieces[-1].start:.2f}-{pieces[-1].end:.2f}")

    print("\n=== ③ 字幕生成（真实 faster-whisper）/ transcribe ===")
    t0 = time.perf_counter()
    r = asr.transcribe(demo, srt=OUT / "demo.srt")
    dt = time.perf_counter() - t0
    print(f"  {dt:.1f}s  RTF={dt / r.duration:.3f}  engine={r.engine} lang={r.language}")
    print(f"  text({len(r.text)} chars): {r.text[:100]}")
    print(f"  segments={len(r.segments)}  首段={r.segments[0].to_dict() if r.segments else None}")
    if r.segments and r.segments[0].words:
        print(f"  words={len(r.segments[0].words)}  前 5: "
              f"{[w.to_dict() for w in r.segments[0].words[:5]]}")
    print(f"  srt -> {OUT / 'demo.srt'}")
    srt_text = (OUT / "demo.srt").read_text(encoding="utf-8")
    print("  srt head:", repr(srt_text[:90]))

    print("\n=== ④ 情绪识别 / emotion ===")
    t0 = time.perf_counter()
    re_ = asr.transcribe(demo, emotion=True)
    print(f"  {time.perf_counter() - t0:.1f}s  summary={re_.emotion_summary}")
    print("  每段情绪:", [(s.start, s.emotion) for s in re_.segments[:6]])
    print("  warnings:", re_.warnings)

    print("\n=== ⑤ 参考音频截取 / clip ===")
    t0 = time.perf_counter()
    clips = asr.clip_reference(demo, transcript=r, top=3, out_dir=OUT / "refs")
    print(f"  {time.perf_counter() - t0:.1f}s  picked={len(clips)}")
    for c in clips:
        print(f"    #{c.index} {c.start:.2f}-{c.end:.2f} ({c.duration:.2f}s) "
              f"score={c.score:.3f} detail={c.score_detail}")
        print(f"       text={c.text[:60]!r}")
        print(f"       wav={Path(c.wav_path).name} txt={Path(c.txt_path or '').name}")

    asr.release()
    print("\n全部功能跑通 / all five functions completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
