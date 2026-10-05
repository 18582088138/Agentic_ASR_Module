"""字幕断句效果对比 / Subtitle segmentation comparison.

用**真实 ASR 输出**对比不同切分参数下的字幕形态，直观回答「哪种看着更舒服」。

运行 / Run:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe scripts\\_smoke_subtitle.py [素材路径]

默认用 `outputs/_smoke/demo_60s.wav`（若不存在则用仓库内的示例音频）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FALLBACK = ROOT / "models" / "SenseVoiceSmall" / "example" / "zh.mp3"
PREFERRED = ROOT / "outputs" / "_smoke" / "demo_60s.wav"


def main() -> int:
    from agentic_asr import ASRModule
    from agentic_asr.subtitle.segment import SegmentParams, resegment

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PREFERRED if PREFERRED.is_file() else FALLBACK)
    if not src.is_file():
        print(f"找不到素材 / no media: {src}")
        return 1

    asr = ASRModule()
    result = asr.transcribe(src, emotion=False)
    asr.release()

    raw = result.segments
    print(f"素材 / media : {src.name}")
    print(f"原始段数 / raw segments : {len(raw)}　总时长 {result.duration:.1f}s")
    if raw:
        print(f"  最长一条 / longest     : {max(s.duration for s in raw):.1f}s / "
              f"{max(len(s.text) for s in raw)} 字")
    print()

    # 字间停顿的真实分布 —— 切分规则全靠它，先确认它是不是合理
    gaps: list[float] = []
    for seg in raw:
        words = seg.words or []
        for a, b in zip(words, words[1:], strict=False):
            gaps.append(b.start - a.end)
    if gaps:
        gaps.sort()
        n = len(gaps)
        print(f"字间停顿 / inter-word gaps（共 {n} 个）")
        print(f"  中位数 {gaps[n // 2]:.3f}s　p90 {gaps[int(n * 0.9)]:.3f}s　"
              f"最大 {gaps[-1]:.3f}s")
        over_hard = sum(1 for g in gaps if g >= 0.28)
        over_soft = sum(1 for g in gaps if g >= 0.12)
        print(f"  ≥0.28s 的 {over_hard} 个（强停顿）　≥0.12s 的 {over_soft} 个（弱停顿）")
    print()

    presets = [
        ("旧默认 (max_chars=42)", SegmentParams(max_chars=42, min_chars=12)),
        ("新默认 (max_chars=20)", SegmentParams(max_chars=20, min_chars=12)),
        ("更碎   (max_chars=14)", SegmentParams(max_chars=14, min_chars=8)),
    ]
    print("参数对比 / parameter comparison")
    print(f"  {'预设':<24} {'条数':>5} {'平均':>7} {'最长':>7} {'最短':>7}")
    for label, params in presets:
        cues = resegment(raw, cfg=params)
        if not cues:
            print(f"  {label:<24} （空）")
            continue
        durations = [c.duration for c in cues]
        print(f"  {label:<24} {len(cues):>5} "
              f"{sum(durations) / len(durations):>6.2f}s "
              f"{max(durations):>6.1f}s {min(durations):>6.1f}s")
    print()

    cues = resegment(raw, cfg=SegmentParams(max_chars=20, min_chars=12))
    print("新规则下的前 10 条 / first 10 cues under the new rules")
    for cue in cues[:10]:
        print(f"  [{cue.start:6.2f} -> {cue.end:6.2f}] ({cue.duration:4.2f}s) {cue.text}")
    if len(cues) > 10:
        print(f"  ...（共 {len(cues)} 条）")

    # 为什么有的条目挂着很久？把最长那条的原始数据摊开看
    longest = max(raw, key=lambda s: s.duration)
    print()
    print(f"最长原始段的内部结构 / internals of the longest segment "
          f"({longest.duration:.2f}s, {len(longest.text)} 字)")
    print(f"  segment: {longest.start:.2f} -> {longest.end:.2f}　text={longest.text!r}")
    words = longest.words or []
    if words:
        print(f"  words[0]  {words[0].start:.2f} -> {words[0].end:.2f}  {words[0].text!r}")
        print(f"  words[-1] {words[-1].start:.2f} -> {words[-1].end:.2f}  {words[-1].text!r}")
        span = words[-1].end - words[0].start
        print(f"  words 跨度 {span:.2f}s（segment 跨度 {longest.duration:.2f}s，"
              f"差额 {longest.duration - span:.2f}s）")
        print("  逐字时长 / per-character durations:")
        for w in words:
            print(f"    {w.start:6.2f} -> {w.end:6.2f}  ({w.end - w.start:4.2f}s)  {w.text!r}")

        # 规则内部到底切了没？把未经 postprocess 的原始切分打印出来
        from agentic_asr.subtitle.segment import _units_from_segment, segment_units

        params = SegmentParams(max_chars=20, min_chars=12)
        units = _units_from_segment(longest)
        print(f"  规则原始输出 / raw cues before postprocess（max_seconds={params.max_seconds}）:")
        for c in segment_units(units, params):
            print(f"    {c.start:6.2f} -> {c.end:6.2f}  ({c.duration:5.2f}s)  {c.text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
