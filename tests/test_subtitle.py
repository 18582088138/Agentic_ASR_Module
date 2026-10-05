"""字幕断句与渲染 / subtitle re-segmentation and rendering.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_subtitle.py -q

覆盖 / covers: SRT/VTT/ASS 格式与时间码、三级断句、时间轴越界防护、卡拉 OK 标签。
说明文档 / docs: docs/01_design.md §3.5
"""

from __future__ import annotations

import re

import pytest

from agentic_asr.core.errors import ASRError
from agentic_asr.core.types import Segment, Transcript, Word
from agentic_asr.subtitle.render import render, to_ass, to_srt, to_vtt
from agentic_asr.subtitle.segment import SegmentParams, resegment, wrap_text

_SRT_TS = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")


def _transcript(segments: list[Segment], duration: float = 30.0) -> Transcript:
    return Transcript(text=" ".join(s.text for s in segments), language="zh",
                      duration=duration, segments=segments, engine="mock")


def test_srt_format_and_numbering() -> None:
    segs = [Segment(id=0, start=0.0, end=2.5, text="第一句"),
            Segment(id=1, start=2.5, end=5.0, text="第二句")]
    out = to_srt(segs, duration=10.0)
    lines = out.strip().splitlines()
    assert lines[0] == "1"
    assert _SRT_TS.match(lines[1]), lines[1]
    assert "2" in lines
    assert out.count("-->") == 2


def test_vtt_uses_dot_in_timestamps() -> None:
    out = to_vtt([Segment(id=0, start=1.0, end=2.0, text="hello")], duration=5.0)
    assert out.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:02.000" in out


def test_ass_has_header_and_dialogue() -> None:
    out = to_ass([Segment(id=0, start=0.5, end=1.5, text="你好")], duration=5.0)
    assert "[Script Info]" in out and "[Events]" in out
    assert out.count("Dialogue:") == 1
    assert "0:00:00.50" in out


def test_ass_karaoke_uses_word_timings() -> None:
    seg = Segment(id=0, start=0.0, end=1.0, text="你好世界",
                  words=[Word(text="你好", start=0.0, end=0.5),
                         Word(text="世界", start=0.5, end=1.0)])
    out = to_ass([seg], duration=2.0, karaoke=True)
    assert "\\k50" in out


def test_negative_timestamp_is_rejected() -> None:
    """时间轴越界必须报错 —— 一个负时间码会让某些播放器整份字幕都不加载。"""
    with pytest.raises(ASRError):
        to_srt([Segment(id=0, start=-1.0, end=2.0, text="bad")], duration=5.0)


def test_beyond_duration_is_rejected() -> None:
    with pytest.raises(ASRError):
        to_srt([Segment(id=0, start=0.0, end=99.0, text="bad")], duration=5.0)


def test_resegment_splits_long_segment_on_words() -> None:
    words = [Word(text=f"字{i}", start=i * 1.0, end=(i + 1) * 1.0) for i in range(12)]
    seg = Segment(id=0, start=0.0, end=12.0, text="".join(w.text for w in words),
                  words=words)
    out = resegment([seg], cfg=None)
    assert len(out) > 1, "12 秒 / 24 字的段应被拆开"
    assert all(s.duration <= 7.0 + 0.01 for s in out)
    # 拆分后首尾必须仍在原段范围内
    assert out[0].start == pytest.approx(seg.start)
    assert out[-1].end == pytest.approx(seg.end)


def test_resegment_keeps_short_segment_intact() -> None:
    seg = Segment(id=0, start=0.0, end=2.0, text="短句。")
    assert resegment([seg], cfg=None)[0].text == "短句。"


def test_segmentation_trims_leading_punctuation_only() -> None:
    """只去**开头**的悬挂标点，句末标点必须保留。

    上游那份实现用的是 `strip()`（首尾都去），会把「短句。」吃成「短句」——
    转写场景下句末标点是原文的一部分。这条用例把这个行为锁死。
    """
    seg = Segment(id=0, start=0.0, end=2.0, text="，开头的逗号要去掉。")
    assert resegment([seg], cfg=None)[0].text == "开头的逗号要去掉。"


def test_numeric_protection_keeps_numbers_whole() -> None:
    """数字内部不许切：`零点三|四四米` 这种切法会把数字读坏。

    这是上游那份切分逻辑里最容易被忽略、也最值得抄过来的一条。
    """
    from agentic_asr.subtitle.segment import SegmentParams, Unit, segment_units

    units = [Unit(start=i * 0.1, end=i * 0.1 + 0.1, text=ch)
             for i, ch in enumerate("零点三四四米")]
    # max_chars 故意设得很小，逼它在数字中间切 —— 数字保护必须挡住
    cues = segment_units(units, SegmentParams(max_chars=1, min_chars=1,
                                              pause_gap=99.0, soft_gap=99.0,
                                              max_seconds=99.0))
    joined = "".join(c.text for c in cues)
    assert joined == "零点三四四米", f"数字被切坏了：{[c.text for c in cues]}"


def test_weight_counts_english_by_words() -> None:
    """英文按词折算，否则一行英文会比一行中文长得多。"""
    from agentic_asr.subtitle.segment import weight

    assert weight("中文四个字") == pytest.approx(5, abs=0.01)
    # 4 个英文词 × 4 = 16，再加句末标点 1
    assert weight("hello world foo bar.") > 12


def test_short_cue_is_merged_into_previous() -> None:
    """过短的尾条要并入上一条，否则会出现一闪而过的字幕。"""
    from agentic_asr.subtitle.segment import SegmentParams

    words = [Word(text="第", start=0.0, end=1.0), Word(text="一", start=1.0, end=2.0),
             Word(text="句", start=2.0, end=3.0), Word(text="。", start=3.0, end=3.05),
             Word(text="尾", start=3.05, end=3.1)]
    seg = Segment(id=0, start=0.0, end=3.1, text="第一句。尾", words=words)
    # merge_below_seconds 设大，逼它合并
    cues = resegment([seg], cfg=SegmentParams(merge_below_seconds=5.0))
    assert len(cues) == 1, f"过短的尾条应被合并，实际：{[c.text for c in cues]}"


def test_soft_ceiling_waits_for_punctuation() -> None:
    """**到软上限后继续走到下一个标点**，而不是就地切。

    中文没有词边界，一到字数就切必然落在某个词中间 —— 实测切开过
    「函 / 数」「基 / 本」。这条规则是从 DailyNewsAssistant 的字幕侧移过来的。
    """
    from agentic_asr.subtitle.segment import SegmentParams, Unit, segment_units

    text = "前面铺垫一句话，后面这句很长但没有内部标点所以只能等到硬上限"
    units = [Unit(start=i * 0.2, end=i * 0.2 + 0.2, text=ch) for i, ch in enumerate(text)]
    params = SegmentParams(max_chars=10, min_chars=4, hard_chars_ratio=3.0,
                           pause_gap=99.0, soft_gap=99.0, max_seconds=99.0)
    cues = segment_units(units, params)
    assert cues
    # 第一个逗号在「话，」处（第 8 字之后）—— 软上限之前就到过它，
    # 但那时还没到 min_chars 之外的软上限，所以真正的切点应当在逗号上
    assert cues[0].text.endswith("，"), f"没有在标点处切：{cues[0].text!r}"


def test_hard_ceiling_is_the_last_resort() -> None:
    """整串没有标点时，只有到**硬上限**才切 —— 不是一到软上限就切。

    软上限 8、硬上限 24：20 字的无标点长串应当**原样保留为一条**。
    """
    from agentic_asr.subtitle.segment import SegmentParams, Unit, segment_units

    text = "这串文字完全没有标点" * 2      # 20 字，一个标点都没有
    units = [Unit(start=i * 0.2, end=i * 0.2 + 0.2, text=ch) for i, ch in enumerate(text)]
    params = SegmentParams(max_chars=8, min_chars=4, hard_chars_ratio=3.0,
                           pause_gap=99.0, soft_gap=99.0, max_seconds=99.0)
    cues = segment_units(units, params)
    assert len(cues) == 1, f"不该在软上限处切开：{[c.text for c in cues]}"
    assert cues[0].text == text, "整串短于硬上限，应当原样保留"


def test_hard_ceiling_does_cut_when_there_is_no_punctuation_at_all() -> None:
    """没有标点可等时，硬上限就是最后一道闸 —— 否则会攒出一条无限长的字幕。"""
    from agentic_asr.subtitle.segment import SegmentParams, Unit, segment_units

    text = "这串文字完全没有标点而且长到超过了硬上限所以必须切开" * 2   # 50 字
    units = [Unit(start=i * 0.2, end=i * 0.2 + 0.2, text=ch) for i, ch in enumerate(text)]
    params = SegmentParams(max_chars=8, min_chars=4, hard_chars_ratio=3.0,
                           pause_gap=99.0, soft_gap=99.0, max_seconds=99.0)
    cues = segment_units(units, params)
    assert len(cues) > 1, "超过硬上限还不切，会攒出一条读不完的字幕"
    assert all(len(c.text) <= 24 for c in cues[:-1]), \
        f"除尾条外不该超过硬上限：{[len(c.text) for c in cues]}"


def test_timeline_stays_monotonic() -> None:
    """切分后时间轴必须单调不重叠 —— 否则某些播放器整份字幕都不加载。"""
    words = [Word(text=ch, start=i * 0.5, end=i * 0.5 + 0.5)
             for i, ch in enumerate("这是很长的一句话需要被切成好几条字幕才行。")]
    seg = Segment(id=0, start=0.0, end=len(words) * 0.5, text="".join(w.text for w in words),
                  words=words)
    cues = resegment([seg], cfg=SegmentParams(max_chars=8, min_chars=4))
    assert len(cues) > 1, "长句应被切成多条"
    for a, b in zip(cues, cues[1:], strict=False):
        assert b.start >= a.end - 1e-6, f"时间轴重叠：{a.end} > {b.start}"


def test_overlength_unit_becomes_a_visible_gap() -> None:
    """ASR 会把静音**归给某个字**，必须钳制成间隙，否则字幕十几秒一条。

    实测：faster-whisper 把 11.68 s 的静音挂在一个「上」字上，于是字间 gap
    看起来是 0，基于停顿的规则完全失效，切出来的字幕依旧 14 秒。见 `issues/007`。
    """
    words = [Word(text="开", start=0.0, end=0.3),
             Word(text="上", start=0.3, end=11.98),   # 异常长：吞掉了 11 s 静音
             Word(text="九", start=11.98, end=12.2)]
    seg = Segment(id=0, start=0.0, end=12.2, text="开上九", words=words)
    cues = resegment([seg], cfg=SegmentParams(max_seconds=6.0))
    assert len(cues) >= 2, f"应在「上」之后切开，实际：{[(c.text, round(c.duration, 2)) for c in cues]}"
    assert all(c.duration <= 6.5 for c in cues), \
        f"仍有超长条目：{[(c.text, round(c.duration, 2)) for c in cues]}"


def test_max_seconds_actually_bounds_cue_length() -> None:
    """`max_seconds` 必须真的兜住单条时长 —— 它是防「一条字幕挂很久」的最后一道闸。"""
    words = [Word(text=ch, start=i * 1.2, end=(i + 1) * 1.2)
             for i, ch in enumerate("一二三四五六七八九十甲乙丙丁")]
    seg = Segment(id=0, start=0.0, end=len(words) * 1.2,
                  text="".join(w.text for w in words), words=words)
    cues = resegment([seg], cfg=SegmentParams(max_seconds=4.0, max_chars=99, min_chars=99))
    assert cues
    assert max(c.duration for c in cues) <= 4.0 + 0.5


def test_resegment_without_words_falls_back_to_punctuation() -> None:
    seg = Segment(id=0, start=0.0, end=20.0,
                  text="第一句话。第二句话。第三句话。第四句话。第五句话。")
    out = resegment([seg], cfg=None)
    assert len(out) >= 2
    assert all(s.start <= s.end for s in out)


def test_render_dispatch_and_bad_format() -> None:
    tr = _transcript([Segment(id=0, start=0.0, end=1.0, text="hi")])
    assert render(tr, "srt").startswith("1")
    assert render(tr, "vtt").startswith("WEBVTT")
    assert "[Script Info]" in render(tr, "ass")
    with pytest.raises(ASRError):
        render(tr, "docx")


def test_wrap_text_respects_limit() -> None:
    long = "一二三四五六七八九十" * 3
    wrapped = wrap_text(long, max_line_chars=10)
    assert "\n" in wrapped
    assert all(len(line) <= 15 for line in wrapped.splitlines())
