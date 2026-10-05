# 002 · 纯音乐/无语音音频会触发 ASR 幻觉

**状态**：已确认，修法已定（开发阶段实施）
**发现**：调研阶段分离功能实测，2026-10-04

---

## 现象 / Symptom

给 Whisper 一段**没有任何人声**的音频（分离后的伴奏轨），它不会返回空结果，
而是编造出一段内容：

```
--- NO_VOCALS (music) ---
  lang=zh p=1.000 segs=1
  text: 优优独播剧场——YoYoTelevisionSeriesExclusive
```

输入是纯正弦波和弦 + 轻微噪声，**不含任何语音**。
复现：`scripts/_probe_separate.py` 生成 `outputs/_probe/`，再对其
`sep/htdemucs/mix_30s/no_vocals.wav` 跑 `faster_whisper`。

## 判定 / Diagnosis

这是 Whisper 系的已知失效模式：**模型被训练成「总是输出一段文本」**，
在无语音输入时会从训练分布里采样出高频片段（中文模型上的典型产物就是
「优优独播剧场」「字幕由 XX 提供」这类片尾声明）。语言识别还给出 `p=1.000`，
**置信度完全不可用于判伪**。

触发场景（本模块都会遇到）：

1. 用户对**纯音乐/纯伴奏**文件跑转写；
2. 用户先用「声音分离」导出伴奏轨，再对伴奏轨跑转写；
3. 长音频里的大段纯音乐间奏；
4. `no_vocals` 这类由分离产出的轨被误当成待转写音频。

## 修法 / Fix

门面层在送引擎之前与之后都设闸门，**不指望模型自己收敛**：

1. **前置静音闸门**：解码后用 VAD 算语音占比；低于阈值
   （默认 `transcribe.min_speech_ratio: 0.02`）直接返回**空 Transcript**
   并在 `warnings` 里说明，**不调用引擎**。
2. **后置压缩比闸门**：复用音频失控检测的思路 —— 若 `压缩比`
   （文本长度 / 音频时长）异常高且语音占比极低，标记 `suspicious=True`
   并写进 `warnings`，不静默丢弃、也不静默接受。
3. **分离产物默认不自动转写**：`separate()` 只产出音频轨；
   要转写必须显式再调 `transcribe()`，避免用户以为「分离完就自动出字幕」。
4. **分段级过滤**：长音频中若某个切片语音占比为 0，该段直接跳过，
   不交给引擎。

阈值都进 `configs/asr.yaml`，不写死在代码里。

## 回归用例 / Regression

`tests/test_guard.py::test_silence_returns_empty`（离线，`--engine mock`）
—— 纯静音与纯噪声输入必须得到空 Transcript，且 `warnings` 非空。

`tests/test_guard_real.py::test_music_only_no_hallucination`（`-m real`）
—— 对 `no_vocals.wav` 断言输出为空或不含「优优独播剧场」类内容。
**这条用例故意写死那段幻觉文本**，改代码时会先撞到它。

## 备注

对应 `docs/00_research.md §4.3`。同类问题在设计文档
`01_design.md §4.2`「时间轴单调校验」里也有位置 —— 那条管的是静默的
**时间轴** bug，这条管的是静默的**内容** bug，两者都要拦。
