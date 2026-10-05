# 003 · sherpa-onnx 的 Silero VAD 必须分块喂入，否则静默漏检

**状态**：已修复（`agentic_asr/audio/vad.py::_silero_intervals`）
**发现**：开发阶段冒烟测试，2026-10-04

---

## 现象 / Symptom

一段 5.6 s 的**干净人声**（峰值 −1.49 dBFS、RMS −17.6 dBFS，内容是
「开放时间早上九点至下午五点」）经 VAD 后：

- `speech_ratio` 只有 **0.0559**
- `estimated_snr_db` 算出来是 **−48.12 dB**（人声比"噪声"还低，显然荒谬）
- 参考音频截取拿不到任何有效候选

进一步定位到 VAD 本身：

```
silero intervals: [(5.286, 5.6)]      # 只认出最后 0.31 s
energy intervals: [(0.68, 5.145)]     # 能量法反而对了
```

## 判定 / Diagnosis

**不是参数问题，是喂入方式问题。** 逐项排查（同一段音频）：

| 喂入方式 | 检出区间 |
|---|---|
| 整段一次性 `accept_waveform(samples)` | `(5.29, 5.60)` ❌ |
| 分块 512 样本 | **`(0.74, 5.13)`** ✅ |
| 分块 1600 样本 | `(0.77, 5.22)` ✅ |
| 整段 + `threshold=0.2` | `(5.29, 5.60)` ❌ |
| 整段 + `buffer_size_in_seconds=10` | `(5.29, 5.60)` ❌ |
| 整段 + `max_speech_duration=5` | `(5.29, 5.60)` ❌ |

⇒ 只要**整段一次性**喂，无论怎么调阈值、buffer、最长语音时长，结果都一样错。
改成按窗口分块喂就立刻正确。这是 sherpa-onnx `VoiceActivityDetector` 的使用陷阱。

## 为什么危险 / Why it matters

它**不报错**。`config.validate()` 返回 True，`accept_waveform` 返回 None，
`flush()` 正常，只是结果变得没用。而 VAD 在本模块是三个功能的共同底座：

1. **幻觉闸门**（语音占比阈值，`issues/002`）—— 占比算错 0.06，会把所有人声
   当成"无语音"直接返回空结果，**表现为转写永远为空**；
2. **音频分段**（静音点优先）—— 切点全错；
3. **参考音频截取**（语音区间 + 打分）—— 挑不出候选。

所以这一个 bug 会以三种完全不同、看起来毫无关联的症状出现，
排查成本极高。必须记档。

## 修法 / Fix

在 `_silero_intervals` 中按 **512 样本**（Silero 的 `window_size`）分块喂入：

```python
self._vad.reset()
for i in range(0, samples.size, 512):
    self._vad.accept_waveform(samples[i:i + 512])
self._vad.flush()
```

能量 VAD 回退路径不受影响（实测其区间本来就正确）。

## 回归用例 / Regression

`tests/test_vad.py::test_silero_detects_speech_in_clean_clip`（`-m real`）
—— 对仓库内固定的样例音频断言 `speech_ratio > 0.5`。
**这条用例故意把阈值写死在 0.5**：一旦有人改回整段喂入，它会立刻变红。

`tests/test_vad.py::test_energy_fallback_ratio`（离线）
—— 不依赖模型文件，断言能量 VAD 在合成正弦+静音上给出的占比落在合理区间，
保证没有 Silero 模型时链路仍然可用。

## 备注

对应 `docs/00_research.md §4`（VAD 是本模块的决策输入）与
`docs/01_design.md §3.4 / §3.8`（分段与幻觉闸门）。
