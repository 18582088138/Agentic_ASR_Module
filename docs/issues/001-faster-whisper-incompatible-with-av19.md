# 001 · faster-whisper 1.2.1 与 av 19 不兼容

**状态**：已定位，修法已定（开发阶段实施）
**发现**：调研阶段实测，`scripts/_probe_asr.py` 首次运行即撞上

---

## 现象 / Symptom

装好 `faster-whisper 1.2.1`（依赖解析自动带上 `av 19.0.1`）后，把**文件路径**交给
`model.transcribe(path)` 直接崩：

```
Traceback (most recent call last):
  File "faster_whisper/transcribe.py", line 876, in transcribe
    audio = decode_audio(audio, sampling_rate=sampling_rate)
  File "faster_whisper/audio.py", line 46, in decode_audio
    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
  File "av/container/core.py", line 479, in open
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

## 判定 / Diagnosis

`av` 19 已移除 `av.open()` 的 `metadata_errors` 参数；而 faster-whisper 1.2.1 的依赖声明是
`av>=11`（**无上界**），所以今天 `pip install faster-whisper` 必然装上 19.x 并必踩。
不是配置问题，是上游版本搭配问题（`pypi` 元数据与 `av` 的实际 API 脱节）。

## 修法 / Fix

**不降级 `av`，改为自己做解码层。** 因为：

1. 这个模块本来就需要一条统一的音频解码通路（文件 / 视频 / 字节流 / 麦克风），
   四条输入要归一成同一份「16 kHz 单声道 float32」；
2. 「音频信息提取」功能本身就要用 `ffprobe`；
3. 降级 `av` 只是把同一个坑推给下一个装环境的人。

具体做法：把 `np.ndarray`（16 kHz mono float32）传给 `model.transcribe()`，
**永不把文件路径交给 faster-whisper**。`transcribe()` 对 ndarray 输入不会走 `av` 分支。

配套：`pyproject.toml` 中把 `av` 钉成 `av>=11,<19` 作为安全网（即使某条路径仍走内部解码），
并在 `tools/check.py` 的环境自检里断言 `faster_whisper` 可用（不实际调 `av`）。

## 回归用例 / Regression

`tests/test_audio_io.py::test_decode_to_16k_mono`（离线）
—— 覆盖 mp3/wav/m4a 三种输入，断言产出为 1 维、dtype `float32`、采样率 16000。

`tests/test_faster_whisper_real.py::test_transcribe_from_array`（`-m real`）
—— 断言传 ndarray 能出结果，且不触发 `av` 路径。

## 备注

实测证据与完整运行记录见 `docs/00_research.md §4.1` 与 `docs/09_dev_log.md`。
