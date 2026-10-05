# 03 · 单元测试说明 / Unit tests

> **什么时候读**：要改测试、要加用例、或想确认"这条断言为什么这么写"时。
> 复测命令在 §2，可直接粘贴。

---

## 1 · 三种运行档位

`pyproject.toml` 里写死了 `addopts = "-m 'not real and not live'"`，所以**默认只跑离线**。

| 档位 | marker | 前置 | 耗时 | 何时跑 |
|---|---|---|---|---|
| 离线（默认） | — | 无（零权重） | 秒级 | 每次改完代码 |
| 真实模型 | `real` | 见 §5 的模型文件 | 约半分钟 | 改动引擎/门面后 |
| 真实云 API | `live` | 有效凭证 + 网络 | 花钱 | 手动，改云适配器后 |

> ⚠️ **`live` marker 已在 `pyproject.toml` 声明，但当前没有任何用例使用它**。
> 云 API 目前只有离线用例（错误映射、切片逻辑）与手工验证，见 §6。

---

## 2 · 复测命令（可直接粘贴）

```bash
# 全部离线（日常）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests -q

# 真实模型（中/英/日转写、情绪、GPU 无 torch 前置、长音频时间轴）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests -q -m real

# 只跑某一个文件（改哪里跑哪里）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests/test_vad.py -q
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests/test_real.py -q -m real

# 只跑一条（调试时最常用）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests/test_guard.py::test_silence_returns_empty_without_calling_engine -q -v

# 看用例清单（不跑）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests --collect-only -q

# 收尾闸口：环境自检 + ruff + 全量离线测试（唯一闸口，日常不跑）
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe tools\check.py
```

---

## 3 · 用例分布

| 文件 | 覆盖什么 | 档位 |
|---|---|---|
| `conftest.py` | 合成素材夹具：`speech_like_wav`（9 s，中间有静音）、`silence_wav`、`long_wav`（75 s）、`module`（mock 引擎门面）、`vad_model` | — |
| `test_audio.py` | 解码归一（任意容器 → **16 kHz 单声道 float32**）、字节流解码、切片保留时间轴、重采样时长不变 | 离线 |
| `test_vad.py` | 能量 VAD 回退路径、区间合并、空输入；**Silero 分块喂入的正确性**（`issues/003` 回归） | 离线 + `real` |
| `test_segment.py` | 分段覆盖全片、时间轴单调、`max_seconds` 约束、连续说话标 `hard_cut`、**`split_gap` 在明显停顿处必切**、重叠时间轴被检出 | 离线 |
| `test_subtitle.py` | SRT/VTT/ASS 格式与时间码、三级断句、**时间轴越界报错**、ASS 卡拉 OK `\k`、折行 | 离线 |
| `test_guard.py` | **幻觉闸门**：静音不进引擎、分段级跳过、压缩比标记、空音频（`issues/002` 回归） | 离线 |
| `test_clip.py` | 参考音频时长规格、产物可解码、分数降序、**`ref_text` 必须来自该片段**、无语音返回空、片段字幕从 0 起算 | 离线 |
| `test_module.py` | 配置加载与 **YAML 布尔陷阱**、注册表、**能力查询不加载权重**、mock 全链路、引擎清单、`doctor`、SRT 落盘、**能力不支持必须报错**、`_shift`/`_clamp`/`_merge` 三个时间轴函数 | 离线 |
| `test_server_cli.py` | CLI 全部子命令、HTTP 全部端点、上传后缀校验、**下载端点不越权** | 离线 |
| `test_gui.py` | **NiceGUI 3.x 上传契约**、文件名防目录穿越、框架字段形状（`issues/006` 回归） | 离线 |
| `test_real.py` | 中/英/日三语关键词、词级时间戳单调、情绪 7 类、情绪补齐链路、**不预先 import torch 也能 GPU 推理**（`issues/004` 回归）、静音不幻觉、长音频时间轴 | `real` |

**离线套件不加载任何权重**：`conftest.py` 的素材全是合成的，`module` 夹具用 `mock` 引擎。
这也是验收项 V1（全链路不加载权重可跑通）的实现方式。

---

## 4 · 几条刻意写死的断言

这些阈值是**故意不给余量**的：一旦有人改回旧的错误实现，它们会立刻变红。

| 用例 | 写死的断言 | 为什么 |
|---|---|---|
| `test_vad.py::test_silero_detects_speech_in_clean_clip` | `speech_ratio > 0.5` | 曾经的 bug 是「整段一次性喂 VAD」，占比会掉到 **0.06** 左右。0.5 这条线让错误实现无处可躲（`issues/003`） |
| `test_vad.py::test_silero_long_audio_does_not_drop_intervals` | `intervals[0][0] < 5.0` | 第二个变体 bug：喂满环形缓冲才取结果，**只剩尾部**。断言首段必须从开头附近开始 |
| `test_clip.py::test_ref_text_matches_the_clip_window` | `c.text != transcript.text` | `ref_text` 与音频错位会让下游 TTS 音色克隆学错对应关系，是**静默失效**。断言片段文本绝不能等于整段文本 |
| `test_guard.py::test_silence_returns_empty_without_calling_engine` | 静音 → `text == ""` 且 `suspicious is True` | 实测 Whisper 对无语音输入会编出「优优独播剧场」且置信度 `p=1.000`。用「一被调用就 raise 的假引擎」验证闸门真的在前置拦截（`issues/002`） |
| `test_gui.py::test_upload_event_arguments_shape` | `"name" not in fields` / `"content" not in fields` | NiceGUI 2.x→3.x 的破坏性变更。这条把**框架契约本身**钉住：将来再变 API 会先在测试里撞红，而不是等用户上传时才炸（`issues/006`） |
| `test_module.py::test_capability_error_is_raised_not_silently_ignored` | 请求词级能力必须抛 `CapabilityError` | 「能力不支持就报错、绝不静默降级」是引擎契约的底线 |

---

## 5 · 真实模型用例的前置

`-m real` 需要这些文件（缺了会 **skip 而不是 fail**，所以看到 skip 不是 bug）：

| 路径 | 谁用 |
|---|---|
| `models/faster-whisper-large-v3-turbo/` | `test_real.py` 的转写用例 |
| `models/sherpa-sensevoice/model.int8.onnx` + `tokens.txt` | 情绪用例 |
| `models/SenseVoiceSmall/example/{zh,en,ja,ko}.mp3` | 三语断言与长音频拼接的**测试素材**（注意：这个仓库的 `model.pt` 引擎不用） |
| `models/sherpa-vad/silero_vad.onnx` | `test_vad.py` 的两条 `real` 用例 |

下载方式见 `05_deployment.md §3`。

一条特殊的：`test_real.py::test_gpu_inference_without_torch_preimport` 会在**独立子进程**里
先断言 `'torch' not in sys.modules`，再跑一次 GPU 转写 —— 它复现的正是
「另一个脚本能用、这个不能用」那个隐性问题（`issues/004`）。

---

## 6 · 没有自动化覆盖的部分

诚实列出，避免误以为"测试全绿 = 什么都验过了"：

| 未覆盖 | 现在的做法 |
|---|---|
| GUI 的真实浏览器交互 | `tests/test_gui.py` 只覆盖上传落盘与框架契约；界面走手工清单（`07_gui_guide.md §5`） |
| 云 API 的**真实**调用 | 无 `live` 用例；靠手工跑一次真音频验证（`test_server_cli.py` 只测错误映射与参数构造） |
| 说话人分离（diarization） | 第一版只留了能力位，**没有实现**（`01_design.md §10.4`） |
| 降噪 / 去混响 | 只留配置位，**没有实现**（`01_design.md §10.5`） |
| 分离模组的功能 | 在**它自己的仓库**里，不在本仓库 |
| 语音质量主观听感 | 需要人工听审，无法自动化 |

---

## 7 · 用例与验收标准的对应

验收标准 V1–V12 的定义见 `01_design.md §9`。对应关系：

| 验收 | 落点 |
|---|---|
| V1 不加载权重跑通 | `test_module.py::test_mock_end_to_end` + 整个离线套件 |
| V2 三语转写 | `test_real.py::test_transcribes_three_languages` |
| V3 词级时间戳单调 | `test_real.py::test_word_timestamps_are_monotonic` |
| V4 SRT 可导入 | `test_subtitle.py`（格式与时间码）+ `test_module.py::test_transcribe_writes_srt` |
| V5 情绪不硬编 | `test_real.py::test_emotion_engine_returns_label` + `test_emotion_backfill_path` |
| V6 参考音频规格 | `test_clip.py` 六条 |
| V7 云 API 可切换 | **仅手工验证**（无 `live` 用例） |
| V8 长音频时间轴 | `test_real.py::test_long_audio_timeline` |
| V9 分离等长 | 在**分离模组自己仓库**的 `tests/test_real.py` |
| V10 无语音不产生字幕 | `test_guard.py` 三条 + `test_real.py::test_silence_does_not_hallucinate` |
| V11 分段行为 | `test_segment.py` 八条 |
| V12 主体人声去除有效 | 在**分离模组自己仓库**的 `tests/test_real.py` |
