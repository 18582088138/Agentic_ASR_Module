# 02 · 开发计划 / Development plan

> **这份文档是回顾式的**：项目已开发完成，这里记录「当初怎么规划的」与
> 「实际怎么做的、哪些地方偏离了」。开工前的规划版不再保留 —— 留两份只会互相矛盾。

**纪律**（当时定的，也确实照做了）：一次做一件事，做完自检一次，过了就往下走；
方案获批后开发 → 测试 → debug 连续推进，不再逐阶段请示。

---

## 1 · 一个重要的范围变更

开工前的计划里，本插件**自带**「人声/背景音分离」（原阶段 6）。开发中途用户提出
要把分离做成**独立的 Agent 插件**，于是：

- 分离能力**整体迁出**到 `Agentic_VoiceSeparate_Module`（独立仓库、独立服务端口 8302）；
- 本插件**不提供**「自动调用分离插件」的开关 —— 两个插件是独立服务，协作由**调用方**组合：
  ```python
  vocals = SeparateModule().extract_vocals("demo.mp4")
  result = ASRModule().transcribe(vocals.stems["vocals"].path)
  ```
  （开发中一度加过 `transcribe.separate` 配置项，但它从未被代码读取，
  属于会误导人的死配置，已删除。）
- 原计划里的 `demucs` 引擎**最终没有采用** —— 选型在那边改成了 sherpa-onnx 的 UVR，
  理由与实测数据记在 `00_research.md §4.2 / §4.4`。

---

## 2 · 阶段划分与实际完成情况

| 阶段 | 计划 | 实际 | 状态 |
|---|---|---|---|
| **0 · 工程骨架** | `git init`、`pyproject.toml`、`.gitignore`、`.env.example`、`tools/check.py`、`.dsh/skills/`、`PROJECT_MEMORY.md` | 全部就位 | ✅ |
| **1 · core + mock 引擎** | 类型/配置/注册表/异常/日志 + `engines/{base,mock}.py` | 另加了 `core/gpu.py`（显存预检必须用 `nvidia-smi`）与 `core/cuda.py`（见偏差 ③） | ✅ |
| **2 · audio + media** | 解码归一、`media/probe.py`、VAD、声学统计与幻觉闸门 | 全部就位；**VAD 踩了两个坑**（见偏差 ②） | ✅ |
| **3 · faster_whisper 引擎** | CT2 加载、词级时间戳、显存预检 | 就位；实测显存 +2.17 GB、长音频 RTF 0.037 | ✅ |
| **4 · subtitle** | 三级断句 + SRT/VTT/ASS + 时间轴校验 | 就位，另加词级卡拉 OK（ASS `\k`） | ✅ |
| **5 · clip 参考音频截取** | VAD 区间 + 滑窗 + 五维打分 + 词边界吸附 | 就位；**新增跨段滑窗**（素材里每句都短于 10 s 时也能凑出目标长度） | ✅ |
| ~~6 · separate~~ | ~~demucs 引擎 + 两轨导出~~ | **迁出到独立插件**（见 §1） | ➡️ 迁出 |
| **6 · sensevoice 情绪** | 先验证取情绪的轻量路径，再实现 `EMOTION` 与门面补齐 | **风险一次解决**：sherpa-onnx 的 `from_sense_voice()` 原生返回 `emotion`/`event`/`lang`，只加 2 个包且跑在 CPU | ✅ |
| **7 · minimax 云引擎** | `POST /v1/speech_to_text`、切片、时间轴平移、错误映射 | 就位；真实调用验证过（含发现日语漏字） | ✅ |
| **8 · openai_compat 云引擎** | 一个适配器覆盖 OpenAI/Groq/硅基流动/OpenRouter | 计划外新增（用户确认要） | ✅ |
| **9 · CLI** | `doctor/engines/probe/segment/transcribe/clip/serve/gui` | 就位 | ✅ |
| **10 · HTTP 服务** | FastAPI，路由与库一一对应 | 就位（8301） | ✅ |
| **11 · GUI** | NiceGUI，五个功能各一个面板 | 就位（8401）；**上传踩了 NiceGUI 3.x 的 API 变化**（见偏差 ④） | ✅ |
| **12 · 收尾** | 文档补齐、`tools/check.py` 全量 | 就位 | ✅ |

---

## 3 · 与计划的四处偏差（都是被实测逼出来的）

### ① 情绪引擎从「三条候选」直接落到第一条

原计划写了三条候选路径（sherpa-onnx → vendored 推理 → funasr），并按顺序试。
**第一条就成功了**：实测 `sherpa_onnx.OfflineRecognizer.from_sense_voice()` 的返回对象
原生带 `emotion` / `event` / `lang` 字段与字级时间戳，`pip install sherpa-onnx` 只新增
2 个包。于是后两条直接放弃，也**没有引入 funasr 的 18 个包**。

顺带白拿了语种识别与音频事件检测，而且它在 **CPU** 上跑，不与 faster-whisper 抢显存。

### ② VAD 有两个静默陷阱（计划里完全没料到）

计划里只把 VAD 当普通工具，实际上它踩了两次：

1. **Silero VAD 必须分块喂**（512 样本）—— 整段一次性 `accept_waveform` 会严重漏检：
   同一段 5.6 s 人声，整段喂只报 0.31 s，分块喂报 4.39 s；
2. **必须边喂边 `pop()`** —— 内部是环形缓冲，喂满后早先的语音段会被挤掉，
   表现为 60 s 素材只剩最后一个 4.5 s 的区间。

这两条会**静默**毁掉分段、幻觉闸门与参考音频截取三处（三个看起来毫不相关的症状），
已记 `issues/003`。

### ③ GPU 推理隐式依赖 torch 被导入

计划外的发现：`ctranslate2` 的 CUDA 后端需要 `cublas64_12.dll`，而 pip 装的 wheel
**不带**它 —— 本机上它来自 `torch/lib`（torch 的 `__init__` 会 `os.add_dll_directory`）。

于是同一个脚本「先 import torch 就能转写、不 import 就报 cublas 找不到」。
本项目**刻意不依赖 torch**，所以这条隐式依赖随时会断。修法是新增
`core/cuda.py`，在加载 GPU 模型前显式注册 CUDA 运行库目录，并让 `doctor` 能诊断它。
已记 `issues/004`。

### ④ NiceGUI 3.x 的上传事件结构变了

计划里没考虑框架版本差异。用户实测报
`'UploadEventArguments' object has no attribute 'name'` —— 代码是照 NiceGUI **2.x**
写的（`e.name` + `e.content`），而机器上是 **3.16**：事件对象只有 `file` 字段，
落盘要用 `await e.file.save(path)`。已记 `issues/006`，并加了把**框架契约本身钉住**的用例。

---

## 4 · 仍然有效的风险与预案

| 风险 | 预案 | 状态 |
|---|---|---|
| `av` 19 与 faster-whisper 不兼容 | 自控解码层，永不把文件路径交给引擎 | 已落实（`issues/001`） |
| 纯音乐/伴奏轨被编出假字幕 | 语音占比闸门 + 压缩比闸门 | 已落实（`issues/002`） |
| 显存读数不能用 `torch.cuda.*` | 一律走 `nvidia-smi`/NVML | 已落实 |
| 8 GB 卡上 TTS 与 ASR 抢显存 | `max_resident: 1` + 换引擎先卸再载；必要时 `--engine minimax` | 已落实 |
| 云 API 的 500 s / 25 MB 上限 | 切片 + 时间轴平移在门面统一实现，对所有引擎共享 | 已落实 |
| 长音频切片切在词中间 | 静音点优先 + `split_gap`；合并后做时间轴单调校验 | 已落实 |
| **Whisper 给出超出音频长度的尾段时间戳** | 每片输出 `_clamp` 回该片区间 | **计划外新增**（实测 123.55 s 素材被标到 131.51 s） |
| 日语质量无权威数字 | 真实素材人工听审 + 关键词断言 | 仍待你复验 |

---

## 5 · 测试策略（成本纪律）

```bash
# 默认：只跑离线，秒级，不加载任何权重
python -m pytest tests -q

# 真实模型（手动，每条几十秒起）
python -m pytest tests -q -m real

# 真实云 API（手动，会花钱）
python -m pytest tests -q -m live

# 收尾闸口：环境自检 + ruff + 全量测试
python tools/check.py
```

`pyproject.toml` 的 `addopts = "-m 'not real and not live'"`。
用例分布与刻意写死的断言见 `03_unit_tests.md`。
