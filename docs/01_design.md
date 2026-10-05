# 01 · 方案设计 / Design

本文是**开发前的契约**。改动核心决策要先改这里，再改代码。

---

## 1 · 定位 / One-liner

> **一个可独立部署与独立调用的 Agent 模组。**

| 模组 | 包 | 回答什么 | 端口 |
|---|---|---|---|
| **本模组** | `agentic_asr` | 「这段音频说了什么、怎么说的」 | 8301 |

**与声音分离模组的关系**：两者**能力正交** —— 一个答「说了什么」，
一个答「这段音频里哪些声音是哪个成分」。所以是两个**平级的独立模组**，
谁都不依赖谁；要一起用就由**应用层**组合，见 §4。

- 硬合并会让能力表无法表达（云 API 能做 ASR 但做不了分离，
  分离模型能分轨但不认识字）；
- 用户明确要求：**独立的 Agent 模组，可独立调用，每个功能都要能在
  service 与 GUI 中作为独立 function 调用**。

**不做**：声音分离（那是另一个模组）、翻译、配音合成、视频剪辑、音色克隆本身
（其余是调用方与 TTS 模组的事）。

### 1.1 ASR 插件的五个功能

| 功能 | 模块 | 一句话 |
|---|---|---|
| 音频信息提取 | `media/probe.py` | ffprobe + 声学统计 → `AudioInfo` |
| **音频分段** | `segment/splitter.py` | 静音点优先切分，超长回退固定时长；出时间轴清单 |
| 字幕生成 | `subtitle/` | `Transcript` → SRT / VTT / ASS，句级+词级 |
| 语音情绪识别 | `engines/sensevoice.py` | 逐段 7 类情绪 + 事件 + 全片汇总 |
| 参考音频截取 | `clip/picker.py` | 自动挑 10~15 s 干净人声 → wav + 字幕（喂 TTS 克隆） |

---

## 2 · 架构总览

```
┌─────────────── 调用方（Agent / 应用 / 人工）───────────────┐
│      HTTP (8301)   ·   CLI   ·   GUI   ·   Python 库      │
└──────────────────────────┬─────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │  agentic_asr    │   ← 本模组（分离是另一个模组，不在此图内）
                  │ ── 门面         │
                  │  解码归一        │
                  │  长音频切片+平移  │
                  │  引擎路由/降级    │
                  │  幻觉闸门        │
                  │ ── 引擎         │
                  │  faster_whisper │
                  │  sensevoice     │
                  │  minimax        │
                  │  openai_compat  │
                  │  mock           │
                  └─────────────────┘
                           │
                  ┌────────▼────────┐
                  │  core/          │  配置、类型、错误、日志
                  └─────────────────┘
```

**引擎接口一律要窄**：解码、切片、时间轴合并、字幕格式、轨道顺序判定这些
**与引擎无关**的事全部留在门面上。一旦下沉到各引擎，就会出现 N 份互相偷偷不一致的实现
——同一段音频换个引擎得到不同切法，调试时看不出是哪一份在起作用。

---

## 3 · ASR 插件设计

### 3.1 能力表

```python
class Capability(str, Enum):
    TIMESTAMPS      = "timestamps"        # 句级时间戳（最小契约）
    WORD_TIMESTAMPS = "word_timestamps"   # 词/字级
    LANGUAGE_ID     = "language_id"
    DIARIZATION     = "diarization"
    EMOTION         = "emotion"
    EVENTS          = "events"            # 音频事件（笑声/掌声/音乐…）
```

`declared_capabilities()` **必须不加载权重就能回答** —— GUI/HTTP 开机的引擎选择器
据此把不支持的开关置灰。不支持就**报错**（`CapabilityError`），绝不静默降级。

| 引擎 | 时间戳 | 词级 | 语种 | 说话人 | 情绪 | 事件 | 单次上限 | 设备 |
|---|---|---|---|---|---|---|---|---|
| `faster_whisper`（默认） | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 本地不限（门面切片） | **GPU 2.17 GB** |
| `sensevoice` | ✅ | ✅(字级) | ✅ | ❌ | **✅** | **✅** | 本地不限 | **CPU（不占显存）** |
| `minimax`（云，专用） | ✅ | ✅ | ✅ | **✅** | ❌ | ❌ | 500 s / 50 MB | — |
| `openai_compat`（云，通用） | 看厂商 | 看厂商 | ✅ | 看厂商 | ❌ | ❌ | 看厂商（OpenAI 25 MB） | — |
| `mock`（离线自测） | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | — | — |

**`sensevoice` 走 sherpa-onnx（CPU）而不是 funasr**，这是**实测结论**：
sherpa-onnx 的 `from_sense_voice()` 返回对象**原生带 `emotion` 与 `event` 字段**
（实测 `emotion: <|NEUTRAL|>`、`event: <|Speech|>`、`lang: <|zh|>`、字级 `timestamps`），
而 `pip install sherpa-onnx` **只新增 2 个包**；相比之下 `funasr` 要新增 **18 个包**。
而且还白拿了语种识别与事件检测，并跑在 CPU 上 —— 正好不与 faster-whisper 抢显存。

`openai_compat` **一个适配器覆盖多家**：填 `base_url` + `model` 即可用于
**OpenAI / Groq / 硅基流动 / OpenRouter**。能力**按配置声明**而非按厂商硬编码 ——
同一厂商内部差异就极大（OpenAI `whisper-1` 有词级时间戳、`gpt-4o-transcribe` 完全没有）。

### 3.2 核心数据类型（跨层契约）

```python
@dataclass
class Word:      text: str; start: float; end: float; confidence: float | None = None
@dataclass
class Segment:   id: int; start: float; end: float; text: str
                 speaker: str | None = None; emotion: str | None = None
                 words: list[Word] | None = None
@dataclass
class Transcript:
    text: str; language: str; duration: float; segments: list[Segment]
    engine: str; emotion_summary: str | None = None
    warnings: list[str] = field(default_factory=list)
    suspicious: bool = False          # 幻觉闸门命中
```

**句级是最小必需契约**，词级/说话人/情绪一律可选 —— 因为 `gpt-4o-transcribe`、
硅基流动、百炼 `qwen3-asr-flash` 都**没有时间戳**，把词级设成必需会让适配器大面积不可用。

### 3.3 音频信息提取

`media/probe.py` 两段：① **容器与流**（ffprobe JSON：容器格式、总时长、码率、
各流的编码/采样率/声道/位深、是否有视频轨）；② **声学统计**（解码后算：峰值 dBFS、
RMS dBFS、**估计 SNR**、VAD 语音占比、是否削波）。合起来返回 `AudioInfo`。

> **估计 SNR 是降噪开关的判据**，**语音占比是幻觉闸门的判据**（见 3.8）。

### 3.4 音频分段（新增）

**目的**：避免过长音频导致 OOM，并为下游 ASR 提供规整的输入单元。
用户已确认策略：**静音点优先，超长时回退固定时长**。

```
解码 → VAD 得语音区间 → 合并相邻区间（gap < merge_gap）
     → 在静音点切分，保证每段 ≤ max_seconds
     → 若某段仍超长（连续说话无停顿）→ 回退到固定时长硬切，并标记 hard_cut=True
     → 输出：段清单（JSON/SRT 时间轴）+ 可选导出音频片段
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_seconds` | 30 | 单段上限（也是送云 API 前的切片依据） |
| `min_seconds` | 0.5 | 太短的段并入相邻段 |
| `merge_gap` | 0.30 | 小于此间隔的语音区间合并，避免碎段 |
| `pad` | 0.10 | 段两侧留白，防止切在辅音上 |
| `export_audio` | false | 是否同时落盘片段 wav |

**输出必须带时间轴**：每段 `{index, start, end, duration, hard_cut}`，
这样下游合并转写结果时能平移回原始时间轴。门面统一负责这件事，对**所有引擎**生效
（云 API 有 500 s / 25 MB 上限，本地长音频也需要分块以免峰值显存失控）。

### 3.5 字幕生成

`subtitle/render.py`：`Transcript` → `srt` / `vtt` / `ass`。三条硬规则：

1. **不依赖厂商导出**。MiniMax 能直出 SRT，但 OpenAI 系不能；为统一**一律本地渲染**。
2. **断句在渲染前**（`subtitle/segment.py`），按优先级切：

   | 优先级 | 规则 | 默认阈值 |
   |---|---|---|
   | 1 | 强停顿（字间静音） | ≥ `pause_gap` 0.28 s |
   | 2 | 句末标点 `。！？…；` | — |
   | 3 | **硬**字数上限（无条件切） | ≥ `max_chars × hard_chars_ratio` = 32 |
   | 4 | 单条时长上限 | ≥ `max_seconds` 6 s |
   | 5 | 软字数上限 **且**落在次级标点上 | ≥ `max_chars` 20 |
   | 6 | 弱停顿（达 `min_chars` 才切） | ≥ `soft_gap` 0.12 s |
   | 7 | 次级标点 `，、,`（达 `min_chars` 才切） | — |

   **软硬两级是为了一件事**：中文没有词边界，一到字数就切必然落在某个词中间 ——
   实测切出过「函 / 数」「基 / 本」。所以到 `max_chars` 只表示「该切了」，
   还要继续往前走到下一个标点；只有超过 `max_chars × 1.6` 仍然没有标点
   （长串英文或数字）才无条件切。这条规则与 `Agentic_ASR_Module` 的字幕侧、
   DailyNewsAssistant 的字幕侧**共用同一套判据**。

   中日文按**字符宽度**算行长；英文按**词**折算字数（否则一行英文比一行中文长得多）。
   另有**数字保护**：`零点三|四四米` 这种切法会把数字读坏，禁止在数字内部切。

3. **时间轴单调校验**：切片合并后校验 `start` 严格递增、无负值、不超音频时长，
   越界即报错（这是最容易出的静默 bug）。

⚠️ **两个反直觉的坑（都实测踩过，见 `issues/007`）**：

- **不能直接拿 ASR 的字级时间戳当停顿依据**。实测 faster-whisper 给出的字间 gap
  **全是 0**，而且它还**把一段 11.68 s 的静音记在了单个「上」字身上** ——
  停顿规则因此完全失效，切出来的字幕依旧 14 秒一条。
  解法是**异常单元钳制**：把超长单元截到语速上限，多出来的时间就变成可见间隙。
- **要更碎调 `max_seconds` 而不是 `max_chars`**。对 faster-whisper 的输出，
  它已按句切分（每段约 13 字），`max_chars` 几乎不触发；真正控制字幕长度的是时长。

> 切分规则是**纯函数**（输入 `(start, end, text)` 序列，与音频、引擎无关），
> 所以 TTS 侧也能复用同一套规则 —— 见 `docs/01_design.md §10` 末尾的说明。

词级时间戳两种出口：ASS 卡拉 OK 标签（`\k`），或旁挂 `*.words.json`。

### 3.6 语音情绪识别

- 引擎层：`sensevoice` 声明 `EMOTION` + `EVENTS`，`transcribe()` 时顺带返回每段情绪。
  实测返回值形如 `<|NEUTRAL|>`，需**解析去掉 `<|` `|>` 标记**，并保留原始值以便回溯。
- 门面层：**若当前 ASR 引擎不支持情绪而用户要情绪**，门面按 segment 的音频区间
  调情绪引擎补齐（组合拳，用户不必关心谁是主力）。云 API 无情绪能力时会自动落到本地。
- 输出：每段 `Segment.emotion` + 全片 `Transcript.emotion_summary`（占比 + 主导情绪）。

⚠️ **情绪是 utterance 级、不是词级**，且对 <1 s 的短段不稳定 ——
门面对过短段返回 `emotion=None` 并在 `warnings` 说明，**不硬给假标签**。

### 3.7 参考音频截取（喂 TTS 音色克隆）

产出 `ref_audio` + `ref_text`，直接用于 Agentic_TTS_Module 的
`SynthRequest(ref_audio=..., ref_text=...)`。

```
解码 → VAD 得语音区间 → 在区间内滑窗枚举 10~15 s 候选 → 打分 → 取 top-N
     → 起止点吸附到词级时间戳边界 → 导出 wav + .txt(=ref_text) + .srt
```

打分项（权重进配置）：语音占比、时长贴合、估计 SNR、**说话人一致**、边界安全。
**默认输出 16 kHz 单声道**（TTS 侧要的就是这个）。
`ref_text` 必须来自**该片段自己的转写**，不能用整段音频的文本 —— 两者必须严格对应，
否则克隆会错位。

### 3.8 无语音防护（幻觉闸门）

**实测**：对纯背景音轨跑 Whisper，它不返回空，而是编出
「优优独播剧场——YoYoTelevisionSeriesExclusive」，且语言置信度 **`p=1.000`**。
⇒ 置信度**不能**用来判伪。门面自己设闸门（详见 `issues/002`）：

- **前置**：VAD 语音占比 < `transcribe.min_speech_ratio`（默认 0.02）→
  返回空 `Transcript` + warning，**不调用引擎**；
- **分段级**：切片语音占比为 0 → 跳过该片；
- **后置**：文本/时长压缩比异常 → 标 `suspicious=True`（不静默丢、不静默收）。

没有这道闸门，「分离出伴奏轨」这个动作会稳定地批量生产假字幕。

---

## 4 · 与声音分离模组的关系

本模组**不负责**声音分离，也**不调用**任何分离实现 —— 两个模组相互独立，
只有**应用层**可以同时使用它们（见 `~/.dsh/AGENTS.md` §七）。

**要分离时由应用层组合**，脚本放模组之外：

```
应用层脚本（不属于任何模组）
  ├─ 分离模组：提取人声 / 去主体人声
  └─ 本模组：对分离出来的轨做转写、字幕、情绪
```

**一个必须由应用层决定的取舍**：分离器质量不够时，**先分离再转写会让 WER 变差**——
有论文实测 SDR 7.97 dB 的分离器会让 WER 一致变差（见 `00_research.md §4.2`）。
所以本模组**不内建**「自动先分离」的开关：要不要分离、用哪个模型，
是调用方看着自己素材做的判断，不该由本模组替他默认做掉。

**关系一句话**：它回答「这段音频里哪些声音是哪个成分」，本模组回答「这些声音说了什么」。
两者的能力正交，所以是两个平级的模组，谁都不依赖谁。

> 分离模组的引擎选型、模型清单、实测数据与问题单，都在**它自己的仓库**里写权威。
> 本文不复制 —— 那些会随对方演进，抄过来必然过期，而且会让人在本仓库里找它们。

---

## 5 · 目录结构

```
agentic_asr/                    # ── 本模组 ──
  core/        config.py types.py registry.py errors.py logging.py
  engines/     base.py faster_whisper.py sensevoice.py minimax.py
               openai_compat.py mock.py
  media/       probe.py                   音频/视频信息提取
  audio/       io.py vad.py features.py   解码归一、VAD、声学统计
  segment/     splitter.py                音频分段
  subtitle/    render.py segment.py       字幕渲染与断句
  clip/        picker.py                  参考音频截取
  asr.py       门面 ASRModule
  server/ gui/ cli.py

configs/asr.yaml   .env.example
models/（gitignore）  outputs/（gitignore）
docs/  tests/  scripts/  tools/
```

单文件 ≤ 500 行；公共 API 与跨层接口**中英双语注释**，内部 helper 用中文。

**跨模组的脚本不在这里**：需要同时调用本模组与分离模组的脚本属于**应用层**，
放 `F:\2026年\Agentic_Pipelines\`（见 §4 与 `~/.dsh/AGENTS.md` §七）。

---

## 6 · 配置

两个包各有自己的配置文件，可独立部署、独立覆盖。

```yaml
# configs/asr.yaml
engine:
  default: faster_whisper        # faster_whisper | sensevoice | minimax | openai_compat | mock
  max_resident: 1
  faster_whisper:
    model_dir: models/faster-whisper-large-v3-turbo
    device: cuda
    compute_type: float16
    beam_size: 5
  sensevoice:
    model_dir: models/sherpa-sensevoice      # sherpa-onnx onnx（CPU）
    provider: cpu
  minimax:
    endpoint: https://api.minimaxi.com       # 【实测】本 key 只在国内端点有效
    model: asr-1.0
  openai_compat:
    base_url: ""                             # OpenAI / Groq / 硅基流动 / OpenRouter
    model: ""
    supports_word_timestamps: null           # 按厂商声明，不硬编码

transcribe:
  language: auto
  word_timestamps: true
  emotion: auto
  separate: false                # true = 先分离只对人声轨做 ASR（默认关闭，见 §10）
  min_speech_ratio: 0.02         # 幻觉闸门
  chunk_seconds: 480             # 云 API 500 s 上限，留余量

segment:                         # 音频分段
  max_seconds: 30
  min_seconds: 0.5
  merge_gap: 0.30
  pad: 0.10
  export_audio: false

subtitle:
  formats: [srt]
  max_line_chars: 20
  max_duration: 7.0
  max_chars: 42

clip:
  target_seconds: 12
  min_seconds: 10
  max_seconds: 15
  top: 3
  sample_rate: 16000
  weights: {speech_ratio: 0.35, duration: 0.25, snr: 0.25, boundary: 0.15}

preprocess:
  denoise: off                   # off | auto | on（第一版只留位，不实现）
  snr_threshold_db: 15
```

```yaml
# configs/asr.yaml（完整清单见该文件本身）
engine:
  default: faster_whisper        # faster_whisper | sensevoice | minimax | openai_compat | mock
```

> **分离模组的配置不在本仓库**：它的 `configs/*.yaml` 与 `.env` 项在**它自己的仓库**里
> 写权威。本模组只认上面这些配置项 —— 两者不共享配置文件，也不共享 `.env`。

`.env`：`MINIMAX_API_KEY`（**用户已提供**）、`ASR_ENGINE`、`ASR_DEVICE`、
`ASR_MODELS_DIR`、`ASR_OUTPUT_DIR`、`ASR_SERVER_PORT=8301`。凭证只从**本项目自己的**
`.env` 读，不跨目录读别的模组的 key；终端/日志/文档/截图里的密钥一律打码。

---

## 7 · 对外接口（库 = HTTP = CLI = GUI 一一对应）

```python
from agentic_asr import ASRModule
asr = ASRModule()
info    = asr.probe("demo.mp4")                       # ① 信息
pieces  = asr.segment("demo.mp4", max_seconds=30)     # ② 分段
r       = asr.transcribe("demo.mp4", words=True, emotion=True)
r.to_srt("outputs/demo.srt")                          # ③ 字幕
clip    = asr.clip_reference("demo.mp4", top=3)       # ④ 参考音频（喂 TTS）
asr.release()
```

**要连分离模组一起用，写在应用层脚本里**（不在本仓库，见 §4）：

| 功能 | 库 | CLI | HTTP |
|---|---|---|---|
| 体检 | `asr.doctor()` | `asr doctor` | `GET /healthz` |
| 引擎表 | `asr.engines()` | `asr engines` | `GET /engines` |
| ① 信息提取 | `probe()` | `asr probe <media>` | `POST /probe` |
| ② 音频分段 | `segment()` | `asr segment <media>` | `POST /segment` |
| ③ 字幕生成 | `transcribe()` / `to_srt()` | `asr transcribe <media> --srt out.srt` | `POST /transcribe` |
| ④ 情绪识别 | `transcribe(emotion=True)` | `--emotion` | 同上 |
| ⑤ 参考音频 | `clip_reference()` | `asr clip <media> --top 3` | `POST /clip` |

**每个功能都是独立 endpoint / 独立 GUI 面板**，可单独调用，不要求走完整流水线。
服务：`python -m agentic_asr.server`（8301）；GUI：`python -m agentic_asr.gui`。

---

## 8 · 资源策略（8 GB 卡）

实测常驻：faster-whisper turbo **2.17 GB**（GPU）、sherpa-onnx SenseVoice **0**（CPU）、
sherpa-onnx UVR **0**（CPU）。⇒ **GPU 只被 ASR 主力占 2.17 GB，其余全走 CPU。**

- `engine.max_resident: 1`：换引擎先卸再载 + `torch.cuda.empty_cache()`；
- 一次请求需要多引擎时**按引擎分组执行**，把轮动次数从「段数」压到「引擎数」；
- 加载前**预检显存**，不够就直接报错并指明该改哪一项；
- **显存必须用 `nvidia-smi`/NVML 读**，不能用 `torch.cuda.memory_allocated()` ——
  CTranslate2 不走 torch 的 CUDA 分配器，torch 会一直报 0（实测踩过）；
- 云 API 引擎不占显存，GPU 被 TTS 占用时可用 `--engine minimax` 兜底。

---

## 9 · 验收标准

| # | 验收 | 判定方式 |
|---|---|---|
| V1 | 全链路不加载权重可跑通 | `pytest -q` 离线用例全绿（`--engine mock`） |
| V2 | 中/英/日短音频转写正确 | `-m real`：三语断言关键词命中 |
| V3 | 词级时间戳可用且单调 | 断言非空、`start<end`、跨段递增 |
| V4 | SRT 可被外部工具导入 | 正则校验时间码格式与序号连续性 |
| V5 | 情绪能出且不硬编 | 断言 7 类之一；过短段为 `None`；`<|NEUTRAL|>` 被正确解析 |
| V6 | 参考音频符合规格 | 10 s ≤ 时长 ≤ 15 s；`ref_text` 与片段转写严格一致；wav 可解码 |
| V7 | 云 API 引擎可切换 | MiniMax 用例打 `live`；手动跑真音频 |
| V8 | 长音频切片后时间轴正确 | 185 s 素材断言末段 `end` ≤ 时长 |
| V9 | 分离输出与输入严格等长 | 断言样本数一致、轨道可解码、`stem_order_resolved_by` 非空 |
| V10 | **无语音输入不产生字幕** | 纯静音与纯伴奏返回空 `Transcript`；断言**不含「优优独播剧场」类幻觉** |
| V11 | **分段行为符合规格** | 断言段长 ≤ `max_seconds`、时间轴单调且覆盖全片、连续说话时标 `hard_cut` |
| V12 | **主体人声去除有效** | 用「主体+背景人声」素材断言：主体轨 ASR 命中主体文本、背景轨命中背景文本 |

V1–V6、V11 离线可验；V9/V12 用合成素材可离线验；V7/V8 需真实调用或真实权重，
打 `real` / `live` marker，在 `pyproject.toml` 的 `addopts` 里默认排除。

---

## 10 · 明确的取舍

1. **本地 ASR 主力是 faster-whisper 而非 Qwen3-ASR**：共享环境的 `transformers==4.57.3`
   被 TTS 钉死，`qwen-asr` 会新增 18 个包并把它拉到 4.57.6。代价是中文方言与
   长音频对齐精度不如 Qwen3-ASR。
2. **情绪只有 7 类且是句级**，没有词级情绪（开源侧天花板）。走 sherpa-onnx/CPU。
3. **分离引擎用 sherpa-onnx UVR 而非 demucs**：CPU、零新增依赖、不占显存、
   且能做 demucs 做不到的「主体/背景人声」区分。代价是 KARA 系训练域是音乐，
   影视对白的泛化性只做过 1 组合成素材验证，**必须用真实素材复验**。
4. **说话人分离（diarization）第一版只留能力位**：中文首选 3D-Speaker/CAM++ 是
   「VAD+嵌入+聚类」组合而非开箱 pipeline。云端的 MiniMax 自带 diarization 可先用。
5. **降噪/去混响只留配置位**，第一版不实现。
6. **不内建自动分离**：分离由**另一个模组**承担，协作由**应用层**组合
   （脚本放模组之外，见 §4）。有论文实测 SDR 7.97 dB 的分离器会让 WER 一致变差
   （见 `00_research.md §4.2`），所以也不该默认替用户做「先分离」这个决定。
7. **「主体人声去除」的预期要写清**：主体与背景**同时说话**的重叠段无法完美分离，
   交付说明里要写明「显著衰减主体人声」而非「彻底移除」。
