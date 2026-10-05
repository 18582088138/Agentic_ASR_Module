# 01 · 方案设计 / Design

本文是**开发前的契约**。改动核心决策要先改这里，再改代码。

---

## 1 · 定位 / One-liner

> **两个平级的、可独立部署与独立调用的 Agent 插件。**
> Two sibling, independently deployable and callable agent plugins.

| 插件 | 包 | 回答什么 | 端口 |
|---|---|---|---|
| **ASR 插件** | `agentic_asr` | 「这段音频说了什么、怎么说的」 | 8301 |
| **分离插件** | `agentic_separate` | 「这段音频里哪些声音是哪个成分」 | 8302 |

两者**能力正交**，因此拆成两套平级抽象，而不是塞进一个模块：

- 一个答「说了什么」，一个答「哪些是人声」；
- 硬合并会让能力表无法表达（云 API 能做 ASR 但做不了分离，
  分离模型能分轨但不认识字）；
- 用户明确要求：**两个独立的 Agent 插件，可独立调用，每个功能都要能在
  service 与 GUI 中作为独立 function 调用**。

**不做**：翻译、配音合成、视频剪辑、音色克隆本身（那是调用方与 Agentic_TTS_Module 的事）。

### 1.1 ASR 插件的五个功能

| 功能 | 模块 | 一句话 |
|---|---|---|
| 音频信息提取 | `media/probe.py` | ffprobe + 声学统计 → `AudioInfo` |
| **音频分段** | `segment/splitter.py` | 静音点优先切分，超长回退固定时长；出时间轴清单 |
| 字幕生成 | `subtitle/` | `Transcript` → SRT / VTT / ASS，句级+词级 |
| 语音情绪识别 | `engines/sensevoice.py` | 逐段 7 类情绪 + 事件 + 全片汇总 |
| 参考音频截取 | `clip/picker.py` | 自动挑 10~15 s 干净人声 → wav + 字幕（喂 TTS 克隆） |

### 1.2 分离插件的三个功能

| 功能 | 用什么模型 | 输出 |
|---|---|---|
| 人声提取 | UVR 通用/Voc 系 | 人声轨（含所有人声） |
| 背景音提取 | UVR 通用/Inst 系 | 伴奏轨（去掉所有人声） |
| **主体人声去除** | **UVR KARA 系** | **去掉主体人声、保留背景人声**的轨 |

---

## 2 · 架构总览

```
┌─────────────── 调用方（Agent / 应用 / 人工）───────────────┐
│   HTTP (8301 / 8302)   ·   CLI   ·   GUI   ·   Python 库   │
└──────────────────────────┬─────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                   ┌────────▼────────┐
│ agentic_asr    │                   │ agentic_separate│
│ ── 门面        │                   │ ── 门面          │
│  解码归一       │                   │  wav 44.1k 归一  │
│  长音频切片+平移 │                   │  轨道顺序判定     │
│  引擎路由/降级   │                   │  等长校验        │
│  幻觉闸门       │                   │                  │
│ ── 引擎        │                   │ ── 引擎          │
│  faster_whisper│                   │  uvr (sherpa)   │
│  sensevoice    │                   │  null           │
│  minimax       │                   │                  │
│  openai_compat │                   │                  │
│  mock          │                   │                  │
└────────────────┘                   └──────────────────┘
        └──────────── 共享：core/（配置、类型、错误、日志）────────────┘
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
   | 3 | 加权字数上限 | ≥ `max_chars` 20 |
   | 4 | 单条时长上限 | ≥ `max_seconds` 6 s |
   | 5 | 弱停顿（达 `min_chars` 才切） | ≥ `soft_gap` 0.12 s |
   | 6 | 次级标点 `，、,`（达 `min_chars` 才切） | — |

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

## 4 · 分离插件设计

### 4.1 引擎抽象

```python
class SeparationEngine(ABC):
    name: str
    @classmethod
    def declared_stems(cls) -> set[str]: ...      # 该引擎能产出的轨
    def supports_mode(self, mode: SeparationMode) -> bool: ...
    @abstractmethod
    def separate(self, chunk: AudioChunk) -> dict[str, np.ndarray]: ...
```

| 引擎 | 后端 | 模型 | 设备 | 实测 RTF | 显存 | 许可 |
|---|---|---|---|---|---|---|
| `uvr`（默认） | sherpa-onnx `OfflineSourceSeparation` | UVR MDX 系 onnx（28–67 MB） | **CPU** | **0.232–0.363**（KARA） | **0** | MIT |
| `null`（占位） | — | — | — | 即时 | 0 | — |

**为什么不用 demucs**（已实测并弃用）：demucs 只能做人声/伴奏二分，
**无法区分主体人声与背景人声** —— 实测「中文主体 + 英文背景」素材时，
它把两种人声全塞进 `vocals`，`no_vocals` 只剩 rms 0.0052 的近静音。
而 sherpa-onnx + UVR 在 **CPU 上比 demucs 在 GPU 上还快**（0.241 vs 0.740）、
**零新增依赖**、**不占显存**（8 GB 卡整块留给 ASR）。
（demucs 的实测数据保留在 `00_research.md §4.2` 作为对照。）

**模型清单**（`models/sherpa-separation/`，gitignore；下载见 `05_deployment.md`）：

| 模型 | 体积 | 用途 |
|---|---|---|
| `UVR_MDXNET_KARA.onnx` | 29.7 MB | **主体人声去除**（默认） |
| `UVR_MDXNET_KARA_2.onnx` | 52.8 MB | 备选 |
| `UVR_MDXNET_Main.onnx` | 66.8 MB | 人声/背景音提取（通用） |
| `UVR-MDX-NET-Voc_FT.onnx` | 66.8 MB | 人声提取（人声专用） |
| `UVR-MDX-NET-Inst_Main.onnx` | 52.8 MB | 背景音提取（伴奏专用） |

### 4.2 人声提取 / 4.3 背景音提取

用通用/Voc/Inst 模型，输出人声轨与伴奏轨。
**实测（素材=人声+背景音乐）**：stem0 = 人声（rms 0.12，ASR 识别正确）、
stem1 = 音乐（rms 0.15，ASR 输出 `/Thank you./` 幻觉 = 无语音）。
⇒ 这两条路径在**正确素材（人声+音乐）**下工作正常。

⚠️ **输入分布边界**：这类模型是为「音乐 + 人声」训练的。喂「人声 + 人声」这种
分布外输入时表现不稳 —— 低能量的背景人声会被当残差丢掉（实测英文碎成片段）。
所以「主体 vs 背景人声」必须走 KARA，不能指望通用模型。

### 4.4 主体人声去除（对标用户的「只去主体人声、保留背景人声」）

**实测证明可行**：构造「中文主体（居中）+ 英文背景（偏侧、−10 dB）」素材，
用 ASR 判读各轨：

| 轨 | rms | ASR 判读 |
|---|---|---|
| 原始混音 | 0.1434 | 中文（英文被掩盖） |
| demucs `vocals` | 0.1428 | 中文 |
| demucs `no_vocals` | **0.0052** | `pause` → 近静音 |
| **kara1 `stem0`** | 0.1083 | **中文（主体）** |
| **kara1 `stem1`** | 0.0651 | **英文（背景）** |
| kara2 `stem0` | 0.0417 | 英文（背景） |
| kara2 `stem1` | 0.1336 | 中文（主体） |

**KARA 系确实能把主体与背景人声分到两条轨，即使在说话而非唱歌场景。**
取「背景」那一轨即为本功能的输出（伴奏 + 背景人声，主体人声已去除）。

> 关于用户设想的「提取人声 → 滤波去背景人声 → 相减」链路：
> **目标可达，但三步手段都不必用**。
> ①「滤波」不可行 —— 主体与背景人声频谱完全重叠（都是人声，基频 80–400 Hz、
> 共振峰 500 Hz–4 kHz），线性滤波无法按「谁在说话」选择；
> ②「相减」不可靠 —— 实测 `vocals + no_vocals` 与原音频的相对 RMS 误差达
> **8.06%（rescale）/ 10.63%（none）**，误差来自模型估计本身；
> ③ KARA 模型直接输出两条轨，**一步到位，不需要相减**。

### 4.5 轨道顺序判定（必须做，不能硬编码）

官方示例写死 `stems[0]=vocals`，但实测 **kara2 的 stem0 是背景**（与官方相反）。
我一度用「能量更高的是主体轨」判定，**已被自己的复测推翻**：
素材为「人声+音乐」时音乐轨 rms（0.1533）反而高于人声轨（0.1114）。

⇒ 采用**三级判定**，从可靠到兜底：

1. **模型约定表**（快，静态）：如 `UVR_MDXNET_KARA` → stem0 = 主体；
2. **VAD 语音占比**（可靠，默认启用）：哪一轨检出语音，哪一轨是主体人声轨；
3. **配置显式覆盖**（`separate.lead_stem_index`）：用户可强制指定。

判定结果写进返回结构（`SeparationResult.stem_order_resolved_by`），
便于事后排查，不静默。

### 4.6 等长与采样率（硬约束）

- **输入**：sherpa-onnx UVR **只接受 wav 44.1 kHz 立体声**
  （官方：`ffmpeg -i in.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav`）。
  门面负责把任意输入归一到这个格式。
- **输出**：实测各轨与输入**严格等长**（`shape[1]` 完全一致）。
  门面仍**强制校验样本数**，不等长就裁剪/补零并写 `warnings` ——
  时间轴漂移是最隐蔽的 bug。
- **不做逐轨峰值归一化**：会破坏轨间相对音量；要归一化由调用方显式开。

---

## 5 · 目录结构

```
agentic_asr/                    # ── ASR 插件 ──
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

agentic_separate/               # ── 分离插件 ──
  core/        （复用 agentic_asr.core 的类型与配置基类）
  engines/     base.py uvr.py null.py
  separate.py  门面 SeparateModule
  server/ gui/ cli.py

configs/asr.yaml   configs/separate.yaml   .env.example
models/（gitignore）  outputs/（gitignore）
docs/  tests/  scripts/  third_party/
```

单文件 ≤ 500 行；公共 API 与跨层接口**中英双语注释**，内部 helper 用中文。

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
# configs/separate.yaml
engine:
  default: uvr                   # uvr | null
  uvr:
    model_dir: models/sherpa-separation
    num_threads: 4
    provider: cpu
    lead_stem_index: null        # null = 自动判定（见 §4.5）

separate:
  mode: lead_removal             # lead_removal | vocals | accompaniment | both
  models:
    lead_removal: UVR_MDXNET_KARA.onnx
    vocals: UVR-MDX-NET-Voc_FT.onnx
    accompaniment: UVR-MDX-NET-Inst_Main.onnx
  output_sample_rate: 44100
  normalize: false               # 不做逐轨峰值归一化
```

`.env`：`MINIMAX_API_KEY`（**用户已提供**）、`ASR_API_ENDPOINT`、`ASR_ENGINE`、
`ASR_DEVICE`、`ASR_MODELS_DIR`、`ASR_OUTPUT_DIR`、`ASR_SERVER_PORT=8301`、
`SEPARATE_SERVER_PORT=8302`。凭证只从**本项目自己的** `.env` 读，不跨目录读 TTS 的 key；
终端/日志/文档/截图里的密钥一律打码。

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

from agentic_separate import SeparateModule
sep = SeparateModule()
v  = sep.extract_vocals("demo.mp4")                   # 人声提取
bg = sep.extract_accompaniment("demo.mp4")            # 背景音提取
lead_removed = sep.remove_lead_vocal("demo.mp4")      # 主体人声去除
sep.release()
```

| 功能 | 库 | CLI | HTTP |
|---|---|---|---|
| ASR 体检 | `asr.doctor()` | `asr doctor` | `GET /healthz` |
| 引擎表 | `asr.engines()` | `asr engines` | `GET /engines` |
| ① 信息提取 | `probe()` | `asr probe <media>` | `POST /probe` |
| ② 音频分段 | `segment()` | `asr segment <media>` | `POST /segment` |
| ③ 字幕生成 | `transcribe()` / `to_srt()` | `asr transcribe <media> --srt out.srt` | `POST /transcribe` |
| ④ 情绪识别 | `transcribe(emotion=True)` | `--emotion` | 同上 |
| ⑤ 参考音频 | `clip_reference()` | `asr clip <media> --top 3` | `POST /clip` |
| 分离·人声 | `extract_vocals()` | `separate vocals <media>` | `POST /vocals` |
| 分离·背景音 | `extract_accompaniment()` | `separate accompaniment <media>` | `POST /accompaniment` |
| 分离·去主体 | `remove_lead_vocal()` | `separate lead-removal <media>` | `POST /lead-removal` |

**每个功能都是独立 endpoint / 独立 GUI 面板**，可单独调用，不要求走完整流水线。
服务：`python -m agentic_asr.server`（8301）、`python -m agentic_separate.server`（8302）；
GUI：`python -m agentic_asr.gui`、`python -m agentic_separate.gui`。

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
6. **不内建自动分离**：分离由独立插件 `Agentic_VoiceSeparate_Module` 承担，
   协作由**调用方**组合（先 `extract_vocals()`，再把 vocals 轨喂进 `transcribe()`）。
   有论文实测 SDR 7.97 dB 的分离器会让 WER 一致变差（见 `00_research.md §4.2`），
   所以也不该默认替用户做「先分离」这个决定。
7. **「主体人声去除」的预期要写清**：主体与背景**同时说话**的重叠段无法完美分离，
   交付说明里要写明「显著衰减主体人声」而非「彻底移除」。
