# 04 · 接口参考 / API reference

三个入口**一一对应**：库 / HTTP / CLI 里的每个功能都是同一个门面方法。
每个功能都能**单独调用**，不要求走完整流水线。

| 功能 | 库 | CLI | HTTP |
|---|---|---|---|
| 体检 | `doctor()` | `asr doctor` | `GET /doctor` |
| 引擎表 | `engines()` | `asr engines` | `GET /engines` |
| ① 音频信息提取 | `probe()` | `asr probe` | `POST /probe` |
| ② 音频分段 | `segment()` | `asr segment` | `POST /segment` |
| ③ 字幕生成 | `transcribe()` | `asr transcribe` | `POST /transcribe` |
| ④ 情绪识别 | `transcribe(emotion=True)` | `asr transcribe --emotion` | 同上 |
| ⑤ 参考音频截取 | `clip_reference()` | `asr clip` | `POST /clip` |

---

## 1 · 库 / Library

```python
from agentic_asr import ASRModule

asr = ASRModule()                       # 读 configs/asr.yaml，引擎用配置默认值
asr = ASRModule(engine="mock")          # 指定引擎
asr = ASRModule(config=my_config)       # 注入配置（测试用）
```

`media` 参数可以是 `str` / `Path`（任意 ffmpeg 能解的容器，含视频），
或已经是 `AudioChunk` 的内存音频。**引擎永远只收到 `AudioChunk`**。

支持上下文管理器：`with ASRModule() as asr: ...`，退出时自动 `release()`。

### ① `probe(media, deep=True, max_deep_seconds=300.0) -> AudioInfo`

音频/视频信息提取。

| 参数 | 默认 | 说明 |
|---|---|---|
| `deep` | `True` | 是否做声学统计（**要解码，慢**）；`False` 时只读容器元信息 |
| `max_deep_seconds` | `300.0` | 深度分析最多解码多少秒（长素材不必全解码才知道有没有削波） |

```python
info = asr.probe("demo.mp4")
print(info.duration, info.sample_rate, info.channels)
print(info.estimated_snr_db, info.speech_ratio, info.clipping)
```

`AudioInfo` 字段：`path` · `duration` · `container` · `bit_rate` · `sample_rate` ·
`channels` · `codec` · `bits_per_sample` · `has_video` · `video` · `peak_dbfs` ·
`rms_dbfs` · `estimated_snr_db` · `speech_ratio` · `clipping` · `warnings`。
`to_dict()` 可直接 `json.dumps`。

> **文件不存在会抛 `DecodeError`**（不返回空结果）—— 否则 CLI/HTTP 会给出
> 「成功」的退出码而结果里什么都没有。
> 但 ffprobe 解析失败只写进 `warnings` 并返回，因为那多半是容器格式问题，不是用户错误。

### ② `segment(media, max_seconds=None, export_audio=None, out_dir=None) -> list[SegmentPiece]`

音频分段。策略是**静音点优先，超长回退固定时长**。

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_seconds` | 配置 `segment.max_seconds`（30） | 单片上限 |
| `export_audio` | 配置 `segment.export_audio`（false） | 是否同时落盘每段 wav |
| `out_dir` | `outputs/segments` | 导出目录，文件名 `{stem}_{序号:04d}.wav` |

```python
for p in asr.segment("demo.mp4", max_seconds=30):
    print(p.index, p.start, p.end, p.duration, p.hard_cut)
```

`SegmentPiece`：`index` · `start` · `end` · `duration` · `hard_cut`
（`hard_cut=True` 表示该处没有静音点、只能按固定时长硬切）。

**返回的段覆盖 `[0, duration]` 全片**（首尾静音也会补成段），所以时间轴可以和原音频对齐。
静音段不会浪费 ASR 调用 —— 门面的分段级闸门会跳过它们。

### ③④ `transcribe(media, *, engine=None, language=None, words=None, emotion=None, srt=None, srt_format="srt") -> Transcript`

转写，可选情绪与字幕落盘。

| 参数 | 默认 | 说明 |
|---|---|---|
| `engine` | 配置 `engine.default`（`faster_whisper`） | 引擎名 |
| `language` | 配置（`auto`） | `auto` 或 `zh` / `en` / `ja` … |
| `words` | 配置（`true`） | 要不要词级时间戳；引擎不支持会抛 `CapabilityError` |
| `emotion` | 配置（`auto` = 能出就出） | 要情绪时，主力引擎不支持会**自动按段调情绪引擎补齐** |
| `srt` | `None` | 给定路径则顺带写字幕 |
| `srt_format` | `"srt"` | `srt` / `vtt` / `ass` |

```python
r = asr.transcribe("demo.mp4", emotion=True, srt="outputs/demo.srt")
print(r.text, r.language, r.duration, len(r.segments))
print(r.emotion_summary)          # 例如 "neutral (9/9)"
print(r.warnings, r.suspicious)   # 幻觉闸门会在这里说话
```

`Transcript`：`text` · `language` · `duration` · `segments` · `engine` ·
`emotion_summary` · `events` · `warnings` · `suspicious` · `raw`；`is_empty` 属性、
`to_dict(words=True)` 方法。

`Segment`：`id` · `start` · `end` · `text` · `speaker` · `emotion` · `words` · `duration` 属性。
`Word`：`text` · `start` · `end` · `confidence`。

**两条要留意的行为**：

- **静音/纯音乐输入会返回空 `Transcript`**（`text=""`、`suspicious=True`、`warnings` 非空），
  而且**不会调用引擎** —— 这是实测出来的必要防护，Whisper 对无语音输入会编内容（`issues/002`）。
- **`suspicious=True` 只标记不丢弃**：压缩比异常（文本长度/音频时长）时置位，
  丢不丢由调用方决定。

### ⑤ `clip_reference(media, transcript=None, top=None, out_dir=None, stem="ref", engine=None) -> list[ClipResult]`

参考音频截取（10~15 s 干净人声 + 对应文本），直接用于 TTS 音色克隆。

| 参数 | 默认 | 说明 |
|---|---|---|
| `transcript` | `None` | **不给就先转写一次** —— 因为 `ref_text` 必须来自该片段自身 |
| `top` | 配置 `clip.top`（3） | 取前 N 个候选 |
| `out_dir` | `outputs/refs` | 产物目录，文件名 `{stem}_{序号:02d}.wav/.txt/.srt` |
| `stem` | `"ref"` | 文件名前缀 |

```python
clips = asr.clip_reference("demo.mp4", top=3)
for c in clips:
    print(c.wav_path, c.text, c.score, c.score_detail)
    # 直接喂 Agentic_TTS_Module：
    #   SynthRequest(ref_audio=c.wav_path, ref_text=c.text)
```

`ClipResult`：`index` · `wav_path` · `start` · `end` · `duration` · `text` ·
`score` · `score_detail`（五维：`speech_ratio` / `duration` / `snr` / `boundary` / `score`
+ `snr_db`）· `srt_path` · `txt_path`。

默认输出 **16 kHz 单声道** wav（TTS 侧要的形状）。片段自己的 `.srt` 时间轴**从 0 起算**。

### 管理方法

| 方法 | 说明 |
|---|---|
| `engines()` | 引擎清单与能力。**不加载权重**，可安全用于开机检查 |
| `doctor()` | 环境体检：引擎、CUDA、显存、模型文件、`checks` 列表、`ok` |
| `release()` | 卸载全部引擎、清显存 |

---

## 2 · HTTP

```bash
python -m agentic_asr.server            # http://127.0.0.1:8301/docs
asr serve --host 0.0.0.0 --port 8301
```

### GET 端点

| 路径 | 返回 |
|---|---|
| `/healthz` | `{ok, version, engine}` |
| `/engines` | `{default, engines:[{name, capabilities, default}]}` |
| `/doctor` | 与库的 `doctor()` 同结构 |
| `/config` | 当前生效配置（JSON） |
| `/download?path=` | 下载**产物**；路径必须落在 `outputs/` 内，否则 403 |

### POST 端点

都是 `multipart/form-data`，字段：

| 路径 | 表单字段 |
|---|---|
| `/probe` | `file` · `deep`(bool, 默认 true) |
| `/segment` | `file` · `max_seconds`(float) · `export_audio`(bool) |
| `/transcribe` | `file` · `engine` · `language` · `words`(bool) · `emotion`(bool) · `subtitle_format`(`srt`/`vtt`/`ass`) |
| `/clip` | `file` · `top`(int) · `engine` |

```bash
# 转写并直接拿回字幕内容
curl -F "file=@demo.mp4" -F "subtitle_format=srt" -F "emotion=true" \
     http://127.0.0.1:8301/transcribe

# 分段
curl -F "file=@demo.mp4" -F "max_seconds=30" http://127.0.0.1:8301/segment

# 参考音频
curl -F "file=@demo.mp4" -F "top=3" http://127.0.0.1:8301/clip
```

`/transcribe` 带 `subtitle_format` 时，响应里会多出 `subtitle_path` 与 `subtitle`（字幕全文）。

> **`/transcribe` 没有「给字幕加说话人前缀」的参数**：说话人分离第一版只留了能力位
> （`docs/01_design.md §10.4`）。只有 MiniMax 引擎会返回 `speaker` 字段，它以 JSON
> 透出、不参与字幕渲染。放一个不生效的参数会误导调用方，所以没有。

### 失败响应

| 状态 | 场景 |
|---|---|
| 415 | 扩展名不在白名单（`wav/mp3/m4a/aac/flac/ogg/opus/aiff/mp4/mkv/mov/webm/avi/ts`） |
| 400 | 分离/转写失败（如缺少凭证、文件读不了） |
| 403 | `/download` 的路径落在 `outputs/` 之外 |
| 404 | `/download` 的文件不存在 |
| 500 | 字幕已声明生成但没落地（不该发生，属于内部错误兜底） |

---

## 3 · CLI

```bash
asr --help          # 或 python -m agentic_asr --help
```

| 命令 | 参数 | 说明 |
|---|---|---|
| `asr version` | — | 打印版本 |
| `asr doctor` | `--config` | 环境与模型体检（不加载权重） |
| `asr engines` | — | 引擎与能力表 |
| `asr probe <media>` | `--deep/--no-deep` · `--config` | ① 信息提取（输出 JSON） |
| `asr segment <media>` | `--max-seconds` · `--export` · `--out-dir` · `--config` | ② 分段 |
| `asr transcribe <media>` | `--engine` · `--language` · `--words/--no-words` · `--emotion/--no-emotion` · `--srt PATH` · `--format srt\|vtt\|ass` · `--karaoke` · `--config` | ③④ 字幕 + 情绪 |
| `asr clip <media>` | `--top` · `--out-dir` · `--engine` · `--config` | ⑤ 参考音频 |
| `asr serve` | `--host` · `--port` · `--reload` · `--config` | 起 HTTP 服务 |
| `asr gui` | `--host` · `--port` · `--config` | 起图形界面（默认端口 = 服务端口 + 100） |

```bash
# 不加载权重先试链路（零依赖 mock 引擎，确定性输出）
asr transcribe demo.mp3 --engine mock

# 真实转写 + 字幕 + 情绪
asr transcribe demo.mp4 --emotion --srt outputs/demo.srt

# 卡拉 OK 字幕（ASS 词级 \k 标签）
asr transcribe demo.mp4 --format ass --karaoke --srt outputs/demo.ass

# 分段并导出片段
asr segment demo.mp4 --max-seconds 20 --export --out-dir outputs/pieces

# 体检
asr doctor
```

命令失败时以非零码退出（例如 `asr probe 不存在的文件` → 退出码 1）。

---

## 4 · 配置速查

完整项见 `configs/asr.yaml`，可被 `.env`（`ASR_*`）与进程环境变量覆盖，优先级
**环境变量 > `.env` > YAML > 内置默认**。

最常改的几项：

| 配置 | 默认 | 什么时候改 |
|---|---|---|
| `engine.default` | `faster_whisper` | 想默认走云端或 CPU 时 |
| `engine.faster_whisper.device` | `cuda` | 显存被占用时改 `cpu` |
| `transcribe.word_timestamps` | `true` | 只要句级字幕时关掉可省时间 |
| `transcribe.min_speech_ratio` | `0.02` | 幻觉闸门阈值 |
| `segment.max_seconds` | `30` | 想要更细的段就调小 |
| `segment.split_gap` | `1.00` | 停顿超过它就一定切段（「静音点优先」） |
| `clip.target_seconds` | `12.0` | 参考音频的目标时长 |
| `clip.weights` | 见文件 | 五维打分的权重 |

`.env` 里只有凭证与运行时覆盖项：`MINIMAX_API_KEY`、`OPENAI_COMPAT_*`、
`ASR_ENGINE`、`ASR_DEVICE`、`ASR_MODELS_DIR`、`ASR_OUTPUT_DIR`、`ASR_SERVER_HOST/PORT`。
**凭证只从本项目自己的 `.env` 读**，不跨目录去读 Agentic_TTS_Module 的 key。
