# 07 · GUI 使用说明 / GUI guide

> **什么时候读**：要用图形界面而不是命令行时；或改动 `agentic_asr/gui/app.py` 之前。

---

## 1 · 起法

```bash
cd F:\2026年\Agentic_ASR_Module
python -m agentic_asr.gui          # 等价于 asr gui
```

- 默认地址 **http://127.0.0.1:8401**（= `configs/asr.yaml` 的 `server.port` + 100）
- 启动后会自动打开浏览器（`ui.run(show=True)`）
- 换端口：`asr gui --port 9401`；换配置：`asr gui --config 某个.yaml`
- **模型是懒加载的，所以界面秒开**；第一次点按钮时才会载权重（几秒）

---

## 2 · 界面结构

```
┌─ 顶栏 ────────────────────────────────────────────────────────┐
│ Agentic ASR <版本>        [引擎 / engine ▾]      状态栏        │
├─ 素材卡片 ────────────────────────────────────────────────────┤
│ [本地路径 / local path ......................]  [上传控件]     │
├─ 页签 ────────────────────────────────────────────────────────┤
│ ① 信息提取 │ ② 音频分段 │ ③ 字幕生成 │ ④ 情绪识别 │ ⑤ 参考音频 │
├───────────────────────────────────────────────────────────────┤
│ （当前页签的参数 + 按钮 + 只读结果框）                          │
└───────────────────────────────────────────────────────────────┘
产物目录 / outputs: <绝对路径>
```

### 2.1 顶栏

| 控件 | 说明 |
|---|---|
| 引擎下拉 | 选项来自 `module.engines()`（**不加载权重**）。默认选中 `configs/asr.yaml` 的 `engine.default`（出厂是 `faster_whisper`）。只对 ③④⑤ 生效 |
| 状态栏 | 绿字=成功、红字=失败。任何失败都只改状态栏与结果框，**界面不会崩** |

### 2.2 素材输入（两种方式，任选其一）

| 方式 | 行为 |
|---|---|
| **本地上传** | 拖/选文件 → 自动存到 `outputs/uploads/`，并把绝对路径填进路径框 |
| **手填路径** | 直接粘贴本地绝对路径（可带首尾引号，会被自动去掉） |

两条校验：路径为空 → 提示「请先指定素材」；路径不存在 → 提示「文件不存在」。
**五个页签共用同一个路径框**，不必每个页签重填。

> 上传的 NiceGUI 3.x 兼容性问题（`'UploadEventArguments' object has no attribute 'name'`）
> 已修复，见 `issues/006`。若你跑的是修复前的进程，请重启 GUI。

---

## 3 · 五个页签

### ① 信息提取

| 项 | 值 |
|---|---|
| 按钮 | 提取信息 / probe |
| 参数 | 无（内部 `deep=True`，即做声学统计） |
| 调用 | `ASRModule.probe(path)` |
| 输出 | `AudioInfo.to_dict()` 的 JSON：时长 / 容器 / 采样率 / 声道 / 编码 / 峰值 dBFS / RMS dBFS / **估计 SNR** / **语音占比** / 是否削波 / warnings |
| 产物 | 无 |

两个统计量是**决策输入**：估计 SNR 是降噪开关的判据，语音占比是幻觉闸门的判据。

### ② 音频分段

| 项 | 值 |
|---|---|
| 参数 | `单片上限(秒)`（默认 30）、`导出片段 / export wav`（默认关） |
| 按钮 | 开始分段 / segment |
| 调用 | `ASRModule.segment(path, max_seconds, export_audio, outputs/gui_segments)` |
| 输出 | 段清单 JSON：`index / start / end / duration / hard_cut` |
| 产物 | 勾了「导出片段」才有：`outputs/gui_segments/{素材名}_{序号:04d}.wav` |

`hard_cut=true` 表示该段落在连续说话区、没有静音点可切，只能按固定时长硬切。

### ③ 字幕生成

| 项 | 值 |
|---|---|
| 参数 | `语种`（默认 `auto`）、`格式`（srt / vtt / ass）、`词级时间戳`（默认开） |
| 按钮 | **转写 / transcribe**（不带情绪）、**转写 + 情绪 / with emotion**（末尾顺带做情绪） |
| 调用 | `ASRModule.transcribe(path, engine=顶栏, language, words, emotion, srt=..., srt_format=...)` |
| 输出 | 头部一行 `[引擎] lang=… segs=… emo=…` + 字幕路径 + warnings，随后是全文 |
| 产物 | `outputs/gui_subtitles/{素材名}.{fmt}` |

这一步的结果会被 **⑤ 参考音频**复用（见下）。

### ④ 情绪识别

| 项 | 值 |
|---|---|
| 参数 | 无 |
| 按钮 | 识别情绪 / detect emotion |
| 调用 | `ASRModule.transcribe(path, engine=顶栏, emotion=True, words=False)` |
| 输出 | `汇总 / summary: <主导情绪> (n/N)` + 每段 `{start, end, emotion, text(前 40 字)}` |
| 产物 | 无（要字幕请用 ③） |

情绪是**句级**的，7 类取值；过短的段（<0.3 s）会留空而不是硬给一个标签。
主力引擎（faster-whisper）没有情绪能力，门面会**自动按段调用本地 SenseVoice 补齐** ——
它跑在 CPU 上，不占显存、不额外花钱。

### ⑤ 参考音频（喂 TTS 音色克隆）

| 项 | 值 |
|---|---|
| 参数 | `取前 N 段 / top`（默认 3）、`目标时长(秒) / target`（默认 12，范围 3–30） |
| 按钮 | 截取参考音频 / pick clips |
| 调用 | `ASRModule.clip_reference(path, transcript, top, outputs/gui_refs, "ref", engine=顶栏)` |
| 输出 | 候选 JSON（含 `start/end/duration/score/score_detail/wav_path/text`）+ **每条内嵌音频播放器** |
| 产物 | 每个候选三件套：`outputs/gui_refs/ref_01.wav` + `.txt`（= `ref_text`）+ `.srt` |

三点注意：

1. **`target` 只在本次界面操作里生效**（临时改 `cfg.clip.target_seconds`），不回写配置文件；
2. 若**没先在 ③ 转过写**，本页签会自动先跑一次转写 —— 因为 `ref_text` 必须来自
   **该片段自己的转写**，不能拿整段文本（否则下游音色克隆会学错对应关系）；
3. 素材里若是每段语音都短于 10 s，截取会跨静音取窗（参考音频中间带一点静音不影响音色）。

---

## 4 · 产物目录

全部落在 `cfg.output_dir`（默认 `F:\2026年\Agentic_ASR_Module\outputs`）：

| 子目录 | 谁写的 | 内容 |
|---|---|---|
| `uploads/` | 顶栏上传 | 用户上传的原始素材 |
| `gui_segments/` | ② 勾选导出片段 | `{素材名}_{序号:04d}.wav` |
| `gui_subtitles/` | ③ | `{素材名}.srt` / `.vtt` / `.ass` |
| `gui_refs/` | ⑤ | `ref_NN.wav` + `ref_NN.txt` + `ref_NN.srt` |

`uploads/` 与 `gui_*` 都在 `outputs/` 之内，已被 `.gitignore` 排除。

---

## 5 · 手工验证清单

GUI 没有浏览器自动化测试（`tests/test_gui.py` 只覆盖上传落盘与框架契约），
所以改界面后按这条清单走一遍：

1. `python -m agentic_asr.gui` → 浏览器自动打开，**状态栏显示「就绪 / ready」**，
   引擎下拉里能看到 `faster_whisper / sensevoice / minimax / openai_compat / mock`；
2. 选一个 `models/SenseVoiceSmall/example/zh.mp3` **上传** → 状态栏显示
   「已上传 / uploaded: zh.mp3」，路径框被自动填上，且 `outputs/uploads/zh.mp3` 真的存在；
3. 不选文件直接点任一按钮 → 状态栏红字「请先指定素材」；
4. 路径框填一个不存在的路径 → 红字「文件不存在」；
5. ① 信息提取 → 结果框出现 JSON，且 `duration ≈ 5.6`、`speech_ratio` 明显大于 0；✅
6. ② 音频分段（上限填 3）→ 段清单非空、`start` 递增、首段 `start = 0`；
7. 勾「导出片段」重跑 → `outputs/gui_segments/` 下真的出现 wav；
8. ③ 字幕生成 → 头部显示 `lang=zh`、全文非空；`outputs/gui_subtitles/zh.srt` 存在
   且能用播放器/编辑器打开；切格式为 `ass` 再跑一次 → `.ass` 也生成；
9. ④ 情绪识别 → `汇总 / summary` 有值，逐段 JSON 里 `emotion` 是 7 类之一（或 null）；
10. ⑤ 参考音频（top=2）→ 候选 JSON 有 score，播放器能出声，
    `outputs/gui_refs/` 下有 `ref_01.wav/.txt/.srt`；
11. 引擎下拉切到 `mock` 重跑 ③ → 应立刻出结果（不载权重），用于确认界面链路本身没问题。

---

## 6 · 已知限制

1. **只有手工验证**：界面行为没有自动化测试覆盖，只有上传逻辑与框架契约有单测；
2. **五个页签共用一个素材路径**：想同时对两个文件操作得来回改，没有"多素材队列"；
3. **⑤ 会自动触发一次转写**：如果没先做 ③，第一次点它会慢（要载权重 + 跑一遍 ASR）；
4. **不显示进度百分比**：长音频转写期间只有「转写中 / transcribing…」的文案，
   没有进度条（`module.transcribe` 目前不是流式 API）；
5. **`target` 不回写配置**：见 §3 ⑤ 的第 1 条；
6. **不做并发**：NiceGUI 默认单进程，同一时刻只服务一个耗时任务；
   多人同时用请起 HTTP 服务（`04_api_reference.md`）而不是 GUI。
