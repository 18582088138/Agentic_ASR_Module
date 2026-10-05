# 09 · 开发与测试记录 / Development log

> **什么时候读**：想知道"实际做出来的和当初方案差在哪""踩过哪些坑""还剩什么没做"时。
> 纯粹的实测数字（RTF / 显存）权威在 `00_research.md §4`，这里只登记偏差与决策。

---

## 1 · 交付内容

| 项 | 位置 |
|---|---|
| 库（门面） | `agentic_asr/asr.py::ASRModule` |
| 五个功能 | `media/probe.py` · `segment/splitter.py` · `subtitle/` · `engines/sensevoice.py` · `clip/picker.py` |
| 五个引擎 | `faster_whisper`（默认）· `sensevoice` · `minimax` · `openai_compat` · `mock` |
| 四种用法 | 库 · CLI(`asr`) · HTTP(8301) · GUI(8401) |
| 文档 | `docs/00_INDEX.md` 是入口 |

**分离功能不在这里**，它被拆成了独立插件 **Agentic_VoiceSeparate_Module**
（人声提取 / 背景音提取 / 主体人声去除，端口 8302 / GUI 8402）。
两个仓库互不依赖，可分别部署。

---

## 2 · 测试结果

不在这里写死条数（会腐烂）。现算：

```bash
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests -q              # 离线
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests -q -m real      # 真实模型
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe -m pytest tests --collect-only -q | Select-Object -Last 1
C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe tools\check.py                  # 收尾闸口
```

**最后一次收尾闸口的状态**：三项全绿 —— 环境自检（解释器 / 依赖 / ffmpeg / 三个模型 /
CUDA 运行库）、ruff、全量离线测试；真实模型档另跑也是全绿。

用例分布、写死的断言、各档位前置见 `03_unit_tests.md`。

---

## 3 · 与方案的偏差

方案见 `01_design.md`，阶段计划见 `02_dev_plan.md`。下面这些是**实际做的时候改了**的。

### 3.1 架构：从一个模块改成两个平级插件

`01_design.md` 的早期版本是"一个 `ASRModule` 挂五个功能（含分离）"。
开发中途用户要求：**两个独立 Agent 插件、可独立部署与独立调用**，于是：

- 分离能力整体移出，成为 `Agentic_VoiceSeparate_Module`；
- 两个插件各有自己的库 / CLI / HTTP / GUI 与端口；
- 引擎抽象也分成两套平级接口（一个答"哪些是人声"，一个答"说了什么"）。

理由：两者能力正交，硬合并会让能力表无法表达（云端 ASR 能做转写但做不了分离，
分离模型能分轨但不认识字）。

### 3.2 分离引擎：demucs → sherpa-onnx UVR

`02_dev_plan.md` 阶段 6 写的是"`SeparationEngine` 抽象 + **demucs** 引擎"。
实际开发中实测后换掉了，理由三条：

1. **demucs 做不到核心需求**：它只能做人声/伴奏二分。实测「中文主体 + 英文背景」素材时，
   它把两种人声全塞进 `vocals`，`no_vocals` 只剩 rms 0.0052 的近静音 ——
   无法"只去掉主体人声、保留背景人声"；
2. **sherpa-onnx 的 UVR 更快且免费**：KARA 系在 **CPU** 上 RTF 0.232，
   而 demucs 在 **GPU** 上才 0.740；
3. **零新增依赖、零显存**：`pip install sherpa-onnx` 只加 2 个包，不占显存
   （8 GB 卡整块留给 ASR）。

数字见 `00_research.md §4.2 / §4.4`。

### 3.3 情绪：sherpa-onnx，不是 funasr

`02_dev_plan.md` 把"怎么在不加 18 个包的前提下拿到 SenseVoice 情绪"列为**首要风险**，
给了三条候选（sherpa-onnx / vendored 推理代码 / funasr）。实际验证结果：

- **`sherpa_onnx.OfflineRecognizer.from_sense_voice()` 原生返回 `emotion` / `event` / `lang`**，
  字段时间戳也有；
- 于是直接走第 1 条，`funasr` 与 vendored 都没用上；
- 顺带白拿了**语种识别**与**音频事件检测**，而且跑在 **CPU** 上，不与 faster-whisper 抢显存。

风险项就此关闭，未产生额外依赖。

### 3.4 轨道判据：能量 → VAD → 都不能信 → 模型约定

这条在分离插件里，但值得在本日志留一句：**两条看起来合理的判据都被实测否决了**。

| 判据 | 反例（实测） |
|---|---|
| 能量（rms） | 「人声 + 背景音乐」素材里，音乐轨 0.1533 **>** 人声轨 0.1114 |
| VAD 语音占比 | 「人声 + 背景人声 + 音乐」素材里，含音乐轨 0.914 **>** 人声轨 0.7624 |

第二条是**跨插件联调时**才暴露的：判定方向整个反了——"去掉主体人声"变成了
"去掉背景人声"，用 ASR 复核才看出来。最终改成"配置 > **模型约定表** > VAD > 兜底"，
详见分离插件的 `docs/issues/005`。

### 3.5 方案里没写、开发中补上的

| 补的东西 | 起因 |
|---|---|
| **音频分段**（第二个功能） | 用户中途追加：避免长音频 OOM，静音点优先 |
| **`_clamp()`**（把片输出裁回该片区间） | 实测 Whisper 在 30 s 解码窗口里给短片的尾段时间戳会**超出实际音频**（123.55 s 素材被标到 131.51 s），字幕凭空长出 8 秒 |
| **`split_gap`** 配置项 | 第一版按 `max_seconds` 贪心合并，把 17 个自然停顿并成 2 个 27 秒长段，不符合"静音点优先" |
| **CUDA 运行库显式注册** | GPU 推理隐式依赖 torch 被导入（`issues/004`） |

---

## 4 · 已记档的问题

**本仓库的问题单是 `001`–`004` 与 `006`。** 编号 `005` 属于
**Agentic_VoiceSeparate_Module**（分离轨道顺序），本仓库不重复建单，所以这里看起来缺一个号。

| 单 | 一句话 | 回归用例 |
|---|---|---|
| [001](issues/001-faster-whisper-incompatible-with-av19.md) | faster-whisper 1.2.1 与 av 19 不兼容（`av.open()` 已移除 `metadata_errors`）→ **自己做解码层**，引擎永不接受文件路径 | `test_audio.py::test_decode_to_16k_mono` 等 |
| [002](issues/002-music-only-hallucination.md) | 纯音乐/无语音输入会被 Whisper 编出假字幕（实测「优优独播剧场」，置信度还 `p=1.000`）→ 语音占比 + 压缩比**双闸门** | `test_guard.py` 三条 |
| [003](issues/003-sherpa-onnx-vad-must-be-fed-in-chunks.md) | sherpa-onnx 的 Silero VAD **必须分块喂、且边喂边取**，否则静默漏检（占比 0.78→0.06）并丢段 | `test_vad.py` 两条 `real` |
| [004](issues/004-gpu-inference-implicitly-needs-torch-imported.md) | GPU 推理隐式依赖 torch 被导入（CTranslate2 不带 `cublas64_12.dll`）→ 加载前**显式注册** CUDA 运行库，`doctor` 可诊断 | `test_real.py::test_gpu_inference_without_torch_preimport` |
| [006](issues/006-nicegui-3-upload-event-has-no-name.md) | NiceGUI 3.x 的 `UploadEventArguments` 没有 `name`（2.x 的 `e.name`/`e.content` 已不存在）→ 改用 `e.file.save()`，并钉住框架契约 | `test_gui.py` 五条 |

`003`、`004`、`006` 的共同点是**静默失效**：不报错，只是结果变得没用或界面不可用，
所以都配了"阈值故意写死"的回归用例（清单见 `03_unit_tests.md §4`）。

---

## 5 · 后续待办

按"已知但没做"与"需要真实素材才能推进"分开。

### 5.1 只留了能力位、明确没实现

| 项 | 现状 | 说明 |
|---|---|---|
| 降噪 / 去混响 | `preprocess.denoise: off` 只留配置位 | 调研结论：**降噪不是默认动作** —— 过度降噪会提升 PESQ 却降低 ASR 准确率；且声纹分支必须走原始音频（双轨）。技术参考见 `10_reference_research.md §2` |
| 说话人分离（diarization） | `Capability.DIARIZATION` 只留位，本地方案未接 | 中文首选 3D-Speaker/CAM++ 是「VAD+嵌入+聚类」组合而非开箱 pipeline。**云端可先用**：MiniMax 的 `verbose_json` 自带 `n_speakers` 与每段 `speaker` |
| `transcribe(separate=True)` | 配置项存在，默认关闭 | 需要 Agentic_VoiceSeparate_Module 配合；默认关的理由是有论文实测 SDR 7.97 dB 的分离器会让 WER **一致变差** |

### 5.2 需要真实素材/额外条件才能推进

| 项 | 卡在哪 |
|---|---|
| **分离泛化性复验** | KARA 系训练域是音乐，目前的"能区分主体/背景人声"只在**合成素材**上验过。真实影视对白需人工听审（分离插件的 `-m real` 用例守着"两条轨都不能是静音"这条底线） |
| **日语质量** | 实测 SenseVoice 与 MiniMax 都会把「持って**いけない**」漏成「持ってい**き**ない」，只有 faster-whisper 对。语种级 CER 无权威数字（`00_research.md §6` 列为未确认） |
| **云 API 的真实调用** | `live` marker 已在 `pyproject.toml` 声明但**无用例使用**；目前靠手工跑一次真音频验证（MiniMax 已手工验过三条样例） |
| **分离对 ASR 的量化收益** | 合成干扰太弱，测不出 WER 变化；需要真实影视素材做 A/B |

### 5.3 明确放弃的

| 项 | 原因 |
|---|---|
| **Qwen3-ASR 作为本地主力** | `pip install qwen-asr` 会新增 18 个包并把共享环境的 `transformers` 从 4.57.3 拉到 4.57.6，而 Agentic_TTS_Module 钉死了这个版本。代价是中文方言与长音频对齐精度不如它（`00_research.md §1`） |
| **RoFormer 系分离模型** | 人声 SDR 更高（12.9 vs KARA 系约 10），但社区权重许可混乱：GPL-3.0 / `license: other` / 干脆未声明。质量差距换不来许可风险 |
| **`funasr` / `demucs` 作为依赖** | 分别被 sherpa-onnx 的情绪路径与 UVR 分离路径取代，详见 §3.2 / §3.3 |
