# 00 · 调研 / Research

调研日期见 `09_dev_log.md`。本文只写**会影响方案的事实**，结论可复现。
标注约定：**【实测】**=本机跑出来的；**【官方】**=官方文档/模型卡；**【三方】**=第三方评测；
**【未确认】**=查不到，不许猜。

---

## 1 · 结论先行

**不存在单一模型同时满足「中英日 + 词级时间轴 + 情绪」，必须组合。**
选定组合（用户已确认）：

| 角色 | 引擎 | 理由 |
|---|---|---|
| 本地主力 ASR | **faster-whisper large-v3-turbo**（CTranslate2） | 【实测】显存 +2.17 GB、RTF 0.037、中英日韩全对、词级时间戳开箱；**只新增 2 个包**，不碰 torch/transformers |
| 语音情绪 | **SenseVoiceSmall** | 唯一轻量的开源 SER 来源（7 类，utterance 级；代码 MIT / 权重 FunASR MODEL_LICENSE） |
| 云 API | **MiniMax `asr-1.0`** | 【官方】有词级时间戳 + 说话人分离 + SRT/VTT 直出，中英日等 20 语种；项目已有 key |

**被否决的选项**（理由都是实测/官方事实，不是偏好）：

- **Qwen3-ASR-0.6B 作本地主力**：中文/方言上限确实更高（Apache-2.0、30 语种、自带
  ForcedAligner，官方 AAS 42.9 ms），但 **【实测】`pip install qwen-asr` 会新增 18 个包
  （Cython/Flask/Werkzeug/nagisa/dyNET38/soynlp…）并把 `transformers` 从 4.57.3 拉到
  4.57.6**，而 TTS 模块在同一个 conda 环境里把 `transformers==4.57.3` 钉死 —— 两个项目
  会来回拉锯。另一条原生 transformers 路线要求 `>=5.13.0`，冲突更大。
  → 用户选择 A 路线：**本地走 faster-whisper，Qwen3-ASR 留给云端 API**。
- **NVIDIA Parakeet-TDT-0.6b-v3**：【官方】模型卡语言列表 25 种全是欧洲语言，**无中文、无日语**。
- **日语专用 Kotoba-Whisper**：【官方】只在域内（ReazonSpeech 11.6 vs 14.9）优于 large-v3，
  域外 CommonVoice(9.2 vs 8.5)、JSUT(8.4 vs 7.1) **反而更差** —— 「日语专用一定更好」不成立。
- **pyannote diarization 作首个说话人方案**：【官方】模型页 gated，需 HF 账号接受条款，
  而 HF 本机直连不可达；中文 DER 12.2% 也高于 3D-Speaker 的 10.30%。
- **FunASR 全家桶**：`pip` 解析显示 `funasr` 要新增 18 个包（hydra-core/umap-learn/jieba/
  tensorboardX/kaldiio/oss2…），与「轻量化」直接冲突。

---

## 2 · 本机环境事实 【实测】

```
GPU      NVIDIA GeForce RTX 4060, 8188 MiB, driver 595.79
Python   3.12.14 @ C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe
torch    2.11.0+cu128  (cuda available, cuda 12.8)
numpy    2.5.2    scipy 1.18.1
已装     transformers 4.57.3 / onnxruntime 1.29.0 / soundfile 0.14.0 / librosa 1.0.0
         torchaudio 2.11.0 / fastapi 0.141.1 / nicegui 3.16.0 / modelscope 1.39.1 / pydub
ffmpeg   C:\Users\75203\Documents\ffmpeg-master\bin\ffmpeg.exe（ffprobe 同目录，均在 PATH）
缺       ctranslate2 / faster-whisper（本次装入）/ sherpa-onnx / pyannote / silero-vad
```

`transformers 4.57.3` 是 **Agentic_TTS_Module 钉的版本**（其 `pyproject.toml` 写死
`transformers==4.57.3`）。两个项目共用一个 conda 环境，所以 ASR 侧**任何**升
`transformers` 的动作都会破坏 TTS。这是本方案选 faster-whisper 的根本原因。

## 3 · 网络与模型源 【实测】

**两种网络状态都测过**：不开 VPN 时 HF 直连超时，开 VPN（用户 2026-10-04 开启）后
HF 与 OpenAI/Groq/DashScope 端点全部可达。所以代码里**不要写死任何一种环境**，
按「先直连、失败退镜像」处理。

| 源 | 不开 VPN | 开 VPN |
|---|---|---|
| `huggingface.co` | ❌ 直连超时 | ✅ 200 / 0.5 s |
| `hf-mirror.com` | ✅ 200 | ✅ 200 |
| `modelscope.cn` | ✅ 200（`snapshot_download` 20 MB/s 级） | ✅ 200 |
| `api.openai.com` / `api.groq.com` / `dashscope.aliyuncs.com` | ❌ 403 / 超时【三方实测】 | ✅ 401（可达，无 key） |
| `api.minimaxi.com`（国内） | — | ✅ 400（**鉴权通过**，仅缺参数） |
| `api.minimax.io`（海外） | — | ❌ 401 `invalid api key (2049)` |

> **凭证端点归属实测**：用户提供的 MiniMax key 在 `.com`（国内）通过鉴权、在 `.io`（海外）
> 被判无效。所以 `configs/asr.yaml` 的 `engine.minimax.endpoint` 默认必须是
> `https://api.minimaxi.com`。

**已下载到 `models/`（gitignore）**：

| 仓库 | 本地目录 | 体积 | 来源 |
|---|---|---|---|
| `pengzhendong/faster-whisper-large-v3-turbo` | `models/faster-whisper-large-v3-turbo` | 1546 MB | ModelScope（HF 上对应 `deepdml/faster-whisper-large-v3-turbo-ct2`） |
| `iic/SenseVoiceSmall` | `models/SenseVoiceSmall` | 897 MB | ModelScope（官方阿里仓库） |

## 4 · 本地引擎实测 【实测】

`scripts/_probe_asr.py`（调研期探针，见 `09_dev_log.md` 的复测命令），
素材是 SenseVoice 自带的 zh/en/ja/ko 示例音频。

| 素材 | 语种识别 | 转写结果 | RTF | 显存 |
|---|---|---|---|---|
| zh 5.62 s | zh `p=0.989` | 开放时间早上九点至下午五点。 | 0.131 | +2.17 GB |
| en 7.18 s | en `p=1.000` | The tribal chieftain called for the boy and presented him with 50 pieces of gold. | 0.073 | 同上 |
| ja 7.22 s | ja `p=1.000` | うちの中学は弁当制で、持っていけない場合は50円の学校販売のパンを買う。 | 0.080 | 同上 |
| ko 4.64 s | ko `p=0.999` | 조금만 생각을 하면서 살면 훨씬 편할 거야. | 0.106 | 同上 |
| **zh 185.3 s** | — | 24 段 | **0.037** | 同上 |

其他要点：
- 模型加载 **1.5 s**；`compute_type="float16"`，`word_timestamps=True`，`beam_size=5`，`vad_filter=True`。
- **词级时间戳可用**：中文按字（`开@0.37 放@0.89`）、英文按词、日文按字。
- 显存只能用 `nvidia-smi` 读 —— **CTranslate2 不走 torch 的 CUDA 分配器**，
  `torch.cuda.max_memory_allocated()` 全程返回 0。这一点写代码时必须记住。
- RTF 0.037 ⇒ 1 小时音频约 **2.2 分钟**转完；8 GB 卡剩余约 6 GB，够再常驻 SenseVoice。

### 4.1 faster-whisper 1.2.1 与 av 19.0.1 不兼容 【实测】

```
TypeError: open() got an unexpected keyword argument 'metadata_errors'
  faster_whisper/audio.py:46  ->  av.container.core.open
```
`av` 19 已移除 `metadata_errors`，而 faster-whisper 的依赖声明是 `av>=11`（无上界），
所以 `pip install faster-whisper` 今天必踩。**解法不是降级 av，而是自己做解码层**
（`ffmpeg`/`soundfile` → 16 kHz 单声道 float32 → 传 `np.ndarray` 给 `transcribe`）——
这本来就是这个模块该有的东西：四条输入通路（文件/视频/字节流/麦克风）都要统一到
同一份音频表示，且「音频信息提取」功能本身就要用 ffprobe。详见
`issues/001-faster-whisper-incompatible-with-av19.md`。

### 4.2 人声/背景音分离实测（demucs htdemucs）【实测】

`scripts/_probe_separate.py`：合成 30 s「人声 + 背景音乐」44.1 kHz 立体声混音，
用 `python -m demucs --two-stems=vocals --segment 7 -d cuda` 分离。

| 指标 | 结果 |
|---|---|
| 依赖成本 | `pip install demucs` 只新增 **4 个包**（demucs / julius / lameenc / sphn） |
| 显存 | **+861 MB**（2048 → 2909 MB） |
| 耗时 | 30 s 音频 → **22.2 s**，RTF **0.740** |
| 输出 | `vocals.wav` / `no_vocals.wav`，30.000 s @ 44.1 kHz 立体声，**与原音频严格等长** |

三条必须记住的事实：

1. **`--segment` 只接受整数**。网上流传的 `--segment 7.8` 会让 CLI 直接报
   `invalid int value: '7.8'`（7.8 是模型内部窗口，不是 CLI 取值口径）。
2. **显存远低于常见说法**。官方 README 的「默认约 7 GB」是默认切分下的量级；
   显式给 `--segment 7` 后实测**只要 861 MB**，8 GB 卡绰绰有余，还能与
   faster-whisper（2.17 GB）同时常驻。
3. **分离对 ASR 的收益本次未能量化**。对照实验里，混音与分离后的 `vocals`
   都能识别出正确内容，差异只是尾部重复与数字写法（「九点」/「9点」，
   是 Whisper 的 ITN 行为）。我合成的背景是纯正弦波，干扰强度不足，
   **因此不能据此宣称 WER 提升** —— 需要真实影视素材才能测。

**分离方案的选型对比**（用户新增需求「人声与背景音分离」的专项调研）：

| 方案 | 人声 SDR | 依赖增量 | 许可 | 结论 |
|---|---|---|---|---|
| **demucs htdemucs** | 8.8~10.0（口径不一） | **+4 包** | MIT（代码与官方权重都干净） | **选它** |
| demucs htdemucs_ft | 9.19~10.8 | 同上 | MIT | 质量档；全 4 专家 bag 要 4× 时间，只取 vocals 单专家则快得多 |
| demucs-onnx（StemSplitio） | 与 PyTorch fp32 数值等价（误差 ≤6.6e-4） | **+2 包**（soxr / hf-hub） | MIT | 备选：纯 onnxruntime、零 torch；但本机 ort 是 CPU 版，要 GPU 得另装 `onnxruntime-gpu`（与 cu128 共存性未验证） |
| UVR MDX-Net | 10.4（UVR 自报口径） | 须自建 STFT/iSTFT | 权重 MIT（上游未逐条核对） | 备选 |
| BS / Mel-Band RoFormer | **12.9 / 12.6（最高）** | — | **混乱**：GPL-3.0 / `other` / 未声明 | ❌ 不用 |
| Spleeter | 5.9~6.9 | — | MIT | ❌ 已停维护 |
| audio-separator | — | **+13 包**（含 `onnx-weekly` 开发版） | MIT | ❌ 太重 |

**论文数据（这条决定流水线设计）**：ICME 2025 Workshop 的
[arXiv 2506.15514](https://arxiv.org/abs/2506.15514) 实测「分离 → Whisper 歌词转写」：

- SDR **7.97 dB** 的分离器让 WER **一致变差**（23.59 → 23.98）
- SDR **8.76 dB** 才一致改善（→ 20.00，约 1.4–3.6 个绝对点）
- 「完美人声」相对原混音改善约 **9 个点**（23.59 → 14.19）

⇒ **不要用低质量分离器做 ASR 前处理**。本模组因此**不内建**自动分离：
分离是**另一个独立模组**的事，要不要用、用哪个模型由**应用层**决定，
而不是替用户默认做掉。（论文未给「收益随 SNR 变化」的定量门槛 → 未确认。）

**三条非直觉的官方事实**（源码/README 核对，不是推断）：

1. **`--two-stems=vocals` 不减显存** —— 官方明确它先跑完 4 轨再相加，
   原文 "this won't be faster or use less memory"。省显存只能靠 `--segment`。
2. **链路会把输入重采样到 44.1 kHz** —— `api.py::_load_audio` 无条件
   `convert_audio(..., self._samplerate, ...)`。所以「等长」只在采样率相同时成立；
   48 kHz / 16 kHz 素材的输出样本数 ≠ 原文件样本数。门面必须自己重采样并对齐样本数。
3. **默认 `clip='rescale'` 是逐轨独立缩放** —— 导致 `vocals + accompaniment ≠ 原混音`
   （官方 README 承认这"breaks the relative volume between stems"）。
   本模块默认改 `--clip-mode none`。

### 4.3 纯音乐轨会触发 ASR 幻觉 【实测】

同一实验里，对 `no_vocals.wav`（已去掉人声，只剩背景音乐）跑 ASR，
Whisper 输出了 **「优优独播剧场——YoYoTelevisionSeriesExclusive」** —— 完全编造的内容。

⇒ 门面对「分离后的人声轨」必须做**无语音防护**（VAD 语音占比阈值），
否则纯伴奏会被编出字幕。已记 `issues/002`。

### 4.4 分离引擎换选：sherpa-onnx UVR **优于** demucs 【实测】

调研「主体人声去除」时发现：**`sherpa-onnx` 自带 `OfflineSourceSeparation`，官方支持
UVR MDX 系 ONNX 模型**（28–64 MB），而本机早就装了 sherpa-onnx。实测对比：

| 方案 | 设备 | RTF | 显存 | 新增依赖 | 输出等长 |
|---|---|---|---|---|---|
| demucs htdemucs | GPU | 0.740 | +861 MB | torch + 4 包 | ✓ |
| **sherpa UVR KARA_2**（50 MB） | **CPU** | **0.363** | **0** | **0** | ✓ |
| **sherpa UVR KARA_1**（28 MB） | **CPU** | **0.241** | **0** | **0** | ✓ |

⇒ **CPU 上比 demucs 在 GPU 上还快，且完全不占显存。** 8 GB 卡可以整块留给 ASR。
官方口径（[sherpa-onnx 文档](https://k2-fsa.github.io/sherpa/onnx/source-separation/models.html)）：
UVR MDX 系单线程 CPU RTF 约 0.59–0.73，官方称「比 Spleeter 慢 10 倍」；
我们 4 线程实测 0.24–0.51，比官方口径更快。

下载：GitHub release `source-separation-models` 前缀
`https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/`，
**用 python `requests` 下；`curl.exe` 会被 Windows schannel 中断**（实测）。

**重大发现：KARA 模型能区分「主体人声」与「背景人声」，而且在说话场景下也成立。**
构造「中文主体（居中）+ 英文背景（偏侧、−10 dB）」素材，用 ASR 判读各轨：

| 轨 | rms | ASR 判读 |
|---|---|---|
| 原始混音 | 0.1434 | 中文（英文被掩盖） |
| demucs `vocals` | 0.1428 | 中文 |
| demucs `no_vocals` | **0.0052** | `pause` → **几乎静音** |
| kara1 `stem0` | 0.1083 | **中文**（主体） |
| kara1 `stem1` | 0.0651 | **英文**（背景） |
| kara2 `stem0` | 0.0417 | **英文**（背景） |
| kara2 `stem1` | 0.1336 | **中文**（主体） |

四条结论：

1. **KARA 系真能把主体与背景人声分到两轨**，即使素材是说话而非唱歌。
   这一点在调研阶段查不到任何官方依据（第三方模型卡把「完整混音」列为 out of scope），
   **只能实测，实测结果是可用**。
2. **demucs 做不到**：它把所有人声都塞进 `vocals`，`no_vocals` 只剩静音。
   ⇒「主体人声去除」必须用 KARA 系，不能用 demucs。
3. **stem 顺序不能按索引硬编码，但「按能量判定」也不可靠（已自我推翻）。**
   官方示例写 `stems[0]=vocals`；实测 kara1 的 stem0 是主体（与官方一致），
   而 kara2 的 stem0 是背景（与官方**相反**）。
   我一度认为「主体轨能量更高」可作判据（在「人声+人声」素材上 0.11 vs 0.07 确实成立），
   **但改用「人声+背景音乐」复测后推翻了它**：音乐轨 rms 0.1533 **高于**人声轨 0.1114。
   ⇒ 可靠判据是**语音占比（VAD）**——哪一轨检出语音，哪一轨就是主体人声轨。
   实现用「VAD 语音占比自动判定 + 配置可覆盖」，**不要用能量**。
4. **通用模型（Main / Voc_FT / Inst_Main）在正确场景下工作正常**：
   素材为「人声 + 背景音乐」时，stem0 是人声（rms 0.12，ASR 识别正确），
   stem1 是音乐（rms 0.15，ASR 输出 `Thank you.` 幻觉 = 无语音）。
   但在「人声 + 人声」这种**分布外输入**上表现不稳：低能量的背景人声会被当残差丢掉
   （英文碎成片段）。所以「人声提取」要用通用模型、喂「人声+音乐」类素材；
   而「主体 vs 背景人声」必须用 KARA 系。
5. 本次单例里 **kara1（28 MB）反而比 kara2（50 MB）更干净也更快**，但样本量只有 1，
   不能据此定论；两者都保留，默认 kara1，可切 kara2。

**使用约束**（官方）：
- **只接受 wav**，且需 44.1 kHz 立体声：
  `ffmpeg -i in.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav`。
- `process(sample_rate=..., samples=samples)` 要求 `samples` 为
  **(num_channels, num_samples)** 的 contiguous `float32`。

---

## 5 · 云 API 调研（用户已定：先 MiniMax 验证）

### 5.1 MiniMax Speech-to-Text 【官方】

`POST https://api.minimaxi.com/v1/speech_to_text`（**国内端点，实测本 key 只在这里有效**；
海外 `api.minimax.io` 返回 401），`multipart/form-data`，`Authorization: Bearer <API_KEY>`。

| 项 | 事实 |
|---|---|
| model | `asr-1.0` |
| 语种 | header `language` 支持 `zh/yue/en/ja/ko/th/vi/id/ms/fil/ar/tr/fr/de/es/it/pt/pl/ru/uk`；**留空 = 混合语种识别** |
| `response_format` | `json`（text+duration）/ `verbose_json`（+segments+n_speakers）/ `srt` / `vtt` |
| `timestamp_level` | `sentence`（默认）/ **`word`**（中文按字、英文按词）；仅 verbose_json/srt/vtt 生效 |
| **说话人分离** | ✅ `verbose_json` 返回 `n_speakers` 与每段 `speaker`（`S1`/`S2`） |
| **情绪** | ❌ 无情绪字段 |
| 流式 | ✅ SSE（`stream=true`，仅 `json`，且与 verbose_json/srt/vtt 互斥） |
| 限制 | **≤500 秒**、**≤50 MB**（超时返 400、超大小返 413，**不静默截断**） |
| 音频格式 | wav/aiff/flac/alac(m4a)/mp3/aac/opus/ogg；**不支持裸 PCM** |
| 建议 | 官方明说识别不受高采样率/立体声益处，**转 16 kHz 单声道**最省 |
| 错误体 | OpenAI 风格 `{"type":"error","error":{...},"request_id":...}` |

**本机实测（真实调用 3 条样例）**：

| 样例 | HTTP | 耗时 | 识别结果 | 词级 | speaker |
|---|---|---|---|---|---|
| zh 5.62 s | 200 | 1.2 s | 开放时间早上九点至下午五点。 **✓** | 中文**字级** | S1 |
| ja 7.22 s | 200 | 1.3 s | うちの中学は弁当制で、持って**いき**ない場合は50円の… **✗ 漏字** | 日文按词 | S1 |
| en 7.18 s | 200 | 2.1 s | …presented him with **fifty** pieces of gold. | 英文按词 | S1 |

- `response_format=verbose_json` + `timestamp_level=word` 工作正常，
  **每个 word 都带 `speaker`** —— 说话人分离的粒度比预期更细。
- **日语质量不如本地 faster-whisper turbo**：本地正确识别出「持って**いけない**」，
  MiniMax 漏成「持っていきない」。这为「本地 faster-whisper 主力、云 API 兜底」
  又添一条实测依据。
- 英文做了 ITN：原文 "50 pieces" 被归一成 "fifty pieces"，而本地 faster-whisper 保留 "50"。
  下游要求数字原样时要注意这个差异。
- 计费按返回的 `duration` 计（官方口径）。

**对本模块的影响**：500 秒上限意味着**长音频切片 + 时间轴平移合并必须由门面统一负责**
（对所有引擎共享），这正好也是 OpenAI `whisper-1`（25 MB）等其它云端的共同需求。

### 5.2 备选与对照（只记要点，详见 `10_reference_research.md` 与调研存档）

- **阿里云百炼**：`paraformer-v2` 0.288 元/小时（全场最低，有说话人分离、句级时间戳）；
  `qwen-audio-3.1-asr-flash-filetrans` 有**词级毫秒时间戳 + 说话人分离**、12 小时/2 GB 异步；
  `qwen3-asr-flash` **有情绪**（7 类）但**无时间戳** —— 需要时间戳与情绪并存时得组合调用。
  直连可达。**用户表示需要时再去申请 key**，本阶段不接入。
- **OpenAI `gpt-4o-transcribe` 无时间戳**【三方交叉验证】→ 字幕场景只能用 `whisper-1`；
  且直连不可达（实测 403），25 MB 上限。
- 各家能力差异在**厂商内部**就很大（同是百炼，qwen3-asr 有情绪无时间戳、qwen-audio 反之），
  所以适配器的能力声明必须按 **(provider, endpoint, model)** 三元组登记，不能按厂商。

---

## 6 · 未确认项（不要当成结论）

1. **日语单语 CER**：Qwen 官方只给 13 语聚合值（0.6B 12.75 / Fleurs 7.57），拆不出日语；
   faster-whisper turbo 的日语第三方 CER 也没找到权威数字。**本机实测只证明了短句正确**。
2. **SenseVoice 的情绪输出怎么在轻量前提下拿到**：官方有 SER 能力，但 `funasr` 会带 18 个包。
   候选：(a) `sherpa-onnx` 的 SenseVoice 绑定是否暴露 emotion；(b) vendored SenseVoice
   推理代码 + torch 直载 `model.pt`。**开发第一阶段必须先验证这条**（见 `02_dev_plan.md` 步骤 1）。
3. **MiniMax `asr-1.0` 的计费单价**与日语实际质量（官方文档未给价格页，需账号内确认）。
4. 火山引擎/讯飞/腾讯国内站的 ASR 单价 —— 官方页动态渲染或 404，未取到。

---

## 7 · 主要来源

- [faster-whisper README 与官方 benchmark](https://github.com/SYSTRAN/faster-whisper)｜
  [faster-whisper PyPI 依赖](https://pypi.org/pypi/faster-whisper/json)
- [SenseVoice 官方仓库（SER/AED 能力、15× 速度声明、许可澄清）](https://github.com/QwenAudio/SenseVoice)｜
  [SenseVoice 论文](https://arxiv.org/abs/2407.04051)
- [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)｜[Qwen3-ASR Blog](https://qwen.ai/blog?id=qwen3asr)
- [MiniMax Speech-to-Text API](https://platform.minimax.io/docs/api-reference/speech-to-text)
- [阿里云百炼 qwen-audio-3.1-asr-flash-filetrans](https://help.aliyun.com/zh/model-studio/qwen-audio-3-1-asr-flash-filetrans)｜
  [paraformer-v2](https://help.aliyun.com/zh/model-studio/paraformer-v2)｜
  [qwen3-asr-flash](https://help.aliyun.com/zh/model-studio/qwen3-asr-flash)
- [modelscope/3D-Speaker（中文 DER 基准，Apache-2.0）](https://github.com/modelscope/3D-Speaker)｜
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [DeepFilterNet（MIT，48 kHz）](https://github.com/Rikorose/DeepFilterNet)｜
  [facebookresearch/demucs（SDR 与显存）](https://github.com/facebookresearch/demucs)｜
  [eka-care/when-denoising-hurts](https://github.com/eka-care/when-denoising-hurts)
- [Huanshere/VideoLingo（Apache-2.0，分层流水线参考）](https://github.com/Huanshere/VideoLingo)
