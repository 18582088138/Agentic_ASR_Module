# 10 · 技术参考调研 / Reference research

> **定位：仅备查，不驱动方案。** 本文三块内容是用户要求的「技术参考」，
> 结论**不进入** `01_design.md` 的实现范围（说话人分离与降噪只在方案里留了能力位）。
> 查不到的项一律标 **【未确认】**，不许猜数字。
>
> 证据等级：【官方】= 模型卡/论文/官方文档；【三方】= 第三方实测或社区整理；【推断】= 本文推论。

---

## 1 · 声音角色区分 / Speaker roles

### 1.1 先分清两类任务（这是最容易混的地方）

| 任务 | 回答什么 | 输出 | 技术路线 |
|---|---|---|---|
| (a) **说话人分离** Diarization | 谁在什么时候说话 | `speaker_0/1/2` + 时间段，**匿名** | 端到端 diarizer，或 VAD + 嵌入 + 聚类 |
| (b) **声纹识别/验证** Verification | 这段音频是不是目标人 | 相似度分数 / 判定 | 说话人嵌入 + 余弦相似度阈值 |

两类模型不通用，但**嵌入模型（CAM++ / ECAPA / ResNet34）是两者的公共底座**。

### 1.2 候选对比

| 方案 | 体积 | 显存·CPU | 许可证 | 语言 | 结论 |
|---|---|---|---|---|---|
| pyannote `speaker-diarization-3.1` | 【未确认】 | 3.x 约 2.6 GB，4.x >9.5 GB【三方】 | MIT 但**模型页 gated** | 多语；AISHELL-4 中文 **DER 12.2%**【官方】 | 生态最成熟，但**强依赖 HF 账号 + 条款同意**；而本机 HF 直连不可达 |
| 阿里 3D-Speaker（FSMN-VAD + CAM++ + 聚类） | CAM++ **7.2 M 参数** | 极低 | **Apache-2.0**【官方】 | 中文强：AISHELL-4 **10.30%**、Meeting-CN-ZH 18.91%（均优于 pyannote）【官方】 | **中文首选**；但不是开箱 pipeline，要自己串 |
| NeMo Sortformer | 【未确认】 | 【未确认】 | CC-BY-4.0【三方】 | 英文为主，中日未确认 | 端到端免聚类，**上限 4 说话人** |
| WeSpeaker `resnet34-LM`（WhisperX 用的那套） | 【未确认】 | 极低 | 工具包 Apache-2.0【官方】 | VoxCeleb 英文为主，亦有中文模型 | **做「声纹注册 + 相似度匹配」最省事** |
| SpeechBrain ECAPA-TDNN | ~20.8 M 参数 | 极低，CPU 可跑 | 工具包 Apache-2.0 | 英文为主 | 经典基线，精度已被 CAM++ 超越 |
| 阿里 CAM++ 中文 `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2 M | 极低 | **Apache-2.0** | 中文 20 万说话人训练 | **「截取某人参考音频」最佳性价比** |

**对本模块两条真实需求的推荐**：

| 需求 | 推荐栈 |
|---|---|
| 给字幕标说话人 | 3D-Speaker 组合（中文优先）；要开箱且能接受 HF 条款时用 pyannote 3.1。**第一版可先用云 API（MiniMax 自带 diarization）** |
| 从长音频截取目标人 10~15 s | **CAM++ 嵌入 + 余弦阈值**，滑窗打分 —— **不要先 diarize 再挑片段**（会叠加聚类误差）。注意这属于「指定某人」，与方案里默认的「自动挑干净片段」是两个档位 |

### 1.3 能不能直接输出「角色」（男/女/年龄/标签）？

**不能。** 所有 diarization 模型只输出匿名 `speaker_N`。性别/年龄有独立小模型可做，
但属于另一条线；「角色名」本质需要语义或先验 —— 可行做法是 diarize 出匿名簇后，
用 LLM 依据台词内容与出现顺序映射到角色，**无开箱模型**。

---

## 2 · 音质优化 / 降噪 / 去环境音 / 人声分离

### 2.1 语音增强

| 模型 | 体积 | 实时率 | 48 kHz | 许可证 | 结论 |
|---|---|---|---|---|---|
| **DeepFilterNet 2/3** | ~1.8 M 参数 | 实时（含 lookahead） | **✅ 专用**（只收 48k wav） | **MIT / Apache-2.0 双许可**【官方】 | 全带宽实时降噪的**最佳平衡点，商用零风险** |
| GTCRN | **48.2 K 参数** | RTF 0.07（i5-12400）【官方】 | 16k 档 | 需核对 | **超轻量首选**；VCTK-DEMAND SISNR 18.83 / PESQ 2.87，官方称大幅优于 RNNoise |
| RNNoise | 0.06 M | 极快 | ❌ | 【未确认】 | 老基线，已被超越 |
| FRCRN / ZipEnhancer（阿里） | 【未确认】 | GPU 更舒适 | ❌（16k） | 需核对 | 中文适配好；**48k 影视素材要先降采样** |
| MossFormer2-SE 48K | 【未确认】 | GPU 建议 | ✅ | **【未确认】** | 质量更高，体积/算力也明显更重 |
| noisereduce（谱减法） | 无模型 | 极快 | 任意 | MIT | 零依赖兜底；只对稳态噪声有效，音乐无效 |

### 2.2 人声/伴奏分离（把对白从影视音里抽出来）

| 模型 | 人声 SDR | 显存 | 许可证 | 结论 |
|---|---|---|---|---|
| Demucs v4 `htdemucs` | 9.00 dB（MUSDB HQ 全源）【官方】 | **官方：≥3 GB；默认参数约 7 GB**【官方】 | MIT | 8 GB 能跑但很紧，**必须 `--segment 7.8`**；`--two-stems=vocals` **不减显存** |
| Demucs `htdemucs_ft` | 9.20 dB | 同上，耗时 ×4 | MIT | 质量最好但慢 |
| MDX-Net（UVR 系 onnx） | 约 9.4~10.4【三方，测试集未注明】 | 低 | **逐个核对** | 轻量、ONNX 部署友好 |
| BS-RoFormer / Mel-Band RoFormer | 11.02 / **11.60 dB**【论文】 | 明显更高 | 权重许可需核对 | 质量标杆，显存需谨慎 |
| Spleeter | 5.9 dB | 低 | MIT | 已被全面超越，**新项目不建议** |

### 2.3 其他组件

- **VAD**：Silero VAD 首选（小、CPU 可跑、多采样率）；中文可备 FunASR FSMN-VAD；
  WebRTC VAD 最轻但精度最低。
- **响度归一化**：`pyloudnorm`（EBU R128 / ITU-R BS.1770-4）。
- **去混响**：UVR 系 `dereverb_mel_band_roformer_*`；影视后期混响属艺术处理，谨慎。
- **AEC**：影视素材一般无回声，除非是通话/扬声器回采。

### 2.4 处理顺序（实践共识）

```
解码 → 降混单声道 →[可选] 人声分离 → VAD 切分 → 降噪 → 重采样 16k → 响度归一化 → ASR
                                      └→ 说话人分离/声纹匹配（用原始或轻度处理音频！）
```

1. **先分离后降噪**：分离是结构性去干扰，降噪是残差处理；反做会让降噪把音乐当噪声压制并留伪影。
2. **VAD 早于降噪**：降噪器在纯静音段会产生 musical noise，先切静音既减伪影又省算力。
3. **重采样位置取决于降噪模型**：16k 模型必须先降到 16k；48k 模型（DFN）则在原采样率处理。
4. **响度归一化最后**，且只做一次。
5. **降噪绝不能用于声纹分支**（会抹掉声道细粒度特征）→ **ASR 用增强音频、说话人用原始音频**（双轨）。

### 2.5 「过度降噪损害 ASR」专项结论

- 学术界明确存在**增强提升感知质量却降低识别率**的现象，有论文观察到增强后
  **PESQ 更高而 WER 更差**；有专门讨论该现象的补充仓库
  [eka-care/when-denoising-hurts](https://github.com/eka-care/when-denoising-hurts)。
- 工程结论：① 只在**估计 SNR 低**（经验阈值约 <15 dB，未确认）时启用；
  ② 一律用**保守档**（DFN 不开 `--pf`），激进抑制会吃掉辅音与气音，中文声调、日语促音受损更明显；
  ③ **验收必须落到 WER**，同素材「原始 vs 增强」各跑一次 —— PESQ 高 ≠ WER 低；
  ④ 优先做**分离**（去掉最大干扰源），降噪作为可选开关。

---

## 3 · 配音软件技术栈

### 3.1 主流软件

| 软件 | 可确认的技术点 | 性质 |
|---|---|---|
| 剪映 / 必剪 | 云端 ASR + 本地时间轴，字幕驱动剪辑；具体模型【未确认】 | 闭源 |
| Premiere Pro | 语音转文本 + Essential Sound + Enhance Speech【官方】；**文本是时间轴的派生视图** | 闭源订阅 |
| DaVinci Resolve | Fairlight 音频引擎 + 自动字幕，字幕是时间轴对象 | 闭源（有免费版） |
| **Descript** | 核心是**「编辑文字即编辑音频」**：转写 → 文本编辑器 → 删/移文字映射为音频剪切/拼接；Overdub 做音色克隆【官方帮助文档】 | 闭源 |
| 讯飞智作 / TTSMaker / 魔音工坊 | 云端 TTS + 音色库 | 闭源 SaaS |

### 3.2 字幕格式与文本-音频对齐

| 格式 | 特性 | 适用 |
|---|---|---|
| SRT | 序号 + 时间码 + 纯文本，最通用 | 交付/交换 |
| WebVTT | SRT 超集，支持样式与 cue 设置 | Web 播放 |
| ASS/SSA | 完整样式、定位、**Karaoke 标签** | 硬字幕、特效 |
| TTML | XML，可扩展，广播级 | 广电/平台规范 |

**「文本编辑即音频编辑」的实现原理**（Descript 未公开实现，以下为推断）：
ASR 产出**词级时间戳** → 文本为主数据结构，每词/段绑定 `[start,end]` → 删词即剪切对应
音频区间，剩余区间重排拼接 → 拼接点需交叉淡化/补静音防爆音。
**工业界词级对齐做法**：Whisper 原生时间戳不够准，主流是「ASR + 强制对齐」
（WhisperX 用 wav2vec2 系；**VideoLingo 当前默认就是 Qwen3-ASR + Qwen3-ForcedAligner**）。

### 3.3 配音工作流的关键技术点

| 环节 | 技术点 | 现状 |
|---|---|---|
| 时长控制 | TTS 时长建模 + **time-stretch**（WSOLA / phase vocoder）拉到原文时长 | 无完美解；VideoLingo 官方明确「不保证自然或完美同步」 |
| 口型同步 | Wav2Lip / MuseTalk / VideoReTalking | 属视频侧 |
| 音色克隆 | GPT-SoVITS / CosyVoice2 / F5-TTS 等 | VideoLingo 支持多家后端 |
| **多角色管理** | **业界普遍弱项**：VideoLingo 官方限制写明「不会为每个说话人自动分配不同声音」 | 说明 **per-speaker 音色路由是差异化空间** |
| SSML | 云端 TTS 普遍支持；开源 TTS 支持程度不一【未确认】 | — |

### 3.4 开源参考项目

| 项目 | 分层 | 许可证 | 借鉴点 |
|---|---|---|---|
| **VideoLingo** | yt-dlp → Qwen3-ASR + ForcedAligner 词级识别 → LLM 断句/术语/翻译 → 字幕排版 → TTS → 合成 | **Apache-2.0** | **分层最清晰**；其「当前限制」清单几乎就是本项目要避的坑 |
| video-subtitle-master | 批量生成字幕 + 多翻译后端 | 【未确认】 | **多后端插件化**，翻译层可插拔 |
| WhisperX + pyannote | ASR → wav2vec2 对齐 → diarization → 按说话人归属词 | 各自许可 | **「词级时间戳 + diarization」对齐的参考实现** |
| KrillinAI / Buzz / Subtitle Edit / Aegisub / Auto-Editor | 单点工具 | 多为 MIT/GPL | 工具形态参考 |

### 3.5 「视频 → 带角色标注的配音字幕轨」标准环节划分

```
1  抽音轨（ffmpeg，保留原始 48k 立体声）
2  人声分离（Demucs / MDX-Net）        ── 影视场景强烈建议
3  VAD 切分（Silero VAD）              ── 与步骤 2 共用同一时间基准
4  ASR → 词级时间戳                     ── 本模块负责 1/3/4
5  说话人分离 → 匿名 speaker_N           ── 与步骤 4 做区间对齐归属
6  [可选] 角色映射（LLM / 人工）
7  翻译（LLM，带上下文与术语表）
8  TTS（per-speaker 音色路由，参考音频来自步骤 5）
9  时长对齐（duration 预测 + time-stretch）
10 混音（对白轨 + 背景/音效轨，EBU R128 响度对齐）
11 导出（SRT/ASS + 混音音轨 + 可选合流）
```

**两条关键工程约束**（推断）：步骤 2/3 必须共用**同一时间基准**，分离/VAD 若改变样本数
会导致后续时间戳全部漂移；步骤 5 的说话人标签必须能回溯到**原始音频时间轴**，
而不是增强后的时间轴。

---

## 4 · 主要来源

- [modelscope/3D-Speaker（EER/DER 基准 + Apache-2.0）](https://github.com/modelscope/3D-Speaker)｜
  [pyannote/speaker-diarization-3.1 模型卡](https://huggingface.co/pyannote/speaker-diarization-3.1)｜
  [pyannote.audio #1963（4.0.3 显存 >9.54 GB vs 3.3.2 的 2.59 GB）](https://github.com/pyannote/pyannote-audio/issues/1963)
- [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)｜
  [NVIDIA NeMo 说话人模型](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html)
- [Rikorose/DeepFilterNet（48 kHz、MIT/Apache 双许可）](https://github.com/Rikorose/DeepFilterNet)｜
  [GTCRN 官方（48.2K 参数、RTF 0.07）](https://github.com/Xiaobin-Rong/gtcrn)｜
  [facebookresearch/demucs（SDR 对照、显存 3GB/7GB、MIT）](https://github.com/facebookresearch/demucs)｜
  [Mel-Band RoFormer 论文（MUSDB18HQ SDR 表）](https://ar5iv.labs.arxiv.org/html/2310.01809)｜
  [eka-care/when-denoising-hurts](https://github.com/eka-care/when-denoising-hurts)
- [Huanshere/VideoLingo（Apache-2.0、流水线分层、已知限制）](https://github.com/Huanshere/VideoLingo)｜
  [Descript 官方帮助：Overdub](https://help.descript.com/hc/en-us/articles/13832955606413-Using-Overdub-to-edit-recorded-audio)｜
  [WhisperX](https://pypi.org/project/whisperx/)
