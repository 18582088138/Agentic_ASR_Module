# Agentic ASR Module

可插拔的**本地优先** ASR 服务插件。五个功能，每个都能在库 / CLI / HTTP / GUI 中
**独立调用**，不必走完整流水线：

| 功能 | 说明 |
|---|---|
| ① 音频信息提取 | ffprobe 容器与流信息 + 声学统计（峰值 / RMS / 估计 SNR / 语音占比 / 削波） |
| ② 音频分段 | 静音点优先，超长回退固定时长；出时间轴清单，避免长音频 OOM |
| ③ 字幕生成 | SRT / VTT / ASS（含词级卡拉 OK），一律本地渲染，不依赖厂商导出 |
| ④ 语音情绪识别 | 逐段 7 类情绪 + 事件检测 + 全片汇总 |
| ⑤ 参考音频截取 | 自动挑 10~15 s 干净人声 → wav + 字幕，直接喂 Agentic_TTS_Module 做音色克隆 |

> 定位：**输入音频或视频，输出带时间戳的字幕、逐段情绪、以及可直接用于 TTS 的参考音频素材。**
> 不做翻译、配音合成、视频剪辑、音色克隆本身。

**与声音分离模组的关系**：两者能力正交（一个答「说了什么」，一个答「哪些声音是哪个
成分」），是两个**相互独立、谁都不依赖谁**的模组。要一起用由**应用层**组合 ——
跨模组的脚本不放在任一模组里，统一放 `F:\2026年\Agentic_Pipelines\`。

---

## 快速开始

```bash
conda activate ov_env_py312
cd Agentic_ASR_Module
pip install -e ".[all]"

# 体检：解释器、依赖、ffmpeg、模型、CUDA 运行库
asr doctor

# 不加载权重先试界面/接口（零依赖的 mock 引擎）
asr transcribe demo.mp3 --engine mock

# 真实转写 + 字幕
asr transcribe demo.mp4 --srt outputs/demo.srt

# 情绪识别
asr transcribe demo.mp4 --emotion

# 参考音频截取（喂 TTS 音色克隆）
asr clip demo.mp4 --top 3 --out-dir outputs/refs

# 图形界面 / HTTP 服务
python -m agentic_asr.gui          # http://127.0.0.1:8401
python -m agentic_asr.server       # http://127.0.0.1:8301/docs
```

## 库用法

```python
from agentic_asr import ASRModule

asr = ASRModule()                                   # 读 configs/asr.yaml

info   = asr.probe("demo.mp4")                      # ① 信息提取
pieces = asr.segment("demo.mp4", max_seconds=30)    # ② 音频分段
result = asr.transcribe("demo.mp4", emotion=True)   # ③④ 字幕 + 情绪
result_dict = result.to_dict()
clips  = asr.clip_reference("demo.mp4", top=3)      # ⑤ 参考音频
print(clips[0].wav_path, clips[0].text)             # -> ref_audio / ref_text

asr.release()                                       # 卸权重、清显存
```

## 引擎

| 引擎 | 设备 | 说明 |
|---|---|---|
| `faster_whisper`（默认） | GPU 2.17 GB | 中英日多语种 + 词级时间戳 |
| `sensevoice` | **CPU（0 显存）** | 情绪 7 类 + 事件 + 语种；走 sherpa-onnx |
| `minimax` | 云 | 词级时间戳 + **说话人分离**；500 s / 50 MB 上限 |
| `openai_compat` | 云 | 一个适配器覆盖 OpenAI / Groq / 硅基流动 / OpenRouter |
| `mock` | — | 零依赖离线占位，用于界面预览与测试 |

**实测性能**（RTX 4060 8 GB，`large-v3-turbo` + `float16` + `beam_size=5`）：
显存 **+2.17 GB**；短音频 RTF 0.073~0.131，**185 s 长音频 RTF 0.037**（≈1 小时音频 2.2 分钟）。
情绪走 CPU，不占显存。

## 目录

```
agentic_asr/
  core/       配置、类型、异常、日志、注册表、GPU/CUDA 运行库定位
  engines/    faster_whisper | sensevoice | minimax | openai_compat | mock
  media/      probe.py         音频/视频信息提取
  audio/      io / vad / features   解码归一、语音活动检测、声学统计
  segment/    splitter.py      音频分段
  subtitle/   render / segment 字幕断句与渲染
  clip/       picker.py        参考音频截取
  asr.py      门面 ASRModule —— 唯一必须记住的类
  server/ gui/ cli.py
configs/asr.yaml    tests/    tools/check.py    docs/
```

## 测试

```bash
python -m pytest tests -q              # 离线，秒级，不加载任何权重
python -m pytest tests -q -m real      # 真实模型，手动触发（每条几十秒）
python tools/check.py                  # 收尾闸口：环境自检 + ruff + 全量测试
```

## 文档

| 文件 | 内容 |
|---|---|
| `docs/00_INDEX.md` | **入口**：哪份文档什么时候加载 |
| `docs/00_research.md` | 调研：选型、**本机实测数据**、未确认项 |
| `docs/01_design.md` | 方案：分层、能力表、功能落点、接口、验收标准 |
| `docs/10_reference_research.md` | 技术参考：声音角色区分 / 降噪 / 配音软件技术栈（仅备查） |
| `docs/issues/` | 开发中发现的四个问题（含回归用例） |

## 环境要求

- Python ≥ 3.12（本机 `C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe`）
- **ffmpeg / ffprobe 必须在 PATH 上**（非 wav 输入、切片、转码都要它）
- GPU 可选；用 GPU 时 CTranslate2 需要 `cublas64_12.dll` —— 本模块会自动定位
  torch 或 `nvidia-*` pip 包里的 CUDA 运行库（见 `docs/issues/004`）
