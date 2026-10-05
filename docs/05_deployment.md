# 05 · 部署 / Deployment

> **什么时候读**：第一次在本机把模块跑起来、或换机器/换环境迁移时。
> 日常开发不需要读它。

---

## 1 · 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **≥ 3.12** | 本机：`C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe` |
| **ffmpeg / ffprobe** | **必须在 PATH** | 非 wav 输入、视频抽音轨、切片、转码全靠它。缺了只有 wav 能用 |
| GPU | 可选 | 有则 faster-whisper 走 CUDA；无则自动退 CPU（`cpu_fallback: true`） |
| 共享环境约束 | **不许动 `transformers`** | 它被 Agentic_TTS_Module 钉在 4.57.3，见 `00_research.md §2` |

本机实测环境（GPU / 已装依赖的完整清单）见 `00_research.md §2`，这里不重抄。

---

## 2 · 安装

```bash
conda activate ov_env_py312
cd F:\2026年\Agentic_ASR_Module

pip install -e ".[all]"        # 库 + server + gui + dev
```

按需装 / optional extras：

| extra | 装什么 | 用途 |
|---|---|---|
| （默认） | numpy / soundfile / PyYAML / pydantic / typer / rich / requests + **faster-whisper + ctranslate2 + av + sherpa-onnx** | 库与 CLI |
| `server` | fastapi / uvicorn / python-multipart | HTTP 服务 |
| `gui` | nicegui | 图形界面 |
| `dev` | pytest / httpx / ruff | 测试与静态检查 |

依赖注入很轻：**faster-whisper 只带进 `ctranslate2` + `av`**（不碰 torch），
情绪走 **sherpa-onnx**（+2 个包）。本项目**不声明 torch 依赖** —— 但 GPU 推理要借它的
CUDA 运行库，见 §4。

---

## 3 · 模型准备

一共四份（前三份是功能必需的，第四份只有测试用）。全部放在 `models/`，该目录已 gitignore。

| 本地目录 | 体积 | 用途 | 来源 |
|---|---|---|---|
| `models/faster-whisper-large-v3-turbo` | 1546 MB | **本地转写主力** | ModelScope `pengzhendong/faster-whisper-large-v3-turbo` |
| `models/sherpa-sensevoice` | 229 MB | **情绪 / 事件 / 语种** | HF `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` |
| `models/sherpa-vad/silero_vad.onnx` | 0.6 MB | VAD（分段、闸门、参考音频都靠它） | GitHub release `asr-models` |
| `models/SenseVoiceSmall` | 897 MB | **引擎不用它**：只提供 `example/*.mp3` 给 `-m real` 用例当测试素材 | ModelScope `iic/SenseVoiceSmall` |

> ⚠️ 别把 `SenseVoiceSmall` 和 `sherpa-sensevoice` 搞混：前者是 ModelScope 的原始
> PyTorch 权重（`model.pt`），**本项目不加载**；后者才是引擎读的 ONNX（`model.int8.onnx`）。

### 3.1 下载脚本

```python
# ① faster-whisper large-v3-turbo（ModelScope，国内直连）
from modelscope import snapshot_download
snapshot_download("pengzhendong/faster-whisper-large-v3-turbo",
                  local_dir="models/faster-whisper-large-v3-turbo")
```

```python
# ② SenseVoice ONNX（HF；只取两个必需文件，别拉整仓 1.1 GB）
from huggingface_hub import snapshot_download
snapshot_download("csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                  local_dir="models/sherpa-sensevoice",
                  allow_patterns=["model.int8.onnx", "tokens.txt"])
```

```python
# ③ Silero VAD（GitHub release）
# ⚠️ 用 python requests，不要用 curl.exe —— 见下方提示
import requests
url = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
       "asr-models/silero_vad.onnx")
r = requests.get(url, timeout=60)
r.raise_for_status()
open("models/sherpa-vad/silero_vad.onnx", "wb").write(r.content)
```

```python
# ④（可选）测试素材（ModelScope）
from modelscope import snapshot_download
snapshot_download("iic/SenseVoiceSmall", local_dir="models/SenseVoiceSmall")
```

### 3.2 两条下载纪律（都是实测踩出来的）

1. **GitHub release 必须用 python `requests` 下。**
   `curl.exe` 在 Windows 上会被 schannel 中断：
   `curl: (56) schannel: server closed abruptly (missing close_notify)`。
   实测同一 URL 换 `requests` 就正常。Agentic_VoiceSeparate_Module 的权重同理。
2. **HF 直连开 VPN 时可达**；不开时用镜像 —— 代码与文档都不写死任何一种网络状态：

   ```bash
   set HF_ENDPOINT=https://hf-mirror.com     # Windows cmd
   $env:HF_ENDPOINT="https://hf-mirror.com"  # PowerShell
   ```

   两种网络状态下的可达性实测表在 `00_research.md §3`。

---

## 4 · GPU 部署的前置条件（必读）

**CTranslate2 自己不带 `cublas64_12.dll` 与 `cudnn*.dll`。** 它们通常来自
`torch/lib`（torch 的 `__init__` 会 `os.add_dll_directory` 注册），或来自
`nvidia-*` pip 包。

不处理的话，GPU 转写会失败并抛出底层错误：

```
段 1 失败: 转写失败 / transcription failed:
Library cublas64_12.dll is not found or cannot be loaded
```

而且因为门面在**单段失败时只记 warning 并继续**（刻意的：长音频里某段失败不该让整次
请求崩掉），对外会表现为「转写结果为空」而不是一个显式错误 —— 很难定位。

**两种解法（任选）**：

```bash
# A. 装 torch（本机已有，torch 2.11.0+cu128），模块会自动找到 torch/lib
python -c "import torch; print(torch.__file__)"

# B. 不装 torch，只装 CUDA 运行库
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

本模块会在**加载 GPU 模型之前自动注册**上述目录（`agentic_asr/core/cuda.py`），
并把这个隐式依赖变成可诊断的一步：

```bash
asr doctor          # 看 cuda_runtime.dirs_with_cublas 是否非空
```

完整分析见 `issues/004`。

---

## 5 · 8 GB 显存策略

**关键事实：GPU 只被 ASR 主力占用，其余全在 CPU。** 实测占用数据见
`00_research.md §4`（faster-whisper ≈2.17 GB；SenseVoice 与 VAD 走 CPU，0 显存）。

- `engine.max_resident: 1`（默认）：换引擎时**先卸再载**并 `empty_cache()`；
- 一次请求里需要多引擎（转写 + 情绪补齐）时，门面**按引擎分组执行**，
  把轮动次数从「段数」压到「引擎数」；
- 加载前**预检显存**，不够就直接报错并指明该改哪一项，不让用户白等；
- **显存只能用 `nvidia-smi` / NVML 读**，不能用 `torch.cuda.memory_allocated()` ——
  CTranslate2 不走 torch 的 CUDA 分配器，后者全程返回 0（实测）；
- 显存被别的模块（如 TTS）占满时，可用云引擎兜底：`--engine minimax`。

---

## 6 · 服务化 / HTTP

```bash
python -m agentic_asr.server                 # http://127.0.0.1:8301/docs
asr serve --host 0.0.0.0 --port 8301         # 对外暴露
asr serve --reload                           # 开发热重载
```

端口来自 `configs/asr.yaml` 的 `server.port`（默认 8301），可用
`.env` 的 `ASR_SERVER_PORT` 覆盖。

端点与库 API 一一对应（完整清单见 `04_api_reference.md`）。
**每个功能都是独立 endpoint，可单独调用**，不必走完整流水线。

---

## 7 · 图形界面

```bash
python -m agentic_asr.gui                    # http://127.0.0.1:8401
asr gui                                      # 等价
```

默认端口 = `server.port + 100`（8301 → 8401）。界面结构、五个面板与手工验证步骤
见 `07_gui_guide.md`。

---

## 8 · 云 API 凭证

本地引擎**不需要任何凭证**。只有用 `--engine minimax` 或 `--engine openai_compat`
时才需要。凭证写在**本项目自己的** `.env`（不入库，模板见 `.env.example`）：

```ini
MINIMAX_API_KEY=            # POST /v1/speech_to_text，model=asr-1.0
ASR_API_ENDPOINT=           # 可选；留空用 configs/asr.yaml 的默认值

OPENAI_COMPAT_API_KEY=      # 通用 OpenAI 兼容端点
OPENAI_COMPAT_BASE_URL=     # 如 https://api.groq.com/openai/v1
OPENAI_COMPAT_MODEL=        # 如 whisper-large-v3
```

三条纪律：

1. **只从本项目自己的 `.env` 读**，不跨目录去读 Agentic_TTS_Module 的 `TTS_API_KEY`
   —— 那会让两个项目耦合，TTS 一改变量名就连带弄坏 ASR；
2. **MiniMax 端点归属有讲究**：实测本机 key 在 `api.minimaxi.com`（国内）通过鉴权、
   在 `api.minimax.io`（海外）返回 `401 invalid api key`。默认值已写对，别乱改；
3. 终端输出、日志、文档、截图里的密钥**一律打码**。

---

## 9 · 部署后自检

```bash
asr doctor              # 解释器 / 依赖 / ffmpeg / 三个模型 / CUDA 运行库 / checks
asr engines             # 引擎与能力清单（不加载权重）
asr probe 任意素材.mp4   # 走一遍真实解码
python tools/check.py   # 环境自检 + ruff + 全量离线测试
```

`doctor` 的 `checks` 字段是给人看的结论，例如：

- `faster-whisper 权重缺失：本地转写不可用（可先用 --engine mock）`
- `Silero VAD 缺失：将回退到能量 VAD（精度略低）`
- `未找到 cublas64_12.dll：GPU 推理会失败 —— 装 torch，或 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`

**完全没有模型也能先跑通链路**：`--engine mock` 是零依赖的确定性占位引擎，
用来验证界面、接口与配置，不加载任何权重。
