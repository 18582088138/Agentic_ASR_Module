# 项目记忆 / Project memory

> 新会话接手先读这里，再读 `docs/00_INDEX.md` 指向的当前阶段文档。
> **同一事实只在一处写权威**，这里只记「不看代码就猜不到」的东西。

## 当前状态（2026-10）

**开发已完成**，`tools/check.py` 全绿（数字现跑，别信任何写死的条数）。

本仓库是**声音转写模组**（`agentic_asr`）。它与**声音分离模组**是两个相互独立、
**谁都不 import 谁**的模组：不同包、不同端口、不同模型目录、不同 `.env`。
要一起用由**应用层**组合，跨模组的脚本统一放 `F:\2026年\Agentic_Pipelines\`
（见 `docs/01_design.md §4` 与 `~/.dsh/AGENTS.md` §七）。

剩下的是**人工动作**，AI 不做：

| 待办 | 看哪 |
|---|---|
| git 提交（分步指令） | `docs/08_git_commands.md` |
| 真实素材验收：日语质量、参考音频是否够干净 | `docs/07_gui_guide.md` 的手工验证清单 |
| 决定「先分离再转写」是否值得（需真实影视素材） | `docs/00_research.md §4.2` |

**接手顺序**：本文件 → `docs/00_INDEX.md` → 按需读一份。
**改代码前必读** `.dsh/skills/agentic-asr-dev/SKILL.md`（分层边界 + 三个静默陷阱）。

## 环境

| 项 | 值 |
|---|---|
| Python | `C:\Users\75203\miniforge3\envs\ov_env_py312\python.exe`（3.12.14） |
| GPU | RTX 4060 8 GB（8188 MiB），driver 595.79 |
| ffmpeg / ffprobe | `C:\Users\75203\Documents\ffmpeg-master\bin`（**已在 PATH**） |
| torch | 2.11.0+cu128（**本项目不依赖它**，但 GPU 推理要借它的 CUDA 运行库） |
| transformers | 4.57.3 —— **被 Agentic_TTS_Module 钉死，本项目不许改** |

> ⚠️ **Windows 上处理本仓库的中文文件，一律用 Python 按 UTF-8 读写。**
> PowerShell 的 `Get-Content` / `Select-String` 按 GBK 读 UTF-8 会乱码，
> 行数统计也会错（把 231 行报成 185 行）；`Set-Content` 回写更会**直接毁掉文件**
> （本会话踩过一次，一个探针脚本被写成乱码）。
> Read and write this repo's Chinese files with Python + UTF-8, never PowerShell text cmdlets.

## 常用命令

```bash
cd F:\2026年\Agentic_ASR_Module
python -m pytest tests -q                # 离线（秒级）
python -m pytest tests -q -m real        # 真实模型（约 30 秒）
python -m pytest tests -q -m live        # 真实云 API（会花钱）
python tools/check.py                    # 收尾闸口：环境自检 + ruff + 全量测试

asr doctor / engines / probe / segment / transcribe / clip / serve / gui
```

跨模组的脚本**不在本仓库** —— 它们属于应用层，放 `F:\2026年\Agentic_Pipelines\`
（见 `docs/01_design.md §4`）。本模组自己的命令全在上面这一块。

## 实测数字（不要凭印象改默认值）

| 项 | 值 | 出处 |
|---|---|---|
| faster-whisper turbo 显存 | **+2.17 GB** | `docs/00_research.md §4` |
| faster-whisper RTF | 短音频 0.073~0.131；185 s 长音频 **0.037** | 同上 |
| sherpa-onnx UVR 分离 RTF | KARA **0.232** / KARA_2 0.363（**CPU**，零显存） | §4.4 |
| demucs htdemucs（对照） | RTF 0.740（GPU）、+861 MB | §4.2 |

## 六条踩过的坑（都有 issues 单与回归用例）

1. **faster-whisper 1.2.1 与 av 19 不兼容** → 自己做解码层，永不把文件路径交给引擎（`issues/001`）
2. **纯音乐轨会被编出假字幕** → 语音占比 + 压缩比双闸门（`issues/002`）
3. **sherpa-onnx 的 Silero VAD 必须分块喂、且边喂边取** → 否则静默漏检/丢段（`issues/003`）
4. **GPU 推理隐式依赖 torch 被导入**（CTranslate2 不带 cublas）→ 显式注册 CUDA 运行库（`issues/004`）
5. **分离轨道顺序不能硬编码**，官方示例与实测不一致 → 三级判定 + 回传依据（分离项目 `issues/005`）
6. **NiceGUI 3.x 的上传事件没有 `name`**（那是 2.x 的写法）→ 用 `e.file.name` +
   `await e.file.save(path)`（`issues/006`）

## 一个反模式（本项目刻意避免的）

**不留「不生效的参数/配置」。** 开发中先后删掉过两个：HTTP `/transcribe` 的
`speakers` 参数（说话人分离未实现）与 `transcribe.separate` 配置项（代码从没读过它）。
它们不会报错，只会让调用方以为设了就有效 —— 比缺功能更糟。

## 不可违反的约束

- **不许动共享环境的 `transformers`**（TTS 钉在 4.57.3）。Qwen3-ASR 会把它拉到 4.57.6，
  所以本地主力选 faster-whisper（详见 `docs/00_research.md §1`）。
- **凭证只从本项目自己的 `.env` 读**，不跨目录读 TTS 的 key；日志/文档/截屏里的密钥一律打码。
- **AI 不执行 `git commit` / `git push`**，指令写在 `docs/08_git_commands.md` 里交人工执行。
- **GPU 下载模型用 python `requests`**，`curl.exe` 在 Windows 上会被 schannel 中断。
