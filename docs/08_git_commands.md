# 08 · Git 提交指令汇总（人工执行）

> **AI 不执行 `git commit` / `git push`**，本文件交人工执行。
> 仓库已 `git init`、`git add` 过但**还没有任何 commit**（`master` 分支为空），
> 所以这是**首次提交**，按功能拆成 9 组。

**提交前务必确认**：

```bash
git status --short          # 不应出现 .env / models/ / outputs/ / *.onnx
git status --ignored --short | head -20    # 这些应当被忽略
```

看到 `.env` 或权重进了暂存区，说明 `.gitignore` 被改动过，**先停下来查**。

commit 类型前缀用英文（feat / fix / docs / test / chore / refactor），正文用中文。

---

## 步骤 1/9 · 工程骨架与依赖

```bash
git reset
git add pyproject.toml .gitignore .env.example tools/ .dsh/
git commit -m "chore: 工程骨架与依赖

- pyproject.toml: 依赖只加 faster-whisper 与 sherpa-onnx；av 钉 <19（见 issues/001）
- tools/check.py: 收尾闸口（环境自检 + ruff + 全量测试）
- .dsh/skills/agentic-asr-dev: 分层边界与三个静默陷阱"
```

## 步骤 2/9 · 调研、方案与项目记忆

```bash
git reset
git add README.md docs/00_INDEX.md docs/00_STAGE_SUMMARY.md docs/PROJECT_MEMORY.md docs/00_research.md docs/01_design.md docs/10_reference_research.md
git commit -m "docs: 调研、方案与项目记忆

- 00_research.md: API/Local ASR 选型与本机实测（RTF/显存/中英日识别、分离实测）
- 01_design.md: 两个平级插件、能力表、五个功能落点、验收标准 V1-V12；
  §3.5 记录字幕的六条切分规则与两个反直觉的坑
- 10_reference_research.md: 声音角色区分/降噪/配音技术栈（仅备查）
- PROJECT_MEMORY.md: 环境、常用命令、硬约束、踩过的六个坑"
```

## 步骤 3/9 · 核心层、音频层与配置

```bash
git reset
git add agentic_asr/__init__.py agentic_asr/core/ agentic_asr/audio/ configs/
git commit -m "feat(core): 配置、类型、异常、注册表与音频层

- 自控解码：引擎只接受 AudioChunk，永不接受文件路径（issues/001）
- VAD: Silero 必须分块喂且边喂边取，否则静默漏检/丢段（issues/003）
- cuda.py: 显式注册 CUDA 运行库，去掉对 torch 的隐式依赖（issues/004）
- gpu.py: 显存一律用 nvidia-smi 读，不用 torch.cuda.*
- SubtitleConfig: 断句参数（max_chars/min_chars/pause_gap/soft_gap/
  merge_below_seconds/max_seconds/chars_per_word）"
```

## 步骤 4/9 · 五个 ASR 引擎

```bash
git reset
git add agentic_asr/engines/
git commit -m "feat(engines): faster_whisper / sensevoice / minimax / openai_compat / mock

- faster_whisper: 本地主力（实测显存 2.17GB、长音频 RTF 0.037、词级时间戳）
- sensevoice: 情绪+事件走 sherpa-onnx 的 CPU，比 funasr 少 16 个包
- minimax: 专用适配器（词级时间戳 + 说话人分离）
- openai_compat: 一个适配器覆盖 OpenAI/Groq/硅基流动/OpenRouter
- mock: 零依赖离线占位，支撑不加载权重的全链路测试"
```

## 步骤 5/9 · 五个功能模块

```bash
git reset
git add agentic_asr/media/ agentic_asr/segment/ agentic_asr/subtitle/ agentic_asr/clip/
git commit -m "feat: 音频信息提取、音频分段、字幕生成、参考音频截取

- media/probe.py: ffprobe + 声学统计（估计 SNR 是降噪判据、语音占比是闸门判据）
- segment/splitter.py: 静音点优先，超长回退固定时长
- subtitle/segment.py: 六条切分规则（强/弱停顿、句末与次级标点、字数与时长上限）
  + 数字保护 + 合并过短尾条；**异常单元钳制**把 ASR 塞进单个字的静音
  还原成可见间隙（issues/007），否则字幕切不碎、一条挂十几秒
- subtitle/render.py: SRT/VTT/ASS（含词级卡拉 OK），一律本地渲染
- clip/picker.py: 五维打分挑 10~15s 参考音频；ref_text 必须来自该片段自身"
```

## 步骤 6/9 · 门面

```bash
git reset
git add agentic_asr/asr.py
git commit -m "feat: 门面 ASRModule 与幻觉闸门

- 切片 + 时间轴平移 + _clamp 裁剪（Whisper 会给出超出音频长度的尾段时间戳）
- 幻觉闸门：语音占比 + 压缩比双闸门，静音输入不进引擎（issues/002）
- 情绪补齐：主力引擎不支持情绪时按段调情绪引擎"
```

## 步骤 7/9 · 三种入口

```bash
git reset
git add agentic_asr/cli.py agentic_asr/__main__.py agentic_asr/server/ agentic_asr/gui/
git commit -m "feat: CLI / HTTP / GUI

- 三个入口共用同一个门面，每个功能都是独立命令 / 端点 / 面板
- GUI 上传按 NiceGUI 3.x 的契约实现（issues/006）
- /transcribe **不提供**「给字幕加说话人前缀」参数：说话人分离第一版只留能力位，
  放一个不生效的参数会误导调用方"
```

## 步骤 8/9 · 测试与调研探针

```bash
git reset
git add tests/ scripts/ docs/issues/
git commit -m "test: 离线套件 + 真实模型用例 + 七个问题的回归

- 每个 issues 单都有阈值故意写死的回归用例
- test_subtitle.py: 数字不被切坏、只去开头悬挂标点、异常单元变成可见间隙、
  max_seconds 真的兜得住时长
- test_gui.py: 把 NiceGUI 3.x 的上传事件契约钉死（事件对象没有 name 字段）
- scripts/_smoke_pipeline.py: 跨插件（分离 → ASR）端到端
- scripts/_smoke_subtitle.py: 字幕切分的效果对比，调参数前先跑它"
```

## 步骤 9/9 · 使用与过程文档

```bash
git reset
git add docs/02_dev_plan.md docs/03_unit_tests.md docs/04_api_reference.md \
        docs/05_deployment.md docs/06_add_engine.md docs/07_gui_guide.md \
        docs/08_git_commands.md docs/09_dev_log.md
git commit -m "docs: 用法、部署、测试、扩展与开发记录

- 04_api_reference: 库 / HTTP / CLI 完整接口参考
- 05_deployment: 模型下载、GPU 前置（cublas）、8GB 显存策略
- 02/09: 与计划的偏差（分离迁出、情绪走 sherpa、CUDA 隐式依赖、NiceGUI API）
- 07_gui_guide: 界面用法与手工验证清单"
```

---

## 核对

```bash
git status          # 应当干净
git log --oneline   # 9 条提交
```

若仍有未提交文件，先判断它是不是本该被忽略的产物：
`git status --ignored --short` 会列出来（`outputs/`、`models/` 属正常忽略）。
