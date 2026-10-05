# 00 · 文档索引 / Index

**默认只读本文件，再按当前阶段读一份。** 不要一次加载整个 `docs/`。

| 何时 | 读哪份 | 里面是什么 |
|---|---|---|
| **接手项目（第一步）** | `PROJECT_MEMORY.md` | 环境、常用命令、硬约束、踩过的六个坑 —— 不看代码就猜不到的东西 |
| 当前进度 | `00_STAGE_SUMMARY.md` §〇 | 已完成什么、下一步、待决事项 |
| 调研 | `00_research.md` | ASR 选型结论、**本机实测数据**、环境事实、未确认项 |
| 方案 | `01_design.md` | 两个插件、能力表、**五个功能**怎么落、接口与验收标准 |
| 开发过程 | `02_dev_plan.md` | 阶段划分与**实际完成情况**（回顾式）、四处与计划的偏差 |
| 测试 | `03_unit_tests.md` | 用例分布、刻意写死的断言、复测命令 |
| **用法** | `04_api_reference.md` | **库 / HTTP / CLI 完整接口参考**（参数、返回、示例、错误码） |
| 部署 | `05_deployment.md` | 模型下载、GPU 前置条件、8 GB 策略、服务化 |
| 接新引擎 | `06_add_engine.md` | 引擎抽象四步与五条硬约束 |
| GUI | `07_gui_guide.md` | 界面使用说明与手工验证清单 |
| 提交 | `08_git_commands.md` | 分步 git 指令（**人工执行**） |
| 开发记录 | `09_dev_log.md` | 交付内容、测试结果、与方案的偏差、后续待办 |
| **技术备查** | `10_reference_research.md` | 声音角色区分 / 音频降噪 / 配音软件技术栈 —— **仅参考，不驱动方案** |
| 问题单 | `issues/NNN-*.md` | 现象 / 判定 / 修法 / 回归用例。**本仓库是 001–004 与 006**（005 属分离插件） |

## 项目一句话

Agentic_ASR_Module —— 两个**平级的、可独立部署与独立调用的 Agent 插件**：

| 插件 | 包 | 功能 | 端口 |
|---|---|---|---|
| **ASR** | `agentic_asr` | 音频信息提取 · **音频分段** · 字幕生成 · 语音情绪识别 · 参考音频截取 | 8301 |
| **分离** | `agentic_separate` | 人声提取 · 背景音提取 · **主体人声去除** | 8302 |

每个功能都能在 service 与 GUI 中作为**独立 function** 调用（不必走完整流水线）。
两者能力正交，所以是两套平级抽象：一个答「说了什么」，一个答「哪些是人声」。

## 三条不可违反的约束

1. **轻量化**：不得引入新的重依赖。本地 ASR 主力只加 2 个包（faster-whisper），
   情绪与分离全部走**已装的 sherpa-onnx + CPU**（零新增依赖、零显存）。
2. **共享 conda 环境** `ov_env_py312` 与 Agentic_TTS_Module 共用，
   改动依赖前先看 `00_research.md §2`（`transformers` 被 TTS 钉在 4.57.3）。
3. **模型源**：HF 直连**开 VPN 时可达**，不开时走 `HF_ENDPOINT=https://hf-mirror.com`
   或 ModelScope；sherpa-onnx 的模型走 GitHub release，且**必须用 python `requests` 下**
   （`curl.exe` 会被 Windows schannel 中断，实测）。代码按「先直连、失败退镜像」写。
