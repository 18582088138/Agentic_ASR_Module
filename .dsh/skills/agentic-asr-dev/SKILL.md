---
name: agentic-asr-dev
description: Agentic_ASR_Module 开发约定 —— 分层边界、8 GB 约束、VAD/CUDA 陷阱、测试与提交纪律。改这个项目的代码前先读。
---

# Agentic_ASR_Module 开发 skill

## 分层边界（不要越界）

```
用法层  cli.py / server/ / gui/     三者调用同一个门面，接口一一对应
门面层  ASRModule (asr.py)          解码归一 · 切片+时间轴平移 · 引擎路由
                                    · 幻觉闸门 · 情绪补齐 · 字幕 · 参考音频
引擎层  engines/*                   唯一必实现：transcribe(chunk) -> Transcript
基础层  core/ audio/ media/ segment/ subtitle/ clip/
```

**五件事必须留在门面，不许下沉到引擎**：解码归一、切片与时间轴平移、字幕渲染、
幻觉闸门、情绪补齐。下沉就会出现 N 份互相不一致的实现。

新增引擎只需两步：实现 `transcribe()` + 声明 `declared_capabilities()`。
能力声明**必须不加载权重就能回答**（GUI 开机要用它置灰开关）。

## 硬约束

- **不许动 `transformers`**：共享环境里 TTS 钉在 4.57.3（`docs/00_research.md §1`）。
- **引擎只接受 `AudioChunk`，永不接受文件路径**（`issues/001`）。
- **显存只能用 `nvidia-smi`/NVML 读**：CTranslate2 与 onnxruntime 不走 torch 分配器，
  `torch.cuda.*` 全程返回 0。
- **模型一律懒加载**：构造引擎实例必须瞬时（GUI 开机要列引擎）。
- 单文件 ≤ 500 行；公共 API 与跨层接口中英双语注释，内部 helper 用中文。

## 三个静默陷阱（改代码前先看）

1. **VAD**：Silero 必须按 **512 样本分块喂**，并且**边喂边 `pop()`**。
   整段喂会漏检（占比从 0.78 掉到 0.06），不取会丢段（长音频只剩尾部）。`issues/003`
2. **幻觉**：无语音输入不进引擎（前置闸门），压缩比异常要标记（后置闸门）。
   置信度 `p=1.000` **不能**用于判伪。`issues/002`
3. **CUDA**：加载 GPU 模型前必须 `register_cuda_dlls()`，否则报
   `cublas64_12.dll is not found`。`issues/004`

## 测试纪律

- 默认只跑离线（`addopts = "-m 'not real and not live'"`）。
- 真实模型打 `real`，真实云调用打 `live`，都不进默认套件。
- 每条 issue 都要有回归用例，阈值**故意写死**（谁改回去谁立刻撞红）。
- 全量只在阶段收尾跑一次 `python tools/check.py`。

## 提交纪律

**AI 不执行 `git commit` / `git push`**。分步指令写到 `docs/08_git_commands.md`，人工执行。
commit 类型前缀用英文（feat/fix/docs/test/chore/refactor），正文用中文。
