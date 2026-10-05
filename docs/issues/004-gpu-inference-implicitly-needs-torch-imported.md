# 004 · GPU 推理隐式依赖 torch 被导入（CTranslate2 不带 cublas）

**状态**：已修复（`agentic_asr/core/cuda.py` + `engines/faster_whisper.py`）
**发现**：开发阶段端到端测试，2026-10-04

---

## 现象 / Symptom

同一段 60 s 音频、同一个模型、同一台机器：

- 在 `scripts/_smoke_e2e.py` 里转写 → **正常**（9 段、113 字）；
- 在另一个脚本里转写 → **全部为空**，`segments=0`，只在 `warnings` 里留一句：

```
段 1 失败: 转写失败 / transcription failed:
Library cublas64_12.dll is not found or cannot be loaded
```

因为我的门面在**单段失败时只记 warning 并继续**（这是刻意的：长音频里某段失败不该
让整次请求崩掉），所以对外表现为「转写结果为空」而不是一个显式错误 —— 很难定位。

## 判定 / Diagnosis

两段脚本唯一的差别是：**前者在造测试素材时 `import torch` 了，后者没有。**

`ctranslate2` 的 CUDA 后端需要 `cublas64_12.dll` 与 `cudnn*.dll`，
而 **pip 装的 `ctranslate2` wheel 里不带这些库**。本机上它们来自
`torch/lib`（`torch/__init__.py` 会调用 `os.add_dll_directory` 把它注册进去）。

于是形成一条**隐式依赖**：GPU 推理能否工作，取决于别的代码有没有先 import torch。
而本项目在 `pyproject.toml` 里**刻意不依赖 torch**（faster-whisper 确实不需要它，
CTranslate2 是独立推理引擎），所以这条隐式依赖随时会断。

## 修法 / Fix

新增 `agentic_asr/core/cuda.py`，在**加载 GPU 模型之前**显式注册 CUDA 运行库目录：

1. `torch/lib`（本机实际来源）；
2. `site-packages/nvidia/*/{bin,lib}`（`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`）；
3. `CUDA_PATH` / `CUDA_HOME` 下的 `bin` 与 `lib/x64`。

`register_cuda_dlls()` 用 `lru_cache` 只做一次，并把实际注册的目录写进 debug 日志；
`faster_whisper` 引擎在 `device=cuda` 时调用它；`doctor()` 增加
`cuda_runtime` 段与一条可读的 check：

```
未找到 cublas64_12.dll：GPU 推理会失败 —— 装 torch，或 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## 回归用例 / Regression

`tests/test_cuda_runtime.py::test_register_returns_dirs`（离线）
—— 断言在 Windows 且装了 torch 时 `register_cuda_dlls()` 非空，
且 `describe()["cublas_available"]` 为 True。

`tests/test_faster_whisper_real.py::test_transcribe_without_torch_preimport`（`-m real`）
—— 在**不 import torch** 的子进程里跑一次转写，断言非空。
这条用例是专门为这个 bug 写的：它复现的正是「另一个脚本能用、这个不能用」。

## 备注

对应 `docs/05_deployment.md`（GPU 部署前置条件）与 `docs/00_research.md §2`。
