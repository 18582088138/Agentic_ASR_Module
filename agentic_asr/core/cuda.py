"""CUDA 运行库定位 / CUDA runtime library discovery.

Windows 上 CTranslate2 需要 `cublas64_12.dll` 与 `cudnn*.dll`，但**它自己不带**，
pip 装的 `ctranslate2` wheel 里没有这些。它们通常来自：

1. `torch` 的 `lib/` 目录（torch 的 `__init__` 会 `os.add_dll_directory`，所以
   **只要先 import torch，一切都正常**）；
2. `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` 等 pip 包的 `bin/` 目录；
3. 系统 CUDA Toolkit（`CUDA_PATH`）。

**实测现象**（不做这件事的后果）：

- 先用 `import torch` 再造素材，然后转写 → 正常；
- 不 import torch 直接转写 → `转写失败: Library cublas64_12.dll is not found`。

也就是说 **GPU 推理隐式依赖 torch 被导入**。本项目刻意不把 torch 列为依赖
（faster-whisper 确实不需要它），所以必须把这个隐式依赖变成**显式、可诊断**的一步。
This module turns an implicit torch side effect into an explicit step.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from agentic_asr.core.logging import get_logger

log = get_logger("core.cuda")


def candidate_dirs() -> list[Path]:
    """按优先级列出可能含 CUDA 运行库的目录 / candidate directories."""
    dirs: list[Path] = []

    # 1) torch 自带的 CUDA 运行库（本机最常见的来源）
    try:
        import torch

        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:  # noqa: BLE001 - torch 没装是正常情况
        pass

    # 2) pip 装的 nvidia-* 运行时包
    for base in (Path(sys.prefix) / "Lib" / "site-packages",
                 Path(sys.prefix) / "lib" / f"python{sys.version_info.major}"
                 f".{sys.version_info.minor}" / "site-packages"):
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for sub in sorted(nvidia.iterdir()):
                for name in ("bin", "lib"):
                    d = sub / name
                    if d.is_dir():
                        dirs.append(d)

    # 3) 系统 CUDA Toolkit
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env_name)
        if root:
            dirs.append(Path(root) / "bin")
            dirs.append(Path(root) / "lib" / "x64")

    # 去重且保留顺序 / dedupe, keep order
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d).lower()
        if key not in seen and d.is_dir():
            seen.add(key)
            out.append(d)
    return out


def has_cublas(directory: Path) -> bool:
    """该目录里有没有 cuBLAS / whether cuBLAS lives here."""
    if not directory.is_dir():
        return False
    for pattern in ("cublas64_*.dll", "libcublas.so*", "libcublas.*.dylib"):
        if any(directory.glob(pattern)):
            return True
    return False


@lru_cache(maxsize=1)
def register_cuda_dlls() -> list[str]:
    """把 CUDA 运行库目录注册进 DLL 搜索路径 / register CUDA dirs for the loader.

    返回实际注册成功的目录（便于日志与排障）。在非 Windows 平台上是空操作。
    Returns the directories actually registered; a no-op off Windows.
    """
    if sys.platform != "win32":
        return []
    added: list[str] = []
    for d in candidate_dirs():
        try:
            os.add_dll_directory(str(d))
            added.append(str(d))
        except (OSError, AttributeError) as exc:  # pragma: no cover
            log.debug("注册 DLL 目录失败 / cannot register %s: %s", d, exc)
    if added:
        log.debug("已注册 CUDA DLL 目录 / registered: %s", "; ".join(added))
    return added


def describe() -> dict[str, object]:
    """给 `doctor` 用的诊断快照 / a diagnostic snapshot for `doctor`."""
    dirs = candidate_dirs()
    cublas = [str(d) for d in dirs if has_cublas(d)]
    return {
        "platform": sys.platform,
        "torch_cuda_dirs": [str(d) for d in dirs if "torch" in str(d).lower()],
        "all_candidate_dirs": [str(d) for d in dirs],
        "dirs_with_cublas": cublas,
        "cublas_available": bool(cublas),
        "hint": ("未找到 cublas64_12.dll：GPU 推理会失败。"
                 "装 torch，或 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        if not cublas else "",
    }


__all__ = ["candidate_dirs", "describe", "has_cublas", "register_cuda_dlls"]
