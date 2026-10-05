"""GPU / 显存探测 / GPU and VRAM probing.

**为什么不用 `torch.cuda.*` 读显存**：本项目的推理全部不走 torch 的 CUDA 分配器
（CTranslate2 自管、sherpa-onnx 走 onnxruntime），实测
`torch.cuda.max_memory_allocated()` 全程返回 0 —— 拿它做显存预检等于没检。
所以这里一律问驱动。
We ask the driver, because CTranslate2 and onnxruntime do not allocate through
PyTorch, so `torch.cuda.*` reports zero (measured).

依赖 / dependency：只用 `nvidia-smi`，不引入 pynvml。
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

_NVIDIA_SMI = shutil.which("nvidia-smi")


def _query(field: str) -> list[float]:
    if _NVIDIA_SMI is None:
        return []
    try:
        out = subprocess.run(
            [_NVIDIA_SMI, f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False).stdout
    except Exception:  # noqa: BLE001 - 探测失败不该让业务崩
        return []
    vals: list[float] = []
    for line in out.strip().splitlines():
        try:
            vals.append(float(line.strip()))
        except ValueError:
            continue
    return vals


def nvidia_smi_available() -> bool:
    return _NVIDIA_SMI is not None


@lru_cache(maxsize=1)
def gpu_name() -> str:
    if _NVIDIA_SMI is None:
        return ""
    try:
        out = subprocess.run([_NVIDIA_SMI, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10,
                             check=False).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return ""
    return out[0].strip() if out else ""


def vram_used_mb() -> float | None:
    """当前已用显存（MB）/ currently used VRAM in MB."""
    vals = _query("memory.used")
    return vals[0] if vals else None


def vram_total_mb() -> float | None:
    vals = _query("memory.total")
    return vals[0] if vals else None


def vram_free_mb() -> float | None:
    """剩余显存（MB）/ free VRAM in MB."""
    used, total = vram_used_mb(), vram_total_mb()
    if used is None or total is None:
        return None
    return max(0.0, total - used)


def cuda_available() -> bool:
    """CUDA 是否可用（从 CTranslate2 的角度）/ whether CTranslate2 can see CUDA.

    用 CTranslate2 的设备计数而不是 torch：ASR 主力就是 CTranslate2，
    torch 可能压根没装。
    """
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def check_headroom(need_mb: float, safety_mb: float = 400.0) -> tuple[bool, str]:
    """加载前预检显存 / pre-flight check before loading weights.

    返回 / Returns:
        (是否够用, 人话说明)。读不到显存时返回 (True, "...")，即**不阻断** ——
        探测不到不代表不够，宁可让加载自己报错，也不要凭空拒绝。
    """
    free = vram_free_mb()
    if free is None:
        return True, "无法读取显存（nvidia-smi 不可用），跳过预检"
    ok = free >= need_mb + safety_mb
    msg = (f"剩余显存 {free:.0f} MB，需要约 {need_mb:.0f} MB"
           f"（安全余量 {safety_mb:.0f} MB）")
    return ok, msg


__all__ = [
    "check_headroom",
    "cuda_available",
    "gpu_name",
    "nvidia_smi_available",
    "vram_free_mb",
    "vram_total_mb",
    "vram_used_mb",
]
