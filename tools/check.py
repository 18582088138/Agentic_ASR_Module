"""收尾闸口 / the single project gate.

日常不跑；**阶段收尾或人工 review 时**才跑这一条：

    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe tools\\check.py

它做三件事 / three stages:
1. **环境自检** —— 解释器、关键依赖、ffmpeg、模型目录、CUDA 运行库；
2. **ruff** —— 静态检查（有则跑，缺则跳过并提示）；
3. **全量测试** —— 默认排除 `real` / `live` 标记的用例。

任一步失败则以非零码退出，方便挂到 CI 或 pre-push。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run(cmd: list[str], allow_fail: bool = False) -> bool:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0 and not allow_fail:
        print(f"[FAIL] 退出码 / exit code: {proc.returncode}")
        return False
    return True


# ── 1. 环境自检 / environment self-check ─────────────────────────────────────


def environment() -> bool:
    section("1/3 环境自检 / environment self-check")
    ok = True
    print(f"python      : {sys.version.split()[0]}  ({PY})")
    if sys.version_info < (3, 12):
        print("[FAIL] 需要 Python >= 3.12")
        ok = False

    print(f"ffmpeg      : {shutil.which('ffmpeg') or '[MISSING]'}")
    print(f"ffprobe     : {shutil.which('ffprobe') or '[MISSING]'}")
    if shutil.which("ffmpeg") is None:
        print("[WARN] 没有 ffmpeg：非 wav 输入、切片与转码将不可用")

    for name in ("numpy", "soundfile", "yaml", "pydantic", "typer", "requests"):
        try:
            mod = __import__(name)
            print(f"{name:<12}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"[FAIL] 缺少依赖 / missing: {name}")
            ok = False

    for name, extra in (("faster_whisper", "faster-whisper"), ("sherpa_onnx", "sherpa-onnx")):
        try:
            mod = __import__(name)
            print(f"{name:<12}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            # 这两个是引擎依赖，缺了会少功能但不至于让项目起不来
            print(f"[WARN] 缺少 {extra}：对应引擎不可用")

    try:
        from agentic_asr.core import cuda as cuda_info
        from agentic_asr.core.gpu import cuda_available, gpu_name, vram_free_mb

        print(f"gpu         : {gpu_name() or '(none)'}  free={vram_free_mb()} MB")
        info = cuda_info.describe()
        print(f"cublas      : {'OK' if info['cublas_available'] else '[MISSING]'}"
              f"  {info['dirs_with_cublas']}")
        if cuda_available() and not info["cublas_available"]:
            print(f"[WARN] {info['hint']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] GPU 自检失败 / GPU check failed: {exc}")

    try:
        from agentic_asr.core.config import load_config

        cfg = load_config(None)
        fw = cfg.faster_whisper_model_path()
        sv_model, _ = cfg.sensevoice_paths()
        vad = cfg.resolve_model_dir("models/sherpa-vad/silero_vad.onnx")
        for label, path in (("faster-whisper", fw), ("sensevoice", sv_model), ("silero-vad", vad)):
            print(f"model {label:<14}: {'OK' if path.exists() else '[MISSING]'}  {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置不可加载 / config failed: {exc}")
        ok = False
    return ok


# ── 2. ruff ──────────────────────────────────────────────────────────────────


def lint() -> bool:
    section("2/3 ruff 静态检查 / lint")
    try:
        import ruff  # noqa: F401
    except ImportError:
        pass
    proc = subprocess.run([PY, "-m", "ruff", "check", "."], cwd=str(ROOT), check=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "No module named" in (proc.stderr or ""):
        print("[SKIP] 未安装 ruff（pip install -e \".[dev]\"）")
        return True
    print(proc.stdout.strip() or "(无输出 / no output)")
    if proc.returncode != 0:
        print("[FAIL] ruff 发现问题")
        return False
    print("[OK] ruff 通过")
    return True


# ── 3. 测试 ──────────────────────────────────────────────────────────────────


def tests() -> bool:
    section("3/3 全量测试 / full test suite（默认排除 real / live）")
    return run([PY, "-m", "pytest", "tests", "-q"])


def main() -> int:
    results = {
        "环境自检 / environment": environment(),
        "ruff": lint(),
        "测试 / tests": tests(),
    }
    section("汇总 / summary")
    for name, passed in results.items():
        print(f"{'[OK]  ' if passed else '[FAIL]'} {name}")
    failed = [n for n, p in results.items() if not p]
    if failed:
        print(f"\n未通过 / failed: {', '.join(failed)}")
        return 1
    print("\n全部通过 / all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
