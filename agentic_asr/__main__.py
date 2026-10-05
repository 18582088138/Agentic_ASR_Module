"""CLI 入口 / console entry point.

`pyproject.toml` 的 `[project.scripts] asr = "agentic_asr.cli:main"` 指向这里，
所以 `python -m agentic_asr` 与安装后的 `asr` 命令行为一致。
"""

from __future__ import annotations

from agentic_asr.cli import main

if __name__ == "__main__":
    main()
