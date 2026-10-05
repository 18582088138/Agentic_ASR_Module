"""`python -m agentic_asr.gui` —— 起图形界面。"""

from __future__ import annotations

from agentic_asr.core.config import load_config
from agentic_asr.gui.app import run

if __name__ == "__main__":
    cfg = load_config()
    run(cfg.server.host, cfg.server.port + 100)
