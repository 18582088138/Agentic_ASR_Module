"""GUI 上传路径 / GUI upload handling.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_gui.py -q

覆盖 / covers: NiceGUI 3.x 的上传事件契约（`UploadEventArguments.file`）、
文件名防目录穿越。**不含**真实浏览器交互（那是手工验证）。
说明文档 / docs: docs/issues/006
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("nicegui")


def _upload(name: str, data: bytes = b"RIFFfakewave"):
    """造一个 NiceGUI 的 FileUpload 对象 / build a NiceGUI FileUpload."""
    from nicegui.elements.upload_files import SmallFileUpload

    return SmallFileUpload(name=name, content_type="audio/wav", _data=data)


def test_save_upload_writes_file(tmp_path: Path) -> None:
    """回归：NiceGUI 3.x 的 `UploadEventArguments` 只有 `file` 字段。

    用户实测报过 `'UploadEventArguments' object has no attribute 'name'` ——
    那是按 NiceGUI 2.x 的 API 写的（2.x 有 `e.name` / `e.content`）。
    这条用例锁住 3.x 的契约：只能通过 `e.file.save(path)` 落盘。
    """
    from agentic_asr.gui.app import save_upload

    dest = asyncio.run(save_upload(_upload("demo.wav"), tmp_path))
    assert dest.is_file()
    assert dest.name == "demo.wav"
    assert dest.read_bytes() == b"RIFFfakewave"


def test_save_upload_strips_client_path(tmp_path: Path) -> None:
    """客户端可能在 name 里塞路径，必须只取 basename / prevent path traversal."""
    from agentic_asr.gui.app import save_upload

    dest = asyncio.run(save_upload(_upload("../../evil.wav"), tmp_path))
    assert dest.parent == tmp_path
    assert dest.name == "evil.wav"


def test_save_upload_handles_missing_name(tmp_path: Path) -> None:
    from agentic_asr.gui.app import save_upload

    dest = asyncio.run(save_upload(_upload(""), tmp_path))
    assert dest.is_file()
    assert dest.name == "upload.bin"


def test_upload_event_arguments_shape() -> None:
    """把框架契约本身也断言下来 —— 将来 NiceGUI 再改 API 会先撞到这里。

    关键的回归点是 **`name` 不在事件对象上**：用户实测报的
    `'UploadEventArguments' object has no attribute 'name'` 正是照着
    NiceGUI 2.x 的写法（2.x 的 `e.name` / `e.content`）写的。
    """
    import dataclasses

    from nicegui import events

    fields = {f.name for f in dataclasses.fields(events.UploadEventArguments)}
    assert "file" in fields, f"上传事件里没有 file 字段：{fields}"
    assert "name" not in fields, (
        f"事件对象重新有了 name 字段（{fields}）—— 请确认 NiceGUI 版本并删掉本断言")
    assert "content" not in fields, f"事件对象重新有了 content 字段（{fields}）"

    from nicegui.elements.upload_files import FileUpload

    for method in ("save", "read", "size"):
        assert hasattr(FileUpload, method), f"FileUpload 缺少 {method}"


def test_gui_module_imports() -> None:
    """模块能导入即可（真正渲染需要浏览器上下文，属手工验证）。"""
    from agentic_asr.gui import app as gui_app

    assert callable(gui_app.build_ui)
    assert callable(gui_app.run)
