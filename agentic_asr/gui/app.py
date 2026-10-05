"""图形界面 / GUI（NiceGUI）。

**五个功能各占一个面板**，都能独立调用 —— 用户可以只做「信息提取」，
也可以只做「参考音频截取」，不必走完整流水线（用户明确要求的形态）。

耗时操作走 `run.io_bound`，避免卡住 UI 事件循环；模型加载是懒的，
所以界面能秒开。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agentic_asr import __version__
from agentic_asr.asr import ASRModule
from agentic_asr.core.config import ASRConfig, load_config
from agentic_asr.core.errors import ASRError


def _make_runner(module: ASRModule) -> Callable[..., Any]:
    """把异常转成可展示的文本 / wrap calls so the UI never dies on an error."""
    from nicegui import run as nice_run

    async def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        try:
            return True, await nice_run.io_bound(fn, *args, **kwargs)
        except ASRError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    return call


async def save_upload(file: Any, dest_dir: str | Path) -> Path:
    """把 NiceGUI 的上传对象存到目标目录 / persist a NiceGUI upload.

    **NiceGUI 3.x 的 API 与 2.x 不同**：`UploadEventArguments` 只有一个字段
    `file`，它是 `FileUpload`（带 `name` / `content_type` / **`await save(path)`**）；
    2.x 那种 `e.name` / `e.content` 已经没有了 —— 用户实测报过
    `'UploadEventArguments' object has no attribute 'name'`。

    文件名一律只取 basename，防止客户端在 `name` 里塞路径做目录穿越。
    """
    name = Path(str(getattr(file, "name", "") or "upload.bin")).name or "upload.bin"
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    await file.save(str(dest))
    return dest


def build_ui(config: ASRConfig | None = None) -> ASRModule:
    """搭出界面并返回门面 / build the page and return the facade."""
    from nicegui import events, ui

    cfg = config or load_config()
    module = ASRModule(cfg)
    call = _make_runner(module)
    state: dict[str, Any] = {"path": "", "transcript": None}

    # ── 顶栏 / header ───────────────────────────────────────────────────────
    with ui.header().classes("items-center justify-between q-px-md"):
        ui.label(f"Agentic ASR  {__version__}").classes("text-h6")
        engine_select = ui.select(
            {e["name"]: e["name"] for e in module.engines()},
            value=cfg.engine.default, label="引擎 / engine").props("dense outlined").classes("w-48")
        status = ui.label("就绪 / ready").classes("text-caption")

    def set_status(text: str, ok: bool = True) -> None:
        status.text = text
        status.classes(replace="text-caption " + ("text-positive" if ok else "text-negative"))

    # ── 素材输入 / media input ──────────────────────────────────────────────
    with ui.card().classes("w-full"):
        ui.label("素材 / media").classes("text-subtitle2")
        with ui.row().classes("items-center w-full"):
            path_input = ui.input("本地路径 / local path",
                                  placeholder=r"F:\path\to\audio.mp4").classes("grow")

            async def on_upload(e: events.UploadEventArguments) -> None:
                try:
                    dest = await save_upload(e.file, cfg.output_dir / "uploads")
                except Exception as exc:  # noqa: BLE001 - 上传失败不该让界面挂掉
                    set_status(f"上传失败 / upload failed: {exc}", ok=False)
                    return
                path_input.value = str(dest)
                set_status(f"已上传 / uploaded: {dest.name}")
                ui.notify(f"已上传 / uploaded: {dest.name}")

            ui.upload(on_upload=on_upload, auto_upload=True,
                      max_files=1).props('flat dense accept="audio/*,video/*"')

    def current_path() -> str | None:
        raw = (path_input.value or "").strip().strip('"')
        if not raw:
            set_status("请先指定素材 / pick a media file first", ok=False)
            return None
        if not Path(raw).exists():
            set_status(f"文件不存在 / not found: {raw}", ok=False)
            return None
        state["path"] = raw
        return raw

    # ── 五个面板 / five panels ──────────────────────────────────────────────
    with ui.tabs().classes("w-full") as tabs:
        t_info = ui.tab("① 信息提取")
        t_seg = ui.tab("② 音频分段")
        t_sub = ui.tab("③ 字幕生成")
        t_emo = ui.tab("④ 情绪识别")
        t_clip = ui.tab("⑤ 参考音频")

    with ui.tab_panels(tabs, value=t_info).classes("w-full"):
        # ① 信息提取
        with ui.tab_panel(t_info):
            info_out = ui.textarea(label="结果 / result").classes("w-full").props("rows=14 readonly")

            async def do_probe() -> None:
                if not (p := current_path()):
                    return
                set_status("探测中 / probing…")
                ok, data = await call(module.probe, p)
                info_out.value = (json.dumps(data.to_dict(), ensure_ascii=False, indent=2)
                                  if ok else data)
                set_status("完成 / done" if ok else "失败 / failed", ok)

            ui.button("提取信息 / probe", on_click=do_probe).props("unelevated")

        # ② 音频分段
        with ui.tab_panel(t_seg):
            with ui.row():
                seg_max = ui.number("单片上限(秒) / max seconds", value=cfg.segment.max_seconds,
                                    min=1, max=600).classes("w-40")
                seg_export = ui.checkbox("导出片段 / export wav", value=False)
            seg_out = ui.textarea(label="分段 / pieces").classes("w-full").props("rows=12 readonly")

            async def do_segment() -> None:
                if not (p := current_path()):
                    return
                set_status("分段中 / segmenting…")
                ok, pieces = await call(module.segment, p, float(seg_max.value or 30),
                                        bool(seg_export.value),
                                        cfg.output_dir / "gui_segments")
                if ok:
                    seg_out.value = json.dumps([x.to_dict() for x in pieces],
                                               ensure_ascii=False, indent=2)
                else:
                    seg_out.value = str(pieces)
                set_status(f"{len(pieces)} 段 / pieces" if ok else "失败 / failed", ok)

            ui.button("开始分段 / segment", on_click=do_segment).props("unelevated")

        # ③ 字幕生成
        with ui.tab_panel(t_sub):
            with ui.row():
                sub_lang = ui.input("语种 / language", value=cfg.transcribe.language).classes("w-32")
                sub_fmt = ui.select(["srt", "vtt", "ass"], value="srt",
                                    label="格式 / format").classes("w-32")
                sub_words = ui.checkbox("词级时间戳 / word timestamps", value=True)
            sub_out = ui.textarea(label="转写 / transcript").classes("w-full").props("rows=12 readonly")

            async def do_transcribe(with_emotion: bool) -> None:
                if not (p := current_path()):
                    return
                set_status("转写中 / transcribing…")
                srt_path = cfg.output_dir / "gui_subtitles" / f"{Path(p).stem}.{sub_fmt.value}"
                ok, result = await call(module.transcribe, p, engine=engine_select.value,
                                        language=sub_lang.value, words=bool(sub_words.value),
                                        emotion=with_emotion, srt=srt_path,
                                        srt_format=str(sub_fmt.value))
                if ok:
                    state["transcript"] = result
                    head = (f"[{result.engine}] lang={result.language} "
                            f"segs={len(result.segments)} emo={result.emotion_summary}\n"
                            f"字幕 / subtitle: {srt_path}\n"
                            f"warnings: {result.warnings}\n{'-' * 40}\n")
                    sub_out.value = head + (result.text or "（空 / empty）")
                    set_status("完成 / done")
                else:
                    sub_out.value = str(result)
                    set_status("失败 / failed", ok=False)

            with ui.row():
                ui.button("转写 / transcribe", on_click=lambda: do_transcribe(False)).props("unelevated")
                ui.button("转写 + 情绪 / with emotion",
                          on_click=lambda: do_transcribe(True)).props("outline")

        # ④ 情绪识别
        with ui.tab_panel(t_emo):
            emo_out = ui.textarea(label="逐段情绪 / per-segment emotion").classes("w-full").props("rows=12 readonly")

            async def do_emotion() -> None:
                if not (p := current_path()):
                    return
                set_status("识别中 / detecting…")
                ok, result = await call(module.transcribe, p, engine=engine_select.value,
                                        emotion=True, words=False)
                if ok:
                    rows = [{"start": s.start, "end": s.end, "emotion": s.emotion,
                             "text": s.text[:40]} for s in result.segments]
                    emo_out.value = (f"汇总 / summary: {result.emotion_summary}\n\n"
                                     + json.dumps(rows, ensure_ascii=False, indent=2))
                    set_status("完成 / done")
                else:
                    emo_out.value = str(result)
                    set_status("失败 / failed", ok=False)

            ui.button("识别情绪 / detect emotion", on_click=do_emotion).props("unelevated")

        # ⑤ 参考音频截取
        with ui.tab_panel(t_clip):
            with ui.row():
                clip_top = ui.number("取前 N 段 / top", value=cfg.clip.top, min=1, max=10).classes("w-32")
                clip_target = ui.number("目标时长(秒) / target", value=cfg.clip.target_seconds,
                                        min=3, max=30).classes("w-40")
            clip_out = ui.textarea(label="候选 / candidates").classes("w-full").props("rows=12 readonly")

            async def do_clip() -> None:
                if not (p := current_path()):
                    return
                set_status("截取中 / picking（首次会先转写）…")
                cfg.clip.target_seconds = float(clip_target.value or 12)
                ok, clips = await call(module.clip_reference, p,
                                       state.get("transcript"), int(clip_top.value or 3),
                                       cfg.output_dir / "gui_refs", "ref",
                                       engine_select.value)
                if ok:
                    rows = [c.to_dict() for c in clips]
                    clip_out.value = json.dumps(rows, ensure_ascii=False, indent=2)
                    for c in clips:
                        if c.wav_path and Path(c.wav_path).is_file():
                            ui.audio(Path(c.wav_path).as_posix()).classes("w-full")
                    set_status(f"{len(clips)} 段 / clips")
                else:
                    clip_out.value = str(clips)
                    set_status("失败 / failed", ok=False)

            ui.button("截取参考音频 / pick clips", on_click=do_clip).props("unelevated")

    ui.label("产物目录 / outputs: " + str(cfg.output_dir)).classes("text-caption q-pa-md")
    return module


def run(host: str = "127.0.0.1", port: int = 8401, config: str | None = None) -> None:
    """起界面 / launch the GUI."""
    from nicegui import ui

    build_ui(load_config(config))
    ui.run(host=host, port=port, title="Agentic ASR", reload=False, show=True)


__all__ = ["build_ui", "run"]
