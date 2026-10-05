"""命令行 / Command line interface.

`asr <命令>`，与库 API、HTTP 路由**一一对应**：
每个功能都能单独调用，不必走完整流水线。

    asr doctor                       # 体检
    asr engines                      # 引擎与能力清单
    asr probe   <media>              # ① 音频信息提取
    asr segment <media>              # ② 音频分段
    asr transcribe <media>           # ③④ 字幕 + 情绪
    asr clip    <media>              # ⑤ 参考音频截取
    asr serve / asr gui
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentic_asr import __version__
from agentic_asr.asr import ASRModule
from agentic_asr.core.config import load_config
from agentic_asr.core.errors import ASRError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Agentic_ASR_Module —— 本地优先的 ASR 服务插件")
console = Console()


def _module(engine: str | None = None, config: str | None = None) -> ASRModule:
    return ASRModule(config=load_config(config), engine=engine)


def _fail(exc: Exception) -> None:
    console.print(f"[red]错误 / error:[/red] {exc}")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """打印版本 / print the version."""
    console.print(f"agentic-asr {__version__}")


@app.command()
def doctor(config: str = typer.Option(None, "--config", help="配置文件路径")) -> None:
    """环境与模型体检 / environment self-check（不加载权重）."""
    report = _module(config=config).doctor()
    console.print_json(json.dumps(report, ensure_ascii=False, default=str))


@app.command()
def engines() -> None:
    """列出引擎与能力 / list engines and their capabilities."""
    table = Table(title="ASR 引擎 / engines")
    table.add_column("引擎"), table.add_column("默认"), table.add_column("能力")
    for item in _module().engines():
        table.add_row(item["name"], "✓" if item.get("default") else "",
                      ", ".join(item.get("capabilities") or []) or (item.get("error") or ""))
    console.print(table)


@app.command()
def probe(media: Path,
          deep: bool = typer.Option(True, help="是否做声学统计（要解码，慢）"),
          config: str = typer.Option(None, "--config")) -> None:
    """① 音频信息提取 / probe media."""
    try:
        info = _module(config=config).probe(media, deep=deep)
    except ASRError as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(info.to_dict(), ensure_ascii=False, default=str))


@app.command()
def segment(media: Path,
            max_seconds: float = typer.Option(None, help="单片上限（秒）"),
            export: bool = typer.Option(False, "--export", help="同时导出每段 wav"),
            out_dir: Path = typer.Option(None, "--out-dir"),
            config: str = typer.Option(None, "--config")) -> None:
    """② 音频分段 / split audio into pieces."""
    try:
        asr = _module(config=config)
        pieces = asr.segment(media, max_seconds=max_seconds, export_audio=export,
                             out_dir=out_dir)
    except ASRError as exc:
        _fail(exc)
        return
    table = Table(title=f"分段 / pieces ({len(pieces)})")
    for col in ("#", "start", "end", "dur", "hard_cut"):
        table.add_column(col)
    for p in pieces:
        table.add_row(str(p.index), f"{p.start:.3f}", f"{p.end:.3f}",
                      f"{p.duration:.3f}", "是" if p.hard_cut else "")
    console.print(table)


@app.command()
def transcribe(media: Path,
               engine: str = typer.Option(None, help="引擎名"),
               language: str = typer.Option(None, help="auto | zh | en | ja …"),
               words: bool = typer.Option(None, "--words/--no-words", help="词级时间戳"),
               emotion: bool = typer.Option(None, "--emotion/--no-emotion"),
               srt: Path = typer.Option(None, "--srt", help="字幕输出路径"),
               fmt: str = typer.Option("srt", "--format", help="srt | vtt | ass"),
               karaoke: bool = typer.Option(False, help="ASS 词级卡拉 OK"),
               config: str = typer.Option(None, "--config")) -> None:
    """③④ 字幕生成 + 情绪识别 / transcribe, optionally with emotion."""
    try:
        asr = _module(engine, config)
        result = asr.transcribe(media, engine=engine, language=language, words=words,
                                emotion=emotion, srt=srt, srt_format=fmt)
    except ASRError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{result.engine}[/bold]  lang={result.language}  "
                  f"duration={result.duration:.2f}s  segments={len(result.segments)}")
    if result.emotion_summary:
        console.print(f"情绪 / emotion: {result.emotion_summary}")
    for w in result.warnings:
        console.print(f"[yellow]warn:[/yellow] {w}")
    if result.suspicious:
        console.print("[yellow]结果被标记为可疑 / flagged suspicious[/yellow]")
    console.print(result.text or "[dim]（空 / empty）[/dim]")
    if srt:
        console.print(f"字幕已写出 / subtitle: {srt}")


@app.command()
def clip(media: Path,
         top: int = typer.Option(None, help="取前 N 段"),
         out_dir: Path = typer.Option(None, "--out-dir"),
         engine: str = typer.Option(None),
         config: str = typer.Option(None, "--config")) -> None:
    """⑤ 参考音频截取（10~15 s，喂 TTS 音色克隆）/ pick reference clips."""
    try:
        clips = _module(engine, config).clip_reference(media, top=top, out_dir=out_dir,
                                                       engine=engine)
    except ASRError as exc:
        _fail(exc)
        return
    if not clips:
        console.print("[yellow]没有找到合适的片段 / no suitable clip found[/yellow]")
        return
    table = Table(title=f"参考音频 / reference clips ({len(clips)})")
    for col in ("#", "start", "end", "dur", "score", "wav", "text"):
        table.add_column(col, overflow="fold")
    for c in clips:
        table.add_row(str(c.index), f"{c.start:.2f}", f"{c.end:.2f}", f"{c.duration:.2f}",
                      f"{c.score:.3f}", Path(c.wav_path).name if c.wav_path else "",
                      c.text[:40])
    console.print(table)


@app.command()
def serve(host: str = typer.Option(None), port: int = typer.Option(None),
          reload: bool = typer.Option(False, help="开发用热重载"),
          config: str = typer.Option(None, "--config")) -> None:
    """起 HTTP 服务 / run the HTTP service."""
    from agentic_asr.server.app import run

    cfg = load_config(config)
    run(host or cfg.server.host, port or cfg.server.port, reload=reload, config=config)


@app.command()
def gui(host: str = typer.Option(None), port: int = typer.Option(None),
        config: str = typer.Option(None, "--config")) -> None:
    """起图形界面 / launch the GUI."""
    from agentic_asr.gui.app import run

    cfg = load_config(config)
    run(host or cfg.server.host, port or (cfg.server.port + 100), config=config)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
