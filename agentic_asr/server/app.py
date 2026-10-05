"""HTTP 服务 / HTTP service.

路由与库 API、CLI **一一对应**，且每个功能都是**独立 endpoint** ——
可以单独调用，不要求走完整流水线（这是用户明确要求的形态）。

    uvicorn agentic_asr.server.app:app --port 8301
    python -m agentic_asr.server
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from agentic_asr import __version__
from agentic_asr.asr import ASRModule
from agentic_asr.core.config import ASRConfig, load_config
from agentic_asr.core.errors import ASRError

_ALLOWED_SUFFIX = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff",
                   ".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts"}


def create_app(config: ASRConfig | None = None) -> FastAPI:
    """构造应用 / build the FastAPI app（便于测试注入配置）."""
    cfg = config or load_config()
    module = ASRModule(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """进程退出时释放权重 / release weights on shutdown.

        用 lifespan 而不是已废弃的 `on_event("shutdown")`。
        """
        try:
            yield
        finally:
            module.release()

    api = FastAPI(title="Agentic_ASR_Module", version=__version__, lifespan=lifespan)

    # ── 工具 / helpers ──────────────────────────────────────────────────────

    def _save(upload: UploadFile) -> Path:
        """把上传落成临时文件（ffprobe 需要真实路径）/ persist an upload."""
        suffix = Path(upload.filename or "").suffix.lower() or ".bin"
        if suffix not in _ALLOWED_SUFFIX:
            raise HTTPException(415, f"不支持的扩展名 / unsupported suffix: {suffix}")
        tmp = Path(tempfile.mkdtemp(prefix="agentic_asr_http_")) / f"input{suffix}"
        with tmp.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        return tmp

    def _guard(fn, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ASRError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ── 元信息 / meta ───────────────────────────────────────────────────────

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "engine": cfg.engine.default}

    @api.get("/engines")
    def engines() -> dict[str, Any]:
        return {"default": cfg.engine.default, "engines": module.engines()}

    @api.get("/doctor")
    def doctor() -> dict[str, Any]:
        return module.doctor()

    @api.get("/config")
    def show_config() -> dict[str, Any]:
        return cfg.model_dump(mode="json")

    # ── ① 音频信息提取 / probe ──────────────────────────────────────────────

    @api.post("/probe")
    async def probe_endpoint(file: UploadFile = File(...),
                             deep: bool = Form(True)) -> dict[str, Any]:
        path = _save(file)
        info = _guard(module.probe, path, deep)
        return info.to_dict()

    # ── ② 音频分段 / segment ────────────────────────────────────────────────

    @api.post("/segment")
    async def segment_endpoint(file: UploadFile = File(...),
                               max_seconds: float | None = Form(None),
                               export_audio: bool = Form(False)) -> dict[str, Any]:
        path = _save(file)
        pieces = _guard(module.segment, path, max_seconds, export_audio,
                        cfg.output_dir / "http_segments")
        return {"pieces": [p.to_dict() for p in pieces]}

    # ── ③④ 转写 + 情绪 / transcribe ─────────────────────────────────────────

    @api.post("/transcribe")
    async def transcribe_endpoint(file: UploadFile = File(...),
                                  engine: str | None = Form(None),
                                  language: str | None = Form(None),
                                  words: bool | None = Form(None),
                                  emotion: bool | None = Form(None),
                                  subtitle_format: str | None = Form(None)) -> JSONResponse:
        """转写（可顺带生成字幕）。

        注意 / note: 这里**没有**「给字幕加说话人前缀」的参数 ——
        说话人分离（`Capability.DIARIZATION`）第一版只留了能力位，未启用
        （见 `docs/01_design.md §10.4`）。只有 MiniMax 引擎会返回 `speaker` 字段，
        它以 JSON 形式透出，不参与字幕渲染。放一个不生效的参数会误导调用方。
        """
        path = _save(file)
        want_srt = bool(subtitle_format)
        srt_path = (cfg.output_dir / "http_subtitles" / f"{path.stem}.{subtitle_format}"
                    if want_srt else None)
        result = _guard(module.transcribe, path, engine=engine, language=language,
                        words=words, emotion=emotion, srt=srt_path,
                        srt_format=subtitle_format or "srt")
        payload = result.to_dict()
        if want_srt:
            if not srt_path or not srt_path.is_file():
                raise HTTPException(500, "字幕生成失败 / subtitle not produced")
            payload["subtitle_path"] = str(srt_path)
            payload["subtitle"] = srt_path.read_text(encoding="utf-8")
        return JSONResponse(payload)

    # ── ⑤ 参考音频截取 / clip ───────────────────────────────────────────────

    @api.post("/clip")
    async def clip_endpoint(file: UploadFile = File(...),
                            top: int | None = Form(None),
                            engine: str | None = Form(None)) -> dict[str, Any]:
        path = _save(file)
        clips = _guard(module.clip_reference, path, None, top,
                       cfg.output_dir / "http_refs", "ref", engine)
        return {"clips": [c.to_dict() for c in clips]}

    @api.get("/download")
    def download(path: str) -> FileResponse:
        """下载产物（仅允许 outputs 之内）/ download an artefact inside outputs."""
        target = Path(path).resolve()
        root = cfg.output_dir.resolve()
        if root not in target.parents and target != root:
            raise HTTPException(403, "只允许下载 outputs 目录内的文件 / outside outputs")
        if not target.is_file():
            raise HTTPException(404, "文件不存在 / not found")
        return FileResponse(str(target), filename=target.name)

    return api


app = create_app()


def run(host: str = "127.0.0.1", port: int = 8301, reload: bool = False,
        config: str | None = None) -> None:
    """起服务 / start uvicorn."""
    import uvicorn

    if reload or config:
        # 带配置或热重载时用工厂模式，避免在导入期固化配置
        import os

        if config:
            os.environ["ASR_CONFIG"] = config
        uvicorn.run("agentic_asr.server.app:app", host=host, port=port, reload=reload)
    else:
        uvicorn.run(app, host=host, port=port)


__all__ = ["app", "create_app", "run"]
