"""CLI 与 HTTP 服务 / CLI and HTTP service.

复测 / rerun:
    C:\\Users\\75203\\miniforge3\\envs\\ov_env_py312\\python.exe -m pytest tests/test_server_cli.py -q

覆盖 / covers: CLI 子命令可调用、HTTP 路由与库一一对应、下载端点越权防护。
不含 / excludes: 真实引擎与真实云调用（默认 mock 引擎即可）。
说明文档 / docs: docs/01_design.md §7
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentic_asr.core.config import load_config


# ── CLI ──────────────────────────────────────────────────────────────────────


def _run_cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "agentic_asr.cli", *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def test_cli_version() -> None:
    proc = _run_cli("version")
    assert proc.returncode == 0
    assert "agentic-asr" in proc.stdout


def test_cli_engines_lists_mock() -> None:
    proc = _run_cli("engines")
    assert proc.returncode == 0
    assert "mock" in proc.stdout


def test_cli_probe_and_segment(speech_like_wav: Path) -> None:
    probe = _run_cli("probe", str(speech_like_wav))
    assert probe.returncode == 0, probe.stderr
    assert "duration" in probe.stdout

    segment = _run_cli("segment", str(speech_like_wav), "--max-seconds", "3")
    assert segment.returncode == 0, segment.stderr


def test_cli_transcribe_with_mock(speech_like_wav: Path, tmp_path: Path) -> None:
    srt = tmp_path / "cli.srt"
    proc = _run_cli("transcribe", str(speech_like_wav), "--engine", "mock",
                    "--srt", str(srt))
    assert proc.returncode == 0, proc.stderr
    assert "mock segment" in proc.stdout
    assert srt.is_file()


def test_cli_reports_missing_file() -> None:
    proc = _run_cli("probe", "Z:/nope/missing.wav")
    assert proc.returncode != 0


# ── HTTP ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from agentic_asr.server.app import create_app

    cfg = load_config(None)
    cfg.engine.default = "mock"
    cfg.output_dir = tmp_path / "outputs"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(cfg)) as c:
        yield c


def test_healthz_and_engines(client) -> None:
    assert client.get("/healthz").json()["ok"] is True
    body = client.get("/engines").json()
    assert any(e["name"] == "mock" for e in body["engines"])


def test_probe_endpoint(client, speech_like_wav: Path) -> None:
    with speech_like_wav.open("rb") as fh:
        resp = client.post("/probe", files={"file": ("a.wav", fh, "audio/wav")},
                           data={"deep": "true"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["duration"] > 0


def test_segment_endpoint(client, speech_like_wav: Path) -> None:
    with speech_like_wav.open("rb") as fh:
        resp = client.post("/segment", files={"file": ("a.wav", fh, "audio/wav")},
                           data={"max_seconds": "3"})
    assert resp.status_code == 200, resp.text
    pieces = resp.json()["pieces"]
    assert pieces
    assert pieces[0]["start"] == pytest.approx(0.0, abs=0.05)


def test_transcribe_endpoint_with_subtitle(client, speech_like_wav: Path) -> None:
    with speech_like_wav.open("rb") as fh:
        resp = client.post("/transcribe", files={"file": ("a.wav", fh, "audio/wav")},
                           data={"subtitle_format": "srt"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["segments"]
    assert "-->" in body["subtitle"]
    assert body["subtitle_path"].endswith(".srt")


def test_clip_endpoint(client, long_wav: Path) -> None:
    with long_wav.open("rb") as fh:
        resp = client.post("/clip", files={"file": ("a.wav", fh, "audio/wav")},
                           data={"top": "2"})
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json()["clips"], list)


def test_unsupported_suffix_rejected(client, tmp_path: Path) -> None:
    bad = tmp_path / "note.txt"
    bad.write_text("not audio", encoding="utf-8")
    with bad.open("rb") as fh:
        resp = client.post("/probe", files={"file": ("note.txt", fh, "text/plain")})
    assert resp.status_code == 415


def test_download_is_confined_to_outputs(client, tmp_path: Path) -> None:
    """下载端点不能变成任意文件读取 / the download route must stay inside outputs."""
    resp = client.get("/download", params={"path": str(Path.home() / ".dsh" / "AGENTS.md")})
    assert resp.status_code in (403, 404)


def test_doctor_endpoint(client) -> None:
    body = client.get("/doctor").json()
    assert "checks" in body and "cuda_runtime" in body
