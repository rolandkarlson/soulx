"""FastAPI wrapper around SoulX-Singer's cli.inference_serve worker.

One long-lived python3.10 subprocess owns the GPU model. Inference is async:
POST /synthesize enqueues a job and returns a job_id immediately; the client
polls GET /jobs/{id} and downloads GET /jobs/{id}/result when done. This keeps
every HTTP request short so RunPod's Cloudflare proxy never hits its ~100s
origin timeout (error 524) on long renders.

Finished renders are also kept under RENDERS_DIR and browsable at GET / with
per-file download and a "Download all" zip.

Run with:
    SOULX_REPO=/workspace/SoulX-Singer \
    SOULX_PYTHON=/workspace/SoulX-Singer/.venv/bin/python \
    SOULX_DEVICE=cuda \
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import datetime
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask


SOULX_REPO = Path(os.environ["SOULX_REPO"]).resolve()
SOULX_PYTHON = os.environ["SOULX_PYTHON"]
SOULX_DEVICE = os.environ.get("SOULX_DEVICE", "cuda")
RENDERS_DIR = Path(os.environ.get("RENDERS_DIR", "/workspace/renders")).resolve()

MODEL_PATH = SOULX_REPO / "pretrained_models" / "SoulX-Singer" / "model.pt"
CONFIG_PATH = SOULX_REPO / "soulxsinger" / "config" / "soulxsinger.yaml"
PHONESET_PATH = SOULX_REPO / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
PROMPT_DIR = SOULX_REPO / "example" / "audio"

PROMPT_PRESETS = {
    "english": ("en_prompt.mp3", "en_prompt.json"),
    "mandarin": ("zh_prompt.mp3", "zh_prompt.json"),
}


class SoulxWorker:
    """One process per pod. Boots SoulX once, serializes infer calls."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _start(self) -> None:
        proc = subprocess.Popen(
            [SOULX_PYTHON, "-u", "-m", "cli.inference_serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            cwd=str(SOULX_REPO),
            text=True,
            bufsize=1,
        )
        boot = {
            "action": "boot",
            "model_path": str(MODEL_PATH),
            "config": str(CONFIG_PATH),
            "device": SOULX_DEVICE,
            "use_fp16": False,
        }
        self._send(proc, boot)
        resp = self._recv(proc)
        if not resp.get("ok"):
            self._kill(proc)
            raise RuntimeError(f"SoulX boot failed: {resp.get('error')}")
        self.proc = proc

    @staticmethod
    def _send(proc: subprocess.Popen, payload: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _recv(proc: subprocess.Popen) -> dict:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("SoulX worker closed its output stream.")
        return json.loads(line)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def infer(self, job: dict) -> None:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            proc = self.proc
            assert proc is not None
            try:
                self._send(proc, {"action": "infer", **job})
                resp = self._recv(proc)
            except (BrokenPipeError, OSError) as exc:
                self._kill(proc)
                self.proc = None
                raise RuntimeError(f"SoulX worker died: {exc}") from exc
            if not resp.get("ok"):
                raise RuntimeError(f"SoulX inference failed: {resp.get('error')}")


app = FastAPI(title="SoulX-Singer Inference")
worker = SoulxWorker()

# Single-worker executor: inference is serialized anyway (one GPU, one worker
# subprocess), and running it here keeps the blocking call off the event loop
# so /jobs polls stay responsive while a render is in flight.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _run_job(job_id: str, infer_args: dict, tmp: Path) -> None:
    _set_job(job_id, status="running")
    try:
        worker.infer(infer_args)
        generated = Path(infer_args["save_dir"]) / "generated.wav"
        if not generated.is_file():
            raise RuntimeError(f"SoulX finished but {generated} missing.")
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        render_name = f"{stamp}-{job_id[:8]}.wav"
        render_path = RENDERS_DIR / render_name
        shutil.copy2(generated, render_path)
        _set_job(job_id, status="done", wav_path=str(render_path), render_name=render_name)
    except Exception as exc:  # noqa: BLE001
        _set_job(job_id, status="error", error=str(exc))
    finally:
        # The per-job working dir (uploads + raw output) is no longer needed;
        # the finished render is preserved in RENDERS_DIR.
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j.get("status") in ("pending", "running"))
    return {
        "ok": True,
        "device": SOULX_DEVICE,
        "repo": str(SOULX_REPO),
        "model_present": MODEL_PATH.is_file(),
        "worker_running": worker.proc is not None and worker.proc.poll() is None,
        "active_jobs": active,
    }


@app.post("/synthesize", status_code=202)
async def synthesize(
    target_metadata: UploadFile = File(...),
    prompt_preset: str = Form("english"),
    pitch_shift: int = Form(0),
    auto_shift: bool = Form(True),
    prompt_wav: UploadFile | None = File(None),
    prompt_metadata: UploadFile | None = File(None),
):
    pitch_shift = max(-24, min(24, int(pitch_shift)))
    preset_key = (prompt_preset or "english").strip().lower()

    tmp = Path(tempfile.mkdtemp(prefix="soulx-job-"))
    try:
        target_path = tmp / "target_metadata.json"
        target_path.write_bytes(await target_metadata.read())

        if preset_key == "custom":
            if prompt_wav is None or prompt_metadata is None:
                raise HTTPException(
                    status_code=400,
                    detail="Custom preset requires prompt_wav and prompt_metadata files.",
                )
            prompt_wav_path = tmp / "prompt.wav"
            prompt_meta_path = tmp / "prompt_metadata.json"
            prompt_wav_path.write_bytes(await prompt_wav.read())
            prompt_meta_path.write_bytes(await prompt_metadata.read())
        else:
            preset = PROMPT_PRESETS.get(preset_key)
            if preset is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown prompt_preset: {preset_key}",
                )
            prompt_wav_path = PROMPT_DIR / preset[0]
            prompt_meta_path = PROMPT_DIR / preset[1]

        save_dir = tmp / "out"
        save_dir.mkdir(parents=True, exist_ok=True)

        infer_args = {
            "prompt_wav_path": str(prompt_wav_path),
            "prompt_metadata_path": str(prompt_meta_path),
            "target_metadata_path": str(target_path),
            "phoneset_path": str(PHONESET_PATH),
            "save_dir": str(save_dir),
            "pitch_shift": pitch_shift,
            "auto_shift": auto_shift,
            "control": "score",
        }
    except HTTPException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    job_id = uuid.uuid4().hex
    _set_job(job_id, status="pending", error=None, wav_path=None, tmp=str(tmp))
    _executor.submit(_run_job, job_id, infer_args, tmp)
    return {"ok": True, "job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "error": job.get("error"),
        "render_name": job.get("render_name"),
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    status = job.get("status")
    if status == "error":
        return JSONResponse(
            status_code=500, content={"ok": False, "error": job.get("error")}
        )
    if status != "done":
        # Not ready yet — tell the client to keep polling.
        return JSONResponse(status_code=409, content={"ok": False, "status": status})

    wav_path = job.get("wav_path")
    if not wav_path or not Path(wav_path).is_file():
        raise HTTPException(status_code=410, detail="Render file no longer available")
    return FileResponse(path=wav_path, media_type="audio/wav", filename="generated.wav")


# --- Render browser ----------------------------------------------------------


def _list_renders() -> list[Path]:
    if not RENDERS_DIR.is_dir():
        return []
    files = [p for p in RENDERS_DIR.glob("*.wav") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _safe_render(name: str) -> Path:
    candidate = (RENDERS_DIR / name).resolve()
    if candidate.parent != RENDERS_DIR or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Unknown render")
    return candidate


def _fmt_size(n: int) -> str:
    return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1024/1024:.1f} MB"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    files = _list_renders()
    rows = []
    for p in files:
        st = p.stat()
        when = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        name = html.escape(p.name)
        rows.append(
            f"<tr><td><a href='/files/{name}'>{name}</a></td>"
            f"<td>{_fmt_size(st.st_size)}</td><td>{when}</td>"
            f"<td><a class='btn' href='/files/{name}' download>Download</a></td></tr>"
        )
    table = (
        "<table><thead><tr><th>File</th><th>Size</th><th>Modified</th><th></th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
        if rows
        else "<p class='empty'>No renders yet.</p>"
    )
    download_all = (
        "<a class='btn primary' href='/download-all'>Download all (zip)</a>"
        if rows
        else ""
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SoulX Renders</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 860px;
         margin: 40px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .bar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e2e2;
           font-size: 0.92rem; }}
  th {{ color:#666; font-weight:600; }}
  a {{ color: #2b6cb0; text-decoration: none; }}
  .btn {{ display:inline-block; padding:5px 10px; border:1px solid #cbd5e0;
         border-radius:6px; font-size:0.85rem; }}
  .btn.primary {{ background:#2b6cb0; color:#fff; border-color:#2b6cb0; }}
  .empty {{ color:#888; margin-top:24px; }}
  .count {{ color:#888; font-size:0.9rem; }}
</style></head>
<body>
  <div class="bar">
    <h1>SoulX Renders <span class="count">({len(files)})</span></h1>
    {download_all}
  </div>
  {table}
</body></html>"""


@app.get("/files/{name}")
def download_file(name: str):
    path = _safe_render(name)
    return FileResponse(path=str(path), media_type="audio/wav", filename=path.name)


@app.get("/download-all")
def download_all():
    files = _list_renders()
    if not files:
        raise HTTPException(status_code=404, detail="No renders to download")
    tmp = Path(tempfile.mkdtemp(prefix="soulx-zip-"))
    zip_path = tmp / "soulx_renders.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="soulx_renders.zip",
        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True),
    )
