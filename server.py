"""FastAPI wrapper around SoulX-Singer's cli.inference_serve worker.

Mirrors the JSONL stdin/stdout protocol that synth.py._SoulxWorker speaks
locally: one long-lived python3.10 subprocess owns the GPU model, this server
serializes inference jobs onto it.

Run with:
    SOULX_REPO=/workspace/SoulX-Singer \
    SOULX_PYTHON=/workspace/SoulX-Singer/.venv/bin/python \
    SOULX_DEVICE=cuda \
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse


SOULX_REPO = Path(os.environ["SOULX_REPO"]).resolve()
SOULX_PYTHON = os.environ["SOULX_PYTHON"]
SOULX_DEVICE = os.environ.get("SOULX_DEVICE", "cuda")

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


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": True,
        "device": SOULX_DEVICE,
        "repo": str(SOULX_REPO),
        "model_present": MODEL_PATH.is_file(),
        "worker_running": worker.proc is not None and worker.proc.poll() is None,
    }


@app.post("/synthesize")
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

        try:
            worker.infer({
                "prompt_wav_path": str(prompt_wav_path),
                "prompt_metadata_path": str(prompt_meta_path),
                "target_metadata_path": str(target_path),
                "phoneset_path": str(PHONESET_PATH),
                "save_dir": str(save_dir),
                "pitch_shift": pitch_shift,
                "auto_shift": auto_shift,
                "control": "score",
            })
        except RuntimeError as exc:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

        generated = save_dir / "generated.wav"
        if not generated.is_file():
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": f"SoulX finished but {generated} missing."},
            )

        # FileResponse will stream the file; the temp dir is cleaned up by a
        # background task after the response is sent.
        return FileResponse(
            path=str(generated),
            media_type="audio/wav",
            filename="generated.wav",
            background=_cleanup_task(tmp),
        )
    except HTTPException:
        _rmtree(tmp)
        raise
    except Exception:
        _rmtree(tmp)
        raise


def _cleanup_task(path: Path):
    from starlette.background import BackgroundTask
    return BackgroundTask(_rmtree, path)


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
