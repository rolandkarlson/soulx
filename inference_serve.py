"""Persistent SoulX inference worker.

Reads JSON-line requests from stdin and writes JSON-line responses to stdout.
Model load happens once on the boot message; subsequent infer messages reuse
the loaded model. Model chatter (build_model prints, tqdm) is redirected to
stderr so it never collides with the response protocol.
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
import types

import torch

from cli.inference import build_model, process
from soulxsinger.utils.file_utils import load_config


def _patch_mel_for_mps(model, device: str) -> None:
    """Route FFT-using submodules (mel encoder, vocoder) through CPU on MPS.

    torch.stft / torch.fft.* hit aten::_fft_r2c, which MPS doesn't implement.
    PYTORCH_ENABLE_MPS_FALLBACK=1 isn't enough: the complex intermediate tensor
    breaks subsequent MPS binary kernels. Running each FFT-heavy submodule
    entirely on CPU and bouncing the real-valued tensor back avoids both
    issues. The cost is one CPU<->MPS copy per submodule per inference -
    nothing inside the diffusion loop.
    """
    if not isinstance(device, str) or not device.startswith("mps"):
        return

    target_device = torch.device(device)

    def to_cpu(_module, args):
        return tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args)

    def to_target(_module, _args, output):
        if torch.is_tensor(output):
            return output.to(target_device)
        return output

    # Mel encoder: result is fed back into the MPS diffusion graph.
    model.mel.to("cpu")
    model.mel.register_forward_pre_hook(to_cpu)
    model.mel.register_forward_hook(to_target)

    # Vocoder: result is the final audio. cli/inference.py immediately moves it
    # to CPU and converts to numpy, so we leave the output on CPU.
    if hasattr(model, "vocoder"):
        model.vocoder.to("cpu")
        model.vocoder.register_forward_pre_hook(to_cpu)


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _read_request() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def _error_payload(exc: BaseException) -> dict:
    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def main() -> int:
    try:
        boot = _read_request()
    except json.JSONDecodeError as exc:
        _emit(_error_payload(exc))
        return 1
    if boot is None:
        return 0
    if boot.get("action") != "boot":
        _emit({"ok": False, "error": "First message must have action='boot'"})
        return 1

    try:
        with contextlib.redirect_stdout(sys.stderr):
            config = load_config(boot["config"])
            model = build_model(
                model_path=boot["model_path"],
                config=config,
                device=boot["device"],
                use_fp16=bool(boot.get("use_fp16", False)),
            )
            _patch_mel_for_mps(model, boot["device"])
    except Exception as exc:  # noqa: BLE001
        _emit(_error_payload(exc))
        return 1

    device = boot["device"]
    use_fp16 = bool(boot.get("use_fp16", False))
    _emit({"ok": True, "event": "ready"})

    while True:
        try:
            req = _read_request()
        except json.JSONDecodeError as exc:
            _emit(_error_payload(exc))
            continue
        if req is None:
            return 0

        action = req.get("action")
        if action == "shutdown":
            return 0
        if action != "infer":
            _emit({"ok": False, "error": f"Unknown action: {action!r}"})
            continue

        args = types.SimpleNamespace(
            prompt_wav_path=req["prompt_wav_path"],
            prompt_metadata_path=req["prompt_metadata_path"],
            target_metadata_path=req["target_metadata_path"],
            phoneset_path=req["phoneset_path"],
            save_dir=req["save_dir"],
            device=device,
            auto_shift=bool(req.get("auto_shift", False)),
            pitch_shift=int(req.get("pitch_shift", 0)),
            control=req.get("control", "score"),
            use_fp16=use_fp16,
        )
        try:
            with contextlib.redirect_stdout(sys.stderr):
                process(args, config, model)
        except Exception as exc:  # noqa: BLE001
            _emit(_error_payload(exc))
            continue
        _emit({"ok": True, "event": "done"})


if __name__ == "__main__":
    sys.exit(main())
