# SoulX-Singer on RunPod (hybrid deployment)

Mac runs the VoiceSynth UI and Ableton bridge unchanged. RunPod runs only the
GPU-bound SoulX inference, exposed as a small FastAPI service. The Mac POSTs
notes + lyrics to the pod and receives a WAV back.

## Files

- `bootstrap.sh` — runs on every pod boot. Installs system deps, clones
  SoulX-Singer, creates a python3.10 venv with CUDA torch, downloads `model.pt`,
  starts `server.py`.
- `server.py` — FastAPI service. One long-lived `cli.inference_serve` subprocess
  holds the GPU model; requests are serialized onto it.

## Hosting the two files

The RunPod template only carries a start command and env vars — it can't push
files. Two options:

**Option 1 — public GitHub repo.** Push this `runpod/` directory to a public
repo (e.g. `https://github.com/<you>/voicesynth-runpod`). Then set the template
start command to:

```
bash -c 'cd /workspace && \
  curl -fsSL https://raw.githubusercontent.com/<you>/voicesynth-runpod/main/server.py -o server.py && \
  curl -fsSL https://raw.githubusercontent.com/<you>/voicesynth-runpod/main/bootstrap.sh | SERVER_URL=https://raw.githubusercontent.com/<you>/voicesynth-runpod/main/server.py bash'
```

**Option 2 — public Gist.** Paste both files into a single Gist with two files.
Use the raw URLs the same way.

## RunPod template settings

Create a new template in the RunPod console with these fields:

| Field | Value |
| --- | --- |
| Container image | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| Container start command | (the `bash -c ...` line above) |
| Container disk | 30 GB (weights are ~5 GB, plus venv + Python wheels) |
| Volume disk | 0 (re-downloading each boot per choice) |
| Expose HTTP ports | `8000` |
| Expose TCP ports | (leave empty) |
| Environment variables | `PORT=8000`, `SOULX_DEVICE=cuda`, optionally `SERVER_URL=...` if not embedding it in the start command |

When you deploy a pod from this template, RunPod gives you a proxy URL of the
shape `https://<pod-id>-8000.proxy.runpod.net`. That's your inference endpoint.

## GPU choice

RTX 4090 (24 GB) is the recommended starting point — fits the model with room
to spare and is the cheapest tier in that class on RunPod's community cloud.

## First boot

Cold start with weight download is roughly:

- system deps install: ~30 s
- pip install (torch + SoulX requirements): ~3–4 min
- HuggingFace weight download (~5 GB): ~1–3 min on most pods
- SoulX model boot on first request: ~10–20 s

So budget ~5–8 minutes between pod start and first successful synth. Subsequent
requests on the same pod are warm.

## Sanity check from your Mac

Once the pod's `/healthz` returns `{"ok": true, ...}` and `worker_running` flips
to `true` after a request, you're good:

```bash
curl https://<pod-id>-8000.proxy.runpod.net/healthz
```

To smoke-test synthesis with a target_metadata file written by VoiceSynth (one
appears at `outputs/soulx_target_metadata.json` after any local generation
attempt):

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/synthesize \
  -F target_metadata=@outputs/soulx_target_metadata.json \
  -F prompt_preset=english \
  -F pitch_shift=0 \
  -F auto_shift=true \
  -o /tmp/pod-generated.wav
```

## Wiring the Mac side

Not done yet — `synth.py`'s `run_soulx_command` still runs SoulX locally. The
follow-up patch adds a `soulxRemoteUrl` config field. When set,
`run_soulx_command` POSTs to `<url>/synthesize` instead of spawning a local
subprocess. Approve that diff separately and the round-trip will work.

## Security

No auth on the endpoint (per the deployment decision). The proxy URL is long
and random but is reachable from the public internet — don't paste it in
public channels or commit it. Stop the pod when you're done so the URL goes
away.
