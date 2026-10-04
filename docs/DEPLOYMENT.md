# Deploying FaceFuel

FaceFuel is one FastAPI process that serves both the API and the web frontend
(`static/index.html`). All models live in memory, so run **one worker per GPU**.

## Requirements

| | Minimum | Tested |
|---|---|---|
| GPU | Any CUDA GPU with ≥ 4 GB VRAM (all models use < 2 GB) | RTX 4070 Super 12 GB |
| CPU-only | Works (`FACEFUEL_DEVICE=cpu`); roughly 1–3 s per request | — |
| RAM | 4 GB to serve; ~10 GB free to retrain YOLO | 16 GB |
| Python | 3.10–3.14 | 3.14 (Windows), 3.12 (Docker image) |
| Disk | ~135 MB weights + ~4 GB for PyTorch/CUDA wheels | |

Warm latency on the RTX 4070 Super is about 0.2 s for `/analyze/combined` (face + eye + tongue).
The first request after start-up is slower (CUDA kernel warm-up). Model loading takes about 6 s.

## Model files

The server needs seven files. `facefuel/paths.py` resolves each one in this order:
environment variable → `$FACEFUEL_WEIGHTS_DIR/<name>` → legacy training-output path.

| Key / env var | Canonical file in `weights/` | Size |
|---|---|---|
| `FACEFUEL_FACE_YOLO` | `face_yolo11m_v5.pt` | 39 MB |
| `FACEFUEL_TONGUE_YOLO` | `tongue_yolo11m_v4.pt` | 39 MB |
| `FACEFUEL_EYE_YOLO` | `eye_yolo11m_v5.pt` | 39 MB |
| `FACEFUEL_FACE_MLP` | `face_severity_mlp_v5.pt` | 7 MB |
| `FACEFUEL_TONGUE_MLP` | `tongue_severity_mlp_v5.pt` | 3.4 MB |
| `FACEFUEL_EYE_MLP` | `eye_severity_mlp_v5.pt` | 3.1 MB |
| `FACEFUEL_FACE_LANDMARKER` | `face_landmarker.task` (MediaPipe, Apache-2.0) | 3.6 MB |

DINOv2 ViT-S/14 (`facebook/dinov2-small`, Apache-2.0) is downloaded from the Hugging Face Hub on first start
and cached; the Docker image bakes it in. Override it with `FACEFUEL_DINOV2`.

```bash
python scripts/collect_weights.py          # copy the 7 files into ./weights (+ MANIFEST.txt)
python scripts/collect_weights.py --check  # show where each file resolves from
```

**Distributing weights:** don't commit them to git. Attach a `facefuel-weights-v4.2.zip` of
`weights/` to a GitHub Release (assets can be up to 2 GB) or push it to a Hugging Face model repo, and
download it during deployment. Check the licences of the training datasets before
publishing weights trained on them (see `docs/PUBLISH_CHECKLIST.md`).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FACEFUEL_HOST` | `0.0.0.0` | Bind address |
| `FACEFUEL_PORT` (or `PORT`) | `8000` | Port |
| `FACEFUEL_DEVICE` | auto (`cuda` if available) | `cuda`, `cuda:1`, `cpu` |
| `FACEFUEL_MAX_UPLOAD_MB` | `15` | Per-image upload limit |
| `FACEFUEL_CORS_ORIGINS` | `*` | Comma-separated origins, needed only if the frontend is hosted elsewhere |
| `FACEFUEL_WEIGHTS_DIR` | `<repo>/weights` | Where to find model files |

## Option A — run directly

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/collect_weights.py
python server.py                       # → http://localhost:8000  (API docs at /docs)
```

## Option B — Docker (GPU)

The host needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
python scripts/collect_weights.py
docker build -t facefuel .
docker run --gpus all -p 8000:8000 facefuel
# mount weights instead of baking them in:
docker run --gpus all -p 8000:8000 -v $PWD/weights:/app/weights facefuel
```

> **Status:** the Dockerfile was written in October 2026 but not yet built. Docker Desktop
> was not running on the development machine. Run the build once and confirm
> `curl localhost:8000/health` reports all three modalities `ready` before relying on it.

## Hosting options

| Host | Fit | Notes |
|---|---|---|
| **Hugging Face Spaces** (Docker SDK, T4 or better) | Best for a public demo | Set `PORT=7860` in Space variables; the image already runs as uid 1000. Free CPU Spaces work but are slow. |
| RunPod / Vast.ai / Lambda | Cheap on-demand GPU | Run the Docker image; put it behind their HTTPS proxy. |
| Google Cloud Run (GPU) / AWS ECS with GPU | Managed, scales to zero (Cloud Run) | Cold start includes model load (~10–20 s). |
| Frontend on Vercel/Netlify + API on a GPU host | Split hosting | Serve `static/index.html` anywhere and point it at the API with `?api=https://your-api` or `window.FACEFUEL_API`; set `FACEFUEL_CORS_ORIGINS` to the frontend origin. |

## Before going public

- Keep the "not medical advice" disclaimer visible. It is in every API response and in the UI.
- Uploaded images are processed in memory and never written to disk or logged; keep it
  that way, and state it in a privacy note.
- Add a reverse proxy with HTTPS and basic rate limiting (e.g. Caddy, nginx, or the host's
  own proxy). Inference is serialised by a lock, so a flood of requests queues rather than
  crashes, but it will slow everyone down.
- Pin the exact versions you tested in `requirements.txt` (`pip freeze`) for reproducible builds.
- Run `python -m pytest tests -q`. The API smoke tests need the weights; they skip without them.
