"""
FaceFuel v4 — Tri-Modal API server
==================================
POST /analyze            selfie → face + eye
POST /analyze/tongue     tongue photo → tongue
POST /analyze/combined   selfie + tongue photo → face + eye + tongue
GET  /health             liveness + per-modality model status
GET  /api/info           class lists, active/inactive classes, coverage, weights
GET  /                   web frontend (static/index.html)

Configuration (environment variables, all optional):
  FACEFUEL_HOST (0.0.0.0)  FACEFUEL_PORT / PORT (8000)  FACEFUEL_DEVICE (cuda|cpu, auto)
  FACEFUEL_MAX_UPLOAD_MB (15)  FACEFUEL_CORS_ORIGINS ("*", comma-separated)
  FACEFUEL_WEIGHTS_DIR and per-file overrides — see facefuel/paths.py

Run:  python server.py      or      uvicorn server:app --host 0.0.0.0 --port 8000
Use a single worker: models live in process memory on one GPU.

FaceFuel is a wellness-awareness research prototype, NOT a medical device.
"""
import base64
import io
import logging
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from facefuel import DISCLAIMER, __version__, eye, face, fusion, paths, schema, tongue
from facefuel.models import default_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
log = logging.getLogger("facefuel.server")

HOST    = os.environ.get("FACEFUEL_HOST", "0.0.0.0")
PORT    = int(os.environ.get("FACEFUEL_PORT", os.environ.get("PORT", 8000)))
DEVICE  = os.environ.get("FACEFUEL_DEVICE") or default_device()
MAX_MB  = float(os.environ.get("FACEFUEL_MAX_UPLOAD_MB", 15))
MAX_SIDE = 2048           # larger photos are downscaled before analysis
CORS    = [o.strip() for o in os.environ.get("FACEFUEL_CORS_ORIGINS", "*").split(",") if o.strip()]
STATIC  = Path(__file__).resolve().parent / "static"

LOADERS = {"face": face.load, "eye": eye.load, "tongue": tongue.load}
MODELS: dict = {}
LOAD_ERRORS: dict = {}
_infer_lock = threading.Lock()   # one GPU, and MC-dropout toggles train/eval mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpu = f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""
    log.info("FaceFuel %s starting on %s%s", __version__, DEVICE, gpu)
    for name, loader in LOADERS.items():
        t0 = time.perf_counter()
        try:
            MODELS[name] = loader(DEVICE)
            log.info("%s models ready (%.1fs)", name, time.perf_counter() - t0)
        except Exception as e:                       # keep serving the other modalities
            LOAD_ERRORS[name] = f"{type(e).__name__}: {e}"
            log.error("%s failed to load: %s", name, LOAD_ERRORS[name])
    yield
    if "face" in MODELS:
        MODELS["face"]["landmarker"].close()


app = FastAPI(title="FaceFuel", version=__version__, lifespan=lifespan,
              description="Tri-modal visual wellness screening — research prototype, not medical advice.")
app.add_middleware(CORSMiddleware, allow_origins=CORS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], allow_credentials=False)
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── helpers ──────────────────────────────────────────────────────────────────
def _require(*mods):
    missing = [m for m in mods if m not in MODELS]
    if missing:
        raise HTTPException(503, {"error": "model_unavailable", "modalities": missing,
                                  "details": {m: LOAD_ERRORS.get(m, "not loaded") for m in missing}})


def decode_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large (max {MAX_MB:g} MB).")
    img = None
    try:                                   # PIL honours EXIF orientation (phone photos)
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image. Use JPEG, PNG or WebP.")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def b64jpg(bgr: np.ndarray, size=(300, 300)) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, size), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode() if ok else ""


def jsonable(o):
    if isinstance(o, dict):          return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.generic):    return o.item()
    if isinstance(o, np.ndarray):    return o.tolist()
    return o


def respond(results: dict, extra: dict, timing: dict, warnings: list):
    report = fusion.build_report(results)
    body = {
        "status": "success",
        "modality": "+".join(m for m in ("face", "eye", "tongue") if m in results),
        "modalities_run": list(results),
        "face_features":   results.get("face", {}).get("features", {}),
        "eye_features":    results.get("eye", {}).get("features", {}),
        "tongue_features": results.get("tongue", {}).get("features", {}),
        # MLP-only signals the detector did not confirm — not used as evidence
        "unconfirmed_signals": {m: r.get("unconfirmed", {}) for m, r in results.items()},
        **report,
        **extra,
        "warnings": warnings,
        "timing_ms": {k: round(v * 1000, 1) for k, v in timing.items()},
        "version": __version__,
        "disclaimer": DISCLAIMER,
    }
    return JSONResponse(jsonable(body))


NO_FACE = {
    "status": "no_face_detected",
    "message": "No face detected. Use a clear, front-facing photo in good, even lighting.",
    "tips": ["Face the camera directly", "Use soft, even light — avoid harsh shadows",
             "Remove glasses, masks and heavy filters", "Keep the photo in focus"],
    "disclaimer": DISCLAIMER,
}


def _face_and_eye(img, timing, warnings):
    fres, al = face.analyze(img, MODELS["face"])
    if fres is None:
        return None, None
    for k, v in fres["timing"].items(): timing[f"face_{k}"] = v
    eres = eye.analyze(al["aligned_big"], MODELS["eye"])
    for k, v in eres["timing"].items(): timing[f"eye_{k}"] = v
    return {"face": fres, "eye": eres}, al


def _tongue(img, timing, warnings):
    tres, crop = tongue.analyze(img, MODELS["tongue"])
    for k, v in tres["timing"].items(): timing[f"tongue_{k}"] = v
    if not tres["tongue_localized"]:
        warnings.append("The tongue outline was not located, so the whole photo was analysed. "
                        "For best results, fill the frame with your extended tongue in good light.")
    return tres, crop


def _run(endpoint, fn):
    try:
        with _infer_lock:
            return fn()
    except HTTPException:
        raise
    except Exception as e:
        log.error("%s failed: %s\n%s", endpoint, e, traceback.format_exc())
        raise HTTPException(500, f"Analysis failed: {type(e).__name__}")


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    idx = STATIC / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return HTMLResponse(f"<h1>FaceFuel {__version__}</h1><p>API docs at <a href='/docs'>/docs</a>.</p>")


@app.get("/health")
def health():
    return {
        "status": "healthy" if not LOAD_ERRORS else "degraded",
        "version": __version__,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "modalities": {m: ("ready" if m in MODELS else f"error: {LOAD_ERRORS.get(m, 'not loaded')}")
                       for m in LOADERS},
    }


@app.get("/api/info")
def info():
    mods = {}
    for m in ("face", "eye", "tongue"):
        mods[m] = {
            "classes": schema.CLASSES[m],
            "active_classes": schema.active_classes(m),
            "inactive_classes": sorted(schema.INACTIVE[m]),
            "coverage": [d for d in schema.ALL_DEFS if d in schema.coverage(m)],
            "fusion_weight": schema.MODALITY_WEIGHTS[m],
        }
    covered = set().union(*(schema.coverage(m) for m in mods))
    return {
        "version": __version__,
        "deficiencies": schema.ALL_DEFS,
        "not_assessable": [d for d in schema.ALL_DEFS if d not in covered],
        "modalities": mods,
        "model_files": {k: str(p) for k in paths.KEYS for p in [next(
            (c for c in paths.candidates(k) if c.is_file()), None)] if p},
        "disclaimer": DISCLAIMER,
    }


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    _require("face", "eye")
    img = decode_image(file)

    def go():
        timing, warnings = {}, []
        results, al = _face_and_eye(img, timing, warnings)
        if results is None:
            return JSONResponse(NO_FACE)
        return respond(results, {"aligned_face_b64": b64jpg(al["aligned"])}, timing, warnings)
    return _run("/analyze", go)


@app.post("/analyze/tongue")
def analyze_tongue(file: UploadFile = File(...)):
    _require("tongue")
    img = decode_image(file)

    def go():
        timing, warnings = {}, []
        tres, crop = _tongue(img, timing, warnings)
        return respond({"tongue": tres}, {"tongue_crop_b64": b64jpg(crop),
                                          "tongue_localized": tres["tongue_localized"]},
                       timing, warnings)
    return _run("/analyze/tongue", go)


@app.post("/analyze/combined")
def analyze_combined(face_file: UploadFile = File(..., alias="face"),
                     tongue_file: UploadFile = File(..., alias="tongue")):
    _require("face", "eye", "tongue")
    face_img, tongue_img = decode_image(face_file), decode_image(tongue_file)

    def go():
        timing, warnings = {}, []
        results, al = _face_and_eye(face_img, timing, warnings)
        if results is None:
            return JSONResponse(NO_FACE)
        tres, crop = _tongue(tongue_img, timing, warnings)
        results["tongue"] = tres
        return respond(results, {"aligned_face_b64": b64jpg(al["aligned"]),
                                 "tongue_crop_b64": b64jpg(crop),
                                 "tongue_localized": tres["tongue_localized"]},
                       timing, warnings)
    return _run("/analyze/combined", go)


if __name__ == "__main__":
    import uvicorn
    print(f"\nFaceFuel {__version__} → http://localhost:{PORT}\n")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, workers=1, log_level="info")
