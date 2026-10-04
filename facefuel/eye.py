"""
Eye modality (v5 YOLO11m + v5 MLP; held-out test: detector mAP50 0.991,
MLP mean F1 0.901, 6% of normal eyes flagged — docs/model_comparison.md).

Runs on the eye band of the RAW aligned face (no extra photo needed).

scleral_icterus has a three-layer false-positive guard, because early eye models
fired it on every image including clearly white sclera:
  1. raised per-class YOLO threshold (0.65 vs 0.30 default),
  2. LAB colour gate — mean B > 145 (genuinely yellow) AND mean L > 140 (not just
     dark/shadowed) on the sclera band. This gate assumes an un-normalised image,
  3. the severity MLP alone can never report it — YOLO must confirm.
"""
import time

import cv2
import numpy as np

from . import schema
from .evidence import collect_features
from .models import embed_regions, get_dino, load_severity_mlp, load_yolo, yolo_boxes

CONF_THRESHOLD = 0.30
PER_CLASS_CONF = {
    "conjunctival_pallor": 0.40,
    "scleral_icterus":     0.65,
    "xanthelasma":         0.45,
    "pterygium":           0.50,
    "conjunctivitis":      0.40,
    "eyelid_drooping":     0.45,
}
REQUIRE_YOLO = ("scleral_icterus",)


def load(device: str) -> dict:
    get_dino(device)
    return {
        "yolo": load_yolo("eye_yolo", schema.EYE_CLASSES),
        "mlp": load_severity_mlp("eye_mlp", schema.EYE_CLASSES, schema.EYE_INACTIVE, device),
        "device": device,
    }


def eye_crop(aligned_bgr: np.ndarray) -> np.ndarray:
    """Brow-to-cheekbone band of an aligned face (eyes sit at 40% height)."""
    h = aligned_bgr.shape[0]
    return aligned_bgr[int(h * 0.15):int(h * 0.50), :]


def lab_gate_scleral(crop_bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    r = lab[int(h * 0.15):int(h * 0.85), int(w * 0.05):int(w * 0.95)]
    if r.size < 100:
        return False
    return float(np.mean(r[:, :, 2])) > 145 and float(np.mean(r[:, :, 0])) > 140


def run_yolo(crop: np.ndarray, models: dict):
    conf = {}
    for cid, c, _ in yolo_boxes(models["yolo"], crop, CONF_THRESHOLD, models["device"]):
        name = schema.EYE_CLASSES[cid]
        if name in schema.EYE_INACTIVE or c < PER_CLASS_CONF.get(name, CONF_THRESHOLD):
            continue
        conf[name] = max(conf.get(name, 0.0), round(c, 3))
    if "scleral_icterus" in conf and not lab_gate_scleral(crop):
        del conf["scleral_icterus"]
    return conf


def analyze(aligned_bgr: np.ndarray, models: dict) -> dict:
    t = {}
    crop = eye_crop(aligned_bgr)
    t0 = time.perf_counter(); yconf = run_yolo(crop, models); t["yolo"] = time.perf_counter() - t0
    t0 = time.perf_counter(); feat = embed_regions(crop, "eye", models["device"]); t["dinov2"] = time.perf_counter() - t0
    t0 = time.perf_counter(); sev, unc = models["mlp"].predict(feat); t["severity_mlp"] = time.perf_counter() - t0
    features, unconfirmed = collect_features(schema.EYE_CLASSES, models["mlp"].inactive, sev, unc,
                                yconf, {k: 1 for k in yconf}, require_yolo=REQUIRE_YOLO)
    return {"modality": "eye", "features": features, "unconfirmed": unconfirmed, "timing": t}
