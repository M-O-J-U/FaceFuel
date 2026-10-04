"""
Tongue modality (v4 — YOLO11m mAP50 0.871, MLP F1 0.761).

  tongue photo → one YOLO pass on the full photo (the detector was trained on
  full photos) → tongue_body box, when found, gives the crop and feature boxes
  outside it are discarded (otherwise the whole photo is used) → DINOv2 on 3
  regions of the 256 px image → v4 MLP.
"""
import time

import cv2
import numpy as np

from . import schema
from .evidence import collect_features
from .models import embed_regions, get_dino, load_severity_mlp, load_yolo, yolo_boxes

CONF_THRESHOLD = 0.25
TONGUE_BODY_CONF = 0.30
PER_CLASS_CONF = {
    "pale_tongue":    0.45,
    "red_tongue":     0.38,
    "yellow_coating": 0.35,
    "white_coating":  0.45,
    "geographic":     0.35,
    "smooth_glossy":  0.35,
}
SMALL_LESION_FEATS = {"fissured", "crenated", "geographic", "oral_ulcer"}
CROP_SIZE = 256


def load(device: str) -> dict:
    get_dino(device)
    return {
        "yolo": load_yolo("tongue_yolo", schema.TONGUE_CLASSES),
        "mlp": load_severity_mlp("tongue_mlp", schema.TONGUE_CLASSES, schema.TONGUE_INACTIVE, device),
        "device": device,
    }


def _inside(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def detect(img_bgr: np.ndarray, models: dict):
    """One YOLO pass → (tongue_box or None, feature confidences, feature counts)."""
    boxes = yolo_boxes(models["yolo"], img_bgr, CONF_THRESHOLD, models["device"])
    body = [(c, b) for cid, c, b in boxes if cid == 0 and c >= TONGUE_BODY_CONF]
    tongue_box = max(body)[1] if body else None

    conf, count = {}, {}
    for cid, c, b in boxes:
        name = schema.TONGUE_CLASSES[cid]
        if cid == 0 or name in schema.TONGUE_INACTIVE:
            continue
        if c < PER_CLASS_CONF.get(name, CONF_THRESHOLD):
            continue
        if tongue_box is not None and not _inside(b, tongue_box):
            continue
        count[name] = count.get(name, 0) + 1
        conf[name] = max(conf.get(name, 0.0), round(c, 3))
    return tongue_box, conf, count


def crop_tongue(img_bgr: np.ndarray, tongue_box):
    h, w = img_bgr.shape[:2]
    if tongue_box is not None:
        x1, y1, x2, y2 = tongue_box
        px, py = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size >= 100:
            return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    # No tongue box: use the whole photo. Most tongue training images carry lesion
    # labels but no tongue_body box (the detector localises the body in only ~22%
    # of validation images), and the MLP features were extracted from whole images.
    return cv2.resize(img_bgr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def analyze(img_bgr: np.ndarray, models: dict):
    """Returns (result, crop_bgr)."""
    t = {}
    t0 = time.perf_counter(); tbox, yconf, ycount = detect(img_bgr, models); t["yolo"] = time.perf_counter() - t0
    crop = crop_tongue(img_bgr, tbox)
    t0 = time.perf_counter(); feat = embed_regions(crop, "tongue", models["device"]); t["dinov2"] = time.perf_counter() - t0
    t0 = time.perf_counter(); sev, unc = models["mlp"].predict(feat); t["severity_mlp"] = time.perf_counter() - t0
    features, unconfirmed = collect_features(schema.TONGUE_CLASSES, models["mlp"].inactive, sev, unc,
                                yconf, ycount, SMALL_LESION_FEATS)
    return {"modality": "tongue", "features": features, "unconfirmed": unconfirmed,
            "tongue_localized": tbox is not None, "timing": t}, crop
