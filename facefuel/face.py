"""
Face modality (v5 YOLO11m + v5 MLP; held-out test: detector mAP50 0.672,
MLP mean F1 0.778 over 9 active classes — docs/model_comparison.md).

  selfie → MediaPipe FaceLandmarker alignment (eyes at fixed positions)
         → YOLO11m v4 on the aligned face (512 px)
         → DINOv2 on 8 fixed regions of the aligned face (256 px) → v4 severity MLP
         → LAB colour analysis (pallor / lip pallor) on landmark regions
         → per-feature severity + evidence over the deficiency framework

The aligned face is NOT LAB-normalised before the models: the v4 YOLO and the v4
feature extraction were both trained on raw images (v2 normalised, v4 does not).
"""
import time

import cv2
import numpy as np

from . import paths, schema
from .models import embed_regions, get_dino, level, load_severity_mlp, load_yolo, yolo_boxes
from .evidence import collect_features

ALIGN_SIZE = 256          # regions, colour analysis, MLP input, thumbnail
YOLO_ALIGN_SIZE = 512     # detector input (more pixels for small lesions)

CONF_THRESHOLD = 0.25
PER_CLASS_CONF = {
    "dark_circle": 0.40,   # easily confused with eye shadow / tired look
    "dark_spot":   0.38,   # confused with freckles, acne shadows
}
SMALL_LESION_FEATS = {"acne", "blackhead", "dark_spot"}   # box COUNT matters

# Landmark-based regions used only for colour analysis (MediaPipe 478-pt indices).
LANDMARK_REGIONS = [
    ("periorbital_left",  [33, 133, 159, 145, 153, 144, 163, 7], 0.35),
    ("periorbital_right", [362, 263, 386, 374, 380, 373, 390, 249], 0.35),
    ("left_cheek",        [234, 93, 132, 58, 172, 136, 150, 149], 0.35),
    ("right_cheek",       [454, 323, 361, 288, 397, 365, 379, 378], 0.35),
    ("forehead",          [10, 338, 297, 332, 284, 251, 389, 356, 70, 63, 105, 66, 107, 9, 336, 296], 0.15),
    ("nose",              [6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 129, 209, 49, 48, 64, 98, 358, 429, 279, 278, 294, 327], 0.3),
    ("lips",              [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146], 0.2),
    ("sclera_left",       [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], 0.1),
]


def load(device: str) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    opts = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(paths.resolve("face_landmarker"))),
        num_faces=1, min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4, min_tracking_confidence=0.4,
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    get_dino(device)
    return {
        "mp": mp,
        "landmarker": FaceLandmarker.create_from_options(opts),
        "yolo": load_yolo("face_yolo", schema.FACE_CLASSES),
        "mlp": load_severity_mlp("face_mlp", schema.FACE_CLASSES, schema.FACE_INACTIVE, device),
        "device": device,
    }


# ── Alignment ────────────────────────────────────────────────────────────────
def _align_matrix(le, re, size):
    tl = np.array([size * 0.35, size * 0.40])
    tr = np.array([size * 0.65, size * 0.40])
    sv, dv = re - le, tr - tl
    sc  = np.linalg.norm(dv) / (np.linalg.norm(sv) + 1e-6)
    ang = np.degrees(np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0]))
    M = cv2.getRotationMatrix2D((float(le[0]), float(le[1])), -ang, sc)
    M[0, 2] += tl[0] - le[0]
    M[1, 2] += tl[1] - le[1]
    return M


def align(img_bgr: np.ndarray, models: dict):
    """Returns None if no face is found."""
    h, w = img_bgr.shape[:2]
    mp = models["mp"]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = models["landmarker"].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None

    pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float32)
    le, re = pts[[33, 133]].mean(0), pts[[362, 263]].mean(0)

    M = _align_matrix(le, re, ALIGN_SIZE)
    aligned = cv2.warpAffine(img_bgr, M, (ALIGN_SIZE, ALIGN_SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M_big = _align_matrix(le, re, YOLO_ALIGN_SIZE)
    aligned_big = cv2.warpAffine(img_bgr, M_big, (YOLO_ALIGN_SIZE, YOLO_ALIGN_SIZE),
                                 flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)

    ones = np.ones((len(pts), 1), dtype=np.float32)
    lm = np.clip((M @ np.hstack([pts, ones]).T).T, 0, ALIGN_SIZE - 1)
    regions = {}
    for name, idxs, pad in LANDMARK_REGIONS:
        pr = lm[idxs]
        x1, y1 = int(pr[:, 0].min()), int(pr[:, 1].min())
        x2, y2 = int(pr[:, 0].max()), int(pr[:, 1].max())
        bw, bh = x2 - x1, y2 - y1
        regions[name] = [max(0, x1 - int(bw * pad)), max(0, y1 - int(bh * pad)),
                         min(ALIGN_SIZE, x2 + int(bw * pad)), min(ALIGN_SIZE, y2 + int(bh * pad))]
    return {"aligned": aligned, "aligned_big": aligned_big, "regions": regions}


# ── LAB colour analysis (ported unchanged from v2 step10_inference.py) ───────
def analyze_color(aligned_bgr: np.ndarray, regions: dict) -> dict:
    """
    Colour cues on the RAW aligned face. Returns {name: (score, confidence)}.
    OpenCV LAB: L 0–255; A 128 neutral (>128 red); B 128 neutral (>128 yellow).
    """
    lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    out = {}

    def box(name):
        b = regions.get(name)
        return b if b and b[2] > b[0] and b[3] > b[1] else None

    def pix(ch, names):
        vals = []
        for n in names:
            b = box(n)
            if b: vals.extend(ch[b[1]:b[3], b[0]:b[2]].flatten())
        return vals

    cl = pix(L, ["left_cheek", "right_cheek"])
    if cl:                                   # pallor: healthy cheek L ≈ 160–200
        score = float(np.clip((155 - float(np.mean(cl))) / 40, 0, 1))
        if score > 0.15:
            out["pallor"] = (score, min(1.0, len(cl) / 2000))

    sa = pix(A, ["left_cheek", "right_cheek", "forehead"])
    if sa:                                   # redness: A > 135
        score = float(np.clip((float(np.mean(sa)) - 135) / 20, 0, 1))
        if score > 0.10:
            out["skin_redness"] = (score, min(1.0, len(sa) / 3000))

    pl = pix(L, ["periorbital_left", "periorbital_right"])
    if pl and cl:                            # dark-circle depth: ≥20 L units darker than cheek
        diff = float(np.mean(cl)) - float(np.mean(pl))
        if diff > 20:
            out["dark_circle_depth"] = (float(np.clip((diff - 20) / 35, 0, 1)), min(1.0, len(pl) / 500))

    b = box("lips")
    if b:                                    # lip pallor: healthy lips A ≈ 140–150
        score = float(np.clip((135 - float(np.mean(A[b[1]:b[3], b[0]:b[2]]))) / 15, 0, 1))
        if score > 0.20:
            out["lip_pallor"] = (score, 0.6)
    return out


# Colour cues that count as evidence (must have a FACE_FEAT_DEF entry) and the
# minimum (score, confidence) for them to be reported — v2 thresholds.
COLOR_EVIDENCE = {"pallor": (0.30, 0.4), "lip_pallor": (0.30, 0.4)}


# ── Detection ────────────────────────────────────────────────────────────────
def run_yolo(aligned_big: np.ndarray, models: dict):
    conf, count = {}, {}
    for cid, c, _ in yolo_boxes(models["yolo"], aligned_big, CONF_THRESHOLD, models["device"]):
        name = schema.FACE_CLASSES[cid]
        if name in schema.FACE_INACTIVE or c < PER_CLASS_CONF.get(name, CONF_THRESHOLD):
            continue
        count[name] = count.get(name, 0) + 1
        conf[name] = max(conf.get(name, 0.0), round(c, 3))
    return conf, count


def analyze(img_bgr: np.ndarray, models: dict):
    """Face analysis. Returns (result, alignment) or (None, None) if no face."""
    t = {}
    t0 = time.perf_counter()
    al = align(img_bgr, models)
    t["align"] = time.perf_counter() - t0
    if al is None:
        return None, None

    t0 = time.perf_counter(); color = analyze_color(al["aligned"], al["regions"]); t["color"] = time.perf_counter() - t0
    t0 = time.perf_counter(); yconf, ycount = run_yolo(al["aligned_big"], models); t["yolo"] = time.perf_counter() - t0
    t0 = time.perf_counter(); feat = embed_regions(al["aligned"], "face", models["device"]); t["dinov2"] = time.perf_counter() - t0
    t0 = time.perf_counter(); sev, unc = models["mlp"].predict(feat); t["severity_mlp"] = time.perf_counter() - t0

    features, unconfirmed = collect_features(schema.FACE_CLASSES, models["mlp"].inactive, sev, unc,
                                yconf, ycount, SMALL_LESION_FEATS)
    for name, (min_s, min_c) in COLOR_EVIDENCE.items():
        if name in color and name not in features:
            s, c = color[name]
            if s >= min_s and c >= min_c:
                features[name] = {"severity": round(s, 3), "level": level(s), "confidence": round(c, 2),
                                  "yolo_count": 0, "detected_by": ["color_analysis"]}

    return {
        "modality": "face",
        "features": features, "unconfirmed": unconfirmed,
        "color_metrics": {k: {"score": round(v[0], 3), "confidence": round(v[1], 2)} for k, v in color.items()},
        "timing": t,
    }, al
