"""
FaceFuel v2 — Step 10: End-to-End Inference Pipeline (Full v2 Integration)
--------------------------------------------------------------------------
All fixes integrated:
  - SeverityMLPv2 (per-feature heads, BCEWithLogitsLoss, raw logits)
  - YOLO conf threshold lowered to 0.20
  - Count-based YOLO severity for small-lesion features
  - LAB color analysis (pallor, redness, yellow sclera, oiliness, lip pallor)
  - bayesian_inference_v2 (recalibrated priors, absence penalty, exp uncertainty)
  - YOLO count tracking passed through entire pipeline
  - Graceful v1/v2 model compatibility
"""

import os, sys, json, argparse, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["GLOG_minloglevel"] = "2"

# ── Paths ─────────────────────────────────────────────────────
YOLO_WEIGHTS      = Path("runs/detect/runs/detect/facefuel_v2/yolo_detector_r2/weights/best.pt")
SEVERITY_MODEL_V2 = Path("facefuel_models/severity_mlp_v2.pt")
SEVERITY_MODEL_V1 = Path("facefuel_models/severity_mlp.pt")
LANDMARK_MODEL    = Path("face_landmarker.task")
OUT_DIR           = Path("facefuel_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
ALIGN_SIZE     = 256
DINOV2_MODEL   = "facebook/dinov2-small"
EMBED_DIM      = 384
FEAT_DIM       = 3072
MC_PASSES      = 25
# Global minimum YOLO confidence
CONF_THRESHOLD = 0.20

# Per-class confidence overrides — classes with ambiguous visual signatures
# need higher thresholds to avoid false positives at 0.20
PER_CLASS_CONF = {
    "dark_circle": 0.40,   # easily confused with eye shadow, tired look
    "dark_spot":   0.38,   # confused with freckles, acne shadows
    "melasma":     0.40,   # rare, needs high confidence
    "eye_bag":     0.35,   # confused with orbital fat pad
    "acne_scar":   0.30,   # confused with pores
    # acne, blackhead, whitehead stay at 0.20 — count-based, safe at low conf
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SKIN_FEATURES = [
    "dark_circle","eye_bag","acne","wrinkle","redness",
    "dark_spot","blackhead","melasma","whitehead","acne_scar","vascular_redness",
]
N_FEATURES = len(SKIN_FEATURES)

YOLO_CLASS_NAMES = {
    0:"dark_circle",1:"eye_bag",2:"acne",3:"wrinkle",4:"redness",
    7:"dark_spot",8:"blackhead",11:"melasma",12:"whitehead",16:"acne_scar",23:"vascular_redness",
}

# Features where COUNT of boxes matters (not area)
SMALL_LESION_FEATS = {"acne","dark_spot","blackhead","melasma","whitehead","acne_scar"}
COUNT_FEAT_IDX     = {2,5,6,7,8,9}

REGION_ORDER = [
    "periorbital_left","periorbital_right","left_cheek",
    "right_cheek","forehead","nose","lips","sclera_left",
]
REGIONS_DEF = [
    ("periorbital_left",  [33,133,159,145,153,144,163,7], 0.35),
    ("periorbital_right", [362,263,386,374,380,373,390,249], 0.35),
    ("left_cheek",        [234,93,132,58,172,136,150,149], 0.35),
    ("right_cheek",       [454,323,361,288,397,365,379,378], 0.35),
    ("forehead",          [10,338,297,332,284,251,389,356,70,63,105,66,107,9,336,296], 0.15),
    ("nose",              [6,197,195,5,4,1,19,94,2,164,129,209,49,48,64,98,358,429,279,278,294,327], 0.3),
    ("lips",              [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146], 0.2),
    ("sclera_left",       [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246], 0.1),
]

DEFICIENCIES = [
    "iron_deficiency","b12_deficiency","vitamin_d_deficiency","zinc_deficiency",
    "omega3_deficiency","vitamin_a_deficiency","vitamin_c_deficiency",
    "poor_sleep_quality","hormonal_imbalance","dehydration","high_stress",
]

FOOD_RECS = {
    "iron_deficiency":      ["spinach","lentils","red meat","tofu","pumpkin seeds"],
    "b12_deficiency":       ["eggs","dairy","salmon","beef liver","fortified cereals"],
    "vitamin_d_deficiency": ["fatty fish","egg yolks","fortified milk","mushrooms"],
    "zinc_deficiency":      ["oysters","beef","chickpeas","cashews","pumpkin seeds"],
    "omega3_deficiency":    ["salmon","walnuts","flaxseed","chia seeds","mackerel"],
    "vitamin_a_deficiency": ["sweet potato","carrots","kale","egg yolks","liver"],
    "vitamin_c_deficiency": ["citrus fruits","bell peppers","broccoli","kiwi"],
    "poor_sleep_quality":   ["improve sleep schedule","reduce caffeine after 2pm","magnesium-rich foods"],
    "hormonal_imbalance":   ["see a doctor","reduce sugar","increase fiber","healthy fats"],
    "dehydration":          ["drink 8+ glasses water daily","cucumber","watermelon"],
    "high_stress":          ["meditation","exercise","B-complex vitamins","magnesium"],
}
LIFESTYLE_ADVICE = {
    "iron_deficiency":      "Pair iron-rich foods with Vitamin C to boost absorption.",
    "b12_deficiency":       "B12 mainly from animal sources. Vegans/vegetarians should supplement.",
    "vitamin_d_deficiency": "Get 15–30 min of sunlight daily. Consider D3 supplement in winter.",
    "zinc_deficiency":      "Soak legumes before cooking to reduce phytates.",
    "omega3_deficiency":    "Aim for 2 servings of fatty fish per week or consider fish oil.",
    "vitamin_a_deficiency": "Fat-soluble vitamin — pair with healthy fats for absorption.",
    "vitamin_c_deficiency": "Cooking destroys Vitamin C. Eat some raw fruits/vegetables daily.",
    "poor_sleep_quality":   "Consistent sleep/wake times matter more than total hours. Aim 7–9 hrs.",
    "hormonal_imbalance":   "Requires medical evaluation. Diet supports, not replaces treatment.",
    "dehydration":          "Thirst is a late signal. Aim for pale yellow urine.",
    "high_stress":          "Chronic stress depletes B vitamins and magnesium. Both diet and stress management needed.",
}

# v2 Recalibrated CPTs
CPT_LIKELIHOOD = np.array([
#    iron   b12   vitD  zinc  om3   vitA  vitC  sleep  horm  dehy  stress
    [0.72,0.68,0.40,0.18,0.20,0.18,0.22,0.68,0.28,0.32,0.48],  # dark_circle
    [0.28,0.22,0.18,0.12,0.18,0.12,0.12,0.82,0.22,0.48,0.62],  # eye_bag
    [0.15,0.12,0.38,0.82,0.52,0.72,0.18,0.35,0.88,0.18,0.42],  # acne ← zinc+hormonal strongly
    [0.20,0.18,0.28,0.18,0.42,0.22,0.52,0.52,0.28,0.38,0.62],  # wrinkle
    [0.18,0.22,0.22,0.28,0.65,0.28,0.18,0.18,0.32,0.22,0.38],  # redness
    [0.28,0.32,0.52,0.18,0.22,0.28,0.62,0.18,0.38,0.18,0.22],  # dark_spot
    [0.12,0.12,0.28,0.58,0.38,0.48,0.12,0.28,0.65,0.12,0.32],  # blackhead
    [0.18,0.18,0.48,0.12,0.18,0.18,0.28,0.12,0.72,0.12,0.18],  # melasma
    [0.12,0.12,0.22,0.55,0.32,0.42,0.12,0.22,0.58,0.12,0.28],  # whitehead
    [0.18,0.18,0.28,0.48,0.32,0.32,0.22,0.28,0.55,0.12,0.28],  # acne_scar
    [0.12,0.12,0.18,0.18,0.58,0.22,0.18,0.12,0.28,0.18,0.42],  # vascular
], dtype=np.float32)

# v2 Recalibrated priors (omega3 was 0.50, now 0.15)
PRIOR_DEFICIENCY = np.array(
    [0.15,0.12,0.22,0.12,0.15,0.08,0.09,0.28,0.14,0.25,0.22],
    dtype=np.float32)


def count_to_severity(n: int) -> float:
    if n == 0:  return 0.0
    if n == 1:  return 0.35
    if n <= 3:  return 0.50
    if n <= 6:  return 0.65
    if n <= 10: return 0.78
    if n <= 20: return 0.88
    return 1.0


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════

class SeverityMLPv2(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_features=N_FEATURES, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256,64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64,1))
            for _ in range(n_features)
        ])
    def forward(self, x):
        s = self.backbone(x)
        return torch.cat([h(s) for h in self.heads], dim=-1)
    def predict_with_uncertainty(self, x, n_passes=25):
        self.train()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.forward(x)) for _ in range(n_passes)])
        self.eval()
        return probs.mean(0), probs.std(0)


class SeverityMLPv1(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_features=N_FEATURES, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.net  = nn.Sequential(
            nn.Linear(feat_dim,512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,128),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128,n_features), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(self.norm(x))
    def predict_with_uncertainty(self, x, n_passes=20):
        self.train()
        with torch.no_grad():
            p = torch.stack([self.forward(x) for _ in range(n_passes)])
        self.eval()
        return p.mean(0), p.std(0)


# ═══════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════

_models = {}

def get_models(device: str) -> dict:
    if _models: return _models

    print("  Loading models...", flush=True)

    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    assert LANDMARK_MODEL.exists(), f"Run step7_preprocessing.py first"
    opts = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARK_MODEL)),
        num_faces=1, min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4, min_tracking_confidence=0.4,
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    _models["face_mesh"] = FaceLandmarker.create_from_options(opts)
    _models["mp"]        = mp
    print("    ✅ MediaPipe FaceLandmarker")

    from ultralytics import YOLO
    assert YOLO_WEIGHTS.exists(), f"YOLO weights not found: {YOLO_WEIGHTS}"
    _models["yolo"] = YOLO(str(YOLO_WEIGHTS))
    print(f"    ✅ YOLOv8 detector  (conf_threshold={CONF_THRESHOLD})")

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(DINOV2_MODEL)
    dinov2.eval().to(device)
    _models["dinov2"] = dinov2
    print("    ✅ DINOv2")

    model_path = SEVERITY_MODEL_V2 if SEVERITY_MODEL_V2.exists() else SEVERITY_MODEL_V1
    assert model_path.exists(), "No severity model. Run: python step9_bayesian_engine.py --train"
    ckpt    = torch.load(str(model_path), map_location=device, weights_only=False)
    version = ckpt.get("version", 1)
    mlp     = SeverityMLPv2() if version == 2 else SeverityMLPv1()
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(device).eval()
    _models["mlp"]        = mlp
    _models["mlp_version"] = version
    _models["feat_mean"]  = np.array(ckpt["mean"], dtype=np.float32)
    _models["feat_std"]   = np.array(ckpt["std"],  dtype=np.float32)
    _models["device"]     = device
    print(f"    ✅ Severity MLP v{version}")
    print("  All models ready.\n")
    return _models


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess_image(img_bgr, models):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp = models["mp"]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = models["face_mesh"].detect(mp_img)
    if not result.face_landmarks: return None

    pts = np.array([[lm.x*w, lm.y*h] for lm in result.face_landmarks[0]], dtype=np.float32)
    le  = pts[[33,133]].mean(0);  re = pts[[362,263]].mean(0)
    tl  = np.array([ALIGN_SIZE*0.35, ALIGN_SIZE*0.40])
    tr  = np.array([ALIGN_SIZE*0.65, ALIGN_SIZE*0.40])
    sv  = re - le; dv = tr - tl
    sc  = np.linalg.norm(dv) / (np.linalg.norm(sv) + 1e-6)
    ang = np.degrees(np.arctan2(dv[1],dv[0]) - np.arctan2(sv[1],sv[0]))
    M   = cv2.getRotationMatrix2D(tuple(le), -ang, sc)
    M[0,2] += tl[0]-le[0]; M[1,2] += tl[1]-le[1]
    aligned = cv2.warpAffine(img_bgr, M, (ALIGN_SIZE,ALIGN_SIZE),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ones   = np.ones((len(pts),1), dtype=np.float32)
    lm_aln = np.clip((M @ np.hstack([pts,ones]).T).T, 0, ALIGN_SIZE-1)

    # LAB normalize (for DINOv2 + YOLO input)
    ar       = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    raw_rgb  = ar.copy()   # ← save raw BEFORE normalization for color analysis
    lab = cv2.cvtColor(ar, cv2.COLOR_RGB2LAB).astype(np.float32)
    for i in range(3):
        p2,p98 = np.percentile(lab[:,:,i],(2,98))
        if p98>p2: lab[:,:,i] = np.clip((lab[:,:,i]-p2)/(p98-p2)*255,0,255)
    norm_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    regions = {}
    for name, idxs, pad in REGIONS_DEF:
        pr = lm_aln[idxs]
        x1,y1 = int(pr[:,0].min()), int(pr[:,1].min())
        x2,y2 = int(pr[:,0].max()), int(pr[:,1].max())
        bw,bh = x2-x1, y2-y1
        x1,y1 = max(0,x1-int(bw*pad)), max(0,y1-int(bh*pad))
        x2,y2 = min(ALIGN_SIZE,x2+int(bw*pad)), min(ALIGN_SIZE,y2+int(bh*pad))
        crop  = norm_rgb[y1:y2, x1:x2]
        if crop.size < 48: crop = np.zeros((16,16,3), dtype=np.uint8)
        regions[name] = {"bbox":[x1,y1,x2,y2], "crop":crop}

    return {"aligned_rgb":norm_rgb, "aligned_bgr":cv2.cvtColor(norm_rgb,cv2.COLOR_RGB2BGR),
            "raw_rgb":raw_rgb,   # ← un-normalized, for color analysis
            "landmarks":lm_aln, "regions":regions}


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — LAB COLOR ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_color_features(raw_rgb, regions):
    """
    Color analysis on the RAW (pre-normalization) aligned image.
    LAB normalization stretches L to fill 0-255 in every image,
    making absolute thresholds meaningless. Raw image preserves
    actual brightness relationships.

    OpenCV LAB ranges:
      L: 0–255  (black→white)
      A: 0–255  (128=neutral, >128=red, <128=green)
      B: 0–255  (128=neutral, >128=yellow, <128=blue)
    """
    lab = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    out = {}

    def crop(name):
        r = regions.get(name, {})
        b = r.get("bbox") if isinstance(r, dict) else None
        if b and b[2] > b[0] and b[3] > b[1]: return b
        return None

    # ── Pallor ────────────────────────────────────────────────
    # Healthy skin L ≈ 160–200. Pale skin L < 155.
    cl = []
    for n in ["left_cheek", "right_cheek"]:
        b = crop(n)
        if b: cl.extend(L[b[1]:b[3], b[0]:b[2]].flatten())
    if cl:
        mL = float(np.mean(cl))
        score = float(np.clip((155 - mL) / 40, 0, 1))
        if score > 0.15:
            out["pallor_color"] = (score, min(1.0, len(cl)/2000))

    # ── Redness ───────────────────────────────────────────────
    # Neutral A ≈ 128. Redness: A > 135.
    sa = []
    for n in ["left_cheek", "right_cheek", "forehead"]:
        b = crop(n)
        if b: sa.extend(A[b[1]:b[3], b[0]:b[2]].flatten())
    if sa:
        mA = float(np.mean(sa))
        score = float(np.clip((mA - 135) / 20, 0, 1))
        if score > 0.10:
            out["redness_color"] = (score, min(1.0, len(sa)/3000))

    # ── Dark circle depth ─────────────────────────────────────
    # On the RAW image, actual dark circles show ≥20 L unit diff.
    # Normal anatomy: 5–15 L units (not dark circles, just orbital shadow).
    pl = []
    for n in ["periorbital_left", "periorbital_right"]:
        b = crop(n)
        if b: pl.extend(L[b[1]:b[3], b[0]:b[2]].flatten())
    if pl and cl:
        abs_diff = float(np.mean(cl)) - float(np.mean(pl))
        if abs_diff > 20:   # genuine pigmentation, not just shadow
            score = float(np.clip((abs_diff - 20) / 35, 0, 1))
            out["dark_circle_color"] = (score, min(1.0, len(pl)/500))

    # ── Yellow sclera ─────────────────────────────────────────
    # Healthy sclera B ≈ 128–138. Jaundice/B12: B > 145.
    b = crop("sclera_left")
    if b:
        sB = B[b[1]:b[3], b[0]:b[2]]
        score = float(np.clip((float(np.mean(sB)) - 145) / 20, 0, 1))
        if score > 0.15:
            out["yellow_sclera_color"] = (score, min(1.0, sB.size/300))

    # ── Skin texture roughness ────────────────────────────────
    stds = []
    for n in ["left_cheek", "right_cheek", "forehead"]:
        b = crop(n)
        if b:
            r = L[b[1]:b[3], b[0]:b[2]]
            if r.size > 100: stds.append(float(np.std(r)))
    if stds:
        score = float(np.clip((float(np.mean(stds)) - 10) / 20, 0, 1))
        if score > 0.15:
            out["skin_texture_color"] = (score, 0.7)

    # ── Oiliness ──────────────────────────────────────────────
    # On the RAW image, oily T-zone has genuine bright specular patches.
    # Strategy: measure fraction of T-zone pixels above BOTH:
    #   (a) regional mean + 2σ  (relative — normalization-robust)
    #   (b) absolute L > 200    (absolute — ensures real brightness)
    # Both conditions must be met for a pixel to count as specular.
    shine_scores = []
    for n in ["forehead", "nose"]:
        b = crop(n)
        if b:
            r = L[b[1]:b[3], b[0]:b[2]]
            if r.size > 100:
                r_mean = float(np.mean(r))
                r_std  = float(np.std(r))
                rel_thresh = r_mean + 2.0 * r_std
                # Both relative AND absolute threshold required
                specular = (r > rel_thresh) & (r > 200)
                shine_scores.append(float(specular.sum()) / r.size)

    if shine_scores:
        mean_spec = float(np.mean(shine_scores))
        # 5% = gaussian baseline → 0.0
        # 20%+ = genuinely oily → 1.0
        score = float(np.clip((mean_spec - 0.05) / 0.15, 0, 1))
        if score > 0.10:
            out["oiliness_color"] = (score, 0.75)

    # ── Lip pallor ────────────────────────────────────────────
    # Healthy lips A ≈ 140–150. Pale lips A < 135.
    b = crop("lips")
    if b:
        lipA = float(np.mean(A[b[1]:b[3], b[0]:b[2]]))
        score = float(np.clip((135 - lipA) / 15, 0, 1))
        if score > 0.20:
            out["lip_pallor_color"] = (score, 0.6)

    return out


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — YOLO DETECTION (count-based)
# ═══════════════════════════════════════════════════════════════

def run_yolo(aligned_bgr, models):
    results = models["yolo"].predict(
        source=aligned_bgr, conf=CONF_THRESHOLD,
        verbose=False, device=models["device"])
    dets   = {}
    counts = {}
    boxes  = results[0].boxes if results else None
    if boxes is not None:
        for cls_id, conf in zip(boxes.cls.cpu().int().tolist(),
                                  boxes.conf.cpu().tolist()):
            fn = YOLO_CLASS_NAMES.get(cls_id)
            if not fn: continue
            # Per-class confidence gate — ambiguous classes need higher conf
            min_conf = PER_CLASS_CONF.get(fn, CONF_THRESHOLD)
            if conf < min_conf: continue
            counts[fn] = counts.get(fn,0) + 1
            if conf > dets.get(fn,0): dets[fn] = round(conf,3)

    # Count-based severity for small lesion classes
    for fn in SMALL_LESION_FEATS:
        n = counts.get(fn,0)
        if n > 0:
            dets[fn] = max(dets.get(fn,0), count_to_severity(n))

    return dets, counts


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — DINOv2 FEATURES
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_dinov2_features(regions, models):
    device = models["device"]
    embs   = []
    for rname in REGION_ORDER:
        crop = regions.get(rname,{}).get("crop", np.zeros((16,16,3),dtype=np.uint8))
        img  = cv2.resize(crop,(224,224)).astype(np.float32)/255.0
        img  = (img - IMAGENET_MEAN) / IMAGENET_STD
        t    = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)
        out  = models["dinov2"](pixel_values=t)
        cls  = F.normalize(out.last_hidden_state[:,0,:], dim=-1)
        embs.append(cls.cpu().float().numpy()[0])
    return np.concatenate(embs)


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — SEVERITY MLP
# ═══════════════════════════════════════════════════════════════

def run_severity_mlp(feat_vec, models):
    fn  = (feat_vec - models["feat_mean"]) / (models["feat_std"] + 1e-6)
    x   = torch.from_numpy(fn).float().unsqueeze(0).to(models["device"])
    s,u = models["mlp"].predict_with_uncertainty(x, n_passes=MC_PASSES)
    return s.squeeze(0).cpu().numpy(), u.squeeze(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — BAYESIAN INFERENCE v2
# ═══════════════════════════════════════════════════════════════

def bayesian_inference_v2(severity, uncertainty, color_features=None, yolo_counts=None):
    if color_features is None: color_features = {}
    if yolo_counts    is None: yolo_counts    = {}

    # exp-based confidence: high uncertainty → low weight
    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0,1).copy()

    # Override with count-based severity for lesion features
    cf_map = {"acne":2,"blackhead":6,"whitehead":8,"acne_scar":9,"dark_spot":5}
    for fn, fi in cf_map.items():
        n = yolo_counts.get(fn, 0)
        if n > 0: evidence[fi] = max(evidence[fi], count_to_severity(n))

    # Supplement with color-based redness + dark circle depth
    if "redness_color" in color_features:
        sc,cf = color_features["redness_color"]
        evidence[4] = float(np.clip(evidence[4]*0.6 + sc*cf*0.4, 0, 1))
    if "dark_circle_color" in color_features:
        sc,cf = color_features["dark_circle_color"]
        evidence[0] = float(np.clip(evidence[0]*0.7 + sc*cf*0.3, 0, 1))

    # Bayesian update
    log_post = np.log(PRIOR_DEFICIENCY + 1e-9).copy()
    for fi in range(N_FEATURES):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_LIKELIHOOD[fi]
        if ev >= 0.15:
            like = ev*p + (1-ev)*(1-p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            # Absent feature penalizes deficiencies that predict it
            log_post += np.where(p>0.4, np.log(1-p*0.3+1e-9), 0.0) * conf * 0.3

    log_post -= log_post.max()
    post = np.exp(log_post)

    # Post-hoc color boosts
    if "pallor_color" in color_features:
        sc,cf = color_features["pallor_color"]
        if sc>0.3: post[0]*=(1+sc*cf*0.8); post[1]*=(1+sc*cf*0.6)
    if "yellow_sclera_color" in color_features:
        sc,cf = color_features["yellow_sclera_color"]
        if sc>0.25: post[1]*=(1+sc*cf*1.0)
    if "oiliness_color" in color_features:
        sc,cf = color_features["oiliness_color"]
        if sc>0.4: post[8]*=(1+sc*cf*0.6); post[3]*=(1+sc*cf*0.4)
    if "skin_texture_color" in color_features:
        sc,cf = color_features["skin_texture_color"]
        if sc>0.4: post[5]*=(1+sc*cf*0.5); post[3]*=(1+sc*cf*0.4)

    post /= (post.sum() + 1e-9)
    return post


# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_output(severity, uncertainty, yolo_dets, yolo_counts,
                   posterior, color_features, timing):
    # Detected features
    features_out = {}
    for i, name in enumerate(SKIN_FEATURES):
        s   = float(severity[i])
        u   = float(uncertainty[i])
        yc  = yolo_dets.get(name, 0.0)
        cnt = yolo_counts.get(name, 0)
        cnt_sev = count_to_severity(cnt) if name in SMALL_LESION_FEATS else 0.0
        combined = max(s, yc * 0.85, cnt_sev)
        if combined > 0.12 or cnt > 0:
            srcs = []
            if s > 0.12:  srcs.append("dinov2")
            if yc > 0:    srcs.append(f"yolo(×{cnt})" if cnt>1 else "yolo")
            features_out[name] = {
                "severity":    round(combined,3),
                "level":       "high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                "confidence":  round(float(np.exp(-u*3)),2),
                "yolo_count":  cnt,
                "detected_by": srcs,
            }

    # Add color-only features
    c2f = {"pallor_color":"pallor","redness_color":"skin_redness",
           "yellow_sclera_color":"yellow_sclera","oiliness_color":"oily_skin",
           "lip_pallor_color":"lip_pallor"}
    for ck, fname in c2f.items():
        if ck in color_features:
            sc,cf = color_features[ck]
            if sc>0.30 and cf>0.4 and fname not in features_out:
                features_out[fname] = {
                    "severity":round(float(sc),3),
                    "level":"high" if sc>0.6 else "moderate" if sc>0.35 else "mild",
                    "confidence":round(float(cf),2),
                    "yolo_count":0, "detected_by":["color_analysis"],
                }

    sdef = sorted(enumerate(posterior), key=lambda x:-x[1])
    deficiencies = {
        DEFICIENCIES[i]: {
            "probability":     round(float(p),4),
            "probability_pct": f"{p*100:.1f}%",
            "priority_rank":   r,
            "foods":           FOOD_RECS.get(DEFICIENCIES[i],[]),
            "advice":          LIFESTYLE_ADVICE.get(DEFICIENCIES[i],""),
            "confidence_band": "high" if p>0.20 else "moderate" if p>0.10 else "low",
        }
        for r,(i,p) in enumerate(sdef,1)
    }
    top = [
        {"rank":r,"issue":DEFICIENCIES[i],"probability":f"{posterior[i]*100:.1f}%",
         "priority":"HIGH" if posterior[i]>0.20 else "MODERATE" if posterior[i]>0.10 else "LOW",
         "top_foods":FOOD_RECS.get(DEFICIENCIES[i],[])[:3],
         "advice":LIFESTYLE_ADVICE.get(DEFICIENCIES[i],""),
        }
        for r,(i,p) in enumerate(sdef[:5],1) if p>0.08
    ]
    return {
        "status":"success",
        "features_detected":   features_out,
        "deficiency_analysis": deficiencies,
        "top_insights":        top,
        "yolo_detections":     yolo_dets,
        "yolo_counts":         yolo_counts,
        "color_features":      {k:{"score":round(v[0],3),"conf":round(v[1],2)}
                                 for k,v in color_features.items()},
        "timing_ms":           {k:round(v*1000,1) for k,v in timing.items()},
        "posterior":           list(posterior),
        "disclaimer": "FaceFuel provides wellness awareness only — not medical diagnosis. "
                      "Visual signs have multiple causes. Consult a qualified healthcare "
                      "provider before making health decisions based on these results.",
    }


# ═══════════════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_inference(image_path, device=None, visualize=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = get_models(device)
    timing = {}

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {"status":"error","message":f"Cannot read: {image_path}"}

    t = time.time()
    prep = preprocess_image(img_bgr, models)
    timing["preprocessing"] = time.time()-t
    if prep is None:
        return {"status":"no_face_detected",
                "message":"No face detected. Use a clear front-facing photo in good lighting.",
                "tips":["Face camera directly","Good lighting","Remove glasses/mask","Keep in focus"]}

    t = time.time()
    color_features = analyze_color_features(prep["raw_rgb"], prep["regions"])
    timing["color_analysis"] = time.time()-t

    t = time.time()
    yolo_dets, yolo_counts = run_yolo(prep["aligned_bgr"], models)
    timing["yolo"] = time.time()-t

    t = time.time()
    feat_vec = extract_dinov2_features(prep["regions"], models)
    timing["dinov2"] = time.time()-t

    t = time.time()
    severity, uncertainty = run_severity_mlp(feat_vec, models)
    timing["severity_mlp"] = time.time()-t

    t = time.time()
    posterior = bayesian_inference_v2(severity, uncertainty, color_features, yolo_counts)
    timing["bayesian"] = time.time()-t

    result = format_output(severity, uncertainty, yolo_dets, yolo_counts,
                            posterior, color_features, timing)
    if visualize:
        vp = OUT_DIR / f"{Path(image_path).stem}_result.jpg"
        save_visualization(prep, result, str(vp))
        result["visualization"] = str(vp)
    return result


def save_visualization(prep, result, out_path):
    vis   = cv2.cvtColor(prep["aligned_rgb"], cv2.COLOR_RGB2BGR).copy()
    vis   = cv2.resize(vis,(512,512))
    scale = 512/ALIGN_SIZE
    colors = [(255,100,100),(100,100,255),(100,255,100),(255,255,100),
               (255,100,255),(100,255,255),(200,150,50),(150,200,50)]
    for ci,(name,_,__) in enumerate(REGIONS_DEF):
        x1,y1,x2,y2 = prep["regions"][name]["bbox"]
        cv2.rectangle(vis,(int(x1*scale),int(y1*scale)),(int(x2*scale),int(y2*scale)),colors[ci],1)
    panel = np.zeros((180,512,3),dtype=np.uint8)
    cv2.putText(panel,"FaceFuel v2 — Top Insights",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
    for i,ins in enumerate(result.get("top_insights",[])[:3]):
        y = 50+i*45
        col = (80,220,80) if ins["priority"]=="HIGH" else (80,180,220) if ins["priority"]=="MODERATE" else (160,160,160)
        cv2.putText(panel,f"#{ins['rank']} {ins['issue'].replace('_',' ')}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,col,1)
        cv2.putText(panel,f"    {ins['probability']}  →  {', '.join(ins.get('top_foods',[])[:2])}",(10,y+18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    cv2.imwrite(out_path, np.vstack([vis,panel]))
    print(f"  Visualization: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nFaceFuel v2 — Inference  [{device}]")
    if args.image:
        r = run_inference(args.image, device, visualize=args.visualize)
        feats = r.get("features_detected",{})
        if feats:
            print(f"\n  Features ({len(feats)}):")
            for n,i in sorted(feats.items(),key=lambda x:-x[1]["severity"]):
                cnt = i.get("yolo_count",0)
                print(f"    {n:<22s}  {i['severity']:.2f}  [{i['level']}]  ×{cnt}" if cnt>1
                      else f"    {n:<22s}  {i['severity']:.2f}  [{i['level']}]")
        print(f"\n  Top insights:")
        for ins in r.get("top_insights",[]):
            print(f"    #{ins['rank']}  {ins['issue'].replace('_',' '):<25s}  {ins['probability']:>6s}  [{ins['priority']}]")
        out = OUT_DIR/f"{Path(args.image).stem}_result.json"
        out.write_text(json.dumps(r,indent=2,default=str))
    elif args.test:
        samples = list(Path("facefuel_datasets/preprocessed/aligned").glob("*_aligned.jpg"))[:5]
        for p in samples:
            r = run_inference(p, device, visualize=True)
            print(f"\n{p.name[:40]}: {len(r.get('features_detected',{}))} features")
            for ins in r.get("top_insights",[])[:2]:
                print(f"  → {ins['issue']}  {ins['probability']}")
    elif args.benchmark:
        samples = list(Path("facefuel_datasets/preprocessed/aligned").glob("*_aligned.jpg"))[:20]
        get_models(device)
        t0=time.time()
        for p in samples: run_inference(p,device)
        el=time.time()-t0
        print(f"\n  {len(samples)} imgs  {el:.2f}s  {el/len(samples)*1000:.0f}ms/img  {len(samples)/el:.1f} FPS")
    else:
        print("  --image FILE [--visualize] | --test | --benchmark")