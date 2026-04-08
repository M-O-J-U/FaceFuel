"""
FaceFuel — Eye Inference Pipeline
===================================
Analyzes eye region from a selfie for visual biomarkers.
Extracted from the SAME selfie as face — no separate image needed.

Eye features detected:
  conjunctival_pallor  → iron deficiency, B12 anemia
  scleral_icterus      → liver stress (yellow sclera)
  xanthelasma          → cholesterol imbalance

Run:
  python eye_inference.py --image selfie.jpg
  python eye_inference.py --test
"""

import os, sys, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["GLOG_minloglevel"] = "2"

# ── Paths ─────────────────────────────────────────────────────
EYE_YOLO = next(Path(".").rglob("eye_v1/weights/best.pt"), None)
EYE_MLP  = Path("facefuel_models/eye_severity_mlp.pt")

DINOV2_MODEL  = "facebook/dinov2-small"
FEAT_DIM      = 1152   # 3 regions × 384
MC_PASSES     = 25
CONF_THRESH   = 0.30
# Per-class minimum confidence (scleral_icterus needs higher confidence)
PER_CLASS_CONF = {
    "conjunctival_pallor": 0.40,
    "scleral_icterus":     0.45,   # moderate bar with LAB gate as backup
    "xanthelasma":         0.45,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EYE_FEATURES = ["conjunctival_pallor", "scleral_icterus", "xanthelasma"]
YOLO_CLASS_NAMES = {0:"conjunctival_pallor", 1:"scleral_icterus", 2:"xanthelasma"}

EYE_DEFICIENCIES = [
    "iron_deficiency", "b12_deficiency", "vitamin_d_deficiency",
    "zinc_deficiency", "omega3_deficiency", "vitamin_a_deficiency",
    "vitamin_c_deficiency", "poor_sleep_quality", "hormonal_imbalance",
    "dehydration", "high_stress", "liver_stress", "gut_dysbiosis",
    "hypothyroid", "folate_deficiency", "cholesterol_imbalance",
]

CPT_EYE = np.array([
    [0.82,0.72,0.15,0.18,0.18,0.18,0.18,0.18,0.18,0.20,0.18,0.25,0.15,0.18,0.42,0.12],
    [0.15,0.15,0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.15,0.88,0.18,0.15,0.12,0.15],
    [0.12,0.12,0.12,0.12,0.52,0.12,0.12,0.12,0.22,0.12,0.18,0.15,0.12,0.18,0.12,0.95],
], dtype=np.float32)

PRIOR_EYE = np.array([
    0.15,0.12,0.22,0.12,0.15,0.08,0.09,
    0.28,0.14,0.25,0.22,0.10,0.08,0.08,0.10,
    0.12,
], dtype=np.float32)

FOOD_RECS = {
    "iron_deficiency":       ["spinach","lentils","red meat","tofu"],
    "b12_deficiency":        ["eggs","dairy","salmon","beef liver"],
    "vitamin_d_deficiency":  ["fatty fish","egg yolks","fortified milk"],
    "zinc_deficiency":       ["oysters","beef","chickpeas","cashews"],
    "omega3_deficiency":     ["salmon","walnuts","flaxseed","mackerel"],
    "vitamin_a_deficiency":  ["sweet potato","carrots","kale","liver"],
    "vitamin_c_deficiency":  ["citrus fruits","bell peppers","broccoli"],
    "poor_sleep_quality":    ["improve sleep schedule","magnesium"],
    "hormonal_imbalance":    ["see a doctor","reduce sugar"],
    "dehydration":           ["drink 8+ glasses water","cucumber"],
    "high_stress":           ["meditation","exercise","B-complex vitamins"],
    "liver_stress":          ["reduce alcohol","leafy greens","beets"],
    "gut_dysbiosis":         ["probiotics","fermented foods","fiber"],
    "hypothyroid":           ["consult doctor","iodine-rich foods"],
    "folate_deficiency":     ["leafy greens","lentils","asparagus"],
    "cholesterol_imbalance": ["oats","beans","avocado","olive oil","salmon"],
}

# Eye region definitions (fraction of image)
REGION_DEFS = [
    ("sclera_left",  0.10, 0.90, 0.00, 0.45),
    ("sclera_right", 0.10, 0.90, 0.55, 1.00),
    ("periorbital",  0.00, 0.35, 0.10, 0.90),
]


# ═══════════════════════════════════════════════════════════════
# LAB COLOR GATES — prevent false positives
# ═══════════════════════════════════════════════════════════════

def validate_eye_detections(eye_crop_bgr: np.ndarray, yolo_dets: dict) -> dict:
    """
    Apply LAB color gates to suppress false positive YOLO detections.
    Each eye condition has a specific color signature that must be present.
    Returns filtered detections.
    """
    if not yolo_dets:
        return yolo_dets

    lab  = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # Focus on sclera zone (center horizontal strip, excluding pupil/iris)
    y1, y2 = int(h*0.15), int(h*0.85)
    x1, x2 = int(w*0.05), int(w*0.95)
    region  = lab[y1:y2, x1:x2]
    if region.size < 100:
        return yolo_dets

    mL = float(np.mean(region[:,:,0]))
    mA = float(np.mean(region[:,:,1]))
    mB = float(np.mean(region[:,:,2]))

    validated = dict(yolo_dets)

    # ── Scleral icterus: requires genuinely yellow sclera ────────
    # Normal sclera: B channel ~128 (neutral), L > 160 (bright white)
    # Icteric sclera: B channel > 145 (yellow shift), often L < 170
    if "scleral_icterus" in validated:
        is_yellow = mB > 138          # genuine yellow shift (lowered for selfie quality)
        is_bright = mL > 130          # not just dark/shadowed
        if not (is_yellow and is_bright):
            del validated["scleral_icterus"]

    # ── Conjunctival pallor: requires genuinely pale (low L, low A) ──
    if "conjunctival_pallor" in validated:
        is_pale = mL < 165 and mA < 138   # pale = low brightness + low redness
        if not is_pale:
            del validated["conjunctival_pallor"]

    # ── Xanthelasma: requires yellowish deposits (high B, moderate L) ──
    # Xanthelasma appears as yellowish raised lesions near eyelid
    if "xanthelasma" in validated:
        has_yellow_deposits = mB > 138 and 120 < mL < 185
        if not has_yellow_deposits:
            # Keep if YOLO was very confident (>0.8) — lesions may be small
            if validated.get("xanthelasma", 0) < 0.80:
                del validated["xanthelasma"]

    return validated


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class EyeSeverityMLP(nn.Module):
    def __init__(self, feat_dim=1152, n_features=3, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim,256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256,128),      nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128,64), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(64,1))
            for _ in range(n_features)
        ])
    def forward(self, x):
        s = self.backbone(x)
        return torch.cat([h(s) for h in self.heads], dim=-1)
    def predict_with_uncertainty(self, x, n_passes=25):
        self.train()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.forward(x))
                                  for _ in range(n_passes)])
        self.eval()
        return probs.mean(0), probs.std(0)


# ═══════════════════════════════════════════════════════════════
# LAB COLOR GATES — prevent false positives
# ═══════════════════════════════════════════════════════════════

def validate_eye_detections(eye_crop_bgr: np.ndarray, yolo_dets: dict) -> dict:
    """
    Apply LAB color gates to suppress false positive YOLO detections.
    Each eye condition has a specific color signature that must be present.
    Returns filtered detections.
    """
    if not yolo_dets:
        return yolo_dets

    lab  = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # Focus on sclera zone (center horizontal strip, excluding pupil/iris)
    y1, y2 = int(h*0.15), int(h*0.85)
    x1, x2 = int(w*0.05), int(w*0.95)
    region  = lab[y1:y2, x1:x2]
    if region.size < 100:
        return yolo_dets

    mL = float(np.mean(region[:,:,0]))
    mA = float(np.mean(region[:,:,1]))
    mB = float(np.mean(region[:,:,2]))

    validated = dict(yolo_dets)

    # ── Scleral icterus: requires genuinely yellow sclera ────────
    # Normal sclera: B channel ~128 (neutral), L > 160 (bright white)
    # Icteric sclera: B channel > 145 (yellow shift), often L < 170
    if "scleral_icterus" in validated:
        is_yellow = mB > 138          # genuine yellow shift (lowered for selfie quality)
        is_bright = mL > 130          # not just dark/shadowed
        if not (is_yellow and is_bright):
            del validated["scleral_icterus"]

    # ── Conjunctival pallor: requires genuinely pale (low L, low A) ──
    if "conjunctival_pallor" in validated:
        is_pale = mL < 165 and mA < 138   # pale = low brightness + low redness
        if not is_pale:
            del validated["conjunctival_pallor"]

    # ── Xanthelasma: requires yellowish deposits (high B, moderate L) ──
    # Xanthelasma appears as yellowish raised lesions near eyelid
    if "xanthelasma" in validated:
        has_yellow_deposits = mB > 138 and 120 < mL < 185
        if not has_yellow_deposits:
            # Keep if YOLO was very confident (>0.8) — lesions may be small
            if validated.get("xanthelasma", 0) < 0.80:
                del validated["xanthelasma"]

    return validated


# ═══════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════

_eye_models = {}

def get_eye_models(device: str) -> dict:
    if _eye_models: return _eye_models

    from ultralytics import YOLO
    from transformers import AutoModel

    assert EYE_YOLO, "Eye YOLO weights not found."
    _eye_models["yolo"]   = YOLO(str(EYE_YOLO))
    print(f"    ✅ Eye YOLO (mAP=0.913)")

    dinov2 = AutoModel.from_pretrained(DINOV2_MODEL)
    dinov2.eval().to(device)
    _eye_models["dinov2"] = dinov2

    assert EYE_MLP.exists(), "Eye MLP not found. Run eye_severity.py --train"
    ckpt = torch.load(str(EYE_MLP), map_location=device, weights_only=False)
    mlp  = EyeSeverityMLP()
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(device).eval()
    _eye_models["mlp"]       = mlp
    _eye_models["feat_mean"] = np.array(ckpt["mean"], dtype=np.float32)
    _eye_models["feat_std"]  = np.array(ckpt["std"],  dtype=np.float32)
    _eye_models["device"]    = device
    print(f"    ✅ Eye Severity MLP (F1={ckpt.get('mean_f1',0):.3f})")
    return _eye_models


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def extract_eye_region(aligned_bgr: np.ndarray,
                        landmarks=None) -> np.ndarray:
    """
    Extract eye region from aligned face image.
    Uses the sclera/periorbital area (top 40%, center crop).
    If MediaPipe landmarks are available, uses exact eye bounding box.
    """
    h, w = aligned_bgr.shape[:2]

    # Eye region = rows 18-48% of aligned face (eyes sit in upper third)
    # Wider horizontal crop to capture sclera (whites of eyes)
    y1_frac, y2_frac = 0.18, 0.48
    x1_frac, x2_frac = 0.03, 0.97

    if landmarks is not None:
        try:
            pts = [(lm.x, lm.y) for lm in landmarks]
            # Eye corner landmarks: 33, 133 (left), 362, 263 (right)
            eye_ys = [pts[i][1] for i in [33,133,362,263] if i < len(pts)]
            eye_xs = [pts[i][0] for i in [33,133,362,263] if i < len(pts)]
            if eye_ys:
                cy = sum(eye_ys)/len(eye_ys)
                y1_frac = max(0.05, cy - 0.15)
                y2_frac = min(0.65, cy + 0.18)
        except Exception:
            pass

    y1, y2 = int(h*y1_frac), int(h*y2_frac)
    x1, x2 = int(w*x1_frac), int(w*x2_frac)
    crop = aligned_bgr[y1:y2, x1:x2]

    if crop.size < 100:
        crop = aligned_bgr[int(h*0.18):int(h*0.48), :]

    return cv2.resize(crop, (256, 128), interpolation=cv2.INTER_LINEAR)


def run_eye_yolo(eye_crop: np.ndarray, models: dict):
    """Run YOLO detection on eye crop."""
    results = models["yolo"].predict(
        source=eye_crop, conf=CONF_THRESH,
        verbose=False, device=models["device"])
    dets = {}
    if results and results[0].boxes is not None:
        for cls_id, conf in zip(
            results[0].boxes.cls.cpu().int().tolist(),
            results[0].boxes.conf.cpu().tolist()
        ):
            fn = YOLO_CLASS_NAMES.get(cls_id)
            min_conf = PER_CLASS_CONF.get(fn, CONF_THRESH)
            if fn and conf >= min_conf and conf > dets.get(fn, 0):
                dets[fn] = round(conf, 3)
    return dets


@torch.no_grad()
def extract_eye_features(eye_crop: np.ndarray, models: dict) -> np.ndarray:
    """Extract 3-region DINOv2 features from eye crop."""
    device   = models["device"]
    crop_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
    h, w     = crop_rgb.shape[:2]
    embs     = []

    for name, y1f, y2f, x1f, x2f in REGION_DEFS:
        y1, y2 = int(h*y1f), int(h*y2f)
        x1, x2 = int(w*x1f), int(w*x2f)
        patch  = crop_rgb[y1:y2, x1:x2]
        if patch.size < 48:
            patch = np.zeros((16,16,3), dtype=np.uint8)
        img = cv2.resize(patch,(224,224)).astype(np.float32)/255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        t   = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)
        out = models["dinov2"](pixel_values=t)
        cls = F.normalize(out.last_hidden_state[:,0,:], dim=-1)
        embs.append(cls.cpu().float().numpy()[0])

    return np.concatenate(embs)


def run_eye_severity(feat_vec: np.ndarray, models: dict):
    mean = models["feat_mean"]
    std  = models["feat_std"]
    fn   = (feat_vec - mean) / (std + 1e-6)
    x    = torch.from_numpy(fn).float().unsqueeze(0).to(models["device"])
    s, u = models["mlp"].predict_with_uncertainty(x, n_passes=MC_PASSES)
    return s.squeeze(0).cpu().numpy(), u.squeeze(0).cpu().numpy()


def eye_bayesian_inference(severity, uncertainty, yolo_dets=None):
    if yolo_dets is None: yolo_dets = {}
    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0, 1).copy()

    for fname, fi in [("conjunctival_pallor",0),
                       ("scleral_icterus",1),
                       ("xanthelasma",2)]:
        yc = yolo_dets.get(fname, 0.0)
        if yc > 0.5:
            evidence[fi] = max(evidence[fi], yc * 0.9)

    log_post = np.log(PRIOR_EYE + 1e-9).copy()
    for fi in range(3):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_EYE[fi]
        if ev >= 0.15:
            like = ev*p + (1-ev)*(1-p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            log_post += np.where(p>0.4, np.log(1-p*0.3+1e-9), 0.0)*conf*0.3

    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / (post.sum() + 1e-9)


# ═══════════════════════════════════════════════════════════════
# MAIN INFERENCE (called from server with aligned face crop)
# ═══════════════════════════════════════════════════════════════

def run_eye_inference(aligned_bgr: np.ndarray,
                       device: str,
                       landmarks=None) -> dict:
    """
    Full eye analysis pipeline.
    Input: aligned face BGR image (256×256) from face pipeline Stage 1.
    Returns: eye features + 16-deficiency posterior.
    """
    models  = get_eye_models(device)
    timing  = {}

    t = time.time()
    eye_crop = extract_eye_region(aligned_bgr, landmarks)
    timing["eye_crop"] = time.time()-t

    t = time.time()
    yolo_dets_raw = run_eye_yolo(eye_crop, models)
    # Apply LAB color gate to suppress false positives
    yolo_dets = validate_eye_detections(eye_crop, yolo_dets_raw)
    timing["yolo"] = time.time()-t

    t = time.time()
    feat_vec = extract_eye_features(eye_crop, models)
    timing["dinov2"] = time.time()-t

    t = time.time()
    severity, uncertainty = run_eye_severity(feat_vec, models)
    timing["severity_mlp"] = time.time()-t

    t = time.time()
    posterior = eye_bayesian_inference(severity, uncertainty, yolo_dets)
    timing["bayesian"] = time.time()-t

    # Format features
    features_out = {}
    for i, name in enumerate(EYE_FEATURES):
        s   = float(severity[i])
        u   = float(uncertainty[i])
        yc  = yolo_dets.get(name, 0.0)
        combined = max(s, yc * 0.85)
        # Scleral icterus: only report if YOLO confirmed it (color-gated)
        # MLP alone is unreliable for this class due to small training set
        if name == "scleral_icterus" and yc == 0.0:
            continue
        # Scleral icterus: only report if YOLO confirmed AND color gate passed
        if name == "scleral_icterus" and yc == 0.0:
            continue
        if combined > 0.12 or yc > 0:
            srcs = []
            if s > 0.12: srcs.append("dinov2")
            if yc > 0:   srcs.append("yolo")
            features_out[name] = {
                "severity":    round(combined, 3),
                "level":       "high" if combined>0.60 else
                               "moderate" if combined>0.35 else "mild",
                "confidence":  round(float(np.exp(-u*3)), 2),
                "detected_by": srcs,
            }

    sdef = sorted(enumerate(posterior), key=lambda x:-x[1])
    deficiencies = {
        EYE_DEFICIENCIES[i]: {
            "probability":     round(float(p), 4),
            "probability_pct": f"{p*100:.1f}%",
            "priority_rank":   r,
            "foods":           FOOD_RECS.get(EYE_DEFICIENCIES[i], []),
            "confidence_band": "high" if p>0.20 else
                               "moderate" if p>0.10 else "low",
        }
        for r,(i,p) in enumerate(sdef, 1)
    }

    # If no features detected, use near-uniform posterior
    # (suppress eye contribution in fusion when nothing found)
    if not features_out:
        posterior = np.full(len(EYE_DEFICIENCIES),
                            1.0/len(EYE_DEFICIENCIES), dtype=np.float32)

    return {
        "status":              "success",
        "modality":            "eye",
        "features_detected":   features_out,
        "deficiency_analysis": deficiencies,
        "yolo_detections":     yolo_dets,
        "timing_ms":           {k:round(v*1000,1) for k,v in timing.items()},
        "posterior":           posterior.tolist(),
        "has_findings":        len(features_out) > 0,
    }


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nFaceFuel — Eye Inference [{device}]")
    print("  Loading models...")
    get_eye_models(device)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ Cannot read: {args.image}")
            exit(1)
        r = run_eye_inference(img, device)
        feats = r.get("features_detected", {})
        if feats:
            print(f"\n  Eye features ({len(feats)}):")
            for n, info in sorted(feats.items(), key=lambda x:-x[1]["severity"]):
                print(f"    {n:<22s}  {info['severity']:.2f}  [{info['level']}]")
        print(f"\n  Top deficiencies:")
        post = r["posterior"]
        for i in np.argsort(post)[::-1][:5]:
            if post[i] > 0.08:
                print(f"    {EYE_DEFICIENCIES[i]:<25s}  {post[i]*100:.1f}%")
        t = r.get("timing_ms", {})
        print(f"\n  Speed: {sum(t.values()):.0f}ms total")

    elif args.test:
        test_imgs = (
            list(Path("eye_datasets/EYE_COMBINED/val/images").glob("*.jpg"))[:5] +
            list(Path("eye_datasets/EYE_COMBINED/val/images").glob("*.png"))[:3]
        )
        if not test_imgs:
            test_imgs = list(Path("eye_datasets").rglob("*.jpg"))[:5]
        for p in test_imgs:
            img = cv2.imread(str(p))
            if img is None: continue
            r = run_eye_inference(img, device)
            feats = r.get("features_detected", {})
            t     = r.get("timing_ms", {})
            top   = EYE_DEFICIENCIES[np.argmax(r["posterior"])]
            print(f"  {p.name[:45]:<47}  "
                  f"feats={len(feats)}  top={top:<25s}  {sum(t.values()):.0f}ms")
    else:
        print("  --image FILE | --test")