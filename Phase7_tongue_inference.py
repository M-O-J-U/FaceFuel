"""
FaceFuel — Tongue Inference Pipeline
=====================================
End-to-end tongue analysis:
  Image → Detect tongue body → Crop → 5 region DINOv2 features
        → Severity MLP → Bayesian inference → Deficiency posterior

Designed to fuse with face pipeline posterior in server.py.

Run:
  python tongue_inference.py --image tongue.jpg
  python tongue_inference.py --test
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
TONGUE_YOLO    = next(Path(".").rglob("tongue_v3_improved/weights/best.pt"),
                  next(Path(".").rglob("tongue_v2_retrain/weights/best.pt"), None))
TONGUE_MLP     = Path("facefuel_models/tongue_severity_mlp.pt")
OUT_DIR        = Path("facefuel_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
DINOV2_MODEL  = "facebook/dinov2-small"
FEAT_DIM      = 1920
MC_PASSES     = 20
CONF_THRESH   = 0.25

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS    = {".jpg", ".jpeg", ".png"}

TONGUE_FEATURES = [
    "tongue_body", "fissured", "crenated", "pale_tongue", "red_tongue",
    "yellow_coating", "white_coating", "thick_coating", "geographic",
    "smooth_glossy", "tooth_marked",
]
YOLO_CLASS_NAMES = {
    0:"tongue_body", 1:"fissured", 2:"crenated", 3:"pale_tongue",
    4:"red_tongue", 5:"yellow_coating", 6:"white_coating", 7:"thick_coating",
    8:"geographic", 9:"smooth_glossy", 10:"tooth_marked",
}
SMALL_LESION_FEATS = {"fissured", "crenated", "geographic", "tooth_marked", "black_hairy_tongue"}

# Per-class confidence thresholds (higher = stricter)
PER_CLASS_CONF = {
    "tongue_body":    0.40,
    "pale_tongue":    0.45,
    "red_tongue":     0.38,
    "yellow_coating": 0.35,
    "white_coating":  0.45,
    "geographic":     0.35,
    "smooth_glossy":  0.35,
}

# Region definitions (fraction of image)
REGION_DEFS = [
    ("tongue_tip",     0.00, 0.35, 0.20, 0.80),
    ("tongue_body",    0.25, 0.75, 0.15, 0.85),
    ("tongue_left",    0.20, 0.80, 0.00, 0.40),
    ("tongue_right",   0.20, 0.80, 0.60, 1.00),
    ("tongue_coating", 0.30, 0.70, 0.25, 0.75),
]

DEFICIENCIES = [
    "iron_deficiency", "b12_deficiency", "vitamin_d_deficiency",
    "zinc_deficiency", "omega3_deficiency", "vitamin_a_deficiency",
    "vitamin_c_deficiency", "poor_sleep_quality", "hormonal_imbalance",
    "dehydration", "high_stress", "liver_stress", "gut_dysbiosis",
    "hypothyroid", "folate_deficiency",
]

FOOD_RECS = {
    "iron_deficiency":      ["spinach", "lentils", "red meat", "tofu"],
    "b12_deficiency":       ["eggs", "dairy", "salmon", "beef liver"],
    "vitamin_d_deficiency": ["fatty fish", "egg yolks", "fortified milk"],
    "zinc_deficiency":      ["oysters", "beef", "chickpeas", "cashews"],
    "omega3_deficiency":    ["salmon", "walnuts", "flaxseed", "mackerel"],
    "vitamin_a_deficiency": ["sweet potato", "carrots", "kale", "liver"],
    "vitamin_c_deficiency": ["citrus fruits", "bell peppers", "broccoli"],
    "poor_sleep_quality":   ["improve sleep schedule", "magnesium", "reduce caffeine"],
    "hormonal_imbalance":   ["see a doctor", "reduce sugar", "healthy fats"],
    "dehydration":          ["drink 8+ glasses water", "cucumber", "watermelon"],
    "high_stress":          ["meditation", "exercise", "B-complex vitamins"],
    "liver_stress":          ["reduce alcohol", "leafy greens", "beets", "milk thistle"],
    "gut_dysbiosis":         ["probiotics", "fermented foods", "fiber", "reduce sugar"],
    "hypothyroid":           ["consult doctor", "iodine-rich foods", "selenium"],
    "folate_deficiency":     ["leafy greens", "lentils", "asparagus", "fortified cereals"],
}

CPT_TONGUE = np.array([
    [0.20,0.18,0.10,0.12,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10],
    [0.18,0.22,0.15,0.28,0.18,0.20,0.15,0.25,0.18,0.45,0.48,0.22,0.18,0.20,0.18],
    [0.15,0.18,0.20,0.72,0.18,0.22,0.15,0.20,0.25,0.28,0.22,0.18,0.18,0.68,0.18],
    [0.82,0.78,0.22,0.18,0.20,0.18,0.18,0.18,0.20,0.15,0.15,0.18,0.15,0.18,0.42],
    [0.28,0.85,0.18,0.22,0.25,0.18,0.22,0.18,0.22,0.18,0.22,0.28,0.22,0.18,0.88],
    [0.15,0.18,0.18,0.22,0.18,0.18,0.18,0.20,0.22,0.18,0.20,0.82,0.42,0.18,0.18],
    [0.12,0.15,0.18,0.22,0.15,0.18,0.15,0.18,0.22,0.18,0.22,0.35,0.88,0.18,0.15],
    [0.15,0.18,0.18,0.22,0.18,0.18,0.18,0.22,0.22,0.18,0.25,0.52,0.62,0.18,0.18],
    [0.18,0.20,0.18,0.72,0.20,0.22,0.18,0.18,0.20,0.18,0.20,0.18,0.18,0.18,0.22],
    [0.72,0.82,0.18,0.22,0.20,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.78],
    [0.18,0.20,0.18,0.68,0.18,0.20,0.18,0.22,0.25,0.35,0.22,0.18,0.20,0.72,0.18],
], dtype=np.float32)

PRIOR_DEFICIENCY = np.array([
    0.15,0.12,0.22,0.12,0.15,0.08,0.09,0.28,
    0.14,0.25,0.22,0.10,0.08,0.08,0.10
], dtype=np.float32)


def count_to_severity(n):
    if n==0: return 0.0
    if n==1: return 0.40
    if n<=3: return 0.55
    if n<=6: return 0.70
    if n<=10: return 0.82
    return 1.0


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class TongueSeverityMLP(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_features=11, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256,64), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(64,1))
            for _ in range(n_features)
        ])
    def forward(self, x):
        s = self.backbone(x)
        return torch.cat([h(s) for h in self.heads], dim=-1)
    def predict_with_uncertainty(self, x, n_passes=20):
        self.train()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.forward(x))
                                  for _ in range(n_passes)])
        self.eval()
        return probs.mean(0), probs.std(0)


# ═══════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════

_tongue_models = {}

def get_tongue_models(device: str) -> dict:
    if _tongue_models: return _tongue_models

    from ultralytics import YOLO
    from transformers import AutoModel

    assert TONGUE_YOLO, "Tongue YOLO weights not found. Run tongue training first."
    _tongue_models["yolo"]   = YOLO(str(TONGUE_YOLO))
    print(f"    ✅ Tongue YOLO ({TONGUE_YOLO.parent.parent.name})")

    dinov2 = AutoModel.from_pretrained(DINOV2_MODEL)
    dinov2.eval().to(device)
    _tongue_models["dinov2"] = dinov2

    assert TONGUE_MLP.exists(), "Tongue MLP not found. Run tongue_severity.py --train"
    ckpt = torch.load(str(TONGUE_MLP), map_location=device, weights_only=False)
    mlp  = TongueSeverityMLP()
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(device).eval()
    _tongue_models["mlp"]       = mlp
    _tongue_models["feat_mean"] = np.array(ckpt["mean"], dtype=np.float32)
    _tongue_models["feat_std"]  = np.array(ckpt["std"],  dtype=np.float32)
    _tongue_models["device"]    = device
    print(f"    ✅ Tongue Severity MLP (F1={ckpt.get('mean_f1',0):.3f})")
    return _tongue_models


# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════

def detect_and_crop_tongue(img_bgr: np.ndarray, models: dict):
    """
    Use YOLO to find tongue_body bounding box and crop it.
    Falls back to center crop if no detection.
    Returns cropped BGR image.
    """
    results = models["yolo"].predict(
        source=img_bgr, conf=0.30, verbose=False,
        device=models["device"], classes=[0])  # class 0 = tongue_body

    h, w = img_bgr.shape[:2]
    crop_bgr = None

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Use highest-confidence tongue_body detection
        boxes = results[0].boxes
        best  = boxes.conf.argmax().item()
        x1,y1,x2,y2 = boxes.xyxy[best].cpu().int().tolist()
        # Add 5% padding
        pad_x = int((x2-x1)*0.05); pad_y = int((y2-y1)*0.05)
        x1 = max(0, x1-pad_x); y1 = max(0, y1-pad_y)
        x2 = min(w, x2+pad_x); y2 = min(h, y2+pad_y)
        crop_bgr = img_bgr[y1:y2, x1:x2]

    if crop_bgr is None or crop_bgr.size < 100:
        # Fallback: assume tongue fills center 70% of image
        y1,y2 = int(h*0.15), int(h*0.85)
        x1,x2 = int(w*0.15), int(w*0.85)
        crop_bgr = img_bgr[y1:y2, x1:x2]

    # Resize to standard size
    crop_bgr = cv2.resize(crop_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    return crop_bgr


def run_tongue_yolo(crop_bgr: np.ndarray, models: dict):
    """Detect features on cropped tongue. Returns (detections, counts)."""
    results = models["yolo"].predict(
        source=crop_bgr, conf=CONF_THRESH,
        verbose=False, device=models["device"])
    dets   = {}
    counts = {}
    boxes  = results[0].boxes if results else None

    if boxes is not None:
        for cls_id, conf in zip(boxes.cls.cpu().int().tolist(),
                                  boxes.conf.cpu().tolist()):
            fn = YOLO_CLASS_NAMES.get(cls_id)
            if not fn or fn == "tongue_body": continue
            min_conf = PER_CLASS_CONF.get(fn, CONF_THRESH)
            if conf < min_conf: continue
            counts[fn] = counts.get(fn, 0) + 1
            if conf > dets.get(fn, 0): dets[fn] = round(conf, 3)

    for fn in SMALL_LESION_FEATS:
        n = counts.get(fn, 0)
        if n > 0: dets[fn] = max(dets.get(fn, 0), count_to_severity(n))

    return dets, counts


@torch.no_grad()
def extract_tongue_features(crop_bgr: np.ndarray, models: dict) -> np.ndarray:
    """Extract 5-region DINOv2 features from cropped tongue."""
    device  = models["device"]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w    = crop_rgb.shape[:2]
    embs    = []

    for name, y1f, y2f, x1f, x2f in REGION_DEFS:
        y1,y2 = int(h*y1f), int(h*y2f)
        x1,x2 = int(w*x1f), int(w*x2f)
        crop  = crop_rgb[y1:y2, x1:x2]
        if crop.size < 48:
            crop = np.zeros((16,16,3), dtype=np.uint8)
        img   = cv2.resize(crop, (224,224)).astype(np.float32)/255.0
        img   = (img - IMAGENET_MEAN) / IMAGENET_STD
        t     = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)
        out   = models["dinov2"](pixel_values=t)
        cls   = F.normalize(out.last_hidden_state[:,0,:], dim=-1)
        embs.append(cls.cpu().float().numpy()[0])

    return np.concatenate(embs)


def run_tongue_severity(feat_vec: np.ndarray, models: dict):
    mean  = models["feat_mean"]
    std   = models["feat_std"]
    fn    = (feat_vec - mean) / (std + 1e-6)
    x     = torch.from_numpy(fn).float().unsqueeze(0).to(models["device"])
    s, u  = models["mlp"].predict_with_uncertainty(x, n_passes=MC_PASSES)
    return s.squeeze(0).cpu().numpy(), u.squeeze(0).cpu().numpy()


def tongue_bayesian_inference(severity, uncertainty, yolo_counts=None):
    if yolo_counts is None: yolo_counts = {}
    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0, 1).copy()

    count_map = {"fissured":1,"crenated":2,"geographic":8,"tooth_marked":10}
    for fn, fi in count_map.items():
        n = yolo_counts.get(fn, 0)
        if n > 0: evidence[fi] = max(evidence[fi], count_to_severity(n))

    log_post = np.log(PRIOR_DEFICIENCY + 1e-9).copy()
    for fi in range(len(TONGUE_FEATURES)):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_TONGUE[fi]
        if ev >= 0.15:
            like = ev*p + (1-ev)*(1-p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            log_post += np.where(p>0.4, np.log(1-p*0.3+1e-9), 0.0)*conf*0.3

    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / (post.sum() + 1e-9)


# ═══════════════════════════════════════════════════════════════
# MAIN INFERENCE
# ═══════════════════════════════════════════════════════════════

def run_tongue_inference(image_path, device=None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    models  = get_tongue_models(device)
    timing  = {}

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {"status":"error","message":f"Cannot read: {image_path}"}

    # Stage 1: detect + crop tongue
    t = time.time()
    crop_bgr = detect_and_crop_tongue(img_bgr, models)
    timing["tongue_crop"] = time.time()-t

    # Stage 2: YOLO features on crop
    t = time.time()
    yolo_dets, yolo_counts = run_tongue_yolo(crop_bgr, models)
    timing["yolo"] = time.time()-t

    # Stage 3: DINOv2
    t = time.time()
    feat_vec = extract_tongue_features(crop_bgr, models)
    timing["dinov2"] = time.time()-t

    # Stage 4: Severity MLP
    t = time.time()
    severity, uncertainty = run_tongue_severity(feat_vec, models)
    timing["severity_mlp"] = time.time()-t

    # Stage 5: Bayesian
    t = time.time()
    posterior = tongue_bayesian_inference(severity, uncertainty, yolo_counts)
    timing["bayesian"] = time.time()-t

    # Format features
    features_out = {}
    for i, name in enumerate(TONGUE_FEATURES):
        if name == "tongue_body": continue
        s   = float(severity[i])
        u   = float(uncertainty[i])
        yc  = yolo_dets.get(name, 0.0)
        cnt = yolo_counts.get(name, 0)
        cnt_sev = count_to_severity(cnt) if name in SMALL_LESION_FEATS else 0.0
        combined = max(s, yc*0.85, cnt_sev)
        if combined > 0.12 or cnt > 0:
            srcs = []
            if s > 0.12: srcs.append("dinov2")
            if yc > 0:   srcs.append(f"yolo(×{cnt})" if cnt>1 else "yolo")
            features_out[name] = {
                "severity":    round(combined, 3),
                "level":       "high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                "confidence":  round(float(np.exp(-u*3)), 2),
                "yolo_count":  cnt,
                "detected_by": srcs,
            }

    # Format deficiencies
    sdef = sorted(enumerate(posterior), key=lambda x:-x[1])
    deficiencies = {
        DEFICIENCIES[i]: {
            "probability":     round(float(p), 4),
            "probability_pct": f"{p*100:.1f}%",
            "priority_rank":   r,
            "foods":           FOOD_RECS.get(DEFICIENCIES[i], []),
            "confidence_band": "high" if p>0.20 else "moderate" if p>0.10 else "low",
        }
        for r,(i,p) in enumerate(sdef, 1)
    }
    top_insights = [
        {
            "rank":        r,
            "issue":       DEFICIENCIES[i],
            "probability": f"{posterior[i]*100:.1f}%",
            "priority":    "HIGH" if posterior[i]>0.20 else "MODERATE" if posterior[i]>0.10 else "LOW",
            "top_foods":   FOOD_RECS.get(DEFICIENCIES[i], [])[:3],
        }
        for r,(i,p) in enumerate(sdef[:5], 1) if p>0.08
    ]

    return {
        "status":              "success",
        "modality":            "tongue",
        "features_detected":   features_out,
        "deficiency_analysis": deficiencies,
        "top_insights":        top_insights,
        "yolo_detections":     yolo_dets,
        "yolo_counts":         yolo_counts,
        "timing_ms":           {k:round(v*1000,1) for k,v in timing.items()},
        "posterior":           posterior.tolist(),
        "disclaimer":          "FaceFuel tongue analysis provides wellness awareness only. "
                               "Consult a healthcare provider for medical decisions.",
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nFaceFuel — Tongue Inference  [{device}]")
    print("  Loading models...")
    get_tongue_models(device)

    if args.image:
        r = run_tongue_inference(args.image, device)
        feats = r.get("features_detected", {})
        if feats:
            print(f"\n  Tongue features ({len(feats)}):")
            for n, info in sorted(feats.items(), key=lambda x:-x[1]["severity"]):
                print(f"    {n:<20s}  {info['severity']:.2f}  [{info['level']}]")
        print(f"\n  Top insights:")
        for ins in r.get("top_insights", []):
            print(f"    #{ins['rank']}  {ins['issue']:<25s}  {ins['probability']:>6s}  [{ins['priority']}]")
        t = r.get("timing_ms", {})
        print(f"\n  Speed: {sum(t.values()):.0f}ms total")
        out = OUT_DIR / f"{Path(args.image).stem}_tongue_result.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"  Saved: {out}")

    elif args.test:
        test_imgs = list(Path("tongue_datasets").rglob("*.jpg"))[:5]
        for p in test_imgs:
            r = run_tongue_inference(p, device)
            feats = r.get("features_detected", {})
            top   = r.get("top_insights", [{}])[0] if r.get("top_insights") else {}
            t     = r.get("timing_ms", {})
            print(f"  {p.name[:40]:<42s}  "
                  f"feats={len(feats)}  "
                  f"top={top.get('issue','—'):<20s}  "
                  f"{sum(t.values()):.0f}ms")
    else:
        print("  --image FILE | --test")