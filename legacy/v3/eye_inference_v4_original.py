"""
FaceFuel v4 — Eye Inference
============================
YOLO: 6 classes  mAP=0.990
MLP:  6 classes  F1=0.985
"""
import os, sys, time, warnings
import numpy as np
import cv2
import torch

os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

EYE_YOLO_PT = r"runs\detect\training_runs\eye_v2\weights\best.pt"
EYE_MLP_PT  = r"facefuel_models\eye_severity_mlp_v4.pt"

EYE_CLASSES = [
    "conjunctival_pallor","scleral_icterus","xanthelasma",
    "pterygium","conjunctivitis","eyelid_drooping",
]
NC = len(EYE_CLASSES)

PER_CLASS_CONF = {
    "conjunctival_pallor": 0.40,
    "scleral_icterus":     0.65,
    "xanthelasma":         0.45,
    "pterygium":           0.50,
    "conjunctivitis":      0.40,
    "eyelid_drooping":     0.45,
}
CONF_THRESH = 0.30

DEFICIENCY_ORDER = [
    "iron_deficiency","b12_deficiency","vitamin_d_deficiency","zinc_deficiency",
    "omega3_deficiency","vitamin_a_deficiency","vitamin_c_deficiency",
    "poor_sleep_quality","hormonal_imbalance","dehydration","high_stress",
    "liver_stress","gut_dysbiosis","hypothyroid","folate_deficiency",
    "cholesterol_imbalance","riboflavin_deficiency","autoimmune_risk",
    "eye_inflammation","copper_deficiency","skin_inflammation","oral_health_risk",
]

EYE_FEAT_DEF = {
    "conjunctival_pallor": ["iron_deficiency","b12_deficiency"],
    "scleral_icterus":     ["liver_stress"],
    "xanthelasma":         ["cholesterol_imbalance"],
    "pterygium":           ["vitamin_a_deficiency","dehydration"],
    "conjunctivitis":      ["vitamin_a_deficiency","vitamin_c_deficiency","eye_inflammation"],
    "eyelid_drooping":     ["hypothyroid"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_models = None


import torch.nn as nn
class SeverityMLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256),    nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128),    nn.GELU(), nn.Dropout(0.20),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128,64), nn.GELU(),
                          nn.Dropout(0.20), nn.Linear(64,1))
            for _ in range(n_cls)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_cls))
    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([head(h) for head in self.heads], dim=-1), self.log_vars


def get_eye_models(device=DEVICE):
    global _models
    if _models is not None:
        return _models
    from ultralytics import YOLO
    from transformers import AutoModel

    print("  Loading eye models...")
    yolo = YOLO(EYE_YOLO_PT)
    print(f"    ✅ Eye YOLO v4  mAP=0.990  6 classes")

    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino.eval()
    print(f"    ✅ DINOv2 ViT-S/14")

    mlp = None
    if os.path.exists(EYE_MLP_PT):
        try:
            ckpt      = torch.load(EYE_MLP_PT, map_location=device)
            n_active  = ckpt.get("n_active", NC)
            mlp       = SeverityMLP(1152, n_active).to(device)
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()
            active    = ckpt.get("active_idx", list(range(n_active)))
            classes   = ckpt.get("classes", EYE_CLASSES)
            print(f"    ✅ Eye MLP v4  F1=0.985  {n_active} active classes")
            _models = {"yolo":yolo,"dino":dino,"mlp":mlp,
                       "active":active,"classes":classes,"device":device}
        except Exception as e:
            print(f"    ⚠ MLP load failed: {e}")
            _models = {"yolo":yolo,"dino":dino,"mlp":None,
                       "active":list(range(NC)),"classes":EYE_CLASSES,"device":device}
    else:
        print(f"    ⚠ MLP not found — YOLO confidence only")
        _models = {"yolo":yolo,"dino":dino,"mlp":None,
                   "active":list(range(NC)),"classes":EYE_CLASSES,"device":device}
    return _models


def lab_gate_scleral(eye_crop_bgr):
    lab = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h,w = lab.shape[:2]
    r   = lab[int(h*0.15):int(h*0.85), int(w*0.05):int(w*0.95)]
    if r.size < 100: return False
    return float(np.mean(r[:,:,2])) > 145 and float(np.mean(r[:,:,0])) > 140


def get_eye_crop(aligned_bgr):
    h,w = aligned_bgr.shape[:2]
    return aligned_bgr[int(h*0.15):int(h*0.50), :]


def run_eye_yolo(eye_crop, models):
    res  = models["yolo"].predict(eye_crop, conf=CONF_THRESH,
                                  verbose=False, device=models["device"])
    dets = {}
    if res and res[0].boxes is not None:
        for cls_id, conf in zip(res[0].boxes.cls.cpu().int().tolist(),
                                res[0].boxes.conf.cpu().tolist()):
            if cls_id >= NC: continue
            name     = EYE_CLASSES[cls_id]
            min_conf = PER_CLASS_CONF.get(name, CONF_THRESH)
            if conf >= min_conf and conf > dets.get(name,0):
                dets[name] = round(conf, 3)
    if "scleral_icterus" in dets and not lab_gate_scleral(eye_crop):
        del dets["scleral_icterus"]
    return dets


def extract_eye_features(eye_crop, models):
    from torchvision import transforms
    device = models["device"]
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = cv2.resize(eye_crop, (256,128))
    h,w = img.shape[:2]
    REGIONS = {
        "sclera_left":  (0.15, 0.85, 0.00, 0.45),
        "sclera_right": (0.15, 0.85, 0.55, 1.00),
        "periorbital":  (0.00, 0.35, 0.10, 0.90),
    }
    feats = []
    with torch.no_grad():
        for _, (r1,r2,c1,c2) in REGIONS.items():
            reg = img[int(h*r1):int(h*r2), int(w*c1):int(w*c2)]
            if reg.size == 0:
                feats.append(torch.zeros(384, device=device))
                continue
            rgb = cv2.cvtColor(reg, cv2.COLOR_BGR2RGB)
            t   = tf(rgb).unsqueeze(0).to(device)
            out = models["dino"](t).last_hidden_state[:,0].squeeze(0)
            feats.append(out)
    return torch.cat(feats)   # 1152-dim


def run_eye_severity(feat_vec, models, T=20):
    mlp = models.get("mlp")
    if mlp is None:
        return np.zeros(NC), np.ones(NC)
    n_active = mlp.heads.__len__()
    results  = []
    with torch.no_grad():
        x = feat_vec.unsqueeze(0).float()
        for _ in range(T):
            mlp.train()
            sev, _ = mlp(x)
            results.append(torch.sigmoid(sev).cpu().numpy())
    mlp.eval()
    arr = np.stack(results)
    return arr.mean(0)[0], arr.std(0)[0]


def run_eye_inference(aligned_bgr, device=DEVICE):
    models   = get_eye_models(device)
    eye_crop = get_eye_crop(aligned_bgr)
    t0       = time.time()

    yolo_dets        = run_eye_yolo(eye_crop, models)
    feat_vec         = extract_eye_features(eye_crop, models)
    severity, uncert = run_eye_severity(feat_vec, models)

    active  = models.get("active", list(range(NC)))
    classes = models.get("classes", EYE_CLASSES)

    # Map active MLP outputs back to full class list
    sev_full = np.zeros(NC)
    unc_full = np.ones(NC)
    for j, ai in enumerate(active):
        if j < len(severity):
            sev_full[ai] = float(severity[j])
            unc_full[ai] = float(uncert[j])

    features_out = {}
    for i, name in enumerate(EYE_CLASSES):
        yc       = yolo_dets.get(name, 0.0)
        s        = sev_full[i]
        u        = unc_full[i]
        combined = max(s, yc * 0.85)
        if name == "scleral_icterus" and yc == 0.0:
            continue
        if combined < 0.12 and yc == 0.0:
            continue
        features_out[name] = {
            "severity":   round(combined, 3),
            "level":      "high" if combined>0.60 else "moderate" if combined>0.35 else "low",
            "confidence": round(float(np.exp(-u*3)) if u < 0.5 else yc, 2),
        }

    # Build posterior over 22 deficiencies
    scores = {d: 0.0 for d in DEFICIENCY_ORDER}
    for feat_name, feat_data in features_out.items():
        w = feat_data["severity"]
        for def_name in EYE_FEAT_DEF.get(feat_name, []):
            if def_name in scores:
                scores[def_name] = max(scores[def_name], w)
    total_s  = sum(scores.values()) + 1e-9
    posterior = [scores[d]/total_s for d in DEFICIENCY_ORDER]

    return {
        "features_detected": features_out,
        "posterior":          posterior,
        "timing_ms":          round((time.time()-t0)*1000, 1),
    }


if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--test",  action="store_true")
    args = ap.parse_args()

    models = get_eye_models(DEVICE)
    test_images = []
    if args.image:
        test_images = [args.image]
    elif args.test:
        test_images = (glob.glob("eye_datasets/EYE_COMBINED/images/*.jpg")[:5]
                      + glob.glob("*.jpg")[:3] + glob.glob("*.jpeg")[:3])

    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None: print(f"  Cannot read: {img_path}"); continue
        result = run_eye_inference(img, DEVICE)
        feats  = result["features_detected"]
        top    = sorted(feats.items(), key=lambda x: -x[1]["severity"])
        print(f"\n  {img_path}  [{result['timing_ms']}ms]")
        if top:
            for name, data in top[:3]:
                print(f"    {name:<25} {data['severity']:.2f}  [{data['level']}]")
        else:
            print(f"    No findings detected")