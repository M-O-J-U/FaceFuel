"""
FaceFuel v4 — Eye Feature Extraction (ground-truth labels)
===========================================================
Output: facefuel_features/eye_features_v4.npz
Run: python 5_eye_features_v4.py
"""
import os, yaml, warnings
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
import random; random.seed(42)

warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"] = "2"

EYE_DATA = "facefuel_eye_v4/data.yaml"
OUT_DIR  = Path("facefuel_features"); OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "eye_features_v4.npz"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMGS = 2000

d       = yaml.safe_load(open(EYE_DATA))
CLASSES = d["names"]
NC      = d["nc"]
print(f"Eye v4 classes ({NC}): {CLASSES}")

EYE_REGIONS = {
    "sclera_left":  (0.15, 0.85, 0.00, 0.45),
    "sclera_right": (0.15, 0.85, 0.55, 1.00),
    "periorbital":  (0.00, 0.35, 0.10, 0.90),
}

def load_dino():
    from transformers import AutoModel
    print("  Loading DINOv2...")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE)
    dino.eval()
    return dino

def extract_dino_eye(img_bgr, dino):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = cv2.resize(img_bgr, (256, 128))
    h, w = img.shape[:2]
    feats = []
    with torch.no_grad():
        for _, (r1,r2,c1,c2) in EYE_REGIONS.items():
            reg = img[int(h*r1):int(h*r2), int(w*c1):int(w*c2)]
            if reg.size == 0:
                feats.append(torch.zeros(384, device=DEVICE))
                continue
            rgb = cv2.cvtColor(reg, cv2.COLOR_BGR2RGB)
            t   = tf(rgb).unsqueeze(0).to(DEVICE)
            out = dino(t).last_hidden_state[:,0].squeeze(0)
            feats.append(out)
    return torch.cat(feats).cpu().numpy()   # 1152-dim

def label_from_file(lbl_path):
    label = np.zeros(NC, dtype=np.float32)
    try:
        for line in Path(lbl_path).read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid < NC: label[cid] = 1.0
    except Exception: pass
    return label

if __name__ == "__main__":
    dino = load_dino()
    base    = Path(yaml.safe_load(open(EYE_DATA)).get("path","facefuel_eye_v4"))
    img_dir = base / "images" / "train"
    lbl_dir = base / "labels" / "train"

    class_to_imgs = defaultdict(list)
    for lp in sorted(lbl_dir.glob("*.txt")):
        for ext in [".jpg",".png",".jpeg"]:
            ip = img_dir / (lp.stem + ext)
            if ip.exists(): break
        else: continue
        for line in lp.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid < NC:
                    class_to_imgs[cid].append((ip, lp))
                    break

    print(f"\n  Class counts:")
    for cid, cls in enumerate(CLASSES):
        print(f"    {cls:<25} {len(class_to_imgs[cid]):>5}")

    selected = {}
    for cid, items in class_to_imgs.items():
        random.shuffle(items)
        for ip, lp in items[:MAX_IMGS]:
            selected[str(ip)] = lp

    print(f"\n  Processing {len(selected):,} images...")
    X, y = [], []
    for i, (ip_str, lp) in enumerate(selected.items()):
        if i % 200 == 0: print(f"  {i}/{len(selected)}...")
        img = cv2.imread(ip_str)
        if img is None: continue
        try:
            feat  = extract_dino_eye(img, dino)
            label = label_from_file(lp)
            X.append(feat); y.append(label)
        except Exception: continue

    X = np.stack(X); y = np.stack(y)
    np.savez_compressed(OUT_FILE, X=X, y=y, classes=np.array(CLASSES))
    print(f"\n  ✅ {X.shape[0]} samples → {OUT_FILE}")
    for i, cls in enumerate(CLASSES):
        n = int(y[:,i].sum())
        bar = "█" * min(40, n//20)
        print(f"    {cls:<25} {n:>5}  {bar}")