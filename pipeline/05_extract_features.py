"""
FaceFuel — Stage 5: DINOv2 region features for severity-MLP training
======================================================================
Uses GROUND-TRUTH YOLO label files (not detector output), so every class with
data gets positive examples. Region layout and preprocessing come from
facefuel.models.embed_regions — the SAME function the server uses at inference,
so training and inference features cannot drift apart (verified identical to the
original v4 scripts in legacy/v4_original/, cosine ≥ 0.99999).

Images whose label file is empty are NEGATIVES (no visible condition) and become
all-zero label rows. The v4 features had none, which is why the v4 MLPs saturate on
real selfies; the clean v5 eye set includes verified normal-eye photos.

Run:
  python pipeline/05_extract_features.py --data v5                  # all modalities, all splits
  python pipeline/05_extract_features.py --data v5 --modality eye
Output: facefuel_features/<modality>_features_<data>_<split>.npz   (X, y, classes, files)
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from facefuel.models import default_device, embed_regions, get_dino

# per-class cap for the TRAIN split only (val/test are always used in full)
MAX_PER_CLASS = {"face": 4000, "tongue": 3000, "eye": 2000}
IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp")


def label_vector(lbl_path: Path, nc: int) -> np.ndarray:
    v = np.zeros(nc, dtype=np.float32)
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(parts[0]) < nc:
                v[int(parts[0])] = 1.0
    return v


def extract(modality: str, data: str, splits, device: str, seed: int = 42):
    data_yaml = ROOT / f"facefuel_{modality}_{data}" / "data.yaml"
    d = yaml.safe_load(open(data_yaml))
    classes, nc = d["names"], d["nc"]
    base = Path(d.get("path", data_yaml.parent))
    get_dino(device)

    for split in splits:
        img_dir, lbl_dir = base / "images" / split, base / "labels" / split
        if not img_dir.exists():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if split == "train":
            # bucket by first labelled class ("neg" for negatives), cap each bucket
            buckets = defaultdict(list)
            for ip in imgs:
                y = label_vector(lbl_dir / (ip.stem + ".txt"), nc)
                buckets[int(np.argmax(y)) if y.any() else "neg"].append(ip)
            rng = random.Random(seed)
            imgs = []
            for k, items in buckets.items():
                rng.shuffle(items)
                imgs += items[:MAX_PER_CLASS[modality]]
        print(f"\n{modality}/{split}: {len(imgs)} images")
        X, Y, files = [], [], []
        for i, ip in enumerate(imgs):
            if i % 500 == 0:
                print(f"  {i}/{len(imgs)}...")
            img = cv2.imread(str(ip))
            if img is None:
                continue
            X.append(embed_regions(img, modality, device).cpu().numpy())
            Y.append(label_vector(lbl_dir / (ip.stem + ".txt"), nc))
            files.append(ip.name)
        X, Y = np.stack(X), np.stack(Y)
        out = ROOT / "facefuel_features" / f"{modality}_features_{data}_{split}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(out, X=X, y=Y, classes=np.array(classes), files=np.array(files))
        neg = int((Y.sum(1) == 0).sum())
        print(f"  ✅ {X.shape[0]} × {X.shape[1]} → {out.name}   negatives={neg}")
        for ci, c in enumerate(classes):
            print(f"    {c:<22} {int(Y[:, ci].sum()):>6} positives")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["face", "tongue", "eye", "all"], default="all")
    ap.add_argument("--data", default="v5", help="dataset version: facefuel_<modality>_<data>")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    dev = default_device()
    for m in (["face", "tongue", "eye"] if a.modality == "all" else [a.modality]):
        extract(m, a.data, a.splits.split(","), dev)
