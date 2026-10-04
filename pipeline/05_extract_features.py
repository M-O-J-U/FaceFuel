"""
FaceFuel v4 — Stage 5: DINOv2 region features for severity-MLP training
=========================================================================
Uses GROUND-TRUTH YOLO label files (not detector output) so every class with
data gets positive examples. Region layout and preprocessing come from
facefuel.models.embed_regions — the SAME function the server uses at inference,
so training and inference features cannot drift apart. (Replaces the three
v4 scripts 5_{face,tongue,eye}_features_v4.py, archived in legacy/v4_original/;
verified to produce the same vectors, cosine ≥ 0.99999.)

Run:
  python pipeline/05_extract_features.py                   # all three
  python pipeline/05_extract_features.py --modality tongue
Output: facefuel_features/{face,tongue,eye}_features_v4.npz  (X, y, classes)

Known limitation: only images with ≥1 label are used, so there are no healthy
negatives. Pass --negatives DIR (images of the same modality with no visible
condition) to add all-zero label rows; see docs/RESEARCH_NOTES.md.
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

MAX_PER_CLASS = {"face": 4000, "tongue": 3000, "eye": 2000}
IMG_EXTS = (".jpg", ".png", ".jpeg")


def label_vector(lbl_path: Path, nc: int) -> np.ndarray:
    v = np.zeros(nc, dtype=np.float32)
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) < nc:
            v[int(parts[0])] = 1.0
    return v


def extract(modality: str, device: str, negatives: Path = None, seed: int = 42):
    data_yaml = ROOT / f"facefuel_{modality}_v4" / "data.yaml"
    d = yaml.safe_load(open(data_yaml))
    classes, nc = d["names"], d["nc"]
    base = Path(d.get("path", data_yaml.parent))
    img_dir, lbl_dir = base / "images" / "train", base / "labels" / "train"
    print(f"\n{modality}: {nc} classes  ({img_dir})")

    # index each image by its FIRST labelled class, then cap per class
    by_class = defaultdict(list)
    for lp in sorted(lbl_dir.glob("*.txt")):
        ip = next((img_dir / (lp.stem + e) for e in IMG_EXTS if (img_dir / (lp.stem + e)).exists()), None)
        if ip is None:
            continue
        for line in lp.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(parts[0]) < nc:
                by_class[int(parts[0])].append((ip, lp))
                break
    rng = random.Random(seed)
    selected = {}
    for cid, items in by_class.items():
        rng.shuffle(items)
        for ip, lp in items[:MAX_PER_CLASS[modality]]:
            selected[str(ip)] = lp
    for cid, c in enumerate(classes):
        print(f"  {c:<25} {len(by_class[cid]):>6}")

    get_dino(device)
    X, y = [], []
    items = list(selected.items())
    if negatives:
        negs = sorted(p for p in Path(negatives).rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"  + {len(negs)} negative (no-condition) images from {negatives}")
        items += [(str(p), None) for p in negs]
    for i, (ip, lp) in enumerate(items):
        if i % 300 == 0:
            print(f"  {i}/{len(items)}...")
        img = cv2.imread(ip)
        if img is None:
            continue
        X.append(embed_regions(img, modality, device).cpu().numpy())
        y.append(label_vector(lp, nc) if lp else np.zeros(nc, dtype=np.float32))

    out = ROOT / "facefuel_features" / f"{modality}_features_v4.npz"
    out.parent.mkdir(exist_ok=True)
    X, y = np.stack(X), np.stack(y)
    np.savez_compressed(out, X=X, y=y, classes=np.array(classes))
    print(f"  ✅ {X.shape[0]} samples × {X.shape[1]} → {out}")
    for i, c in enumerate(classes):
        print(f"    {c:<25} {int(y[:, i].sum()):>6} positives")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["face", "tongue", "eye", "all"], default="all")
    ap.add_argument("--negatives", type=Path, help="folder of no-condition images (single modality only)")
    a = ap.parse_args()
    if a.negatives and a.modality == "all":
        ap.error("--negatives needs a single --modality")
    dev = default_device()
    for m in (["face", "tongue", "eye"] if a.modality == "all" else [a.modality]):
        extract(m, dev, a.negatives)
