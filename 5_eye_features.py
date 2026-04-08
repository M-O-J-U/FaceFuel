"""
FaceFuel — Eye Module: DINOv2 Feature Extraction
==================================================
Extracts DINOv2 features from eye images using 3 anatomical regions:
  - sclera_left    (white of eye, left side — pallor + icterus)
  - sclera_right   (white of eye, right side)
  - periorbital    (around the eye — xanthelasma location)

Each region → 384-dim DINOv2 CLS token → 3×384 = 1152-dim vector

Output:
  eye_datasets/features/
    eye_feature_matrix.npy    ← (N, 1152) float32
    eye_feature_index.json
    eye_feature_stats.json

Run: python eye_features.py
"""

import json, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

BASE     = Path("eye_datasets")
COMBINED = BASE / "EYE_COMBINED"
FEAT_DIR = BASE / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_MATRIX = FEAT_DIR / "eye_feature_matrix.npy"
FEAT_INDEX  = FEAT_DIR / "eye_feature_index.json"
FEAT_STATS  = FEAT_DIR / "eye_feature_stats.json"

DINOV2_MODEL  = "facebook/dinov2-small"
EMBED_DIM     = 384
N_REGIONS     = 3
FEAT_DIM      = N_REGIONS * EMBED_DIM   # 1152

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS    = {".jpg", ".jpeg", ".png"}

# 3 anatomical eye regions (fraction of image)
REGION_DEFS = [
    ("sclera_left",   0.10, 0.90, 0.00, 0.45),   # left half — pallor + icterus
    ("sclera_right",  0.10, 0.90, 0.55, 1.00),   # right half — pallor + icterus
    ("periorbital",   0.00, 0.35, 0.10, 0.90),   # upper periorbital — xanthelasma
]


def extract_regions(img_rgb: np.ndarray) -> list:
    h, w = img_rgb.shape[:2]
    crops = []
    for name, y1f, y2f, x1f, x2f in REGION_DEFS:
        y1, y2 = int(h*y1f), int(h*y2f)
        x1, x2 = int(w*x1f), int(w*x2f)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size < 48:
            crop = np.zeros((16, 16, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


@torch.no_grad()
def extract_batch_features(imgs_bgr: list, model, device: str) -> np.ndarray:
    batch_vecs = []
    for img_bgr in imgs_bgr:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crops   = extract_regions(img_rgb)
        region_embs = []
        for crop in crops:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_f   = resized.astype(np.float32) / 255.0
            img_f   = (img_f - IMAGENET_MEAN) / IMAGENET_STD
            tensor  = torch.from_numpy(img_f.transpose(2,0,1)).unsqueeze(0).to(device)
            out     = model(pixel_values=tensor)
            cls_tok = F.normalize(out.last_hidden_state[:,0,:], dim=-1)
            region_embs.append(cls_tok.cpu().float().numpy()[0])
        batch_vecs.append(np.concatenate(region_embs))
    return np.stack(batch_vecs)


def run(batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print("Eye Module — DINOv2 Feature Extraction")
    print(f"  Device  : {device}")
    print(f"  Regions : {N_REGIONS} × {EMBED_DIM}d = {FEAT_DIM}d vectors")
    print("="*60)

    print("\n  Loading DINOv2...")
    model = AutoModel.from_pretrained(DINOV2_MODEL)
    model.eval().to(device)
    print("  ✅ DINOv2 loaded")

    # Collect all images from EYE_COMBINED
    all_imgs = []
    for split in ["train", "val"]:
        img_dir = COMBINED / split / "images"
        if img_dir.exists():
            for ext in IMAGE_EXTS:
                all_imgs.extend(img_dir.glob(f"*{ext}"))

    # Deduplicate
    seen = set(); unique_imgs = []
    for p in all_imgs:
        if p.stem not in seen:
            seen.add(p.stem); unique_imgs.append(p)

    print(f"  Images: {len(unique_imgs):,}  Batch size: {batch_size}")

    features = []; index = {}; failed = 0
    t0 = time.time()

    for i in range(0, len(unique_imgs), batch_size):
        batch_paths = unique_imgs[i:i+batch_size]
        batch_imgs  = []
        batch_valid = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None: failed += 1; continue
            batch_imgs.append(img); batch_valid.append(p)

        if not batch_imgs: continue
        try:
            vecs = extract_batch_features(batch_imgs, model, device)
            for vec, p in zip(vecs, batch_valid):
                idx = len(features)
                features.append(vec)
                index[idx] = p.stem
        except Exception as e:
            print(f"  ⚠ Batch error: {e}")
            failed += len(batch_imgs); continue

        done = min(i+batch_size, len(unique_imgs))
        if done % 300 < batch_size or done == len(unique_imgs):
            elapsed = time.time()-t0
            rate = done/elapsed
            eta  = (len(unique_imgs)-done)/(rate+1e-6)
            print(f"  [{done:>4}/{len(unique_imgs)}]  "
                  f"{rate:.1f} img/s  ETA {eta/60:.1f}min")

    feat_matrix = np.stack(features).astype(np.float32)
    np.save(str(FEAT_MATRIX), feat_matrix)
    with open(FEAT_INDEX, "w") as f: json.dump(index, f)
    mean = feat_matrix.mean(0).tolist()
    std  = feat_matrix.std(0).tolist()
    with open(FEAT_STATS, "w") as f: json.dump({"mean":mean,"std":std}, f)

    elapsed = time.time()-t0
    print(f"\n{'='*60}")
    print(f"DONE  shape={feat_matrix.shape}  "
          f"size={feat_matrix.nbytes/1024**2:.1f}MB  "
          f"time={elapsed/60:.1f}min")
    print(f"  Next: python eye_severity.py --train")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run(args.batch)