"""
FaceFuel — Phase 5: Tongue DINOv2 Feature Extraction
=====================================================
Extracts DINOv2 features from tongue images using 5 anatomical regions:
  - tongue_tip      (front 25%)
  - tongue_body     (center)
  - tongue_left     (left lateral edge)
  - tongue_right    (right lateral edge)
  - tongue_coating  (dorsal surface — center strip)

Each region → 384-dim DINOv2 CLS token → 5×384 = 1920-dim feature vector

Output:
  tongue_datasets/features/
    tongue_feature_matrix.npy    ← (N, 1920) float32
    tongue_feature_index.json    ← {i: stem}
    tongue_feature_stats.json    ← {mean, std} for normalisation

Run:
  python tongue_features.py
  python tongue_features.py --batch 64   ← larger batch if VRAM allows
"""

import argparse, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

# ── Paths ─────────────────────────────────────────────────────
BASE       = Path("tongue_datasets")
COMBINED   = BASE / "TONGUE_COMBINED"
FEAT_DIR   = BASE / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_MATRIX = FEAT_DIR / "tongue_feature_matrix.npy"
FEAT_INDEX  = FEAT_DIR / "tongue_feature_index.json"
FEAT_STATS  = FEAT_DIR / "tongue_feature_stats.json"

DINOV2_MODEL  = "facebook/dinov2-small"
EMBED_DIM     = 384
N_REGIONS     = 5
FEAT_DIM      = N_REGIONS * EMBED_DIM   # 1920

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS    = {".jpg", ".jpeg", ".png"}


# ── Tongue region definitions ─────────────────────────────────
# Each entry: (name, y_start_frac, y_end_frac, x_start_frac, x_end_frac)
# Tongue is assumed to be cropped/centered in the image
REGION_DEFS = [
    ("tongue_tip",     0.00, 0.35, 0.20, 0.80),   # front tip
    ("tongue_body",    0.25, 0.75, 0.15, 0.85),   # central body
    ("tongue_left",    0.20, 0.80, 0.00, 0.40),   # left lateral
    ("tongue_right",   0.20, 0.80, 0.60, 1.00),   # right lateral
    ("tongue_coating", 0.30, 0.70, 0.25, 0.75),   # dorsal coating zone
]


def extract_regions(img_rgb: np.ndarray) -> list:
    """Extract 5 anatomical tongue regions from image."""
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
    """
    Extract DINOv2 CLS features for a batch of images.
    Each image → 5 region crops → 5 × 384-dim embeddings → 1920-dim vector.
    Returns (batch_size, 1920) array.
    """
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

        batch_vecs.append(np.concatenate(region_embs))   # 1920-dim

    return np.stack(batch_vecs)   # (B, 1920)


def run(batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("Phase 5 — Tongue DINOv2 Feature Extraction")
    print(f"  Device  : {device}")
    print(f"  Regions : {N_REGIONS} × {EMBED_DIM}d = {FEAT_DIM}d vectors")
    if device == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load DINOv2
    print("\n  Loading DINOv2 (facebook/dinov2-small)...")
    model = AutoModel.from_pretrained(DINOV2_MODEL)
    model.eval().to(device)
    print("  ✅ DINOv2 loaded")

    # Collect all labeled images from COMBINED
    all_imgs = []
    for split in ["train", "val"]:
        img_dir = COMBINED / split / "images"
        if img_dir.exists():
            for ext in IMAGE_EXTS:
                all_imgs.extend(img_dir.glob(f"*{ext}"))

    # Deduplicate by stem
    seen_stems = set()
    unique_imgs = []
    for p in all_imgs:
        stem = p.stem.lstrip("p_")   # remove pseudo prefix
        if stem not in seen_stems:
            seen_stems.add(stem)
            unique_imgs.append(p)

    print(f"\n  Images to process: {len(unique_imgs):,}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Estimated time   : ~{len(unique_imgs)*0.003/60:.1f} min\n")

    features = []
    index    = {}
    failed   = 0
    t_start  = time.time()

    for i in range(0, len(unique_imgs), batch_size):
        batch_paths = unique_imgs[i:i+batch_size]
        batch_imgs  = []
        batch_valid = []

        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                failed += 1
                continue
            batch_imgs.append(img)
            batch_valid.append(p)

        if not batch_imgs:
            continue

        try:
            vecs = extract_batch_features(batch_imgs, model, device)
            for j, (vec, p) in enumerate(zip(vecs, batch_valid)):
                idx = len(features)
                features.append(vec)
                index[idx] = p.stem

        except Exception as e:
            print(f"  ⚠ Batch {i//batch_size} error: {e}")
            failed += len(batch_imgs)
            continue

        # Progress
        done = min(i + batch_size, len(unique_imgs))
        if done % 500 < batch_size or done == len(unique_imgs):
            elapsed = time.time() - t_start
            rate    = done / elapsed
            eta     = (len(unique_imgs) - done) / (rate + 1e-6)
            print(f"  [{done:>5}/{len(unique_imgs)}]  "
                  f"{rate:.1f} img/s  ETA {eta/60:.1f}min  "
                  f"features={len(features)}")

    # Stack and save
    print(f"\n  Stacking {len(features):,} feature vectors...")
    feat_matrix = np.stack(features).astype(np.float32)

    print(f"  Shape: {feat_matrix.shape}")
    np.save(str(FEAT_MATRIX), feat_matrix)

    with open(FEAT_INDEX, "w") as f:
        json.dump(index, f)

    # Compute normalisation stats
    mean = feat_matrix.mean(axis=0).tolist()
    std  = feat_matrix.std(axis=0).tolist()
    with open(FEAT_STATS, "w") as f:
        json.dump({"mean": mean, "std": std}, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images    : {len(features):,}")
    print(f"  Failed          : {failed}")
    print(f"  Feature shape   : {feat_matrix.shape}")
    print(f"  Matrix size     : {feat_matrix.nbytes/1024**2:.1f} MB")
    print(f"  Time            : {elapsed/60:.1f} min")
    print(f"  Saved to        : {FEAT_DIR.resolve()}")
    print(f"\n  Next: python tongue_severity.py --train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run(args.batch)