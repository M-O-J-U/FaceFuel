"""
FaceFuel v2 — Step 8: DINOv2 Feature Extraction
-------------------------------------------------
Extracts deep semantic feature vectors from each of the 8 face regions
using DINOv2 ViT-S/14 as the backbone.

Why DINOv2 over EfficientNet / ResNet:
  - Trained self-supervised on 142M images → richer texture + color features
  - Spatially structured patch tokens → captures local skin micro-texture
  - No classification head → pure feature extractor, domain-agnostic
  - ViT-S/14 = 21M params, ~170MB VRAM per batch → fits easily on 12GB

Output per image:
  One 3,072-dim feature vector =
    8 regions × 384-dim DINOv2 CLS token embedding

All vectors saved as a single matrix:
  facefuel_datasets/features/
    feature_matrix.npy     ← (N, 3072) float32
    feature_index.json     ← {row_index: image_stem} mapping
    feature_stats.json     ← mean/std per dimension (for normalisation)

These feed directly into step9_bayesian_engine.py.

Usage:
  python step8_dinov2_features.py
  python step8_dinov2_features.py --batch_size 64
  python step8_dinov2_features.py --test        ← 10 images only
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ── Paths ─────────────────────────────────────────────────────
ALIGNED_DIR  = Path("facefuel_datasets/preprocessed/aligned")
REGIONS_DIR  = Path("facefuel_datasets/preprocessed/regions")
OUT_DIR      = Path("facefuel_datasets/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_MATRIX  = OUT_DIR / "feature_matrix.npy"
FEAT_INDEX   = OUT_DIR / "feature_index.json"
FEAT_STATS   = OUT_DIR / "feature_stats.json"

# ── DINOv2 config ─────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-small"   # 21M params, 384-dim embeddings
EMBED_DIM    = 384                        # CLS token dim for ViT-S
N_REGIONS    = 8
FEAT_DIM     = EMBED_DIM * N_REGIONS     # 3,072 total per image

REGION_ORDER = [
    "periorbital_left",
    "periorbital_right",
    "left_cheek",
    "right_cheek",
    "forehead",
    "nose",
    "lips",
    "sclera_left",
]

# ── DINOv2 input normalization (ImageNet stats) ───────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINOV2_SIZE   = 224   # ViT-S/14 expects 224×224


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_dinov2(device: str) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 from HuggingFace Hub."""
    print(f"  Loading DINOv2 ({DINOV2_MODEL})...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        DINOV2_MODEL,
        # Use local cache — avoids re-downloading
    )
    model.eval()
    model.to(device)

    # Verify output dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, DINOV2_SIZE, DINOV2_SIZE).to(device)
        out = model(pixel_values=dummy)
        actual_dim = out.last_hidden_state[:, 0, :].shape[-1]
        assert actual_dim == EMBED_DIM, \
            f"Expected {EMBED_DIM}-dim but got {actual_dim}"

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ DINOv2 loaded — {param_count:.1f}M params on {device}")
    return model


# ═══════════════════════════════════════════════════════════════
# REGION CROP LOADING
# ═══════════════════════════════════════════════════════════════

def load_region_crops(aligned_path: Path,
                      regions_json: dict) -> dict[str, np.ndarray]:
    """
    Load the aligned face image and crop each region from it.
    Returns {region_name: crop_rgb_uint8}.
    """
    img_bgr = cv2.imread(str(aligned_path))
    if img_bgr is None:
        return {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    crops = {}
    for name, info in regions_json["regions"].items():
        x1, y1, x2, y2 = info["bbox"]
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            # Degenerate region — use a blank patch
            crop = np.zeros((16, 16, 3), dtype=np.uint8)
        crops[name] = crop
    return crops


def preprocess_crop(crop_rgb: np.ndarray) -> torch.Tensor:
    """
    Resize crop to 224×224, normalize with ImageNet stats.
    Returns (1, 3, 224, 224) float32 tensor.
    """
    # Resize
    resized = cv2.resize(crop_rgb, (DINOV2_SIZE, DINOV2_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    # Normalize
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC → CHW → NCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features_batch(model: torch.nn.Module,
                            batch_tensors: list[torch.Tensor],
                            device: str) -> np.ndarray:
    """
    Extract CLS token embeddings for a batch of image tensors.
    batch_tensors: list of (1, 3, 224, 224) tensors
    Returns: (batch_size, 384) numpy array
    """
    batch = torch.cat(batch_tensors, dim=0).to(device)  # (B, 3, 224, 224)

    outputs = model(pixel_values=batch)
    # CLS token = first token of last hidden state
    cls_tokens = outputs.last_hidden_state[:, 0, :]      # (B, 384)

    # L2-normalize each embedding
    cls_tokens = F.normalize(cls_tokens, dim=-1)

    return cls_tokens.cpu().float().numpy()


def build_image_feature_vector(model: torch.nn.Module,
                                aligned_path: Path,
                                regions_json: dict,
                                device: str) -> np.ndarray | None:
    """
    Build the full 3,072-dim feature vector for one image.
    Extracts each of the 8 regions individually and concatenates.
    Returns (3072,) float32 array or None on failure.
    """
    crops = load_region_crops(aligned_path, regions_json)
    if not crops:
        return None

    region_embeddings = []
    for region_name in REGION_ORDER:
        if region_name in crops:
            crop = crops[region_name]
        else:
            # Missing region — use zeros
            region_embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
            continue

        tensor = preprocess_crop(crop)
        emb = extract_features_batch(model, [tensor], device)  # (1, 384)
        region_embeddings.append(emb[0])

    return np.concatenate(region_embeddings, axis=0)  # (3072,)


# ═══════════════════════════════════════════════════════════════
# MAIN EXTRACTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_extraction(batch_size: int = 32, test_mode: bool = False):
    """Extract features for all preprocessed images."""

    # ── GPU setup ─────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        print("   Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*60}")
    print(f"FaceFuel v2 — DINOv2 Feature Extraction")
    print(f"{'='*60}")
    print(f"  GPU    : {gpu_name}")
    print(f"  Model  : {DINOV2_MODEL}  ({EMBED_DIM}-dim, {N_REGIONS} regions)")
    print(f"  Output : {FEAT_DIM}-dim vector per image")

    # ── Find all preprocessed images ─────────────────────────
    aligned_images = sorted(ALIGNED_DIR.glob("*_aligned.jpg"))
    if test_mode:
        aligned_images = aligned_images[:10]

    print(f"  Images : {len(aligned_images):,}")

    if not aligned_images:
        print(f"❌ No aligned images found in {ALIGNED_DIR}")
        print("   Run step7_preprocessing.py --batch first.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────
    model = load_dinov2(device)

    # ── Pre-allocate output matrix ────────────────────────────
    N = len(aligned_images)
    feature_matrix = np.zeros((N, FEAT_DIM), dtype=np.float32)
    index_map = {}   # {row_idx: image_stem}
    failed = []

    print(f"\n  Extracting features...")
    t0 = time.time()

    for i, aligned_path in enumerate(aligned_images):
        stem = aligned_path.stem.replace("_aligned", "")

        # Load corresponding regions JSON
        regions_path = REGIONS_DIR / f"{stem}_regions.json"
        if not regions_path.exists():
            failed.append(stem)
            continue

        with open(regions_path, "r") as f:
            regions_json = json.load(f)

        # Extract feature vector
        feat = build_image_feature_vector(
            model, aligned_path, regions_json, device)

        if feat is None:
            failed.append(stem)
            continue

        feature_matrix[i] = feat
        index_map[i] = stem

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (N - i - 1) / rate
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  [{i+1:,}/{N:,}]  "
                  f"{rate:.1f} img/s  "
                  f"ETA {eta/60:.1f}min  "
                  f"VRAM {vram:.2f}GB")

    # ── Trim to valid rows only ───────────────────────────────
    valid_rows = sorted(index_map.keys())
    feature_matrix = feature_matrix[valid_rows]
    index_map = {new_i: index_map[old_i]
                 for new_i, old_i in enumerate(valid_rows)}

    # ── Compute stats for normalisation ──────────────────────
    feat_mean = feature_matrix.mean(axis=0).tolist()
    feat_std  = feature_matrix.std(axis=0).clip(1e-6).tolist()

    # ── Save outputs ─────────────────────────────────────────
    np.save(str(FEAT_MATRIX), feature_matrix)

    with open(FEAT_INDEX, "w") as f:
        json.dump(index_map, f)

    with open(FEAT_STATS, "w") as f:
        json.dump({"mean": feat_mean, "std": feat_std,
                   "n_images": len(index_map),
                   "feat_dim": FEAT_DIM,
                   "regions":  REGION_ORDER,
                   "model":    DINOV2_MODEL}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Succeeded      : {len(index_map):,}")
    print(f"  Failed         : {len(failed):,}")
    print(f"  Feature matrix : {feature_matrix.shape}  "
          f"({feature_matrix.nbytes/1024**2:.1f} MB)")
    print(f"  Time           : {elapsed/60:.1f} minutes")
    print(f"  Rate           : {len(index_map)/elapsed:.1f} img/s")
    print(f"\n  Saved:")
    print(f"    {FEAT_MATRIX}")
    print(f"    {FEAT_INDEX}")
    print(f"    {FEAT_STATS}")
    print(f"""
NEXT STEP:
  Run step9_bayesian_engine.py to build the nutrient inference model
  using these feature vectors.
""")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FaceFuel v2 — DINOv2 Feature Extraction")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Images per GPU batch (default 32)")
    parser.add_argument("--test", action="store_true",
                        help="Run on 10 images only")
    args = parser.parse_args()

    run_extraction(batch_size=args.batch_size, test_mode=args.test)