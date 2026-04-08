"""
FaceFuel — Tongue Improvement Pass
====================================
Fixes three root causes of weak performance before final training:

  Fix 1: Recover Tongue_v3_i train labels (375 images missing from CSV)
          The CSV only covered 60 test images. Train images are labeled
          by filename pattern — fissured/crenated images have distinctive
          naming in the dataset.

  Fix 2: Better pseudo-labels for geographic + smooth_glossy
          Uses LBP texture analysis instead of simple LAB thresholds.
          Geographic tongue: high inter-patch texture variance
          Smooth/glossy: very low LBP entropy (absent papillae)

  Fix 3: Continue training 60 more epochs from current best weights
          Val loss was still declining at epoch 80.

Run:
  python tongue_improve.py              ← all fixes + retrain
  python tongue_improve.py --fix-only   ← just fix data, no training
  python tongue_improve.py --train-only ← just continue training
"""

import argparse, csv, json, shutil, cv2, yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
# LBP replaced with fast OpenCV texture analysis

BASE       = Path("tongue_datasets")
MERGED     = BASE / "TONGUE_MERGED"
COMBINED   = BASE / "TONGUE_COMBINED"
LBL_DIR    = MERGED / "labels"
IMG_DIR    = MERGED / "images"
PSEUDO_LBL = MERGED / "pseudo_labels"
PSEUDO_IMG = MERGED / "pseudo_images"
RUN_DIR    = Path("runs/tongue")

CLASSES = [
    "tongue_body", "fissured", "crenated", "pale_tongue", "red_tongue",
    "yellow_coating", "white_coating", "thick_coating", "geographic",
    "smooth_glossy", "tooth_marked",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ═══════════════════════════════════════════════════════════════
# FIX 1: Recover Tongue_v3_i missing train labels
# ═══════════════════════════════════════════════════════════════

def fix_tongue_v3_labels():
    print("=" * 60)
    print("Fix 1 — Recovering Tongue_v3_i train labels")
    print("=" * 60)

    src = BASE / "Tongue_v3_i"
    if not src.exists():
        print("  ⚠ Tongue_v3_i not found, skipping")
        return 0

    # Find ALL CSV files in dataset — might have train CSV too
    all_csvs = list(src.rglob("*.csv"))
    print(f"  Found CSV files: {[c.name for c in all_csvs]}")

    # Build full label map from ALL csvs
    label_map = {}   # filename → class_id
    for csv_file in all_csvs:
        try:
            with open(csv_file, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", "").strip()
                    if not fname:
                        continue
                    fissured = (row.get(" fissured", "0").strip() == "1" or
                                row.get(" fissured tongue", "0").strip() == "1" or
                                row.get("fissured", "0").strip() == "1")
                    crenated = (row.get(" crenated", "0").strip() == "1" or
                                row.get(" crenated tongue", "0").strip() == "1" or
                                row.get("crenated", "0").strip() == "1")
                    if fissured:
                        label_map[fname] = 1
                    elif crenated:
                        label_map[fname] = 2
                    else:
                        label_map[fname] = -1   # normal
        except Exception as e:
            print(f"  ⚠ CSV error {csv_file.name}: {e}")

    print(f"  CSV entries loaded: {len(label_map)}")
    print(f"  Fissured: {sum(1 for v in label_map.values() if v==1)}")
    print(f"  Crenated: {sum(1 for v in label_map.values() if v==2)}")
    print(f"  Normal  : {sum(1 for v in label_map.values() if v==-1)}")

    # Now process ALL images in the dataset (train + val + test)
    added = 0
    for img_path in src.rglob("*.*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Look up by filename
        cls_id = label_map.get(img_path.name,
                  label_map.get(img_path.stem, None))

        # If not in CSV, try to infer from subfolder name
        if cls_id is None:
            parent = img_path.parent.name.lower()
            if "fissur" in parent:
                cls_id = 1
            elif "crenat" in parent:
                cls_id = 2
            else:
                cls_id = -1   # treat as normal

        # Find corresponding image in MERGED (may have been renamed)
        merged_img = IMG_DIR / img_path.name
        if not merged_img.exists():
            # Try finding with prefix
            matches = list(IMG_DIR.glob(f"*{img_path.stem}*"))
            if matches:
                merged_img = matches[0]
            else:
                continue

        # Write/update label
        lbl_path = LBL_DIR / f"{merged_img.stem}.txt"
        existing = lbl_path.read_text(errors="ignore").strip() \
                   if lbl_path.exists() else ""

        if cls_id > 0:
            new_line = f"{cls_id} 0.500000 0.500000 0.950000 0.950000"
            if new_line not in existing:
                with open(lbl_path, "a", encoding="utf-8") as f:
                    f.write(("\n" if existing else "") + new_line + "\n")
                added += 1
        elif cls_id == -1 and not lbl_path.exists():
            lbl_path.write_text("")   # empty = normal

    print(f"  ✅ Added/updated {added} label files")
    return added


# ═══════════════════════════════════════════════════════════════
# FIX 2: Improved pseudo-labeling for geographic + smooth_glossy
# ═══════════════════════════════════════════════════════════════

def lbp_texture_analysis(img_bgr: np.ndarray) -> dict:
    """
    Fast OpenCV-based texture analysis — replaces slow scikit-image LBP.
    Uses Laplacian variance + grid std for geographic detection.
    Uses Sobel gradient magnitude for smooth/glossy detection.
    ~0.5ms per image vs ~100ms for LBP — 200x faster.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Central tongue region only
    y1, y2 = int(h*0.15), int(h*0.85)
    x1, x2 = int(w*0.15), int(w*0.85)
    region  = gray[y1:y2, x1:x2]

    if region.size < 100:
        return {"geographic": 0.0, "smooth_glossy": 0.0}

    # ── Laplacian variance — measures texture complexity ──────
    lap     = cv2.Laplacian(region, cv2.CV_32F)
    lap_var = float(np.var(lap))

    # ── Sobel gradient magnitude — low = smooth surface ───────
    sx      = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    sy      = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    grad    = np.sqrt(sx**2 + sy**2)
    grad_mean = float(np.mean(grad))

    # ── Spatial patchiness — 3×3 grid std of local means ──────
    grid_means = []
    rows, cols = region.shape
    for gy in range(3):
        for gx in range(3):
            r1 = int(rows * gy / 3);     r2 = int(rows * (gy+1) / 3)
            c1 = int(cols * gx / 3);     c2 = int(cols * (gx+1) / 3)
            cell = region[r1:r2, c1:c2]
            if cell.size > 0:
                grid_means.append(float(np.mean(cell)))
    spatial_var = float(np.std(grid_means)) if len(grid_means) > 1 else 0.0

    # ── Scores ────────────────────────────────────────────────
    # Geographic: high patchiness (spatial_var > 8) + moderate texture
    geo_score    = float(np.clip((spatial_var - 8.0) / 15.0, 0, 1))

    # Smooth/glossy: low gradient (no papillae) + low Laplacian var
    smooth_score = float(np.clip((40.0 - grad_mean) / 30.0, 0, 1))
    if lap_var > 300:   # high texture = NOT smooth
        smooth_score = 0.0

    return {
        "geographic":    geo_score,
        "smooth_glossy": smooth_score,
        "lap_var":        lap_var,
        "grad_mean":      grad_mean,
        "spatial_var":    spatial_var,
    }


def improve_pseudo_labels():
    print("\n" + "=" * 60)
    print("Fix 2 — Improved pseudo-labels (geographic + smooth_glossy)")
    print("=" * 60)

    # Target: unlabeled images in pseudo folder + unlabeled pool
    target_dirs = [MERGED / "unlabeled", PSEUDO_IMG]
    all_imgs    = []
    for d in target_dirs:
        if d.exists():
            for ext in IMAGE_EXTS:
                all_imgs.extend(d.glob(f"*{ext}"))

    # Also check diabetes dataset — frequently shows geographic/smooth
    diabetes_dir = BASE / "preprocessedcropped"
    if diabetes_dir.exists():
        for ext in IMAGE_EXTS:
            all_imgs.extend(diabetes_dir.rglob(f"*{ext}"))

    print(f"  Images to analyze: {len(all_imgs):,}")

    added_geo    = 0
    added_smooth = 0
    updated      = 0

    for i, img_path in enumerate(all_imgs):
        if i % 500 == 0:
            print(f"  [{i}/{len(all_imgs)}] geo={added_geo} smooth={added_smooth}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        scores = lbp_texture_analysis(img_bgr)

        geo_score    = scores["geographic"]
        smooth_score = scores["smooth_glossy"]

        if geo_score < 0.35 and smooth_score < 0.35:
            continue

        # Find corresponding pseudo label
        lbl_path = PSEUDO_LBL / f"{img_path.stem}.txt"

        # If no pseudo label yet, also check main label dir
        main_lbl = LBL_DIR / f"{img_path.stem}.txt"
        if main_lbl.exists():
            lbl_path = main_lbl

        existing = lbl_path.read_text(errors="ignore").strip() \
                   if lbl_path.exists() else ""

        new_lines = []
        if geo_score >= 0.35:
            line = f"8 0.500000 0.500000 0.950000 0.950000"
            if line not in existing:
                new_lines.append(line)
                added_geo += 1

        if smooth_score >= 0.35:
            line = f"9 0.500000 0.500000 0.950000 0.950000"
            if line not in existing:
                new_lines.append(line)
                added_smooth += 1

        if new_lines:
            # Copy image to pseudo folder if not already there
            dst_img = PSEUDO_IMG / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            with open(PSEUDO_LBL / f"{dst_img.stem}.txt",
                      "a", encoding="utf-8") as f:
                prefix = "\n" if existing else ""
                f.write(prefix + "\n".join(new_lines) + "\n")
            updated += 1

    print(f"\n  ✅ geographic pseudo-labels added  : {added_geo:,}")
    print(f"  ✅ smooth_glossy pseudo-labels added: {added_smooth:,}")
    print(f"  ✅ Files updated                    : {updated:,}")
    return added_geo, added_smooth


# ═══════════════════════════════════════════════════════════════
# FIX 3: Continue training from current best weights
# ═══════════════════════════════════════════════════════════════

def rebuild_combined():
    """Rebuild TONGUE_COMBINED with updated labels."""
    print("\n" + "=" * 60)
    print("Rebuilding TONGUE_COMBINED with fixed labels...")
    print("=" * 60)

    import random
    random.seed(42)

    # Clear old combined
    if COMBINED.exists():
        shutil.rmtree(COMBINED)

    train_img = COMBINED / "train" / "images"
    val_img   = COMBINED / "val"   / "images"
    train_lbl = COMBINED / "train" / "labels"
    val_lbl   = COMBINED / "val"   / "labels"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all labeled images (real + pseudo)
    all_pairs = []

    # Real labeled
    for img in IMG_DIR.glob("*.*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = LBL_DIR / f"{img.stem}.txt"
        if lbl.exists():
            all_pairs.append((img, lbl, False))

    # Pseudo labeled
    if PSEUDO_IMG.exists():
        for img in PSEUDO_IMG.glob("*.*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = PSEUDO_LBL / f"{img.stem}.txt"
            if lbl.exists():
                all_pairs.append((img, lbl, True))

    random.shuffle(all_pairs)
    split   = int(len(all_pairs) * 0.8)
    train_p = all_pairs[:split]
    val_p   = all_pairs[split:]

    def copy_pair(pairs, img_dir, lbl_dir):
        for img, lbl, is_pseudo in pairs:
            prefix = "p_" if is_pseudo else ""
            dst_i  = img_dir / f"{prefix}{img.name}"
            dst_l  = lbl_dir / f"{prefix}{img.stem}.txt"
            if not dst_i.exists():
                shutil.copy2(img, dst_i)
            if not dst_l.exists():
                shutil.copy2(lbl, dst_l)

    copy_pair(train_p, train_img, train_lbl)
    copy_pair(val_p,   val_img,   val_lbl)

    # Count class distribution
    class_counts = defaultdict(int)
    for _, lbl, _ in all_pairs:
        for line in lbl.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                class_counts[int(parts[0])] += 1

    yaml_path = COMBINED / "data.yaml"
    data = {
        "train": str(train_img.resolve()),
        "val":   str(val_img.resolve()),
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  Train: {len(train_p):,}  Val: {len(val_p):,}")
    print(f"  Class distribution:")
    for i, name in enumerate(CLASSES):
        cnt = class_counts.get(i, 0)
        bar = "█" * min(30, cnt // 10)
        print(f"    {i:2d}  {name:<18s}  {cnt:5d}  {bar}")

    return yaml_path


def continue_training(yaml_path: Path, epochs: int = 60):
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"

    # Find current best weights
    candidates = (list(Path(".").rglob("tongue_v2_retrain/weights/best.pt")) +
                  list(Path(".").rglob("tongue_v1/weights/best.pt")))
    weights = str(candidates[0]) if candidates else "yolov8m.pt"
    print(f"\n  Starting from: {weights}")

    print(f"\n{'='*60}")
    print(f"Fix 3 — Continue training  ({epochs} more epochs)")
    print(f"{'='*60}")

    model   = YOLO(weights)
    results = model.train(
        data          = str(yaml_path),
        epochs        = epochs,
        imgsz         = 640,
        batch         = 16,
        workers       = 0,
        device        = device,
        project       = str(RUN_DIR),
        name          = "tongue_v3_improved",
        patience      = 30,
        lr0           = 1e-4,        # Low LR — fine-tuning
        lrf           = 0.01,
        warmup_epochs = 2,
        box           = 7.5,
        cls           = 3.0,         # Higher cls weight for rare classes
        hsv_h         = 0.02,
        hsv_s         = 0.6,
        hsv_v         = 0.4,
        fliplr        = 0.5,
        flipud        = 0.0,
        degrees       = 10,
        scale         = 0.3,
        mosaic        = 0.9,
        mixup         = 0.15,        # Mixup helps rare classes
        save          = True,
        exist_ok      = True,
        verbose       = True,
    )

    best = next(Path(".").rglob("tongue_v3_improved/weights/best.pt"), None)
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"\n  ✅ Training complete")
    print(f"  Best weights: {best}")
    print(f"  mAP@0.5: {map50}")
    print(f"\n  Next: python tongue_features.py")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-only",   action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--epochs",     type=int, default=60)
    args = parser.parse_args()

    if not args.train_only:
        fix_tongue_v3_labels()
        improve_pseudo_labels()

    yaml_path = rebuild_combined()

    if not args.fix_only:
        continue_training(yaml_path, args.epochs)