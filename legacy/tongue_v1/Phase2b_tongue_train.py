"""
FaceFuel — Phase 2b: Fix RF Labels + Train Tongue YOLO
=======================================================
Step 1: Patches the TONGUE_MERGED folder by finding and copying
        the missing Roboflow label files (they sit in labels/split/
        not next to images).

Step 2: Trains YOLOv8m on the merged tongue dataset.

Unified classes:
  0  tongue_body      1  fissured       2  crenated
  3  pale_tongue      4  red_tongue     5  yellow_coating
  6  white_coating    7  thick_coating  8  geographic
  9  smooth_glossy   10  tooth_marked

Run:
  python tongue_train.py               ← fix labels + train
  python tongue_train.py --train-only  ← skip fix, just train
  python tongue_train.py --fix-only    ← just fix labels, no training
"""

import argparse, json, shutil, yaml
from pathlib import Path
from collections import defaultdict

BASE    = Path("tongue_datasets")
MERGED  = BASE / "TONGUE_MERGED"
LBL_DIR = MERGED / "labels"
IMG_DIR = MERGED / "images"
RUN_DIR = Path("runs/tongue")

CLASSES = [
    "tongue_body", "fissured", "crenated", "pale_tongue", "red_tongue",
    "yellow_coating", "white_coating", "thick_coating", "geographic",
    "smooth_glossy", "tooth_marked",
]

RF_SOURCES = [
    BASE / "rf_tongue_seg_75",
    BASE / "rf_tongue_general_46",
    BASE / "rf_oral_tongue_96",
    BASE / "tongue_disease_clf",
]


# ═══════════════════════════════════════════════════════════════
# PHASE 2a — Fix Roboflow label paths
# ═══════════════════════════════════════════════════════════════

def fix_rf_labels():
    print("=" * 60)
    print("Phase 2a — Fixing Roboflow label paths")
    print("=" * 60)

    fixed = 0
    for src_rf in RF_SOURCES:
        if not src_rf.exists():
            continue

        # RF YOLO structure: images/split/img.jpg → labels/split/img.txt
        lbl_root = src_rf / "labels"
        if not lbl_root.exists():
            print(f"  ⚠ No labels/ folder in {src_rf.name}")
            continue

        for lbl_file in lbl_root.rglob("*.txt"):
            if lbl_file.name in ("classes.txt", "notes.txt"):
                continue

            # Find the corresponding merged image
            stem    = lbl_file.stem
            dst_lbl = LBL_DIR / f"{stem}.txt"

            # Skip if label already exists and has content
            if dst_lbl.exists() and dst_lbl.stat().st_size > 0:
                continue

            # Confirm the image exists in merged folder
            img_exists = any(
                (IMG_DIR / f"{stem}{ext}").exists()
                for ext in [".jpg", ".jpeg", ".png"]
            )
            if not img_exists:
                # The image may have been renamed with a prefix
                matches = list(IMG_DIR.glob(f"*{stem}*"))
                if not matches:
                    continue
                stem = matches[0].stem

            # Read label and remap class 0 → 0 (tongue_body)
            try:
                lines  = lbl_file.read_text(errors="ignore").strip().splitlines()
                boxes  = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append(
                            f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                        )
                if boxes:
                    dst_lbl = LBL_DIR / f"{stem}.txt"
                    dst_lbl.write_text("\n".join(boxes) + "\n", encoding="utf-8")
                    fixed += 1
            except Exception as e:
                print(f"  ⚠ {lbl_file.name}: {e}")

        print(f"  ✅ {src_rf.name}: patched labels")

    print(f"\n  Fixed: {fixed} label files added")

    # Recount
    labeled   = sum(1 for l in LBL_DIR.glob("*.txt") if l.stat().st_size > 0)
    unlabeled = sum(1 for l in LBL_DIR.glob("*.txt") if l.stat().st_size == 0)
    print(f"  Labeled (with boxes): {labeled:,}")
    print(f"  Empty (normal tongue): {unlabeled:,}")

    # Update class distribution
    class_counts = defaultdict(int)
    for lbl in LBL_DIR.glob("*.txt"):
        for line in lbl.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                class_counts[int(parts[0])] += 1

    print(f"\n  Class distribution after fix:")
    for i, name in enumerate(CLASSES):
        cnt = class_counts.get(i, 0)
        bar = "█" * min(30, cnt // 10)
        print(f"    {i:2d}  {name:<18s}  {cnt:5d}  {bar}")

    return fixed


# ═══════════════════════════════════════════════════════════════
# PHASE 2b — Build data.yaml + Train YOLO
# ═══════════════════════════════════════════════════════════════

def build_train_val_split():
    """Create train/val image lists (80/20 split from merged images)."""
    import random
    random.seed(42)

    all_imgs = []
    for ext in [".jpg", ".jpeg", ".png"]:
        all_imgs.extend(IMG_DIR.glob(f"*{ext}"))

    # Only include images that have a label file (labeled or empty=normal)
    labeled_imgs = [
        p for p in all_imgs
        if (LBL_DIR / f"{p.stem}.txt").exists()
    ]

    random.shuffle(labeled_imgs)
    split    = int(len(labeled_imgs) * 0.8)
    train    = labeled_imgs[:split]
    val      = labeled_imgs[split:]

    train_dir = MERGED / "train"
    val_dir   = MERGED / "val"
    train_img = train_dir / "images"
    val_img   = val_dir   / "images"
    train_lbl = train_dir / "labels"
    val_lbl   = val_dir   / "labels"

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    for p in train:
        dst = train_img / p.name
        if not dst.exists(): shutil.copy2(p, dst)
        lbl_src = LBL_DIR / f"{p.stem}.txt"
        lbl_dst = train_lbl / f"{p.stem}.txt"
        if lbl_src.exists() and not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)

    for p in val:
        dst = val_img / p.name
        if not dst.exists(): shutil.copy2(p, dst)
        lbl_src = LBL_DIR / f"{p.stem}.txt"
        lbl_dst = val_lbl / f"{p.stem}.txt"
        if lbl_src.exists() and not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)

    print(f"  Train: {len(train):,}  Val: {len(val):,}")
    return train_dir, val_dir


def write_yaml(train_dir: Path, val_dir: Path):
    data = {
        "train": str((train_dir / "images").resolve()),
        "val":   str((val_dir   / "images").resolve()),
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = MERGED / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml written: {yaml_path}")
    return yaml_path


def train_yolo(yaml_path: Path, epochs: int = 100, imgsz: int = 640):
    print(f"\n{'='*60}")
    print(f"Phase 2b — Training YOLOv8m on tongue dataset")
    print(f"  Epochs: {epochs}  Image size: {imgsz}")
    print(f"{'='*60}")

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")

    results = model.train(
        data       = str(yaml_path),
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = 16,
        workers    = 0,          # Windows safe
        device     = device,
        project    = str(RUN_DIR),
        name       = "tongue_v1",
        patience   = 30,
        lr0        = 1e-3,
        lrf        = 0.01,
        warmup_epochs = 5,
        box        = 7.5,
        cls        = 2.0,        # Higher cls loss — many classes unbalanced
        # Augmentation — important for small dataset
        hsv_h      = 0.02,       # Slight hue shift (tongue color is diagnostic)
        hsv_s      = 0.5,
        hsv_v      = 0.4,
        fliplr     = 0.5,
        flipud     = 0.0,        # No vertical flip — tongue orientation matters
        degrees    = 15,
        scale      = 0.3,
        mosaic     = 0.8,
        mixup      = 0.1,
        copy_paste = 0.0,
        save       = True,
        exist_ok   = True,
        verbose    = True,
    )

    best = RUN_DIR / "tongue_v1" / "weights" / "best.pt"
    print(f"\n  ✅ Training complete")
    print(f"  Best weights: {best}")
    print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return best


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true",
                        help="Skip label fix, go straight to training")
    parser.add_argument("--fix-only",   action="store_true",
                        help="Only fix labels, don't train")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--imgsz",      type=int, default=640)
    args = parser.parse_args()

    if not args.train_only:
        fixed = fix_rf_labels()

    if not args.fix_only:
        print(f"\n{'='*60}")
        print("Building train/val split...")
        print(f"{'='*60}")
        train_dir, val_dir = build_train_val_split()
        yaml_path = write_yaml(train_dir, val_dir)
        best_weights = train_yolo(yaml_path, args.epochs, args.imgsz)

        print(f"""
{'='*60}
DONE — Next steps:
  1. python tongue_pseudolabel.py
     Runs trained YOLO on 3,818 unlabeled images to generate
     pseudo-labels for pale_tongue, red_tongue, coating classes.

  2. python tongue_retrain.py
     Retrains with pseudo-labeled data for full class coverage.

  3. python tongue_features.py
     Extracts DINOv2 features for severity MLP training.
{'='*60}
""")