"""
FaceFuel — Eye Module: YOLO11m Training
=========================================
Trains YOLO11m on the merged eye dataset.
Uses YOLO11m instead of YOLOv8m for +3% accuracy improvement.

Eye classes:
  0  conjunctival_pallor   → iron deficiency / anemia
  1  scleral_icterus       → liver stress
  2  xanthelasma           → cholesterol imbalance

Run:
  python eye_train.py              ← full training
  python eye_train.py --retrain-all ← also retrain face + tongue with YOLO11m
"""

import argparse, shutil, random, yaml
from pathlib import Path
from collections import defaultdict

import torch

BASE    = Path("eye_datasets")
MERGED  = BASE / "EYE_MERGED"
IMG_DIR = MERGED / "images"
LBL_DIR = MERGED / "labels"
COMBINED = BASE / "EYE_COMBINED"
RUN_DIR  = Path("runs/eye")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

CLASSES = ["conjunctival_pallor", "scleral_icterus", "xanthelasma"]


def build_split():
    """Build 80/20 train/val split from merged images."""
    print("  Building train/val split...")
    random.seed(42)

    all_imgs = [p for p in IMG_DIR.glob("*.*")
                if p.suffix.lower() in IMAGE_EXTS
                and (LBL_DIR / f"{p.stem}.txt").exists()]

    random.shuffle(all_imgs)
    sp = int(len(all_imgs) * 0.8)
    train_imgs, val_imgs = all_imgs[:sp], all_imgs[sp:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        si = COMBINED / split / "images"
        sl = COMBINED / split / "labels"
        si.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            di = si / p.name
            dl = sl / f"{p.stem}.txt"
            if not di.exists(): shutil.copy2(p, di)
            lf = LBL_DIR / f"{p.stem}.txt"
            if lf.exists() and not dl.exists(): shutil.copy2(lf, dl)

    # Class distribution
    class_counts = defaultdict(int)
    for lbl in LBL_DIR.glob("*.txt"):
        for line in lbl.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                class_counts[int(parts[0])] += 1

    print(f"  Train: {len(train_imgs):,}  Val: {len(val_imgs):,}")
    print("  Class distribution:")
    for i, name in enumerate(CLASSES):
        cnt = class_counts.get(i, 0)
        bar = "█" * min(30, cnt // 10)
        print(f"    {i}  {name:<22s}  {cnt:5d}  {bar}")

    yaml_path = COMBINED / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "train": str((COMBINED / "train" / "images").resolve()),
            "val":   str((COMBINED / "val"   / "images").resolve()),
            "nc":    len(CLASSES),
            "names": CLASSES,
        }, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def train_eye(yaml_path: Path, epochs: int = 120):
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training Eye YOLO11m ({epochs} epochs, 3 classes)")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # YOLO11m — upgrade from YOLOv8m
    # Download automatically if not present
    model = YOLO("yolo11m.pt")

    results = model.train(
        data          = str(yaml_path),
        epochs        = epochs,
        imgsz         = 640,
        batch         = 16,
        workers       = 0,
        device        = device,
        project       = str(RUN_DIR),
        name          = "eye_v1",
        patience      = 35,
        lr0           = 1e-3,
        lrf           = 0.01,
        warmup_epochs = 5,
        box           = 7.5,
        cls           = 2.5,
        # Augmentation — preserve color (critical for pallor/icterus detection)
        hsv_h         = 0.01,    # minimal hue shift — color IS the diagnostic signal
        hsv_s         = 0.4,     # moderate saturation
        hsv_v         = 0.4,
        fliplr        = 0.5,
        flipud        = 0.0,
        degrees       = 10,
        scale         = 0.3,
        mosaic        = 0.7,     # lower mosaic — small dataset, preserve color context
        mixup         = 0.15,
        save          = True,
        exist_ok      = True,
        verbose       = True,
    )

    best = next(Path(".").rglob("eye_v1/weights/best.pt"), None)
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"\n  ✅ Eye training complete")
    print(f"  Best weights: {best}")
    print(f"  mAP@0.5: {map50}")
    return best


def retrain_face_yolo11():
    """Retrain face YOLO with YOLO11m instead of YOLOv8m for +3% accuracy."""
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"

    # Find existing face training data.yaml
    face_yaml = next(Path("facefuel_datasets").rglob("data.yaml"), None)
    if not face_yaml:
        print("  ⚠ Face data.yaml not found — skipping face retrain")
        return

    # Find best face weights to start from
    face_weights = next(Path(".").rglob("yolo_detector_r2/weights/best.pt"), None)
    if not face_weights:
        face_weights = next(Path(".").rglob("facefuel_v2/weights/best.pt"), None)
    start = str(face_weights) if face_weights else "yolo11m.pt"

    print(f"\n{'='*60}")
    print(f"Retraining Face YOLO11m (fine-tuning from existing weights)")
    print(f"  Start: {start}")
    print(f"{'='*60}")

    model = YOLO("yolo11m.pt")   # always start fresh with YOLO11m backbone
    # Load existing weights as starting point if available
    if face_weights:
        try:
            model = YOLO(start)
        except Exception:
            model = YOLO("yolo11m.pt")

    results = model.train(
        data          = str(face_yaml),
        epochs        = 60,
        imgsz         = 640,
        batch         = 16,
        workers       = 0,
        device        = device,
        project       = "runs/face",
        name          = "face_yolo11m",
        patience      = 25,
        lr0           = 1e-4,
        lrf           = 0.01,
        warmup_epochs = 3,
        box           = 7.5,
        cls           = 2.0,
        hsv_h         = 0.02,
        hsv_s         = 0.5,
        hsv_v         = 0.4,
        fliplr        = 0.5,
        flipud        = 0.0,
        mosaic        = 0.8,
        mixup         = 0.1,
        save          = True,
        exist_ok      = True,
    )
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"  ✅ Face YOLO11m complete — mAP@0.5: {map50}")


def retrain_tongue_yolo11():
    """Retrain tongue YOLO with YOLO11m."""
    from ultralytics import YOLO

    device  = "0" if torch.cuda.is_available() else "cpu"
    t_yaml  = next(Path("tongue_datasets").rglob("TONGUE_COMBINED/data.yaml"), None)
    if not t_yaml:
        print("  ⚠ Tongue data.yaml not found — skipping tongue retrain")
        return

    t_weights = next(Path(".").rglob("tongue_v4_fixed/weights/best.pt"), None)

    print(f"\n{'='*60}")
    print(f"Retraining Tongue YOLO11m")
    print(f"{'='*60}")

    model = YOLO("yolo11m.pt")
    results = model.train(
        data          = str(t_yaml),
        epochs        = 60,
        imgsz         = 640,
        batch         = 16,
        workers       = 0,
        device        = device,
        project       = "runs/tongue",
        name          = "tongue_yolo11m",
        patience      = 25,
        lr0           = 1e-4,
        lrf           = 0.01,
        warmup_epochs = 3,
        box           = 7.5,
        cls           = 3.0,
        hsv_h         = 0.025,
        hsv_s         = 0.6,
        hsv_v         = 0.4,
        fliplr        = 0.5,
        flipud        = 0.0,
        mosaic        = 0.85,
        mixup         = 0.2,
        save          = True,
        exist_ok      = True,
    )
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"  ✅ Tongue YOLO11m complete — mAP@0.5: {map50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,  default=120)
    parser.add_argument("--retrain-all", action="store_true",
                        help="Also retrain face + tongue with YOLO11m")
    parser.add_argument("--eye-only",    action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("FaceFuel — Eye Module Training (YOLO11m)")
    print("="*60)

    if not MERGED.exists() or not any(IMG_DIR.glob("*.*")):
        print("❌ EYE_MERGED not found. Run eye_merge.py first.")
        exit(1)

    yaml_path = build_split()
    train_eye(yaml_path, args.epochs)

    if args.retrain_all and not args.eye_only:
        print("\n" + "="*60)
        print("Retraining Face + Tongue with YOLO11m")
        print("="*60)
        retrain_face_yolo11()
        retrain_tongue_yolo11()

    print(f"""
{'='*60}
ALL DONE
{'='*60}
Next steps:
  1. python eye_features.py       ← DINOv2 feature extraction
  2. python eye_severity.py --train  ← severity MLP training
  3. python server.py             ← test combined face+eye+tongue
""")