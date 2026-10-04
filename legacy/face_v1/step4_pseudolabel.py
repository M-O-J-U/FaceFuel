"""
FaceFuel v2 — Step 4: Pseudo-Labeling
--------------------------------------
Uses a trained YOLOv8 model to auto-label the unlabeled images,
turning them into training data without any manual annotation.

Strategy:
  1. Load your trained model (from step6_train_yolo.py)
  2. Run it on all unlabeled images (listed in MERGED_V2/unlabeled_images.txt)
  3. Keep only HIGH-CONFIDENCE predictions (>= CONFIDENCE_THRESHOLD)
  4. Save them as YOLO label files
  5. These pseudo-labels are added to the training set for round 2

This approach (FixMatch / self-training) typically recovers 70-80% of
the accuracy you'd get from fully labeling everything manually.
Cost: $0. Time: ~30 minutes.

Run AFTER your first training round (step6_train_yolo.py).
"""

from pathlib import Path

MERGED           = Path("facefuel_datasets/MERGED_V2")
IMG_DIR          = MERGED / "images"
LBL_DIR          = MERGED / "labels"
PSEUDO_LBL_DIR   = MERGED / "pseudo_labels"
UNLABELED_LIST   = MERGED / "unlabeled_images.txt"
MODEL_PATH       = "runs/detect/facefuel_v1/weights/best.pt"  # update after training
CONFIDENCE_THRESHOLD = 0.60  # only keep predictions >= 60% confidence

PSEUDO_LBL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

print("=" * 65)
print("FaceFuel v2 — Pseudo-Labeling")
print("=" * 65)

# ── Load model ───────────────────────────────────────────────
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    print(f"   Make sure you've trained the model first (step6_train_yolo.py)")
    print(f"   and that {MODEL_PATH} exists.")
    exit(1)

# ── Get list of unlabeled images ──────────────────────────────
if UNLABELED_LIST.exists():
    stems = [l.strip() for l in UNLABELED_LIST.read_text().splitlines() if l.strip()]
    unlabeled_paths = []
    for stem in stems:
        for ext in IMAGE_EXTS:
            p = IMG_DIR / (stem + ext)
            if p.exists():
                unlabeled_paths.append(p)
                break
else:
    # Fall back: any image without a matching label
    labeled_stems = {f.stem for f in LBL_DIR.glob("*.txt")}
    unlabeled_paths = [
        f for f in IMG_DIR.glob("*.*")
        if f.suffix.lower() in IMAGE_EXTS and f.stem not in labeled_stems
    ]

print(f"Unlabeled images to process: {len(unlabeled_paths):,}")

if not unlabeled_paths:
    print("No unlabeled images found. All images are already labeled.")
    exit(0)

# ── Run inference ─────────────────────────────────────────────
saved = 0
skipped_low_conf = 0
skipped_no_det = 0

BATCH = 32  # process in batches to save memory

for i in range(0, len(unlabeled_paths), BATCH):
    batch = unlabeled_paths[i : i + BATCH]
    results = model.predict(
        source=[str(p) for p in batch],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        save=False,
    )

    for img_path, result in zip(batch, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            skipped_no_det += 1
            continue

        # Only use predictions above the threshold
        high_conf_mask = boxes.conf >= CONFIDENCE_THRESHOLD
        if not high_conf_mask.any():
            skipped_low_conf += 1
            continue

        # Write YOLO label file
        lbl_out = PSEUDO_LBL_DIR / (img_path.stem + ".txt")
        lines = []
        for cls_id, xywhn in zip(
            boxes.cls[high_conf_mask].cpu().int().tolist(),
            boxes.xywhn[high_conf_mask].cpu().tolist()
        ):
            cx, cy, w, h = xywhn
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        saved += 1

    if (i // BATCH) % 10 == 0:
        print(f"  Progress: {min(i+BATCH, len(unlabeled_paths)):,} / {len(unlabeled_paths):,}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PSEUDO-LABELING RESULTS")
print("=" * 65)
print(f"  Pseudo-labels created : {saved:,}")
print(f"  No detection found    : {skipped_no_det:,}")
print(f"  Low confidence        : {skipped_low_conf:,}")
print(f"  Output folder         : {PSEUDO_LBL_DIR.resolve()}")

print("""
NEXT STEPS:
  1. REVIEW a sample of pseudo-labels visually (optional but recommended).
     Use: python -c "from ultralytics import YOLO; YOLO('...').val()"
     or open a few images in LabelImg to spot-check.

  2. MERGE pseudo-labels into your main labels directory:
     import shutil
     for f in Path("facefuel_datasets/MERGED_V2/pseudo_labels").glob("*.txt"):
         dest = Path("facefuel_datasets/MERGED_V2/labels") / f.name
         if not dest.exists():
             shutil.copy(f, dest)

  3. RE-TRAIN with the expanded labeled set (step6_train_yolo.py again).
     You should see a meaningful accuracy improvement in round 2.

  4. Optionally repeat: train → pseudo-label → add → retrain.
     2-3 cycles is usually sufficient.
""")
