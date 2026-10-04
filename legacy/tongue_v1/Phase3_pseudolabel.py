"""
FaceFuel — Phase 3: Tongue Pseudo-Labeling
==========================================
Uses the trained tongue YOLO + LAB color analysis to generate
pseudo-labels for the 3,818 unlabeled tongue images.

Covers the zero-annotation classes:
  3  pale_tongue      → LAB L channel < threshold
  4  red_tongue       → LAB A channel > threshold
  5  yellow_coating   → LAB B channel > threshold in coating zone
  6  white_coating    → Low saturation + high L in coating zone
  7  thick_coating    → Texture roughness in coating zone
  8  geographic       → Patchy L variance pattern
  9  smooth_glossy    → Very low texture std (absent papillae)

Run:
  python tongue_pseudolabel.py
  python tongue_pseudolabel.py --conf 0.35   ← stricter threshold
"""

import argparse, json, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE    = Path("tongue_datasets")
MERGED  = BASE / "TONGUE_MERGED"
UNL_DIR = MERGED / "unlabeled"
IMG_DIR = MERGED / "images"
LBL_DIR = MERGED / "labels"

PSEUDO_IMG = MERGED / "pseudo_images"
PSEUDO_LBL = MERGED / "pseudo_labels"
PSEUDO_IMG.mkdir(exist_ok=True)
PSEUDO_LBL.mkdir(exist_ok=True)

CLASSES = [
    "tongue_body", "fissured", "crenated", "pale_tongue", "red_tongue",
    "yellow_coating", "white_coating", "thick_coating", "geographic",
    "smooth_glossy", "tooth_marked",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ═══════════════════════════════════════════════════════════════
# LAB COLOR PSEUDO-LABELING
# ═══════════════════════════════════════════════════════════════

def analyze_tongue_color(img_bgr: np.ndarray) -> list:
    """
    Analyze tongue image in LAB space on the RAW image.
    Returns list of YOLO-format boxes for detected color features.
    Uses a full-image box since we're doing classification, not localization.
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return []

    # Convert to LAB
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    # Focus on central tongue region (60% crop — avoids background)
    y1 = int(h * 0.15); y2 = int(h * 0.85)
    x1 = int(w * 0.15); x2 = int(w * 0.85)
    Lc = L[y1:y2, x1:x2]
    Ac = A[y1:y2, x1:x2]
    Bc = B[y1:y2, x1:x2]

    if Lc.size == 0:
        return []

    mL  = float(np.mean(Lc))
    mA  = float(np.mean(Ac))
    mB  = float(np.mean(Bc))
    std = float(np.std(Lc))
    boxes = []

    # Full-image box (cx cy w h) = 0.5 0.5 0.95 0.95
    full_box = (0.5, 0.5, 0.95, 0.95)

    # ── pale_tongue (cls 3) ───────────────────────────────────
    # Iron/B12 deficiency: L < 150 AND low A (not red)
    if mL < 145 and mA < 132:
        score = (150 - mL) / 40
        if score > 0.25:
            boxes.append((3, *full_box))

    # ── red_tongue (cls 4) ────────────────────────────────────
    # B12/inflammation: high A channel
    if mA > 150:
        boxes.append((4, *full_box))

    # ── yellow_coating (cls 5) ────────────────────────────────
    # Liver/digestive: high B channel
    if mB > 152:
        boxes.append((5, *full_box))

    # ── white_coating (cls 6) ─────────────────────────────────
    # Candida: high L (bright white) + low A/B (neutral)
    if mL > 170 and abs(mA - 128) < 6 and abs(mB - 128) < 6:
        boxes.append((6, *full_box))

    # ── thick_coating (cls 7) ─────────────────────────────────
    # High texture variance in coating zone (rough coat = thick)
    if std > 30 and mL > 150:
        boxes.append((7, *full_box))

    # ── smooth_glossy (cls 9) ─────────────────────────────────
    # Iron/B12/folate: very smooth surface (absent papillae)
    # Low std + high L = glossy
    if std < 12 and mL > 155:
        boxes.append((9, *full_box))

    # ── geographic (cls 8) ────────────────────────────────────
    # Zinc/B-complex: patchy pattern = high spatial variance
    # Measure variance in 3x3 grid
    grid_stds = []
    for gy in range(3):
        for gx in range(3):
            gy1 = int(h * (0.15 + gy * 0.23))
            gy2 = int(h * (0.15 + (gy+1) * 0.23))
            gx1 = int(w * (0.15 + gx * 0.23))
            gx2 = int(w * (0.15 + (gx+1) * 0.23))
            patch = L[gy1:gy2, gx1:gx2]
            if patch.size > 0:
                grid_stds.append(float(np.std(patch)))
    if grid_stds:
        std_of_stds = float(np.std(grid_stds))
        if std_of_stds > 15 and 10 < std < 35:
            boxes.append((8, *full_box))

    return boxes


# ═══════════════════════════════════════════════════════════════
# MAIN PSEUDO-LABELING
# ═══════════════════════════════════════════════════════════════

def run_pseudolabel(conf_threshold: float = 0.30):
    print("=" * 60)
    print("Phase 3 — Tongue Pseudo-Labeling")
    print("=" * 60)

    # Load trained YOLO
    # Search for weights anywhere under runs/ — handles nested path variations
    candidates = (list(Path(".").rglob("tongue_v1/weights/best.pt")) +
                  list(Path(".").rglob("tongue_v1/weights/last.pt")))
    yolo_weights = candidates[0] if candidates else None
    if not yolo_weights:
        print("  ❌ No trained YOLO found. Run tongue_train.py first.")
        print("     Running color-only pseudo-labeling instead...")
        use_yolo = False
    else:
        use_yolo = True
        import torch
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo   = YOLO(str(yolo_weights))
        print(f"  ✅ YOLO loaded: {yolo_weights}")

    # Collect unlabeled images
    # Clear previous pseudo labels so we start fresh
    import shutil as _shutil
    if PSEUDO_IMG.exists(): _shutil.rmtree(PSEUDO_IMG); PSEUDO_IMG.mkdir()
    if PSEUDO_LBL.exists(): _shutil.rmtree(PSEUDO_LBL); PSEUDO_LBL.mkdir()

    unl_imgs = []
    for ext in [".jpg", ".jpeg", ".png"]:
        unl_imgs.extend(UNL_DIR.glob(f"*{ext}"))
        unl_imgs.extend(IMG_DIR.glob(f"*{ext}"))   # also recheck main folder

    # Filter: only images WITHOUT existing label files
    unl_imgs = [
        p for p in unl_imgs
        if not (LBL_DIR / f"{p.stem}.txt").exists()
    ]

    print(f"  Unlabeled images to process: {len(unl_imgs):,}")
    print(f"  YOLO conf threshold: {conf_threshold}")

    stats     = defaultdict(int)
    cls_counts = defaultdict(int)

    for i, img_path in enumerate(unl_imgs):
        if i % 200 == 0:
            print(f"  [{i}/{len(unl_imgs)}] processing...")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        boxes = []

        # YOLO detection for structural features
        if use_yolo:
            try:
                results = yolo.predict(
                    source=img_bgr, conf=conf_threshold,
                    verbose=False, device=device)
                if results and results[0].boxes is not None:
                    for cls_id, conf in zip(
                        results[0].boxes.cls.cpu().int().tolist(),
                        results[0].boxes.conf.cpu().tolist()
                    ):
                        if conf >= conf_threshold:
                            b = results[0].boxes.xywhn[
                                results[0].boxes.cls.cpu().int().tolist().index(cls_id)
                            ].cpu().tolist()
                            boxes.append((cls_id, b[0], b[1], b[2], b[3]))
                            cls_counts[cls_id] += 1
            except Exception:
                pass

        # LAB color analysis for color/coating features (always run)
        color_boxes = analyze_tongue_color(img_bgr)
        for cb in color_boxes:
            # Don't duplicate if YOLO already detected this class
            if not any(b[0] == cb[0] for b in boxes):
                boxes.append(cb)
                cls_counts[cb[0]] += 1

        if boxes:
            # Copy image to pseudo folder
            dst_img = PSEUDO_IMG / img_path.name
            if not dst_img.exists():
                import shutil
                shutil.copy2(img_path, dst_img)

            # Write pseudo label
            lbl_path = PSEUDO_LBL / f"{img_path.stem}.txt"
            lines    = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                        for b in boxes]
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["pseudo_labeled"] += 1
        else:
            stats["skipped"] += 1

    print(f"\n{'='*60}")
    print("PSEUDO-LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Images pseudo-labeled : {stats['pseudo_labeled']:,}")
    print(f"  Skipped (no signal)   : {stats['skipped']:,}")
    print(f"\n  Pseudo class distribution:")
    for cls_id, cls_name in enumerate(CLASSES):
        cnt = cls_counts.get(cls_id, 0)
        bar = "█" * min(30, cnt // 5)
        print(f"    {cls_id:2d}  {cls_name:<18s}  {cnt:5d}  {bar}")

    # Save report
    report = {
        "pseudo_labeled": stats["pseudo_labeled"],
        "skipped":        stats["skipped"],
        "class_counts":   {CLASSES[k]: v for k, v in cls_counts.items()},
        "conf_threshold": conf_threshold,
    }
    (MERGED / "pseudolabel_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"""
  Next: python tongue_retrain.py
  This will combine real labels + pseudo-labels and retrain
  YOLO with full class coverage.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()
    run_pseudolabel(args.conf)