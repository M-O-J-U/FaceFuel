"""
FaceFuel v2 — Tongue Module: Dataset Merge + Label Unification
==============================================================
Converts ALL tongue datasets into a single unified YOLO-format folder.

Source datasets and their label types:
  1. tongue-segmentation-1    2476  COCO JSON → YOLO (tongue_body)
  2. kaggle_biohit              900  PNG masks  → YOLO (tongue_body)
  3. kaggle_tooth_marked       1250  Folders    → YOLO (tooth_marked)
  4. preprocessedcropped       2750  Folders    → image-level class CSV
  5. Tongue_v3_i                477  CSV        → YOLO (fissured, crenated, normal)
  6. rf_tongue_seg_75            75  YOLO       → remap classes
  7. rf_tongue_general_46        46  YOLO       → remap classes
  8. rf_oral_tongue_96           42  YOLO       → remap classes
  9. Tongue_color_v20          1068  No labels  → unlabeled pool
 10. tongue_disease_clf          13  YOLO       → remap classes

Unified class list (tongue features → nutrient deficiencies):
  0  tongue_body          — presence/segmentation (always)
  1  fissured             → Vitamin B3, dehydration, chronic stress
  2  crenated             → Zinc deficiency, fluid retention, thyroid
  3  pale_tongue          → Iron deficiency, B12, anemia
  4  red_tongue           → B12/folate deficiency, inflammation
  5  yellow_coating       → Digestive issues, liver stress
  6  white_coating        → Candida, gut dysbiosis
  7  thick_coating        → Digestive stagnation, poor diet
  8  geographic           → Zinc, B-complex deficiency
  9  smooth_glossy        → Iron, B12, folate deficiency
 10  tooth_marked         → Zinc, fluid retention, thyroid

Output:
  tongue_datasets/TONGUE_MERGED/
    images/        ← all images
    labels/        ← YOLO .txt labels
    unlabeled/     ← images with no labels (for pseudo-labeling)
    class_labels.csv  ← image-level classification labels
    data.yaml
    merge_report.json
"""

import json, csv, shutil, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────
BASE    = Path("tongue_datasets")
OUT     = BASE / "TONGUE_MERGED"
OUT_IMG = OUT / "images"
OUT_LBL = OUT / "labels"
OUT_UNL = OUT / "unlabeled"
CLF_CSV = OUT / "class_labels.csv"

for d in [OUT_IMG, OUT_LBL, OUT_UNL]:
    d.mkdir(parents=True, exist_ok=True)

# ── Unified classes ────────────────────────────────────────────
CLASSES = [
    "tongue_body",    # 0
    "fissured",       # 1
    "crenated",       # 2
    "pale_tongue",    # 3
    "red_tongue",     # 4
    "yellow_coating", # 5
    "white_coating",  # 6
    "thick_coating",  # 7
    "geographic",     # 8
    "smooth_glossy",  # 9
    "tooth_marked",   # 10
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── State ──────────────────────────────────────────────────────
seen      = set()
stats     = defaultdict(int)
clf_rows  = []   # (image_stem, class_name, source)

print("=" * 65)
print("FaceFuel — Tongue Dataset Merge")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def safe_copy(src: Path, dst: Path):
    """Copy with dedup by filename."""
    if dst.name in seen:
        # Add prefix to avoid collision
        dst = dst.with_name(f"{src.parent.parent.name}_{dst.name}")
    seen.add(dst.name)
    shutil.copy2(src, dst)
    return dst


def write_yolo(label_path: Path, boxes: list):
    """Write YOLO format: class cx cy w h (all normalised 0-1)."""
    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
             for b in boxes]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_to_yolo_box(mask_path: Path) -> tuple | None:
    """Convert binary segmentation mask to YOLO bounding box."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = mask.shape
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (0, cx, cy, bw, bh)   # class 0 = tongue_body


def coco_to_yolo(img_w: int, img_h: int,
                 bbox: list, cls_id: int) -> tuple:
    """COCO bbox [x,y,w,h] → YOLO [cls cx cy w h] normalised."""
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return (cls_id, cx, cy, nw, nh)


# ═══════════════════════════════════════════════════════════════
# 1. tongue-segmentation-1 — COCO JSON → YOLO tongue_body
# ═══════════════════════════════════════════════════════════════

print("\n[1] tongue-segmentation-1 (COCO JSON → YOLO tongue_body)...")

src1 = BASE / "tongue-segmentation-1"
for split in ["train", "valid"]:
    img_dir  = src1 / split / "images" if (src1 / split / "images").exists() \
               else src1 / split
    ann_file = src1 / split / "_annotations.coco.json"
    if not ann_file.exists():
        # Try finding it
        found = list((src1 / split).rglob("*.json"))
        if found: ann_file = found[0]
        else:
            print(f"  ⚠ No JSON in {split}")
            continue

    try:
        data   = json.loads(ann_file.read_text(encoding="utf-8"))
        id_map = {img["id"]: img for img in data["images"]}
        anns   = defaultdict(list)
        for ann in data["annotations"]:
            anns[ann["image_id"]].append(ann)

        copied = 0
        for img_id, img_info in id_map.items():
            img_name = img_info["file_name"]
            src_img  = img_dir / img_name
            if not src_img.exists():
                src_img = next((src1.rglob(img_name)), None)
            if not src_img: continue

            dst_img  = safe_copy(src_img, OUT_IMG / src_img.name)
            iw, ih   = img_info["width"], img_info["height"]
            boxes    = []
            for ann in anns.get(img_id, []):
                boxes.append(coco_to_yolo(iw, ih, ann["bbox"], 0))

            if boxes:
                write_yolo(OUT_LBL / (dst_img.stem + ".txt"), boxes)
                stats["labeled"] += 1
            else:
                shutil.copy2(dst_img, OUT_UNL / dst_img.name)
                stats["unlabeled"] += 1
            copied += 1

        print(f"  ✅ {split}: {copied} images")
        stats["total"] += copied

    except Exception as e:
        print(f"  ❌ {split}: {e}")


# ═══════════════════════════════════════════════════════════════
# 2. kaggle_biohit — PNG segmentation masks → YOLO tongue_body
# ═══════════════════════════════════════════════════════════════

print("\n[2] kaggle_biohit (mask → YOLO tongue_body)...")

src2     = BASE / "kaggle_biohit" / "TongeImageDataset"
img_dir2 = src2 / "dataset"
msk_dir2 = src2 / "groundtruth" / "mask"

if img_dir2.exists() and msk_dir2.exists():
    imgs2 = list(img_dir2.glob("*.*"))
    ok = 0
    for img_path in imgs2:
        if img_path.suffix.lower() not in IMAGE_EXTS: continue
        stem      = img_path.stem
        # Masks may have same name or _gt suffix
        mask_path = next((msk_dir2 / f"{stem}{e}" for e in [".png",".jpg",".bmp"]
                          if (msk_dir2 / f"{stem}{e}").exists()), None)
        if mask_path is None:
            mask_path = next(msk_dir2.glob(f"{stem}*"), None)

        dst_img = safe_copy(img_path, OUT_IMG / img_path.name)
        stats["total"] += 1

        if mask_path:
            box = mask_to_yolo_box(mask_path)
            if box:
                write_yolo(OUT_LBL / (dst_img.stem + ".txt"), [box])
                stats["labeled"] += 1
                ok += 1
            else:
                stats["unlabeled"] += 1
        else:
            shutil.copy2(dst_img, OUT_UNL / dst_img.name)
            stats["unlabeled"] += 1

    print(f"  ✅ {ok} images with masks converted")
else:
    print(f"  ⚠ Expected dirs not found: {img_dir2}, {msk_dir2}")


# ═══════════════════════════════════════════════════════════════
# 3. kaggle_tooth_marked — folder names → YOLO tooth_marked (cls 10)
# ═══════════════════════════════════════════════════════════════

print("\n[3] kaggle_tooth_marked (folders → YOLO tooth_marked)...")

src3 = BASE / "kaggle_tooth_marked"
ok3  = 0
for cls_name, yolo_cls in [("marked", 10), ("unmarked", -1)]:
    folder = src3 / cls_name
    if not folder.exists(): continue
    for img_path in folder.rglob("*.*"):
        if img_path.suffix.lower() not in IMAGE_EXTS: continue
        dst_img = safe_copy(img_path, OUT_IMG / img_path.name)
        stats["total"] += 1

        if yolo_cls >= 0:
            # Full-image box for classification-style label
            write_yolo(OUT_LBL / (dst_img.stem + ".txt"),
                       [(yolo_cls, 0.5, 0.5, 0.9, 0.9)])
            stats["labeled"] += 1
        else:
            # Normal tongue — still a labeled negative example
            # Write empty label file (background class)
            (OUT_LBL / (dst_img.stem + ".txt")).write_text("")
            stats["labeled"] += 1

        clf_rows.append((dst_img.stem, cls_name, "kaggle_tooth_marked"))
        ok3 += 1

print(f"  ✅ {ok3} images with tooth-mark labels")


# ═══════════════════════════════════════════════════════════════
# 4. preprocessedcropped — diabetes/nondiabetes → image-level CSV
# ═══════════════════════════════════════════════════════════════

print("\n[4] preprocessedcropped (diabetes/nondiabetes → CSV + unlabeled)...")

src4 = BASE / "preprocessedcropped"
ok4  = 0
for split in ["train", "valid", "test"]:
    for cls_name in ["diabetes", "nondiabetes"]:
        folder = src4 / split / cls_name
        if not folder.exists(): continue
        for img_path in folder.rglob("*.*"):
            if img_path.suffix.lower() not in IMAGE_EXTS: continue
            dst_img = safe_copy(img_path, OUT_IMG / img_path.name)
            stats["total"] += 1
            # No YOLO box — image-level label only
            shutil.copy2(dst_img, OUT_UNL / dst_img.name)
            stats["unlabeled"] += 1
            clf_rows.append((dst_img.stem, cls_name, "mendeley_diabetes"))
            ok4 += 1

print(f"  ✅ {ok4} images added with image-level diabetes labels")


# ═══════════════════════════════════════════════════════════════
# 5. Tongue_v3_i — CSV → YOLO (fissured=1, crenated=2)
# ═══════════════════════════════════════════════════════════════

print("\n[5] Tongue_v3_i (CSV → YOLO fissured/crenated)...")

src5     = BASE / "Tongue_v3_i"
csv_file = next(src5.rglob("*.csv"), None)

csv_map = {}
if csv_file:
    try:
        with open(csv_file, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "").strip()
                # Determine class from columns
                if row.get(" fissured", "0").strip() == "1" or \
                   row.get(" fissured tongue", "0").strip() == "1":
                    csv_map[fname] = 1   # fissured
                elif row.get(" crenated", "0").strip() == "1" or \
                     row.get(" crenated tongue", "0").strip() == "1":
                    csv_map[fname] = 2   # crenated
                else:
                    csv_map[fname] = -1  # normal
        print(f"  CSV loaded: {len(csv_map)} entries")
    except Exception as e:
        print(f"  ⚠ CSV error: {e}")

ok5 = 0
for img_path in src5.rglob("*.*"):
    if img_path.suffix.lower() not in IMAGE_EXTS: continue
    dst_img = safe_copy(img_path, OUT_IMG / img_path.name)
    stats["total"] += 1

    cls_id = csv_map.get(img_path.name, csv_map.get(img_path.stem, None))
    if cls_id is not None and cls_id >= 0:
        write_yolo(OUT_LBL / (dst_img.stem + ".txt"),
                   [(cls_id, 0.5, 0.5, 0.9, 0.9)])
        stats["labeled"] += 1
    elif cls_id == -1:
        # Normal tongue — empty label
        (OUT_LBL / (dst_img.stem + ".txt")).write_text("")
        stats["labeled"] += 1
    else:
        stats["unlabeled"] += 1

    clf_rows.append((dst_img.stem, "fissured" if cls_id==1 else
                     "crenated" if cls_id==2 else "normal", "tongue_v3"))
    ok5 += 1

print(f"  ✅ {ok5} images processed")


# ═══════════════════════════════════════════════════════════════
# 6-8. Roboflow YOLO sets — remap to unified classes
# ═══════════════════════════════════════════════════════════════

print("\n[6-8] Roboflow YOLO datasets (remap classes)...")

# For all RF sets, class 0 = tongue/tongue_body → unified class 0
rf_sources = [
    BASE / "rf_tongue_seg_75",
    BASE / "rf_tongue_general_46",
    BASE / "rf_oral_tongue_96",
    BASE / "tongue_disease_clf",
]

for src_rf in rf_sources:
    if not src_rf.exists(): continue
    ok_rf = 0
    for img_path in src_rf.rglob("*.*"):
        if img_path.suffix.lower() not in IMAGE_EXTS: continue
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path = (img_path.parent.parent / "labels" /
                        img_path.parent.name / img_path.with_suffix(".txt").name)

        dst_img = safe_copy(img_path, OUT_IMG / img_path.name)
        stats["total"] += 1

        if lbl_path.exists():
            lines = lbl_path.read_text(encoding="utf-8",
                                        errors="ignore").strip().splitlines()
            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # All RF tongue sets: class 0 → unified class 0 (tongue_body)
                        boxes.append((0, float(parts[1]), float(parts[2]),
                                      float(parts[3]), float(parts[4])))
                    except ValueError:
                        pass
            if boxes:
                write_yolo(OUT_LBL / (dst_img.stem + ".txt"), boxes)
                stats["labeled"] += 1
                ok_rf += 1
            else:
                stats["unlabeled"] += 1
        else:
            stats["unlabeled"] += 1

    print(f"  ✅ {src_rf.name}: {ok_rf} labeled images")


# ═══════════════════════════════════════════════════════════════
# 9. Tongue_color_v20 — no labels → unlabeled pool
# ═══════════════════════════════════════════════════════════════

print("\n[9] Tongue_color_v20 (unlabeled → unlabeled pool)...")

src9 = BASE / "Tongue_color_v20"
ok9  = 0
for img_path in src9.rglob("*.*"):
    if img_path.suffix.lower() not in IMAGE_EXTS: continue
    dst = OUT_UNL / img_path.name
    if dst.name not in seen:
        seen.add(dst.name)
        shutil.copy2(img_path, dst)
        stats["unlabeled"] += 1
        stats["total"] += 1
        ok9 += 1

print(f"  ✅ {ok9} images added to unlabeled pool")


# ═══════════════════════════════════════════════════════════════
# WRITE data.yaml + class_labels.csv + report
# ═══════════════════════════════════════════════════════════════

# data.yaml
yaml_content = (
    f"train: {(OUT / 'images').resolve().as_posix()}\n"
    f"val:   {(OUT / 'images').resolve().as_posix()}\n\n"
    f"nc: {len(CLASSES)}\n"
    f"names: {CLASSES}\n"
)
(OUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

# class_labels.csv
with open(CLF_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_stem", "class_label", "source"])
    w.writerows(clf_rows)

# Count labeled images (have non-empty label files)
actual_labeled = sum(
    1 for lbl in OUT_LBL.glob("*.txt")
    if lbl.stat().st_size > 0
)
actual_empty_lbl = sum(
    1 for lbl in OUT_LBL.glob("*.txt")
    if lbl.stat().st_size == 0
)

# Class distribution
class_counts = defaultdict(int)
for lbl in OUT_LBL.glob("*.txt"):
    for line in lbl.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            class_counts[int(parts[0])] += 1

# report
report = {
    "total_images":        stats["total"],
    "labeled_with_boxes":  actual_labeled,
    "labeled_empty":       actual_empty_lbl,
    "unlabeled":           len(list(OUT_UNL.glob("*.*"))),
    "classification_rows": len(clf_rows),
    "class_distribution":  {CLASSES[k]: v for k, v in class_counts.items()},
}
(OUT / "merge_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("MERGE COMPLETE")
print(f"{'='*65}")
print(f"  Total images copied    : {stats['total']:,}")
print(f"  With YOLO box labels   : {actual_labeled:,}")
print(f"  Empty labels (normal)  : {actual_empty_lbl:,}")
print(f"  Unlabeled pool         : {len(list(OUT_UNL.glob('*.*'))):,}")
print(f"  Image-level clf labels : {len(clf_rows):,}  → {CLF_CSV}")
print(f"\n  Class distribution:")
for cls_id, cls_name in enumerate(CLASSES):
    cnt = class_counts.get(cls_id, 0)
    bar = "█" * min(30, cnt // 5)
    print(f"    {cls_id:2d}  {cls_name:<18s}  {cnt:5d}  {bar}")
print(f"\n  Output: {OUT.resolve()}")
print(f"\nNEXT: python tongue_train.py")