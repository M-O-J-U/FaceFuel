"""
FaceFuel v2 — Step 3: Verify + Unlabeled Audit
-----------------------------------------------
After running step2_smart_merge.py, run this to:

  1. Confirm image / label pairing (which images have labels, which don't)
  2. Show per-class annotation counts
  3. Spot-check 5 random label files for sanity
  4. Print a clear "what to do next" summary
"""

from pathlib import Path
import random

MERGED = Path("facefuel_datasets/MERGED_V2")
IMG_DIR = MERGED / "images"
LBL_DIR = MERGED / "labels"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

UNIFIED_CLASSES = [
    "dark_circle", "eye_bag", "acne", "wrinkle", "redness",
    "dry_skin", "oily_skin", "dark_spot", "blackhead", "pallor",
    "lip_dry", "melasma", "whitehead", "pore", "eye_redness",
    "yellow_sclera", "acne_scar", "pigmentation", "skin_dullness",
    "forehead_wrinkle", "nasolabial_fold", "crow_feet",
    "skin_texture_rough", "vascular_redness", "lip_pallor",
]

print("=" * 65)
print("FaceFuel v2 — Verification Report")
print("=" * 65)

# ------ Collect images and labels ------
all_images = {f.stem: f for f in IMG_DIR.glob("*.*")
              if f.suffix.lower() in IMAGE_EXTS}
all_labels = {f.stem: f for f in LBL_DIR.glob("*.txt")}

labeled   = {s for s in all_images if s in all_labels}
unlabeled = {s for s in all_images if s not in all_labels}

print(f"\n  Total images    : {len(all_images):,}")
print(f"  With labels     : {len(labeled):,}  ({100*len(labeled)/max(len(all_images),1):.1f}%)")
print(f"  Without labels  : {len(unlabeled):,}  ({100*len(unlabeled)/max(len(all_images),1):.1f}%)")

# ------ Per-class counts ------
from collections import defaultdict
class_counts = defaultdict(int)
empty_labels = 0
for lbl in LBL_DIR.glob("*.txt"):
    lines = [l for l in lbl.read_text(encoding="utf-8", errors="ignore")
             .splitlines() if l.strip()]
    if not lines:
        empty_labels += 1
        continue
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            class_counts[int(parts[0])] += 1

print(f"\n  Empty label files : {empty_labels}")
print(f"\n  Annotation boxes per class:")
total_boxes = sum(class_counts.values())
for cid in sorted(class_counts.keys()):
    name = UNIFIED_CLASSES[cid] if cid < len(UNIFIED_CLASSES) else f"class_{cid}"
    count = class_counts[cid]
    pct   = 100 * count / max(total_boxes, 1)
    bar   = "█" * min(35, int(pct))
    print(f"    {cid:2d}  {name:<22s}  {count:5d}  ({pct:5.1f}%)  {bar}")

print(f"\n  Total boxes : {total_boxes:,}")

# ------ Spot-check 5 random label files ------
print("\n" + "─" * 65)
print("  Spot-check: 5 random label files")
print("─" * 65)
sample = random.sample(sorted(labeled), min(5, len(labeled)))
for stem in sample:
    lbl_path = all_labels[stem]
    lines = [l for l in lbl_path.read_text(encoding="utf-8", errors="ignore")
             .splitlines() if l.strip()]
    print(f"\n  {lbl_path.name}  ({len(lines)} annotation(s))")
    for line in lines[:4]:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            cid  = int(parts[0])
            name = UNIFIED_CLASSES[cid] if cid < len(UNIFIED_CLASSES) else f"??({cid})"
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            print(f"    class={cid} ({name:<22s})  cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")

# ------ What to do with unlabeled images ------
print("\n" + "=" * 65)
print("WHAT TO DO WITH UNLABELED IMAGES")
print("=" * 65)
pct_unlabeled = 100 * len(unlabeled) / max(len(all_images), 1)

if len(unlabeled) == 0:
    print("  ✅ All images are labeled. Skip to training.")
elif pct_unlabeled < 20:
    print(f"  {pct_unlabeled:.0f}% unlabeled — small enough to ignore for now.")
    print("  Training on labeled subset is fine as a starting point.")
elif pct_unlabeled < 60:
    print(f"  {pct_unlabeled:.0f}% unlabeled — use pseudo-labeling (see step4_pseudolabel.py)")
    print("  Train on labeled → run on unlabeled → use confident predictions as new labels.")
else:
    print(f"  {pct_unlabeled:.0f}% unlabeled — significant gap.")
    print("  Priority: run step4_pseudolabel.py for auto-labeling.")
    print("  Also consider adding more datasets (see step5_add_datasets.py).")

# ------ Save unlabeled list ------
if unlabeled:
    unlabeled_list_path = MERGED / "unlabeled_images.txt"
    unlabeled_list_path.write_text("\n".join(sorted(unlabeled)), encoding="utf-8")
    print(f"\n  Unlabeled image stems saved to: {unlabeled_list_path}")

print("\n✅ Verification complete.")
if total_boxes < 3000:
    print("⚠  Total boxes < 3,000 — add more datasets before training (see step5_add_datasets.py).")
elif total_boxes < 10000:
    print("⚠  Total boxes < 10,000 — usable for prototyping but add more data for best accuracy.")
else:
    print("✅  Box count is good for initial training.")
