"""
Quick audit of eye_datasets folder structure.
Run: python audit_eye.py
"""
from pathlib import Path
from collections import defaultdict

BASE = Path("eye_datasets")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".json", ".xml", ".csv", ".xlsx"}

if not BASE.exists():
    print("❌ eye_datasets folder not found.")
    exit()

print("=" * 65)
print("eye_datasets — Full Audit")
print("=" * 65)

total_imgs   = 0
total_labels = 0
folder_stats = {}

# Walk all subfolders
all_dirs = sorted([d for d in BASE.rglob("*") if d.is_dir()])
all_dirs.insert(0, BASE)

for d in all_dirs:
    imgs   = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    labels = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in LABEL_EXTS]
    others = [f for f in d.iterdir() if f.is_file()
              and f.suffix.lower() not in IMAGE_EXTS | LABEL_EXTS]

    # Only print dirs that have files OR are top-level subfolders of BASE
    is_direct_child = d.parent == BASE or d == BASE
    has_files = imgs or labels or others

    if has_files or is_direct_child:
        rel = d.relative_to(BASE) if d != BASE else Path(".")
        depth = len(rel.parts)
        indent = "  " * depth
        img_str   = f"{len(imgs):4d} imgs" if imgs   else "       "
        lbl_str   = f"{len(labels):3d} labels" if labels else "          "
        empty_tag = " ← EMPTY" if not has_files else ""
        print(f"{indent}{rel}/  {img_str}  {lbl_str}{empty_tag}")

        # Show label filenames for small folders
        if labels and len(labels) <= 5:
            for lf in labels:
                print(f"{indent}  [{lf.suffix}] {lf.name}")

        total_imgs   += len(imgs)
        total_labels += len(labels)

# Extension breakdown
print(f"\n{'='*65}")
print("Image extension breakdown:")
ext_counts = defaultdict(int)
for f in BASE.rglob("*"):
    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
        ext_counts[f.suffix.lower()] += 1
for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
    print(f"  {ext:<8} {cnt:,}")

# Class distribution from YOLO label files
print(f"\n{'='*65}")
print("YOLO class distribution (from .txt label files):")
class_counts = defaultdict(int)
for lf in BASE.rglob("*.txt"):
    try:
        for line in lf.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                class_counts[int(parts[0])] += 1
    except: pass

if class_counts:
    for cls_id, cnt in sorted(class_counts.items()):
        print(f"  Class {cls_id:2d}:  {cnt:,} boxes")
else:
    print("  No YOLO .txt label files found yet.")

# Check for yaml/data config files
print(f"\n{'='*65}")
print("Config files found:")
yamls = list(BASE.rglob("*.yaml")) + list(BASE.rglob("data.yaml"))
for y in yamls[:10]:
    print(f"  {y.relative_to(BASE)}")
    try:
        content = y.read_text(errors="ignore")[:300]
        print(f"    {content[:200]}")
    except: pass

print(f"\n{'='*65}")
print(f"TOTALS:  {total_imgs:,} images   {total_labels:,} label/metadata files")
print(f"{'='*65}")