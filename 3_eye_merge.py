"""
FaceFuel — Eye Module: Dataset Merge
=====================================
Converts all downloaded eye datasets from classification format
to unified YOLO detection format.

Eye classes:
  0  conjunctival_pallor   → iron deficiency / anemia / B12
  1  scleral_icterus       → liver stress / elevated bilirubin
  2  xanthelasma           → cholesterol imbalance / omega-3 deficit

Hb threshold for anemia labeling:
  Hb < 110 g/L  → anemic   → label as conjunctival_pallor (class 0)
  Hb >= 120 g/L → healthy  → empty label (normal eye)
  Hb 110-119    → borderline → skip (ambiguous)

Output:
  eye_datasets/EYE_MERGED/
    images/    ← all images
    labels/    ← YOLO .txt labels
    class_labels.csv  ← image-level classification info

Run: python eye_merge.py
"""

import shutil, csv, re
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("  ⚠ pandas not installed. Run: pip install pandas openpyxl")
    print("  Continuing without Hb-based labeling...")

BASE    = Path("eye_datasets")
MERGED  = BASE / "EYE_MERGED"
IMG_DIR = MERGED / "images"
LBL_DIR = MERGED / "labels"
for d in [IMG_DIR, LBL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = ["conjunctival_pallor", "scleral_icterus", "xanthelasma"]
FULL_BOX = "0.500000 0.500000 0.950000 0.950000"

HB_ANEMIA_THRESHOLD  = 11.0  # Hgb < 11.0 g/dL → anemic → pallor (WHO)
HB_NORMAL_THRESHOLD  = 12.0  # Hgb >= 12.0 g/dL → normal → empty label

stats = defaultdict(int)


def write_label(stem: str, cls_id: int):
    """Write a full-image YOLO label."""
    lbl = LBL_DIR / f"{stem}.txt"
    lbl.write_text(f"{cls_id} {FULL_BOX}\n", encoding="utf-8")


def write_empty(stem: str):
    """Write empty label (normal / no pathology)."""
    (LBL_DIR / f"{stem}.txt").write_text("", encoding="utf-8")


def copy_image(src: Path, prefix: str = "") -> str:
    """Copy image to merged folder, return stem."""
    safe_name = re.sub(r'[^\w.-]', '_', src.name)
    dst_name  = f"{prefix}{safe_name}"
    dst       = IMG_DIR / dst_name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst.stem


def all_images(folder: Path):
    return [f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


import shutil as _sh
if IMG_DIR.exists(): _sh.rmtree(IMG_DIR); IMG_DIR.mkdir(parents=True)
if LBL_DIR.exists(): _sh.rmtree(LBL_DIR); LBL_DIR.mkdir(parents=True)

print("="*65)
print("Eye Module Dataset Merge")
print("="*65)


# ═══════════════════════════════════════════════════════════════
# SOURCE 1: conjunctiva_anemia_defy (India + Italy)
# Clinical Hb-based labeling
# ═══════════════════════════════════════════════════════════════

def load_hb_map(xlsx_path: Path) -> dict:
    """Read patient Hb values from Excel. Returns {patient_id: hb_value}."""
    if not HAS_PANDAS or not xlsx_path.exists():
        return {}
    try:
        df = pd.read_excel(str(xlsx_path))
        # Look for a column containing Hb or Haemoglobin
        # Column is 'Hgb' in India/Italy datasets, id column is 'Number'
        hb_col = None
        for col in df.columns:
            if str(col).strip().lower() in ['hgb', 'hb', 'haemoglobin', 'hemoglobin']:
                hb_col = col
                break
        if hb_col is None:
            # fallback: second numeric column
            for col in df.columns[1:]:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    hb_col = col
                    break
                except: pass
        if hb_col is None:
            return {}
        id_col = 'Number' if 'Number' in df.columns else df.columns[0]
        hb_map = {}
        for _, row in df.iterrows():
            try:
                pid = str(int(float(row[id_col])))
                hb  = float(row[hb_col])
                hb_map[pid] = hb
            except (ValueError, TypeError):
                continue
        return hb_map
    except Exception as e:
        print(f"  ⚠ Error reading {xlsx_path.name}: {e}")
        return {}


print("\n[1] conjunctiva_anemia_defy (India + Italy) — Hb-based labeling")

for country in ["India", "Italy"]:
    country_dir = BASE / "conjunctiva_anemia_defy" / "dataset anemia" / country
    if not country_dir.exists():
        print(f"  ⚠ {country} folder not found")
        continue

    # Load Hb map
    xlsx_path = country_dir / f"{country}.xlsx"
    hb_map    = load_hb_map(xlsx_path)
    print(f"  {country}: {len(hb_map)} patients with Hb data")

    labeled = normal = skipped = 0

    for patient_dir in sorted(country_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        pid  = patient_dir.name.lstrip("0") or "0"
        hb   = hb_map.get(pid)

        imgs = [f for f in patient_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        for img in imgs:
            stem = copy_image(img, f"conjdefy_{country.lower()}_p{pid}_")

            if hb is not None:
                if hb < HB_ANEMIA_THRESHOLD:
                    write_label(stem, 0)   # conjunctival_pallor
                    labeled += 1
                elif hb >= HB_NORMAL_THRESHOLD:
                    write_empty(stem)      # normal
                    normal += 1
                else:
                    write_empty(stem)      # borderline — treat as normal
                    skipped += 1
            else:
                # No Hb data — use all images as unlabeled pool
                write_empty(stem)
                skipped += 1

    print(f"    Pallor labels: {labeled}  Normal: {normal}  Borderline/unknown: {skipped}")
    stats["conjunctival_pallor"] += labeled
    stats["normal"] += normal


# ═══════════════════════════════════════════════════════════════
# SOURCE 2: palpebral_conjunctiva (183 images, all anemia-related)
# No Hb data — label all as conjunctival_pallor
# ═══════════════════════════════════════════════════════════════

print("\n[2] palpebral_conjunctiva — all labeled as conjunctival_pallor")

src = BASE / "palpebral_conjunctiva"
added = 0
for img in all_images(src):
    stem = copy_image(img, "palp_")
    write_label(stem, 0)
    added += 1

print(f"  Added: {added} images as class 0 (conjunctival_pallor)")
stats["conjunctival_pallor"] += added


# ═══════════════════════════════════════════════════════════════
# SOURCE 3: rf_conjunctiva_detector (218 images with CSV labels)
# Check _classes.csv for label mapping
# ═══════════════════════════════════════════════════════════════

print("\n[3] rf_conjunctiva_detector — YOLO CSV labels")

src = BASE / "rf_conjunctiva_detector"
added = 0

for split_dir in [src / "train", src / "valid", src / "test"]:
    if not split_dir.exists():
        continue

    # Check if there are YOLO .txt label files in a labels/ subfolder
    lbl_subdir = split_dir / "labels"
    has_yolo   = lbl_subdir.exists() and list(lbl_subdir.glob("*.txt"))

    for img in all_images(split_dir):
        stem_src = img.stem
        stem_dst = copy_image(img, "rfconj_")

        if has_yolo:
            lbl_src = lbl_subdir / f"{stem_src}.txt"
            if lbl_src.exists():
                # Remap all classes to 0 (conjunctival_pallor)
                lines = lbl_src.read_text(errors="ignore").strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        new_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                if new_lines:
                    (LBL_DIR / f"{stem_dst}.txt").write_text(
                        "\n".join(new_lines) + "\n", encoding="utf-8")
                    added += 1
                    continue
        # No YOLO labels — label whole image as conjunctival_pallor
        write_label(stem_dst, 0)
        added += 1

print(f"  Added: {added} images as class 0 (conjunctival_pallor)")
stats["conjunctival_pallor"] += added


# ═══════════════════════════════════════════════════════════════
# SOURCE 4: rf_eye_disease_yolo
# Jaundice folder → class 1 (scleral_icterus)
# Normal folder  → empty label
# ═══════════════════════════════════════════════════════════════

print("\n[4] rf_eye_disease_yolo — Jaundice=class1, Normal=empty")

src = BASE / "rf_eye_disease_yolo"
jaundice_added = normal_added = 0

for split in ["train", "valid", "test"]:
    for cls_folder, cls_id, label_fn in [
        ("Jaundice", 1, write_label),
        ("Normal",  -1, None),
    ]:
        folder = src / split / cls_folder
        if not folder.exists():
            continue
        for img in all_images(folder):
            stem = copy_image(img, f"rfjaun_{split}_")
            if cls_id == 1:
                write_label(stem, 1)
                jaundice_added += 1
            else:
                write_empty(stem)
                normal_added += 1

print(f"  Jaundice (class 1): {jaundice_added}  Normal: {normal_added}")
stats["scleral_icterus"] += jaundice_added
stats["normal"] += normal_added


# ═══════════════════════════════════════════════════════════════
# SOURCE 5: rf_xanthelasma
# All images in Xanthelasma folder → class 2
# ═══════════════════════════════════════════════════════════════

print("\n[5] rf_xanthelasma — all Xanthelasma images → class 2")

src = BASE / "rf_xanthelasma"
added = 0

for img in all_images(src):
    # Only label images inside a "Xanthelasma" folder
    if "Xanthelasma" in img.parts or "xanthelasma" in str(img).lower():
        stem = copy_image(img, "xanth_")
        write_label(stem, 2)
        added += 1

if added == 0:
    # Fallback: label all images in the dataset
    for img in all_images(src):
        if img.suffix.lower() in IMAGE_EXTS:
            stem = copy_image(img, "xanth_")
            write_label(stem, 2)
            added += 1

print(f"  Added: {added} images as class 2 (xanthelasma)")
stats["xanthelasma"] += added


# ═══════════════════════════════════════════════════════════════
# SUMMARY + data.yaml
# ═══════════════════════════════════════════════════════════════

total_imgs   = len(list(IMG_DIR.glob("*.*")))
total_labels = len([l for l in LBL_DIR.glob("*.txt") if l.stat().st_size > 0])
total_empty  = len([l for l in LBL_DIR.glob("*.txt") if l.stat().st_size == 0])

print(f"\n{'='*65}")
print("MERGE COMPLETE")
print(f"{'='*65}")
print(f"  Total images      : {total_imgs:,}")
print(f"  Labeled images    : {total_labels:,}")
print(f"  Normal/empty      : {total_empty:,}")
print(f"\n  Class distribution:")
for cls_name in CLASSES:
    cnt = stats.get(cls_name, 0)
    bar = "█" * min(30, cnt // 10)
    print(f"    {CLASSES.index(cls_name):2d}  {cls_name:<22s}  {cnt:5d}  {bar}")
print(f"    Normal images: {stats.get('normal', 0):,}")

# Write data.yaml (for training reference — actual split done in eye_train.py)
import yaml
yaml_content = {
    "path":  str(MERGED.resolve()),
    "train": str((MERGED / "train" / "images").resolve()),
    "val":   str((MERGED / "val"   / "images").resolve()),
    "nc":    len(CLASSES),
    "names": CLASSES,
}
with open(MERGED / "data.yaml", "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
print(f"\n  data.yaml written: {MERGED / 'data.yaml'}")
print(f"\n  Next: python eye_train.py")