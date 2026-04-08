"""
FaceFuel — Tongue Dataset Audit
Counts images + detects annotation format in every folder.
Run: python tongue_audit.py
"""

import json, csv
from pathlib import Path
from collections import defaultdict

BASE = Path("tongue_datasets")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
LABEL_EXTS = {".txt", ".xml", ".json", ".csv", ".yaml", ".yml"}


def detect_annotation_format(folder: Path) -> list:
    """Detect what annotation formats exist in a folder tree."""
    formats = []

    txts  = list(folder.rglob("*.txt"))
    xmls  = list(folder.rglob("*.xml"))
    jsons = list(folder.rglob("*.json"))
    csvs  = list(folder.rglob("*.csv"))

    # YOLO .txt — check if content looks like class x y w h
    valid_yolo = 0
    for t in txts[:20]:
        try:
            lines = t.read_text(errors="ignore").strip().splitlines()
            if lines and all(len(l.split()) >= 5 and l.split()[0].isdigit() for l in lines[:3]):
                valid_yolo += 1
        except: pass
    if valid_yolo > 0:
        formats.append(f"YOLO .txt ({len(txts)} files, {valid_yolo}/20 valid)")

    # COCO JSON
    for j in jsons[:5]:
        try:
            d = json.loads(j.read_text(errors="ignore"))
            if isinstance(d, dict) and "annotations" in d:
                n_ann = len(d.get("annotations", []))
                n_img = len(d.get("images", []))
                cats  = [c["name"] for c in d.get("categories", [])]
                formats.append(f"COCO JSON — {n_img} images, {n_ann} annotations, "
                                f"classes: {cats[:8]}")
                break
        except: pass

    # Pascal VOC XML
    voc_count = 0
    for x in xmls[:10]:
        try:
            if "<annotation>" in x.read_text(errors="ignore"):
                voc_count += 1
        except: pass
    if voc_count > 0:
        formats.append(f"Pascal VOC XML ({len(xmls)} files)")

    # CSV
    for c in csvs[:3]:
        try:
            text = c.read_text(errors="ignore")
            rows = list(csv.reader(text.splitlines()))
            if rows:
                formats.append(f"CSV — header: {rows[0][:6]}, rows: {len(rows)-1}")
                break
        except: pass

    # Plain JSON (non-COCO)
    non_coco_json = []
    for j in jsons[:5]:
        try:
            d = json.loads(j.read_text(errors="ignore"))
            if isinstance(d, dict) and "annotations" not in d:
                non_coco_json.append(j.name)
            elif isinstance(d, list):
                non_coco_json.append(j.name)
        except: pass
    if non_coco_json and not any("COCO" in f for f in formats):
        formats.append(f"JSON (non-COCO): {non_coco_json[:3]}")

    # Classes.txt or data.yaml — detect class names
    for p in list(folder.rglob("classes.txt")) + list(folder.rglob("data.yaml")):
        try:
            content = p.read_text(errors="ignore")
            if "names" in content or (p.name == "classes.txt" and content.strip()):
                formats.append(f"Class config: {p.relative_to(BASE)}")
            break
        except: pass

    if not formats:
        formats.append("❌ NO ANNOTATIONS FOUND")

    return formats


def scan_folder(folder: Path, depth: int = 0) -> dict:
    """Recursively scan for images, return counts per subfolder."""
    result = {
        "path":       folder,
        "images":     0,
        "subfolders": [],
        "formats":    [],
    }

    direct_imgs = [f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    result["images"] = len(direct_imgs)

    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            sub_imgs = sum(1 for f in sub.rglob("*")
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            result["subfolders"].append((sub.name, sub_imgs))

    result["formats"] = detect_annotation_format(folder)
    return result


# ── Main audit ────────────────────────────────────────────────

if not BASE.exists():
    print(f"❌ Folder not found: {BASE.resolve()}")
    exit(1)

print("=" * 70)
print("FaceFuel — Tongue Dataset Audit")
print("=" * 70)
print(f"Root: {BASE.resolve()}\n")

grand_total = 0
dataset_rows = []

for ds_folder in sorted(BASE.iterdir()):
    if not ds_folder.is_dir():
        continue

    total_imgs = sum(1 for f in ds_folder.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    grand_total += total_imgs

    # Subfolder breakdown
    subs = []
    for sub in sorted(ds_folder.iterdir()):
        if sub.is_dir():
            n = sum(1 for f in sub.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if n > 0:
                subs.append((sub.name, n))

    formats = detect_annotation_format(ds_folder)
    dataset_rows.append((ds_folder.name, total_imgs, subs, formats))

# ── Print table ───────────────────────────────────────────────

for name, total, subs, formats in dataset_rows:
    status = "✅" if total > 0 else "⚠️ "
    print(f"{status} {name}")
    print(f"   Total images : {total:,}")

    if subs:
        print(f"   Subfolders   :")
        for sname, scount in subs:
            bar = "█" * min(20, scount // 10)
            print(f"     {sname:<30s} {scount:>5}  {bar}")

    print(f"   Annotations  :")
    for fmt in formats:
        print(f"     • {fmt}")
    print()

# ── Summary ───────────────────────────────────────────────────

print("=" * 70)
print("TOTAL SUMMARY")
print("=" * 70)
print(f"\n  {'Dataset':<40s} {'Images':>7}  {'Has Labels?'}")
print(f"  {'-'*60}")
for name, total, _, formats in sorted(dataset_rows, key=lambda x: -x[1]):
    has_labels = "✅ YES" if "NO ANNOTATIONS" not in formats[0] else "❌ NO"
    print(f"  {name:<40s} {total:>7,}  {has_labels}")

print(f"\n  {'GRAND TOTAL':<40s} {grand_total:>7,}")
print(f"\n  Note: Images without labels need manual annotation or")
print(f"        pseudo-labeling before they can be used for training.")