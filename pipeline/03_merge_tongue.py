"""
FaceFuel v4 — Tongue Dataset Merger (complete rewrite)
Run: python pipeline/03_merge_tongue.py
"""
import os as _os, pathlib as _pl; _os.chdir(_pl.Path(__file__).resolve().parent.parent)  # paths below are repo-root relative
import os, shutil, random, yaml
from pathlib import Path
from collections import Counter

random.seed(42)
VAL_RATIO = 0.15
OUT = Path("facefuel_tongue_v4")
for s in ["train", "val"]:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

CLASSES = [
    "tongue_body", "white_coating", "yellow_coating", "thick_coating",
    "no_coating", "red_tongue", "pale_tongue", "purple_tongue",
    "fissured", "geographic", "smooth_glossy", "crenated",
    "oral_ulcer", "angular_stomatitis", "median_rhomboid",
    "lichen_planus", "leukoplakia", "hairy_leukoplakia",
]
C2I     = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
counter = Counter()
total   = [0]


def copy_img(src, stem, split, cls_id):
    ext = Path(src).suffix.lower()
    if ext not in IMG_EXT:
        return False
    di = OUT / "images" / split / f"{stem}{ext}"
    dl = OUT / "labels" / split / f"{stem}.txt"
    if di.exists():
        return False
    try:
        shutil.copy2(src, di)
        dl.write_text(f"{cls_id} 0.500000 0.500000 0.990000 0.990000\n")
        counter[CLASSES[cls_id]] += 1
        total[0] += 1
        return True
    except Exception:
        return False


def add_flat(folder, cls_name, prefix, max_imgs=2000):
    folder = Path(folder)
    if not folder.exists():
        print(f"  skip: {folder}")
        return
    if cls_name not in C2I:
        return
    cid  = C2I[cls_name]
    imgs = [f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXT]
    random.shuffle(imgs)
    added = 0
    for img in imgs[:max_imgs]:
        sp = "val" if random.random() < VAL_RATIO else "train"
        if copy_img(img, f"{prefix}_{total[0]:07d}", sp, cid):
            added += 1
    print(f"  {folder.name} -> {cls_name}: {added}")


def add_cls(base, class_map, prefix, max_pc=2000):
    base = Path(base)
    if not base.exists():
        print(f"  skip: {base}")
        return
    added = Counter()
    for folder in sorted(base.rglob("*")):
        if not folder.is_dir():
            continue
        cls = class_map.get(folder.name)
        if cls is None or cls not in C2I:
            continue
        cid  = C2I[cls]
        imgs = [f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXT]
        random.shuffle(imgs)
        for img in imgs[:max_pc]:
            sp = "val" if random.random() < VAL_RATIO else "train"
            if copy_img(img, f"{prefix}_{total[0]:07d}", sp, cid):
                added[cls] += 1
    print(f"  {base.name}: {sum(added.values())}  {dict(added)}")


def add_yolo_flat(base, src_classes, prefix, name_remap=None, max_per_class=3000):
    """YOLO-format dataset: images/ + labels/ (flat or split)."""
    base = Path(base)
    if not base.exists():
        print(f"  skip: {base}")
        return
    all_imgs = (list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
                + list(base.rglob("*.bmp")))
    if not all_imgs:
        print(f"  skip (no images): {base}")
        return
    per_cls = Counter()
    added   = Counter()
    for img in all_imgs:
        lbl = None
        p1  = Path(str(img).replace(
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep)).with_suffix(".txt")
        p3  = img.parent.parent / "labels" / (img.stem + ".txt")
        if p1.exists():
            lbl = p1
        elif p3.exists():
            lbl = p3
        if lbl is None:
            continue
        new_lines = []
        for line in lbl.read_text(errors="ignore").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            sid = int(parts[0])
            if sid >= len(src_classes):
                continue
            sname = src_classes[sid]
            tname = name_remap.get(sname, sname) if name_remap else sname
            if tname is None:
                continue
            tid = C2I.get(tname)
            if tid is None:
                continue
            if per_cls[tname] >= max_per_class:
                continue
            new_lines.append(f"{tid} {' '.join(parts[1:])}")
            per_cls[tname] += 1
        if not new_lines:
            continue
        sp   = "val" if random.random() < VAL_RATIO else "train"
        stem = f"{prefix}_{total[0]:07d}"
        di   = OUT / "images" / sp / f"{stem}{img.suffix.lower()}"
        if di.exists():
            continue
        try:
            shutil.copy2(img, di)
            (OUT / "labels" / sp / f"{stem}.txt").write_text(
                "\n".join(new_lines) + "\n")
            for nl in new_lines:
                cn = CLASSES[int(nl.split()[0])]
                counter[cn] += 1
                added[cn]   += 1
            total[0] += 1
        except Exception:
            pass
    print(f"  {base.name}: {sum(added.values())}  {dict(added)}")


# ── Actual class lists from data.yaml (verified by diagnostic) ─
TONGUE_SRC = [
    "tongue_body", "fissured", "crenated", "pale_tongue", "red_tongue",
    "yellow_coating", "white_coating", "thick_coating", "geographic",
    "smooth_glossy", "tooth_marked", "black_hairy_tongue",
]
TONGUE_REMAP = {
    "tooth_marked":       "crenated",
    "black_hairy_tongue": "hairy_leukoplakia",
}

MOD_MAP = {
    "CaS": "oral_ulcer",   # Canker Sores
    "CoS": "oral_ulcer",   # Cold Sores
    "Gum": None,
    "MC":  "leukoplakia",  # Mouth Cancer
    "OC":  "oral_ulcer",
    "OLP": "lichen_planus",
    "OT":  "white_coating",
}

ORAL_DISEASE_MAP = {
    "Mouth Ulcer": "oral_ulcer", "Gingivitis": "oral_ulcer",
    "Ulcer": "oral_ulcer", "ulcer": "oral_ulcer",
    "Leukoplakia": "leukoplakia", "Lichen Planus": "lichen_planus",
    "Angular Cheilitis": "angular_stomatitis",
    "Candidiasis": "white_coating", "Thrush": "white_coating",
    "Normal": "tongue_body", "Healthy": "tongue_body",
    "Calculus": None, "Caries": None, "Hypodontia": None,
    "Tooth Discoloration": None,
}

# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("FaceFuel v4 — Tongue Merge")
print("=" * 65)

print("\n[1] TONGUE_MERGED — actual 12 classes")
add_yolo_flat("tongue_datasets/TONGUE_MERGED", TONGUE_SRC,
              "orig", name_remap=TONGUE_REMAP, max_per_class=3000)

print("\n[2] TONGUE_COMBINED — 9,083 imgs")
add_yolo_flat("tongue_datasets/TONGUE_COMBINED", TONGUE_SRC,
              "tc", name_remap=TONGUE_REMAP, max_per_class=2000)

print("\n[3] Oral ulcer (4,416 imgs)")
add_cls("tongue_datasets_v4/oral_ulcer", {
    "HERPETIFORM ULCERATION": "oral_ulcer",
    "Infectious ulcer":       "oral_ulcer",
    "MAJOR RAS":              "oral_ulcer",
    "MINOR RAS":              "oral_ulcer",
    "OSCC":                   "leukoplakia",
    "Traumatic ulcer":        "oral_ulcer",
    "empty":                  None,
}, "ou")

print("\n[4] Oral diseases salmansajid (use once, skip duplicate)")
add_cls("tongue_datasets_v4/oral_diseases", ORAL_DISEASE_MAP, "od")

print("\n[5] Mouth oral diseases MOD")
add_cls("tongue_datasets_v4/mouth_oral_diseases_mod", MOD_MAP, "mod")

print("\n[6] Oral ulcers v2 (Mendeley Karnataka clinical)")
add_cls("tongue_datasets_v4/oral_ulcers_v2", {
    "benign_lesions":    "oral_ulcer",
    "augmented_benign":  "oral_ulcer",
    "malignant_lesions": "leukoplakia",
    "augmented_malignant": "leukoplakia",
}, "mend")

print("\n[7] Oral cancer tongue")
add_cls("tongue_datasets_v4/oral_cancer_tongue", {
    "Oral Ulcer": "oral_ulcer", "oral_ulcer": "oral_ulcer",
    "Leukoplakia": "leukoplakia", "Lichen Planus": "lichen_planus",
    "Normal": "tongue_body", "Candidiasis": "white_coating",
    "Angular Cheilitis": "angular_stomatitis",
}, "oct")

print("\n[8] Ataturk 623 patients (if emailed)")
add_yolo_flat("tongue_datasets_v4/ataturk_tongue_623", TONGUE_SRC,
              "atk", name_remap=TONGUE_REMAP, max_per_class=2000)

# ── data.yaml ──────────────────────────────────────────────────
ti = len(list((OUT / "images" / "train").glob("*")))
vi = len(list((OUT / "images" / "val").glob("*")))
yaml.dump({"path": str(OUT.resolve()), "train": "images/train",
           "val": "images/val", "nc": len(CLASSES), "names": CLASSES},
          open(OUT / "data.yaml", "w"), default_flow_style=False)

print(f"\n{'='*65}")
print(f"TONGUE COMPLETE  train={ti:,}  val={vi:,}  total={ti+vi:,}")
print(f"{'='*65}")
for c in CLASSES:
    n   = counter.get(c, 0)
    bar = "█" * min(50, n // 50)
    print(f"  {c:<25} {n:>6}  {bar}")
print("\n  ⚠ ZERO classes:")
for c in CLASSES:
    if counter.get(c, 0) == 0:
        print(f"    {c}")