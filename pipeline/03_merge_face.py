"""
FaceFuel v4 — Face Dataset Merger (complete rewrite)
Run: python pipeline/03_merge_face.py
"""
import os as _os, pathlib as _pl; _os.chdir(_pl.Path(__file__).resolve().parent.parent)  # paths below are repo-root relative
import os, shutil, random, yaml, csv
from pathlib import Path
from collections import Counter

random.seed(42)
VAL_RATIO = 0.15
OUT = Path("facefuel_face_v4")
for s in ["train", "val"]:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

CLASSES = [
    "dark_circle", "eye_bag", "pallor", "acne", "blackhead", "wrinkle",
    "redness", "yellow_skin", "dark_spot", "periorbital_puffiness", "dull_skin",
    "rosacea", "vitiligo", "eczema", "angular_cheilitis", "perioral_dermatitis",
    "butterfly_rash", "enlarged_pores", "oily_skin", "perifollicular_hem",
]
C2I     = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def add_cls(base, class_map, prefix, max_pc=3000):
    """Classification-style folder tree — folder name = class."""
    base = Path(base)
    if not base.exists():
        print(f"  skip (not found): {base}")
        return
    added = Counter()
    for folder in sorted(base.rglob("*")):
        if not folder.is_dir():
            continue
        cls = class_map.get(folder.name)
        if cls is None or cls not in C2I:
            continue
        imgs = [f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXT]
        random.shuffle(imgs)
        for img in imgs[:max_pc]:
            sp = "val" if random.random() < VAL_RATIO else "train"
            if copy_img(img, f"{prefix}_{total[0]:07d}", sp, C2I[cls]):
                added[cls] += 1
    print(f"  {base.name}: {sum(added.values())}  {dict(added)}")


def add_yolo_flat(base, src_classes, prefix, name_remap=None, max_per_class=3000):
    """YOLO-format dataset: images/ + labels/ (flat or train/val/test split)."""
    base = Path(base)
    if not base.exists():
        print(f"  skip (not found): {base}")
        return
    all_imgs = (list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
                + list(base.rglob("*.bmp")))
    if not all_imgs:
        print(f"  skip (no images): {base}")
        return
    per_cls = Counter()
    added   = Counter()
    for img in all_imgs:
        # find label
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


# ── class maps ─────────────────────────────────────────────────
DERMNET_MAP = {
    "Acne and Rosacea Photos":                            "acne",
    "Atopic Dermatitis Photos":                           "eczema",
    "Bullous Disease Photos":                             "eczema",
    "Eczema Photos":                                      "eczema",
    "Exanthems and Drug Eruptions":                       "redness",
    "Light Diseases and Disorders of Pigmentation":       "vitiligo",
    "Lupus and other Connective Tissue diseases":         "butterfly_rash",
    "Melanoma Skin Cancer Nevi and Moles":                "dark_spot",
    "Poison Ivy Photos and other Contact Dermatitis":     "eczema",
    "Psoriasis pictures Lichen Dermatosis and related diseases": "eczema",
    "Seborrheic Keratoses and other Benign Tumors":       "dark_spot",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "eczema",
    "Urticaria Hives":                                    "redness",
    "Vascular Tumors":                                    "redness",
    "Vasculitis Photos":                                  "redness",
}

COMP_MAP = {
    "1. Eczema 1677":                                     "eczema",
    "2. Melanoma 15.75k":                                 "dark_spot",
    "3. Atopic Dermatitis - 1.25k":                       "eczema",
    "4. Basal Cell Carcinoma (BCC) 3323":                 "dark_spot",
    "5. Melanocytic Nevi (NV) - 7970":                    "dark_spot",
    "6. Benign Keratosis-like Lesions (BKL) 2624":        "dark_spot",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "eczema",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "dark_spot",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "eczema",
    "10. Warts Molluscum and other Viral Infections - 2103": "dark_spot",
}

AMELLIA_MAP = {
    "Acne": "acne", "Oily Skin": "oily_skin", "Dry Skin": "dull_skin",
    "Normal": None, "Rosacea": "rosacea", "Vitiligo": "vitiligo", "Eczema": "eczema",
}

HAM_MAP = {
    "melanocytic_nevi": "dark_spot", "melanoma": "dark_spot",
    "benign_keratosis-like_lesions": "dark_spot", "basal_cell_carcinoma": "dark_spot",
    "actinic_keratoses": "dark_spot", "vascular_lesions": "redness",
    "dermatofibroma": "dark_spot",
}

ISIC9_MAP = {
    "MEL": "dark_spot", "NV": "dark_spot", "BCC": "dark_spot",
    "AKIEC": "dark_spot", "BKL": "dark_spot", "DF": "dark_spot", "VASC": "redness",
}

AUG_MAP = {
    "Clinical Actinic Keratosis": "dark_spot",
    "Clinical Basal Cell Carcinoma": "dark_spot",
    "Clinical Dermatofibroma": "dark_spot",
    "Clinical Melanoma": "dark_spot",
    "Clinical Nevus": "dark_spot",
    "Clinical Pigmented Benign Keratosis": "dark_spot",
    "Clinical Seborrheic Keratosis": "dark_spot",
    "Clinical Squamous Cell Carcinoma": "dark_spot",
    "Clinical Vascular Lesion": "redness",
    "Derma Acitinic Keratosis": "dark_spot",
    "Derma Basal Cell Carcinoma": "dark_spot",
    "Derma Dermatofibroma": "dark_spot",
    "Derma Melanoma": "dark_spot",
    "Derma Nevus": "dark_spot",
    "Derma Pigmented Benign Keratosis": "dark_spot",
    "Derma Seborrheic Keratosis": "dark_spot",
    "Derma Squamous Cell Carcinoma": "dark_spot",
    "Derma Vascular Lesion": "redness",
}

# MERGED_V2 actual 25 classes (from data.yaml)
MERGED_V2_SRC = [
    "dark_circle", "eye_bag", "acne", "wrinkle", "redness", "dry_skin",
    "oily_skin", "dark_spot", "blackhead", "pallor", "lip_dry", "melasma",
    "whitehead", "pore", "eye_redness", "yellow_sclera", "acne_scar",
    "pigmentation", "skin_dullness", "forehead_wrinkle", "nasolabial_fold",
    "crow_feet", "skin_texture_rough", "vascular_redness", "lip_pallor",
]
MERGED_V2_REMAP = {
    "dry_skin": "dull_skin", "lip_dry": None, "melasma": "dark_spot",
    "whitehead": "blackhead", "pore": "enlarged_pores",
    "eye_redness": "redness", "yellow_sclera": "yellow_skin",
    "acne_scar": "dark_spot", "pigmentation": "dark_spot",
    "skin_dullness": "dull_skin", "forehead_wrinkle": "wrinkle",
    "nasolabial_fold": "wrinkle", "crow_feet": "wrinkle",
    "skin_texture_rough": "dull_skin", "vascular_redness": "redness",
    "lip_pallor": "pallor",
}

# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("FaceFuel v4 — Face Merge")
print("=" * 65)

print("\n[1] DermNet")
add_cls("facefuel_datasets_v4/dermnet", DERMNET_MAP, "dn", max_pc=2000)

print("\n[2] Skin diseases comprehensive")
add_cls("facefuel_datasets_v4/skin_diseases_comprehensive/IMG_CLASSES",
        COMP_MAP, "sdc", max_pc=2000)

print("\n[3] Face skin amellia")
add_cls("facefuel_datasets_v4/face_skin_amellia", AMELLIA_MAP, "fa")

print("\n[4] HAM10000")
ham = Path("facefuel_datasets_v4/ham10000")
if ham.exists():
    csvs = list(ham.rglob("*.csv"))
    id2dx = {}
    if csvs:
        with open(csvs[0]) as f:
            for row in csv.DictReader(f):
                id2dx[row.get("image_id", "")] = row.get("dx", "")
    added = Counter()
    for img in ham.rglob("*.jpg"):
        cls = HAM_MAP.get(id2dx.get(img.stem, ""))
        if not cls:
            continue
        sp = "val" if random.random() < VAL_RATIO else "train"
        if copy_img(img, f"ham_{total[0]:07d}", sp, C2I[cls]):
            added[cls] += 1
    print(f"  HAM10000: {sum(added.values())}  {dict(added)}")
else:
    print(f"  skip: {ham}")

print("\n[5] Skin disease multiclass")
add_cls("facefuel_datasets_v4/skin_disease_multiclass",
        {"chickenpox": "redness", "cowpox": "redness",
         "measles": "redness", "monkeypox": "redness", "healthy": None}, "smc")

print("\n[6] Skin cancer ISIC9")
add_cls("facefuel_datasets_v4/skin_cancer_isic9", ISIC9_MAP, "isic9")

print("\n[7] Vitiligo")
add_cls("facefuel_datasets_v4/vitiligo",
        {"vitiligo": "vitiligo", "Vitiligo": "vitiligo",
         "Healthy Skin": None, "non_vitiligo": None}, "vit")

print("\n[8] Augmented skin conditions")
add_cls("facefuel_datasets_v4/augmented_skin_conditions", AUG_MAP, "aug")

print("\n[9] Multimodal augmented (HAM10000-based)")
add_cls("facefuel_datasets_v4/multimodal_augmented_skin", ISIC9_MAP, "mas", max_pc=500)

print("\n[10] ORIGINAL face MERGED_V2 — 25 classes YOLO format")
add_yolo_flat("facefuel_datasets/MERGED_V2", MERGED_V2_SRC,
              "orig", name_remap=MERGED_V2_REMAP, max_per_class=3000)

# ── data.yaml ──────────────────────────────────────────────────
ti = len(list((OUT / "images" / "train").glob("*")))
vi = len(list((OUT / "images" / "val").glob("*")))
yaml.dump({"path": str(OUT.resolve()), "train": "images/train",
           "val": "images/val", "nc": len(CLASSES), "names": CLASSES},
          open(OUT / "data.yaml", "w"), default_flow_style=False)

print(f"\n{'='*65}")
print(f"FACE COMPLETE  train={ti:,}  val={vi:,}  total={ti+vi:,}")
print(f"{'='*65}")
for c in CLASSES:
    n   = counter.get(c, 0)
    bar = "█" * min(50, n // 100)
    print(f"  {c:<25} {n:>6}  {bar}")
print("\n  ⚠ ZERO classes:")
for c in CLASSES:
    if counter.get(c, 0) == 0:
        print(f"    {c}")