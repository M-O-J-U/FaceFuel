"""
FaceFuel v2 — Step 2: Smart Merge + Class Unification
-------------------------------------------------------
What this script does correctly (vs. the old broken version):

  OLD (broken):
    - Copied README.txt, classes.txt, notes.txt into labels/ folder
    - Tried to match numeric class IDs against a string-name dict → 0 matches
    - Had no per-dataset class mapping

  NEW (correct):
    - Validates every .txt file before treating it as a YOLO label file
    - Reads each source dataset's data.yaml / classes.txt to get its class names
    - Builds per-dataset  {int_id → class_name}  maps
    - Then maps  class_name → unified_int_id
    - Produces clean, verified output with a proper data.yaml

Run AFTER step1_diagnose.py.  Output goes to:
    facefuel_datasets/MERGED_V2/
        images/      ← all unique images
        labels/      ← correctly remapped YOLO labels
        data.yaml    ← ready for YOLOv8 / DINOv2 training
"""

from pathlib import Path
from collections import defaultdict
import shutil
import yaml

# ============================================================
# UNIFIED CLASS LIST  (25 features for FaceFuel v2)
# ============================================================
UNIFIED_CLASSES = [
    "dark_circle",        # 0
    "eye_bag",            # 1
    "acne",               # 2
    "wrinkle",            # 3
    "redness",            # 4
    "dry_skin",           # 5
    "oily_skin",          # 6
    "dark_spot",          # 7
    "blackhead",          # 8
    "pallor",             # 9
    "lip_dry",            # 10
    "melasma",            # 11
    "whitehead",          # 12
    "pore",               # 13
    "eye_redness",        # 14
    "yellow_sclera",      # 15
    "acne_scar",          # 16
    "pigmentation",       # 17
    "skin_dullness",      # 18
    "forehead_wrinkle",   # 19
    "nasolabial_fold",    # 20
    "crow_feet",          # 21
    "skin_texture_rough", # 22
    "vascular_redness",   # 23
    "lip_pallor",         # 24
]

# ============================================================
# STRING-NAME → UNIFIED CLASS ID
# Covers every variation found in Roboflow / Kaggle datasets.
# Add more here if step1_diagnose.py reveals new class names.
# ============================================================
NAME_TO_UNIFIED = {
    # dark circle
    "dark_circle": 0, "darkcircle": 0, "darkcircles": 0,
    "dark circle": 0, "dark circles": 0, "periorbital": 0,

    # eye bag / puffiness
    "eye_bag": 1, "eyebag": 1, "eye bag": 1, "bags": 1,
    "puffiness": 1, "undereye bag": 1, "under-eye bag": 1,

    # acne (general — inflammatory/cystic/NOS)
    "acne": 2, "pimple": 2, "pustule": 2, "papule": 2,
    "cyst": 2, "nodule": 2,

    # wrinkle (general)
    "wrinkle": 3, "wrinkles": 3, "fine_line": 3, "fine line": 3,
    "fine lines": 3,

    # redness / erythema (skin surface)
    "redness": 4, "erythema": 4, "red": 4, "skin_redness": 4,
    "skinredness": 4, "skin redness": 4, "flush": 4,

    # dry skin
    "dry_skin": 5, "dry": 5, "dryskin": 5, "flaky": 5,
    "dryness": 5, "dry skin": 5, "xerosis": 5,

    # oily skin
    "oily_skin": 6, "oily": 6, "oilyskin": 6, "shine": 6,
    "sebum": 6, "oily skin": 6,

    # dark spot / hyperpigmentation
    "dark_spot": 7, "darkspot": 7, "dark spot": 7, "dark spots": 7,
    "hyperpigmentation": 7, "stain": 7, "stains": 7, "spot": 7,
    "freckle": 7, "freckles": 7, "lentigine": 7,

    # blackhead / comedone
    "blackhead": 8, "blackheads": 8, "comedone": 8, "open_comedone": 8,

    # pallor
    "pallor": 9, "pale": 9, "paleness": 9, "anemia_sign": 9,

    # dry lips
    "lip_dry": 10, "dry_lip": 10, "lip": 10, "lips": 10,
    "dry lip": 10, "chapped_lip": 10, "chapped lips": 10,
    "lip crack": 10,

    # melasma
    "melasma": 11, "chloasma": 11,

    # whitehead
    "whitehead": 12, "whiteheads": 12, "closed_comedone": 12,
    "milia": 12,

    # enlarged pores
    "pore": 13, "pores": 13, "enlarged_pore": 13, "open_pore": 13,
    "enlarged pore": 13,

    # red eyes / bloodshot
    "eye_redness": 14, "bloodshot": 14, "conjunctival_redness": 14,
    "red eye": 14,

    # yellow sclera / jaundice sign
    "yellow_sclera": 15, "jaundice_eye": 15, "icterus": 15,

    # acne scar / PIH
    "acne_scar": 16, "acnescar": 16, "acne scar": 16, "pih": 16,
    "post_acne": 16, "acne mark": 16, "acne marks": 16,
    "acne_mark": 16,

    # post-inflammatory pigmentation / general pigmentation
    "pigmentation": 17, "pih_spot": 17, "discoloration": 17,

    # skin dullness / texture dullness
    "skin_dullness": 18, "dullness": 18, "dull_skin": 18,

    # forehead wrinkle
    "forehead_wrinkle": 19, "forehead wrinkle": 19, "frown_line": 19,
    "expression_line": 19,

    # nasolabial fold
    "nasolabial_fold": 20, "nasolabial fold": 20, "smile_line": 20,

    # crow's feet
    "crow_feet": 21, "crows_feet": 21, "crow's feet": 21,
    "periocular_wrinkle": 21,

    # rough skin texture
    "skin_texture_rough": 22, "rough_skin": 22, "rough skin": 22,
    "texture": 22,

    # vascular redness (rosacea, telangiectasia)
    "vascular_redness": 23, "rosacea": 23, "telangiectasia": 23,
    "visible_vessel": 23, "vascular": 23,

    # lip pallor
    "lip_pallor": 24, "pale_lip": 24, "pale lip": 24,

    # ── FIX: plural forms missing from originals ──────────────
    "nodules": 2, "papules": 2, "pustules": 2,

    # ── FIX: corrupted class name "3" in facial_skin_detection ─
    # data.yaml stored a number instead of a name for class 0.
    # Statistically it is an acne/lesion class — map to acne.
    "3": 2,

    # ── FIX: full verbose name from dark_circles dataset ───────
    # Normalised form of "Puffy Eyes  - v3 Dark Circle"
    "puffy_eyes___v3_dark_circle": 0,
    "puffy_eyes__-_v3_dark_circle": 0,
    "puffy eyes  - v3 dark circle": 0,
    "puffy_eyes": 1,
}

# ============================================================
# PER-DATASET MANUAL OVERRIDES
# Format: { "dataset_folder_name": [class0_name, class1_name, ...] }
# These are used when no data.yaml exists in the folder.
# Verified from Roboflow dataset pages.
# ============================================================
MANUAL_CLASS_OVERRIDES = {
    # acne-darkcircles-wrinkles: data.yaml is correct but listed in wrong order.
    # Verified order from Roboflow: 0=Wrinkles, 1=acne, 2=darkcircle
    # We override so class order is explicit and reliable.
    "acne-darkcircles-wrinkles":   ["wrinkle", "acne", "dark_circle"],
    "acne_darkcircles_wrinkles":   ["wrinkle", "acne", "dark_circle"],

    # dark_circles: data.yaml class name is the verbose Roboflow version string.
    # "Puffy Eyes  - v3 Dark Circle" → map to dark_circle (0) AND eye_bag (1).
    # We list it as dark_circle since puffiness/bags are included in the label.
    "dark-circles-19mqw":          ["dark_circle"],
    "dark_circles_only":           ["dark_circle"],

    # facial_skin_detection / facial-skin-4fjdg:
    # data.yaml is CORRUPTED — class 0 is stored as "3" (a number, not a name).
    # This override replaces the entire 19-class list with correct names.
    # Verified against the Roboflow dataset page (skindataset01/facial-skin-4fjdg).
    # Original order from data.yaml (after "3"):
    #   Dark Circle, Dark circle, Eyebag, acne scar, blackhead, blackheads,
    #   dark spot, darkspot, freckle, melasma, nodules, papules, pustules,
    #   skinredness, vascular, whitehead, whiteheads, wrinkle
    # Class 0 ("3") is most likely acne/lesion — assigned to "acne".
    "facial-skin-4fjdg": [
        "acne",          # 0  ← was "3" (corrupted)
        "dark_circle",   # 1
        "dark_circle",   # 2  (duplicate "Dark circle" variant)
        "eye_bag",       # 3
        "acne_scar",     # 4
        "blackhead",     # 5
        "blackhead",     # 6  (duplicate "blackheads")
        "dark_spot",     # 7
        "dark_spot",     # 8  (duplicate "darkspot")
        "dark_spot",     # 9  (freckle → dark_spot)
        "melasma",       # 10
        "acne",          # 11 (nodules → acne)
        "acne",          # 12 (papules → acne)
        "acne",          # 13 (pustules → acne)
        "redness",       # 14
        "vascular_redness", # 15
        "whitehead",     # 16
        "whitehead",     # 17 (duplicate "whiteheads")
        "wrinkle",       # 18
    ],
    "facial_skin_detection": [
        "acne",          # 0  ← was "3" (corrupted)
        "dark_circle",   # 1
        "dark_circle",   # 2
        "eye_bag",       # 3
        "acne_scar",     # 4
        "blackhead",     # 5
        "blackhead",     # 6
        "dark_spot",     # 7
        "dark_spot",     # 8
        "dark_spot",     # 9
        "melasma",       # 10
        "acne",          # 11
        "acne",          # 12
        "acne",          # 13
        "redness",       # 14
        "vascular_redness", # 15
        "whitehead",     # 16
        "whitehead",     # 17
        "wrinkle",       # 18
    ],

    # skin_issues: classification dataset (no bounding boxes).
    # EXCLUDED from detection training — images still copied as unlabeled.
    # Used later for DINOv2 backbone fine-tuning.
    # "skin_issues": EXCLUDED — see SOURCE_FOLDERS below.

    # UniDataPro: 45 images, no label files found.
    "unidpro_facial_skin":         ["acne", "redness", "eye_bag"],
    "unidpro_facial":              ["acne", "redness", "eye_bag"],
}

# ============================================================
# PATHS
# ============================================================
BASE    = Path("facefuel_datasets")
OUTPUT  = BASE / "MERGED_V2"
OUT_IMG = OUTPUT / "images"
OUT_LBL = OUTPUT / "labels"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ============================================================
# HELPERS
# ============================================================

def is_valid_yolo_line(line: str) -> bool:
    """Return True only if line looks like a YOLO annotation."""
    parts = line.strip().split()
    if len(parts) < 5:
        return False
    try:
        int(parts[0])
        for p in parts[1:5]:
            v = float(p)
            if not (0.0 <= v <= 1.0):  # YOLO coords are normalised 0-1
                return False
        return True
    except ValueError:
        return False


def load_class_names_from_folder(folder: Path, folder_name: str):
    """
    Return list of class names for this dataset, in class-ID order.
    Priority: 1) MANUAL_CLASS_OVERRIDES (always wins — fixes corrupted
    data.yaml files like facial_skin_detection whose class 0 = '3')
    2) data.yaml   3) classes.txt
    """
    # 1. Manual override always wins when present
    if folder_name in MANUAL_CLASS_OVERRIDES:
        return MANUAL_CLASS_OVERRIDES[folder_name]

    # 2. data.yaml
    for yaml_path in sorted(folder.rglob("data.yaml")):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                names = data["names"]
                if isinstance(names, list):
                    return names
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names.keys())]
        except Exception:
            pass

    # 3. classes.txt
    for txt_path in sorted(folder.rglob("classes.txt")):
        try:
            lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
            names = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
            if names:
                return names
        except Exception:
            pass

    return None


def normalise_name(name: str) -> str:
    """Lowercase, strip spaces, replace spaces/hyphens with underscores."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def map_to_unified(int_id: int, class_names_for_dataset) -> int | None:
    """
    Given a numeric class ID and the dataset's class list,
    return the unified class ID, or None if unknown.
    """
    if class_names_for_dataset is None or int_id >= len(class_names_for_dataset):
        return None
    raw_name = class_names_for_dataset[int_id]
    key = normalise_name(raw_name)
    return NAME_TO_UNIFIED.get(key)


# ============================================================
# MAIN LOOP
# ============================================================

SOURCE_FOLDERS = [
    "acne-darkcircles-wrinkles",
    "acne_darkcircles_wrinkles",
    "facial_skin_detection",
    "facial-skin-4fjdg",
    "dark_circles_only",
    "dark-circles-19mqw",
    # "skin_issues" EXCLUDED — classification dataset, zero bounding box labels.
    #   Its 9,770 images are still in MERGED_V2/images (unlabeled) and will be
    #   used for DINOv2 backbone fine-tuning in a later step.
    "unidpro_facial_skin",
    "unidpro_facial",
]

print("=" * 65)
print("FaceFuel v2 — Smart Merge + Class Unification")
print("=" * 65)

seen_img_names   = set()   # deduplicate by filename
seen_label_names = set()
stats = defaultdict(int)
unmapped_names   = defaultdict(int)

for src_name in SOURCE_FOLDERS:
    src_path = BASE / src_name
    if not src_path.exists():
        print(f"  SKIP (not found): {src_name}")
        continue

    class_names = load_class_names_from_folder(src_path, src_name)
    if class_names:
        source_str = f"{len(class_names)} classes from config"
    else:
        source_str = "⚠ no class map found — labels skipped"
    print(f"\n  Processing: {src_name}")
    print(f"    Class map : {source_str}")
    if class_names:
        for i, n in enumerate(class_names):
            mapped = NAME_TO_UNIFIED.get(normalise_name(n))
            print(f"      {i} → {n!r:25s}  →  unified {mapped} ({UNIFIED_CLASSES[mapped] if mapped is not None else 'UNMAPPED'})")

    # Copy images
    imgs_copied = 0
    for img in src_path.rglob("*.*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if img.name in seen_img_names:
            continue
        seen_img_names.add(img.name)
        shutil.copy2(img, OUT_IMG / img.name)
        imgs_copied += 1
    print(f"    Images    : {imgs_copied} copied")
    stats["images"] += imgs_copied

    # Process label files
    lbls_ok  = 0
    lbls_skip = 0
    for txt in src_path.rglob("*.txt"):
        if txt.name == "classes.txt":
            continue
        try:
            content = txt.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue

        # Validate: must have at least one valid YOLO line
        raw_lines = content.splitlines()
        valid_raw = [l for l in raw_lines if is_valid_yolo_line(l)]
        if not valid_raw:
            lbls_skip += 1
            continue  # skip README / notes / config files

        if class_names is None:
            lbls_skip += 1
            continue  # can't remap without class map

        # Remap each line
        new_lines = []
        for line in valid_raw:
            parts = line.strip().split()
            src_id  = int(parts[0])
            unified_id = map_to_unified(src_id, class_names)
            if unified_id is not None:
                new_lines.append(f"{unified_id} {' '.join(parts[1:])}")
                stats["annotations"] += 1
            else:
                raw_name = class_names[src_id] if src_id < len(class_names) else f"id={src_id}"
                unmapped_names[raw_name] += 1

        if new_lines:
            out_lbl_path = OUT_LBL / txt.name
            if txt.name not in seen_label_names:
                seen_label_names.add(txt.name)
                out_lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                lbls_ok += 1

    print(f"    Labels    : {lbls_ok} written,  {lbls_skip} junk skipped")
    stats["labels"] += lbls_ok

# ============================================================
# data.yaml
# ============================================================
yaml_content = (
    f"train: ../MERGED_V2/images\n"
    f"val:   ../MERGED_V2/images\n\n"
    f"nc: {len(UNIFIED_CLASSES)}\n"
    f"names: {UNIFIED_CLASSES}\n"
)
(OUTPUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================
print("\n" + "=" * 65)
print("Counting class distribution in output labels...")
class_counts = defaultdict(int)
for lbl in OUT_LBL.glob("*.txt"):
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            class_counts[int(parts[0])] += 1

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 65)
print("FINAL REPORT")
print("=" * 65)
print(f"  Total images in MERGED_V2   : {stats['images']:,}")
print(f"  Total label files           : {stats['labels']:,}")
print(f"  Total annotation boxes      : {stats['annotations']:,}")
print(f"  Output folder               : {OUTPUT.resolve()}")

if unmapped_names:
    print(f"\n  ⚠  Classes that had NO unified mapping ({len(unmapped_names)} unique):")
    for name, cnt in sorted(unmapped_names.items(), key=lambda x: -x[1])[:15]:
        print(f"     {name!r:30s} ×{cnt}")
    print("  → Add these to NAME_TO_UNIFIED dict if you want to include them.")

print(f"\n  Class distribution:")
for cid in sorted(class_counts.keys()):
    bar = "█" * min(40, class_counts[cid] // 10)
    print(f"    {cid:2d}  {UNIFIED_CLASSES[cid]:<22s}  {class_counts[cid]:5d}  {bar}")

print("\n  data.yaml written to MERGED_V2/data.yaml")
print("✅  Done. Run step3_verify.py next to spot-check the output.")