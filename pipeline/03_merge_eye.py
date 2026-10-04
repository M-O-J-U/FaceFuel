"""
FaceFuel v4 — Eye Dataset Merger (VERIFIED paths from scan)
============================================================
Output: facefuel_eye_v4/
Run:    python pipeline/03_merge_eye.py

CRITICAL NOTES from scan:
  - mendeley_eye_disease = FUNDUS IMAGES (retinal, requires ophthalmoscope)
    CANNOT be used for selfie eye surface detection
    ONLY Pterygium folder has relevant images — use those only
  - mendeley_uveitis = GOOD surface eye images:
    Subfolder: "Image Dataset on Eye Diseases Classification..."
    Conjunctivitis(357) + Eyelid(525) + Uveitis(223) + Normal(649) + Cataract(544)
  - eye_diseases_classification == eye_diseases_maria (EXACT DUPLICATES — use only one)
  - eye_disease_kondwani == kaggle_eye_disease (EXACT DUPLICATES — use only one)
"""
import os as _os, pathlib as _pl; _os.chdir(_pl.Path(__file__).resolve().parent.parent)  # paths below are repo-root relative

import shutil, random, yaml, json, csv
from pathlib import Path
from collections import Counter
import os

random.seed(42)
VAL_RATIO = 0.15
OUT = Path("facefuel_eye_v4")
for s in ["train","val"]:
    (OUT/"images"/s).mkdir(parents=True, exist_ok=True)
    (OUT/"labels"/s).mkdir(parents=True, exist_ok=True)

CLASSES = [
    "conjunctival_pallor",   # 0  iron/B12 deficiency
    "scleral_icterus",       # 1  liver stress
    "xanthelasma",           # 2  cholesterol imbalance
    "pterygium",             # 3  vitamin A / UV (102 imgs from mendeley)
    "conjunctivitis",        # 4  immune / vitamin A / C (357 imgs from mendeley_uveitis)
    "eyelid_drooping",       # 5  thyroid / neuromuscular (525 imgs from mendeley_uveitis)
    # pinguecula REMOVED — 0 training images (add when Roboflow downloads work)
    # dry_eye    REMOVED — 0 training images (add when Roboflow downloads work)
]
C2I = {c:i for i,c in enumerate(CLASSES)}

def add_yolo_flat(base, src_classes, tgt_classes, prefix, max_per_class=3000):
    """
    Import a YOLO-format dataset (images/ + labels/ flat or train/val/test split).
    src_classes: list of class names as they appear in the source data.yaml
    tgt_classes: the C2I dict of the target dataset
    """
    base = Path(base)
    if not base.exists():
        print(f"  ⏭ Not found: {base}")
        return
    # Find all images recursively (handles flat + split formats)
    all_imgs = list(base.rglob("*.jpg")) + list(base.rglob("*.png")) + list(base.rglob("*.bmp"))
    if not all_imgs:
        print(f"  ⏭ No images in: {base}")
        return
    per_class_count = Counter()
    added = Counter()
    for img in all_imgs:
        # Find label: same stem, replace /images/ with /labels/ in path
        lbl = None
        try:
            lbl_path = str(img).replace(os.sep + "images" + os.sep,
                                        os.sep + "labels" + os.sep)
            lbl_path = lbl_path.rsplit(".", 1)[0] + ".txt"
            lbl = Path(lbl_path)
            if not lbl.exists():
                lbl = img.with_suffix(".txt")
            if not lbl.exists():
                lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                continue
        except Exception:
            continue
        try:
            lines = lbl.read_text().strip().splitlines()
        except Exception:
            continue
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            src_id = int(parts[0])
            if src_id >= len(src_classes):
                continue
            src_name = src_classes[src_id]
            tgt_id   = tgt_classes.get(src_name)
            if tgt_id is None:
                continue
            if per_class_count[src_name] >= max_per_class:
                continue
            new_lines.append(f"{tgt_id} {' '.join(parts[1:])}")
            per_class_count[src_name] += 1
        if not new_lines:
            continue
        sp   = "val" if random.random() < VAL_RATIO else "train"
        stem = f"{prefix}_{total[0]:07d}"
        ext  = img.suffix.lower()
        di   = OUT / "images" / sp / f"{stem}{ext}"
        dl   = OUT / "labels" / sp / f"{stem}.txt"
        if di.exists():
            continue
        try:
            shutil.copy2(img, di)
            dl.write_text("\n".join(new_lines) + "\n")
            for l in new_lines:
                cid = int(l.split()[0])
                counter[CLASSES[cid]] += 1
            total[0] += 1
            added[src_classes[int(new_lines[0].split()[0])] 
                  if new_lines else "?"] += 1
        except Exception:
            pass
    print(f"  {base.name}: {sum(added.values())}  {dict(added)}")

IMG_EXT = {".jpg",".jpeg",".png",".bmp"}
counter = Counter()
total = [0]

def copy_img(src, stem, split, cls_id):
    ext = Path(src).suffix.lower()
    if ext not in IMG_EXT: return False
    di = OUT/"images"/split/f"{stem}{ext}"
    dl = OUT/"labels"/split/f"{stem}.txt"
    if di.exists(): return False
    try:
        shutil.copy2(src, di)
        dl.write_text(f"{cls_id} 0.500000 0.500000 0.990000 0.990000\n")
        counter[CLASSES[cls_id]] += 1
        total[0] += 1
        return True
    except: return False

def add_flat(folder, cls_name, prefix, max_imgs=2000):
    folder = Path(folder)
    if not folder.exists(): print(f"  ⏭ {folder}"); return
    if cls_name not in C2I: return
    cid = C2I[cls_name]
    imgs = [f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXT]
    random.shuffle(imgs); added=0
    for img in imgs[:max_imgs]:
        sp="val" if random.random()<VAL_RATIO else "train"
        if copy_img(img, f"{prefix}_{total[0]:07d}", sp, cid): added+=1
    print(f"  {folder.name} → {cls_name}: {added}")

def add_cls(base, class_map, prefix, max_pc=2000):
    base = Path(base)
    if not base.exists(): print(f"  ⏭ {base}"); return
    added = Counter()
    for folder in sorted(base.rglob("*")):
        if not folder.is_dir(): continue
        cls = class_map.get(folder.name)
        if cls is None: continue
        if cls not in C2I: continue
        cid = C2I[cls]
        imgs=[f for f in folder.iterdir()
              if f.is_file() and f.suffix.lower() in IMG_EXT]
        random.shuffle(imgs)
        for img in imgs[:max_pc]:
            sp="val" if random.random()<VAL_RATIO else "train"
            if copy_img(img, f"{prefix}_{total[0]:07d}", sp, cid): added[cls]+=1
    print(f"  {base.name}: {sum(added.values())}  {dict(added)}")

print("="*65)
print("FaceFuel v4 — Eye Merge (VERIFIED PATHS)")
print("="*65)

# ── 1. Original eye dataset (EYE_COMBINED) YOLO flat format ───
print("\n[1] Original eye EYE_COMBINED (v3 classes, YOLO format)")
V3_EYE = ["conjunctival_pallor","scleral_icterus","xanthelasma"]
add_yolo_flat("eye_datasets/EYE_COMBINED", V3_EYE, C2I, "orig", max_per_class=3000)

# Existing individual source folders
print("\n[1b] Original source: rf_xanthelasma (1,043 imgs)")
add_flat("eye_datasets/rf_xanthelasma", "xanthelasma", "xan", max_imgs=1043)

print("\n[1c] Original source: rf_conjunctiva_detector (218 imgs)")
add_flat("eye_datasets/rf_conjunctiva_detector", "conjunctival_pallor", "rfc")

print("\n[1d] Original source: rf_eye_disease_yolo (191 imgs)")
add_cls("eye_datasets/rf_eye_disease_yolo", {
    "Jaundice":"scleral_icterus","jaundice":"scleral_icterus",
    "Normal":None,"normal":None,
}, "rfyd")

print("\n[1e] palpebral conjunctiva (183 imgs)")
add_flat("eye_datasets/palpebral_conjunctiva", "conjunctival_pallor", "pal")

# ── 2. mendeley_uveitis — GOOD surface images ─────────────────
# Subfolder name (exact from scan):
# "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis,
#  Cataract, Eyelid) with Symptoms and SMOTE Validation"
print("\n[2] Mendeley uveitis/conjunctivitis/eyelid (2,298 imgs)")
mendeley_uv_inner = None
base = Path("eye_datasets_v4/mendeley_uveitis")
if base.exists():
    for d in base.rglob("*"):
        if d.is_dir() and "Image Dataset" in d.name:
            mendeley_uv_inner = d
            break
    if mendeley_uv_inner:
        add_cls(mendeley_uv_inner, {
            "Conjunctivitis":   "conjunctivitis",
            "Eyelid":           "eyelid_drooping",
            "Uveitis":          "conjunctivitis",
            "Cataract":         None,
            "Normal":           None,
        }, "muv")
    else:
        # Try direct subfolders
        add_cls(base, {
            "Conjunctivitis":"conjunctivitis","conjunctivitis":"conjunctivitis",
            "Eyelid":"eyelid_drooping","eyelid_drooping":"eyelid_drooping",
            "Uveitis":"conjunctivitis","uveitis":"conjunctivitis",
            "Cataract":None,"Normal":None,
        }, "muv")

# ── 3. mendeley_eye_disease — FUNDUS ONLY use Pterygium ───────
# SKIP all retinal classes. Only Pterygium (102 augmented imgs).
print("\n[3] Mendeley eye disease — PTERYGIUM ONLY (fundus images — rest skipped)")
add_cls("eye_datasets_v4/mendeley_eye_disease", {
    "Pterygium":                            "pterygium",
    # All others are fundus/retinal — NOT usable for selfie surface eye
    "Diabetic Retinopathy":                 None,
    "Glaucoma":                             None,
    "Macular Scar":                         None,
    "Myopia":                               None,
    "Retinal Detachment":                   None,
    "Retinitis Pigmentosa":                 None,
    "Healthy":                              None,
    "Central Serous Chorioretinopathy [Color Fundus]": None,
    "Disc Edema":                           None,
}, "mend_pt")

# ── 4. eye_diseases_classification (4,217 imgs) ───────────────
# USE THIS ONE — skip eye_diseases_maria (exact duplicate)
print("\n[4] Eye diseases classification (4-class)")
add_cls("eye_datasets_v4/eye_diseases_classification", {
    "cataract":None,"Cataract":None,
    "glaucoma":None,"Glaucoma":None,
    "normal":None,"Normal":None,
    "diabetic_retinopathy":None,"Diabetic Retinopathy":None,
    # If it has conjunctivitis class
    "conjunctivitis":"conjunctivitis","Conjunctivitis":"conjunctivitis",
    "dry_eye":"dry_eye","dry eye":"dry_eye",
}, "ec4")

# ── 5. eye_disease_kondwani (766 imgs) ────────────────────────
# USE THIS ONE — skip kaggle_eye_disease (exact duplicate)
print("\n[5] Eye disease kondwani (conjunctivitis, cataract, normal)")
add_cls("eye_datasets_v4/eye_disease_kondwani", {
    "conjunctivitis":"conjunctivitis","Conjunctivitis":"conjunctivitis",
    "Cataract":None,"cataract":None,"Normal":None,"normal":None,"glaucoma":None,
}, "ekond")

# ── 6. ODIR-5K — extract conjunctivitis from JSON labels ──────
print("\n[6] ODIR-5K (extract relevant classes from JSON)")
odir = Path("eye_datasets_v4/odir5k")
if odir.exists():
    json_files = list(odir.rglob("*.json"))
    img_label  = {}
    for jf in json_files:
        try:
            data = json.load(open(jf))
            for entry in data:
                for side in ["Left","Right"]:
                    fname = entry.get(f"{side}-Fundus","")
                    kw    = entry.get(f"{side}-Diagnostic Keywords","")
                    if fname: img_label[fname] = kw.lower()
        except: pass

    # Also check CSV
    csvf = odir/"full_df.csv"
    if csvf.exists():
        with open(csvf) as f:
            for row in csv.DictReader(f):
                for side in ["Left-Fundus","Right-Fundus"]:
                    fname = row.get(side,"")
                    kw    = (row.get("Left-Diagnostic Keywords","") +
                             row.get("Right-Diagnostic Keywords","")).lower()
                    if fname: img_label[fname] = kw

    added = Counter()
    for img in odir.rglob("*.jpg"):
        diag = img_label.get(img.name,"")
        # Only conjunctivitis-like keywords from surface eye
        cls = None
        if "conjunctivitis" in diag: cls = "conjunctivitis"
        elif "pterygium" in diag:    cls = "pterygium"
        elif "pinguecula" in diag:   cls = "pinguecula"
        if cls is None: continue
        sp = "val" if random.random()<VAL_RATIO else "train"
        stem = f"odir_{total[0]:07d}"
        if copy_img(img, stem, sp, C2I[cls]): added[cls]+=1
    print(f"  ODIR-5K: {sum(added.values())} usable  {dict(added)}")
else:
    print(f"  ⏭ {odir}")

# ── 7. eye_periorbital (6,027 imgs — monkeypox periorbital) ───
# These are periorbital skin lesions — map rash patterns around eye to redness
# Not useful for our eye classes — skip
print("\n[7] eye_periorbital — skipping (periorbital skin, not eye surface)")

# ── Write data.yaml ───────────────────────────────────────────
ti = len(list((OUT/"images/train").glob("*")))
vi = len(list((OUT/"images/val").glob("*")))
yaml.dump({"path":str(OUT.resolve()),"train":"images/train","val":"images/val",
           "nc":len(CLASSES),"names":CLASSES},
          open(OUT/"data.yaml","w"), default_flow_style=False)

print(f"\n{'='*65}")
print(f"EYE MERGE COMPLETE  train={ti:,}  val={vi:,}  total={ti+vi:,}")
print(f"{'='*65}")
for c in CLASSES:
    n=counter.get(c,0); bar="█"*min(50,n//30)
    print(f"  {c:<25} {n:>6}  {bar}")
print(f"\n  ⚠ LOW CLASSES (need Roboflow downloads):")
for c in CLASSES:
    if counter.get(c,0) < 100:
        print(f"    {c}: {counter.get(c,0)} — run Roboflow after setting key")
print(f"""
  NOTE: To get pterygium/pinguecula/dry_eye images:
  PowerShell: $env:ROBOFLOW_API_KEY = "your_key_here"
  then: python pipeline/01_download_datasets.py --eye-only
""")