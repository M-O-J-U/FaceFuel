"""
FaceFuel v4 — Dataset Diagnostic
==================================
Run BEFORE retraining to find:
  1. What's inside tongue_datasets/ other subfolders
  2. What class IDs exist in TONGUE_MERGED labels
  3. Why face MERGED_V2 reads 0 images
  4. What other tongue class data exists
Run: python pipeline/02_diagnose_datasets.py
"""
import os as _os, pathlib as _pl; _os.chdir(_pl.Path(__file__).resolve().parent.parent)  # paths below are repo-root relative

import os, yaml, collections
from pathlib import Path

IMG_EXT = {".jpg",".jpeg",".png",".bmp"}

def sep(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def count_imgs(p): return sum(1 for f in Path(p).rglob("*") if f.suffix.lower() in IMG_EXT) if Path(p).exists() else 0

# ── 1. Face MERGED_V2 diagnostic ──────────────────────────────
sep("FACE MERGED_V2 DIAGNOSTIC")
p = Path("facefuel_datasets/MERGED_V2")
if p.exists():
    imgs  = list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
    lbls  = list(p.rglob("*.txt"))
    yamls = list(p.rglob("*.yaml"))
    print(f"  Images found:  {len(imgs)}")
    print(f"  Labels found:  {len(lbls)}")
    print(f"  YAMLs found:   {[str(y.relative_to(p)) for y in yamls]}")
    if yamls:
        for yf in yamls:
            try:
                d = yaml.safe_load(open(yf))
                print(f"  {yf.name} classes: {d.get('names',[])}")
            except: pass
    # Check first image has a paired label
    if imgs:
        img = imgs[0]
        print(f"\n  Sample image: {img.relative_to(p)}")
        # Try label path
        lbl1 = Path(str(img).replace(os.sep+"images"+os.sep, os.sep+"labels"+os.sep)).with_suffix(".txt")
        lbl2 = img.with_suffix(".txt")
        lbl3 = img.parent.parent / "labels" / (img.stem + ".txt")
        print(f"  Label path 1: {lbl1.relative_to(p) if p in lbl1.parents else lbl1}  exists={lbl1.exists()}")
        print(f"  Label path 2: {lbl2.relative_to(p) if p in lbl2.parents else lbl2}  exists={lbl2.exists()}")
        print(f"  Label path 3: {lbl3.relative_to(p) if p in lbl3.parents else lbl3}  exists={lbl3.exists()}")
        if lbl1.exists():
            print(f"  Label content: {lbl1.read_text()[:200]}")
        elif lbl3.exists():
            print(f"  Label content: {lbl3.read_text()[:200]}")
    # Check class distribution in labels
    class_counts = collections.Counter()
    for lbl in lbls[:500]:
        try:
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if parts: class_counts[int(parts[0])] += 1
        except: pass
    print(f"\n  Class ID distribution (first 500 labels): {dict(class_counts)}")
else:
    print(f"  NOT FOUND: {p.resolve()}")

# ── 2. Tongue datasets full scan ───────────────────────────────
sep("TONGUE DATASETS — ALL SUBFOLDERS")
td = Path("tongue_datasets")
if td.exists():
    for d in sorted(td.iterdir()):
        if not d.is_dir(): continue
        imgs = count_imgs(d)
        yamls = list(d.rglob("*.yaml"))
        print(f"\n  📁 {d.name}  [{imgs} imgs]")
        for yf in yamls:
            try:
                dat = yaml.safe_load(open(yf))
                names = dat.get("names",[])
                print(f"     {yf.name}: classes={names}")
            except: pass
        # Count class IDs in labels
        class_counts = collections.Counter()
        for lbl in list(d.rglob("*.txt"))[:300]:
            try:
                for line in lbl.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5: class_counts[int(parts[0])] += 1
            except: pass
        if class_counts:
            print(f"     Class IDs (sample 300 labels): {dict(class_counts)}")
else:
    print(f"  NOT FOUND: {td.resolve()}")

# ── 3. TONGUE_MERGED class distribution (full scan) ───────────
sep("TONGUE_MERGED — FULL CLASS ID SCAN")
tm = Path("tongue_datasets/TONGUE_MERGED")
if tm.exists():
    # Read data.yaml
    yamls = list(tm.rglob("*.yaml"))
    names = []
    for yf in yamls:
        try:
            d = yaml.safe_load(open(yf))
            names = d.get("names",[])
            print(f"  data.yaml classes: {names}")
        except: pass
    # Count ALL label files
    class_counts = collections.Counter()
    all_lbls = list(tm.rglob("*.txt"))
    for lbl in all_lbls:
        try:
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5: class_counts[int(parts[0])] += 1
        except: pass
    print(f"\n  Total label files: {len(all_lbls)}")
    print(f"  Class ID distribution (ALL labels):")
    for cid in sorted(class_counts.keys()):
        name = names[cid] if cid < len(names) else f"class_{cid}"
        bar = "█" * min(40, class_counts[cid]//50)
        print(f"    {cid} {name:<25} {class_counts[cid]:>5}  {bar}")
else:
    print(f"  NOT FOUND: {tm.resolve()}")

# ── 4. tongue_datasets_v4 subfolders ──────────────────────────
sep("TONGUE_DATASETS_V4 — ALL SUBFOLDERS")
td4 = Path("tongue_datasets_v4")
if td4.exists():
    for d in sorted(td4.iterdir()):
        if not d.is_dir(): continue
        imgs = count_imgs(d)
        print(f"  📁 {d.name:<45} {imgs:>5} imgs")

# ── 5. Eye dataset check (duplicate detection) ────────────────
sep("EYE — DUPLICATE CHECK")
ec = Path("eye_datasets/EYE_COMBINED")
rx = Path("eye_datasets/rf_xanthelasma")
if ec.exists() and rx.exists():
    ec_imgs = {f.name for f in ec.rglob("*") if f.suffix.lower() in IMG_EXT}
    rx_imgs = {f.name for f in rx.rglob("*") if f.suffix.lower() in IMG_EXT}
    overlap = ec_imgs & rx_imgs
    print(f"  EYE_COMBINED: {len(ec_imgs)} unique filenames")
    print(f"  rf_xanthelasma: {len(rx_imgs)} unique filenames")
    print(f"  Overlap (duplicates): {len(overlap)}")
    print(f"  Verdict: {'⚠ DUPLICATES EXIST' if overlap else '✅ No duplicates'}")

print("\n" + "="*60)
print("  DIAGNOSTIC COMPLETE")
print("="*60)