"""
FaceFuel v2 — Tongue Module: Dataset Downloader (v3 — clean)
=============================================================
All downloads use direct ZIP (no git required) or Kaggle/Roboflow APIs.
No chest X-rays. No irrelevant datasets.

Expected total: ~4,000–5,000 tongue images automatically
                ~3,000+ more from manual section D
"""

import os, sys, zipfile, urllib.request, time
from pathlib import Path

OUT_BASE = Path("tongue_datasets")
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Replace with your Roboflow API key ──────────────────────
ROBOFLOW_API_KEY = "O3UYShcuMuCuuWzo9697"

print("=" * 65)
print("FaceFuel — Tongue Dataset Downloader v3")
print("=" * 65)
print(f"Output: {OUT_BASE.resolve()}\n")


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def download_zip(url: str, dest: Path, desc: str):
    """Download a ZIP directly — no git needed."""
    if dest.exists() and count_images(dest) > 0:
        print(f"  SKIP (exists): {dest.name}  [{count_images(dest)} imgs]")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "_dl.zip"
    try:
        print(f"  Downloading: {desc} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
        n = count_images(dest)
        print(f"  ✅ {dest.name}: {n} images")
    except Exception as e:
        if zip_path.exists(): zip_path.unlink()
        print(f"  ❌ {dest.name}: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION A — Direct GitHub ZIP downloads (no git required)
# ═══════════════════════════════════════════════════════════════

print("[A] GitHub direct ZIP downloads...")

github_zips = [
    (
        "https://github.com/BioHit/TongeImageDataset/archive/refs/heads/master.zip",
        "biohit_300",
        "BioHit — 300 tongue images + segmentation masks"
    ),
    (
        "https://github.com/BioHit/tongue-image-dataset/archive/refs/heads/master.zip",
        "biohit_v2",
        "BioHit v2 — additional tongue images"
    ),
    (
        "https://github.com/cshan-github/TongueSAM/archive/refs/heads/main.zip",
        "tonguesam_full",
        "TongueSAM — contains TongueSet3 (1000 in-the-wild tongue images)"
    ),
    (
        "https://github.com/pengjianqiang/FDU-TC/archive/refs/heads/main.zip",
        "fdu_tc_cracks",
        "FDU-TC — tongue crack/fissure images (B3/dehydration markers)"
    ),
    (
        "https://github.com/pariskang/ZhongJing-OMNI/archive/refs/heads/main.zip",
        "zhongjing_tcm",
        "ZhongJing-OMNI — TCM tongue diagnosis with expert Q&A (Fudan Univ.)"
    ),
    (
        "https://github.com/Hamed-Aghapanah/Tongue_Detection_Classification/archive/refs/heads/main.zip",
        "tongue_disease_clf",
        "Tongue disease classification dataset"
    ),
]

for url, folder, desc in github_zips:
    download_zip(url, OUT_BASE / folder, desc)


# ═══════════════════════════════════════════════════════════════
# SECTION B — Kaggle (tongue-specific only)
# ═══════════════════════════════════════════════════════════════

print("\n[B] Kaggle datasets (tongue only)...")

try:
    import kaggle

    kaggle_tongue = [
        (
            "thngdngvn/biohit-tongue-image-dataset",
            "kaggle_biohit",
            "BioHit tongue — 300 images with masks (Kaggle mirror)"
        ),
        (
            "clearhanhui/biyesheji",
            "kaggle_tooth_marked",
            "Tooth-marked tongue — 564 tooth-marked + 704 normal = 1268 images"
        ),
    ]

    for slug, folder, desc in kaggle_tongue:
        dest = OUT_BASE / folder
        if dest.exists() and count_images(dest) > 0:
            print(f"  SKIP (exists): {folder}  [{count_images(dest)} imgs]")
            continue
        try:
            print(f"  Downloading: {folder}  [{desc}]")
            kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True)
            print(f"  ✅ {folder}: {count_images(dest)} images")
        except Exception as e:
            print(f"  ❌ {folder}: {e}")

except ImportError:
    print("  ⚠ pip install kaggle")
except Exception as e:
    print(f"  ❌ Kaggle: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION C — Roboflow (tongue-specific, correct formats)
# ═══════════════════════════════════════════════════════════════

print("\n[C] Roboflow datasets...")

if ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY_HERE":
    print("  ⚠ Add your Roboflow API key at the top of this script.")
else:
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)

        # Detection format (yolov8) — tongue bounding boxes
        det_sets = [
            ("tongue-heso0",    "tongue-0f5gm",             1, "rf_tongue_general_46"),
            ("whateverunusued", "tongue-segmentation-0foqr", 1, "rf_tongue_seg_75"),
            ("oral-cancer-synsj","tongue-segmentation-uorhh",1, "rf_oral_tongue_96"),
        ]
        for ws, proj, ver, folder in det_sets:
            dest = OUT_BASE / folder
            if dest.exists() and count_images(dest) > 0:
                print(f"  SKIP (exists): {folder}  [{count_images(dest)} imgs]")
                continue
            try:
                print(f"  Downloading: {folder}")
                rf.workspace(ws).project(proj).version(ver).download(
                    "yolov8", location=str(dest))
                print(f"  ✅ {folder}: {count_images(dest)} images")
            except Exception as e:
                print(f"  ❌ {folder}: {e}")

        # Segmentation format — must use coco-segmentation, not yolov8
        seg_sets = [
            ("minh-ha-tixet", "tongue-segmentation-deq4x", 5, "rf_tongue_seg_2476"),
        ]
        for ws, proj, ver, folder in seg_sets:
            dest = OUT_BASE / folder
            if dest.exists() and count_images(dest) > 0:
                print(f"  SKIP (exists): {folder}  [{count_images(dest)} imgs]")
                continue
            try:
                print(f"  Downloading: {folder}  (coco-segmentation format)")
                rf.workspace(ws).project(proj).version(ver).download(
                    "coco-segmentation", location=str(dest))
                print(f"  ✅ {folder}: {count_images(dest)} images")
            except Exception as e:
                print(f"  ❌ {folder}: {e}")
                print(f"     Manual: https://universe.roboflow.com/{ws}/{proj}")

    except ImportError:
        print("  ⚠ pip install roboflow")
    except Exception as e:
        print(f"  ❌ Roboflow: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION D — Manual downloads (open in browser)
# ═══════════════════════════════════════════════════════════════

print("\n[D] MANUAL DOWNLOADS — open these URLs in your browser:")

manual = [
    {
        "name":  "⭐ SciDB TCM Tongue — 1,194 images (BEST DATASET)",
        "desc":  "Full TCM annotations: color, shape, cracks, tooth marks, coating. "
                 "Annotated by 3 TCM doctors. Best label quality for nutrient mapping.",
        "url":   "https://www.scidb.cn/en/detail?dataSetId=8417299de5ef4f3db5ec62e01a969d54",
        "save":  "tongue_datasets/scidb_tcm_1194/",
    },
    {
        "name":  "⭐ Mendeley Diabetes Tongue — ~500 images",
        "desc":  "Diabetic vs healthy tongue. Blood test confirmed. 5 shots/person, 48MP.",
        "url":   "https://data.mendeley.com/datasets/hyb44jf936/2",
        "save":  "tongue_datasets/mendeley_diabetes_tongue/",
    },
    {
        "name":  "IEEE DataPort Annotated Tongue Images",
        "desc":  "Clinical tongue images with patient data. Free IEEE account required.",
        "url":   "https://ieee-dataport.org/open-access/annotated-dataset-tongue-images",
        "save":  "tongue_datasets/ieee_tongue/",
    },
    {
        "name":  "Roboflow: tongue-segmentation-deq4x — 2,476 images",
        "desc":  "If section C failed, download manually. Select version + PNG Mask format.",
        "url":   "https://universe.roboflow.com/minh-ha-tixet/tongue-segmentation-deq4x",
        "save":  "tongue_datasets/rf_tongue_seg_2476/",
    },
    {
        "name":  "Roboflow: tongue features (fissured/crenated) — 477 images",
        "desc":  "Download latest version as YOLOv8 format.",
        "url":   "https://universe.roboflow.com/my-testing/tongue-aggwh",
        "save":  "tongue_datasets/rf_tongue_fissured/",
    },
    {
        "name":  "Roboflow: tongue_color classification",
        "desc":  "Red/pale/purple/normal tongue color classes.",
        "url":   "https://universe.roboflow.com/jl-lk6uw/tongue_color",
        "save":  "tongue_datasets/rf_tongue_color/",
    },
]

for i, ds in enumerate(manual, 1):
    print(f"\n  {i}. {ds['name']}")
    print(f"     {ds['desc']}")
    print(f"     URL : {ds['url']}")
    print(f"     Save: {ds['save']}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
total = 0
for d in sorted(OUT_BASE.iterdir()):
    if d.is_dir():
        n = count_images(d)
        total += n
        if n > 0:
            print(f"  {d.name:<40s} {n:>5} images")

print(f"\n  Automated total : {total:,} images")
print(f"  After manual D  : +~4,200 more")
print(f"  Expected final  : ~{total+4200:,} tongue images")
print(f"\n  Folder: {OUT_BASE.resolve()}")


import os, sys, shutil, zipfile, subprocess, urllib.request
from pathlib import Path

OUT_BASE = Path("tongue_datasets")
OUT_BASE.mkdir(parents=True, exist_ok=True)

ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY_HERE"

print("=" * 65)
print("FaceFuel — Tongue Dataset Downloader v2")
print("=" * 65)
print(f"Output: {OUT_BASE.resolve()}\n")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def count_images(folder: Path) -> int:
    return len(list(folder.rglob("*.jpg"))) + \
           len(list(folder.rglob("*.png"))) + \
           len(list(folder.rglob("*.jpeg")))

def clone_repo(url: str, dest: Path):
    """Full clone — needed for Git LFS repos."""
    if dest.exists() and count_images(dest) > 0:
        print(f"  SKIP (exists): {dest.name}  [{count_images(dest)} imgs]")
        return
    try:
        subprocess.run(["git", "clone", "--depth=1", url, str(dest)],
                       check=True, capture_output=True)
        # Pull LFS objects if available
        try:
            subprocess.run(["git", "lfs", "pull"], check=True,
                           cwd=str(dest), capture_output=True)
        except Exception:
            pass
        n = count_images(dest)
        print(f"  ✅ {dest.name}: {n} images")
    except Exception as e:
        print(f"  ❌ {dest.name}: {e}")

def download_zip(url: str, dest: Path, desc: str):
    """Download and extract a ZIP file."""
    if dest.exists() and count_images(dest) > 0:
        print(f"  SKIP (exists): {dest.name}  [{count_images(dest)} imgs]")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "_download.zip"
    try:
        print(f"  Downloading: {desc}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
        n = count_images(dest)
        print(f"  ✅ {dest.name}: {n} images")
    except Exception as e:
        print(f"  ❌ {dest.name}: {e}")


def rf_download(rf, workspace, project, version, folder, fmt="yolov8"):
    dest = OUT_BASE / folder
    if dest.exists() and count_images(dest) > 0:
        print(f"  SKIP (exists): {folder}  [{count_images(dest)} imgs]")
        return
    try:
        print(f"  Downloading: {folder}")
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download(fmt, location=str(dest))
        print(f"  ✅ {folder}: {count_images(dest)} images")
    except Exception as e:
        print(f"  ❌ {folder}: {e}")