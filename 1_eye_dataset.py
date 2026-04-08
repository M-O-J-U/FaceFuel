"""
FaceFuel — Eye Module Dataset Downloader
=========================================
Downloads all publicly available eye biomarker datasets for training
the eye analysis module.

Datasets:
  1. Eyes-Defy-Anemia (Kaggle) — conjunctival pallor, 218 images + Hb labels
  2. Roboflow eye-conjunctiva-detector — 218 annotated conjunctiva images
  3. Roboflow conjunctiva-pallor datasets — anemia classification
  4. Roboflow sclera-detection — sclera segmentation
  5. Roboflow xanthelasma-detection — cholesterol deposits near eyelid
  6. Harvard Dataverse conjunctiva dataset — clinical conjunctiva pallor
  7. Roboflow jaundice/icteric-sclera — yellow sclera detection
  8. Additional Roboflow eye condition datasets

Run:
  pip install kaggle roboflow requests
  python eye_dataset_download.py

  # Kaggle requires API key — set up at kaggle.com/account
  # Roboflow requires free account at roboflow.com
"""

import os, sys, json, zipfile, shutil
from pathlib import Path

BASE_DIR = Path("eye_datasets")
BASE_DIR.mkdir(exist_ok=True)

print("="*60)
print("FaceFuel — Eye Module Dataset Downloader")
print("="*60)
print(f"Target directory: {BASE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
# 1. KAGGLE DATASETS
# ═══════════════════════════════════════════════════════════════

def download_kaggle(dataset_slug: str, out_dir: Path, desc: str):
    """Download a Kaggle dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] {desc}")
    print(f"  Dataset: {dataset_slug}")
    try:
        import kaggle
        kaggle.api.dataset_download_files(dataset_slug,
                                           path=str(out_dir),
                                           unzip=True)
        files = list(out_dir.rglob("*.*"))
        print(f"  ✅ Downloaded: {len(files)} files")
        return True
    except ImportError:
        print("  ❌ kaggle package not installed: pip install kaggle")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        print(f"  Manual download: https://www.kaggle.com/datasets/{dataset_slug}")
    return False


# ═══════════════════════════════════════════════════════════════
# 2. ROBOFLOW DATASETS
# ═══════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = os.environ.get("O3UYShcuMuCuuWzo9697", "")

def download_roboflow(workspace: str, project: str, version: int,
                      out_dir: Path, desc: str, fmt: str = "yolov8"):
    """Download a Roboflow dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Roboflow] {desc}")
    print(f"  Project: {workspace}/{project} v{version}")

    if not ROBOFLOW_API_KEY:
        print("  ⚠ No API key. Set: set ROBOFLOW_API_KEY=your_key")
        print(f"  Manual: https://universe.roboflow.com/{workspace}/{project}")
        return False

    try:
        from roboflow import Roboflow
        rf   = Roboflow(api_key=ROBOFLOW_API_KEY)
        proj = rf.workspace(workspace).project(project)
        ds   = proj.version(version).download(fmt, location=str(out_dir))
        print(f"  ✅ Downloaded to {out_dir}")
        return True
    except ImportError:
        print("  ❌ roboflow package not installed: pip install roboflow")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        print(f"  Manual: https://universe.roboflow.com/{workspace}/{project}")
    return False


# ═══════════════════════════════════════════════════════════════
# 3. DIRECT HTTP DOWNLOADS
# ═══════════════════════════════════════════════════════════════

def download_http(url: str, out_path: Path, desc: str):
    """Download a file via HTTP."""
    import requests
    print(f"\n[HTTP] {desc}")
    print(f"  URL: {url}")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✅ Saved: {out_path}")
        return True
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
    return False


# ═══════════════════════════════════════════════════════════════
# DOWNLOAD ALL DATASETS
# ═══════════════════════════════════════════════════════════════

results = {}

print("\n" + "="*60)
print("PART 1 — CONJUNCTIVAL PALLOR (Iron/B12/Anemia)")
print("="*60)

# Dataset 1a: Eyes-Defy-Anemia (Kaggle)
# 218 conjunctiva images with actual Hb measurements — clinical ground truth
results["eyes_defy_anemia"] = download_kaggle(
    "harshwardhanfartale/eyes-defy-anemia",
    BASE_DIR / "conjunctiva_anemia_defy",
    "Eyes-Defy-Anemia — 218 conjunctiva images with Hb lab measurements"
)

# Dataset 1b: Conjunctiva anemia (Kaggle, palpebral)
results["palpebral_conjunctiva"] = download_kaggle(
    "guptajanavi/palpebral-conjunctiva-to-detect-anaemia",
    BASE_DIR / "palpebral_conjunctiva",
    "Palpebral conjunctiva anemia detection dataset"
)

# Dataset 1c: Roboflow eye conjunctiva detector
results["rf_conjunctiva_detector"] = download_roboflow(
    "eyeconjunctivadetector", "eye-conjunctiva-detector", 2,
    BASE_DIR / "rf_conjunctiva_detector",
    "Roboflow conjunctiva detector — 218 images with YOLO annotations"
)

# Dataset 1d: Roboflow conjunctiva segmentation datasets
for rf_ds in [
    ("conjunctiva-segmentation", "conjunctiva-segmentation", 1,
     "Roboflow conjunctiva segmentation"),
    ("conjunctiva-4yq5u", "conjunctiva", 1,
     "Roboflow conjunctiva classification"),
]:
    ws, proj, ver, desc = rf_ds
    results[f"rf_{proj}"] = download_roboflow(
        ws, proj, ver,
        BASE_DIR / f"rf_{proj}",
        desc
    )

print("\n" + "="*60)
print("PART 2 — SCLERAL ICTERUS / YELLOW SCLERA (Liver stress)")
print("="*60)

# Dataset 2a: Jaundice / icteric sclera detection
for rf_ds in [
    ("jaundice-detection-3utrm", "jaundice-detection", 1,
     "Jaundice / icteric sclera detection"),
    ("yellow-eyes-ggjnz", "yellow-eyes", 1,
     "Yellow sclera classification"),
    ("scleral-icterus", "scleral-icterus", 1,
     "Scleral icterus detection"),
]:
    ws, proj, ver, desc = rf_ds
    results[f"rf_{proj}"] = download_roboflow(
        ws, proj, ver,
        BASE_DIR / f"rf_{proj}",
        desc
    )

print("\n" + "="*60)
print("PART 3 — SCLERA DETECTION (General eye region)")
print("="*60)

results["rf_sclera"] = download_roboflow(
    "sclera-detection", "sclera", 3,
    BASE_DIR / "rf_sclera",
    "Sclera detection and segmentation"
)

results["rf_eye_disease"] = download_roboflow(
    "eye-diseases-classification", "eye-diseases-classification", 1,
    BASE_DIR / "rf_eye_disease",
    "General eye disease classification"
)

print("\n" + "="*60)
print("PART 4 — XANTHELASMA (Cholesterol deposits near eyes)")
print("="*60)

for rf_ds in [
    ("xanthelasma-detection", "xanthelasma", 1,
     "Xanthelasma detection — cholesterol deposits near eyelid"),
    ("xanthelasma-qxcuf", "xanthelasma-detection", 1,
     "Xanthelasma classification v2"),
]:
    ws, proj, ver, desc = rf_ds
    results[f"rf_{proj}_{ver}"] = download_roboflow(
        ws, proj, ver,
        BASE_DIR / f"rf_{proj}_{ver}",
        desc
    )

print("\n" + "="*60)
print("PART 5 — SUBCONJUNCTIVAL HEMORRHAGE (Vitamin C deficiency)")
print("="*60)

results["rf_subconj"] = download_roboflow(
    "subconjunctival-hemorrhage", "subconjunctival-hemorrhage", 1,
    BASE_DIR / "rf_subconjunctival_hemorrhage",
    "Subconjunctival hemorrhage — vitamin C deficiency marker"
)

print("\n" + "="*60)
print("PART 6 — PERIORBITAL / EYE BAG (Existing face pipeline)")
print("="*60)

# These supplement existing dark circle / eye bag datasets already in face pipeline
results["rf_dark_circles_eye"] = download_roboflow(
    "eye-condition-6jb16", "eye-condition", 1,
    BASE_DIR / "rf_eye_condition_general",
    "General eye condition detection (dark circles, puffiness)"
)

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("DOWNLOAD SUMMARY")
print(f"{'='*60}")
success = sum(1 for v in results.values() if v)
total   = len(results)
print(f"  Downloaded: {success}/{total} datasets")

for name, ok in results.items():
    status = "✅" if ok else "⚠ "
    print(f"  {status} {name}")

# Count files
all_imgs = []
for ext in [".jpg",".jpeg",".png"]:
    all_imgs.extend(BASE_DIR.rglob(f"*{ext}"))
print(f"\n  Total images downloaded: {len(all_imgs):,}")
print(f"  Directory: {BASE_DIR.resolve()}")

print(f"""
{'='*60}
MANUAL DOWNLOAD LINKS (if automated download failed)
{'='*60}

Eyes-Defy-Anemia (with real Hb measurements):
  https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia
  Also: https://ieee-dataport.org/documents/eyes-defy-anemia

Harvard Dataverse conjunctiva pallor dataset:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4MDKC

Roboflow eye-conjunctiva-detector:
  https://universe.roboflow.com/eyeconjunctivadetector/eye-conjunctiva-detector

Roboflow sclera detection:
  https://universe.roboflow.com/sclera-detection/sclera

Roboflow jaundice detection:
  Search "jaundice" on universe.roboflow.com

{'='*60}
NEXT STEP:
  python eye_merge.py     <- merge all sources into unified dataset
  python eye_train.py     <- train YOLOv8m eye detector
{'='*60}
""")