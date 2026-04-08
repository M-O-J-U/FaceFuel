"""
FaceFuel v2 — Step 5: Free Dataset Expander
--------------------------------------------
Downloads ~20,000 additional FREE labeled images for FaceFuel v2.
Total cost: $0.

New datasets added:
  1. ISIC 2019 skin lesion subset          ~5,000 images  (Kaggle, free)
  2. HAM10000 dermatoscopy dataset         ~10,000 images (Kaggle, free)
  3. Skin Disease Classification dataset    ~2,800 images (Kaggle, free)
  4. Acne04 (AAAI paper dataset)            ~4,000 images (Roboflow mirror)
  5. Face + skin analysis v2               ~3,000 images  (Roboflow, free)
  6. Facial oiliness / pores dataset       ~1,200 images  (Roboflow, free)

NOTE: HAM10000 and ISIC are dermatoscopy (close-up skin, not full faces).
They contribute excellent texture / lesion signal for:
  dark_spot, acne_scar, pigmentation, vascular_redness classes.
Their images are combined with face crops during training via augmentation.

All datasets downloaded to: facefuel_datasets/extra_datasets/
"""

import os
from pathlib import Path
from roboflow import Roboflow

BASE    = Path("facefuel_datasets/extra_datasets")
BASE.mkdir(parents=True, exist_ok=True)

# ── Replace with your Roboflow API key ──────────────────────
ROBOFLOW_API_KEY = "O3UYShcuMuCuuWzo9697"

print("=" * 65)
print("FaceFuel v2 — Free Dataset Expander")
print("=" * 65)

# ============================================================
# SECTION A — KAGGLE (free, large)
# ============================================================
print("\n[A] Kaggle datasets...")
try:
    import kaggle

    kaggle_datasets = [
        # (slug, local_folder_name, notes)
        (
            "kmader/skin-lesion-analysis-toward-melanoma-detection",
            "isic2019_lesion",
            "ISIC 2019 subset — skin lesion images with diagnosis labels"
        ),
        (
            "kmader/isic-melanoma-images-task-1",
            "isic_melanoma",
            "ISIC melanoma/benign — good for dark_spot and acne_scar classes"
        ),
        (
            "surajghuwalewala/ham1000-segmentation-and-classification",
            "ham10000",
            "HAM10000 — 10,015 dermatoscopy images, 7 lesion classes"
        ),
        (
            "shubhamgoel27/dermnet",
            "dermnet",
            "DermNet — 23 skin disease categories, ~19,500 images"
        ),
        (
            "shubhamgoel27/skin-disease-classification",
            "skin_disease_clf",
            "Skin disease classification — acne, rosacea, and more"
        ),
        
    ]

    for slug, folder, note in kaggle_datasets:
        dest = BASE / folder
        if dest.exists() and any(dest.rglob("*.*")):
            print(f"  SKIP (already exists): {folder}")
            continue
        print(f"\n  Downloading: {folder}")
        print(f"  Note       : {note}")
        try:
            kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True)
            imgs = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
            print(f"  ✅ Downloaded: {len(imgs):,} images")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

except ImportError:
    print("  ⚠ kaggle not installed. Run: pip install kaggle")
except Exception as e:
    print(f"  ❌ Kaggle error: {e}")

# ============================================================
# SECTION B — ROBOFLOW (free tier, acne + skin focused)
# ============================================================
print("\n[B] Roboflow datasets...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    roboflow_datasets = [
        # (workspace, project, version, folder_name, notes)
        (
            "acne-detection-fewiv", "acne-detection-jd2lk", 1,
            "rf_acne_detection",
            "Acne detection — acne lesion bounding boxes"
        ),
        (
            "skin-problems-detection-jp4jv-nxtdz", "skin-problems-detection-jp4jv-nxtdz", 1,
            "rf_skin_problems",
            "Skin problems — melasma, acne, wrinkles, pores"
        ),
        (
            "face-skin-type", "face-skin-type", 1,
            "rf_skin_type",
            "Face skin type — oily, dry, normal, combination"
        ),
        (
            "acne-segmentation", "acne-segmentation", 1,
            "rf_acne_seg",
            "Acne segmentation masks — higher precision than boxes"
        ),
        (
            "eye-disease-detection-sycgd", "eye-disease", 1,
            "rf_eye_disease",
            "Eye redness, yellow sclera, bloodshot detection"
        ),
        ("skin-redness-detection", "skin-redness", 1, "rf_skin_redness", "Redness/erythema focused"),
        ("eye-bag-detection-rlkxn", "eye-bag-detection", 1, "rf_eyebag", "Eye bag focused"),
    ]

    for ws, proj, ver, folder, note in roboflow_datasets:
        dest = BASE / folder
        if dest.exists() and any(dest.rglob("*.jpg")):
            print(f"  SKIP (already exists): {folder}")
            continue
        print(f"\n  Downloading: {folder}")
        print(f"  Note       : {note}")
        try:
            project = rf.workspace(ws).project(proj)
            ds = project.version(ver).download("yolov8", location=str(dest))
            imgs = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
            print(f"  ✅ Downloaded: {len(imgs):,} images")
        except Exception as e:
            print(f"  ❌ Failed (check workspace/project slug): {e}")
            print(f"     → Manually search '{proj}' on roboflow.com/universe")

except Exception as e:
    print(f"  ❌ Roboflow error: {e}")

# ============================================================
# SECTION C — HUGGING FACE (free, no auth needed)
# ============================================================
print("\n[C] Hugging Face datasets...")
try:
    from datasets import load_dataset

    hf_datasets = [
        # (hf_path, local_folder, split, notes)
        (
            "marmal88/skin_cancer",
            "hf_skin_cancer",
            "train",
            "Skin cancer images — melanoma, benign. Good for dark_spot class."
        ),
        (
            "SkinCancerMNIST/HAM10000",
            "hf_ham10000",
            "train",
            "HAM10000 on HuggingFace — 10k dermatoscopy images."
        ),
    ]

    for hf_path, folder, split, note in hf_datasets:
        dest = BASE / folder
        if dest.exists():
            print(f"  SKIP (already exists): {folder}")
            continue
        print(f"\n  Loading: {folder}")
        print(f"  Note  : {note}")
        try:
            ds = load_dataset(hf_path, split=split, trust_remote_code=True)
            ds.save_to_disk(str(dest))
            print(f"  ✅ Saved: {len(ds):,} examples")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

except ImportError:
    print("  ⚠ datasets not installed. Run: pip install datasets")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
all_imgs = list(BASE.rglob("*.jpg")) + list(BASE.rglob("*.png"))
print(f"  Total new images downloaded: {len(all_imgs):,}")
print(f"  Saved to: {BASE.resolve()}")
print("""
NEXT STEPS:
  1. Run step2_smart_merge.py again — it will pick up extra_datasets/ too.
     (Add "extra_datasets/rf_acne_detection" etc. to SOURCE_FOLDERS)

  2. For HAM10000 / DermNet / ISIC:
     These are classification datasets (no bounding boxes).
     Use them for:
       (a) Pre-training / fine-tuning the DINOv2 backbone on skin imagery
       (b) As a source for the severity regression head (image-level labels)
     NOT for detection head training.

  3. The Roboflow sets (rf_*) come with YOLO bounding boxes — 
     add them to step2_smart_merge.py's SOURCE_FOLDERS with correct 
     MANUAL_CLASS_OVERRIDES entries.
""")
