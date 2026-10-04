"""
FaceFuel v4 — Complete Dataset Expansion Downloader
=====================================================
Downloads all new datasets for expanding FaceFuel from:
  Face: 11 classes → 20 classes
  Tongue: 12 classes → 18 classes
  Eye: 3 classes → 8 classes

Total new datasets: 25+
Run: python expand_datasets.py
Run: python expand_datasets.py --face-only
Run: python expand_datasets.py --tongue-only
Run: python expand_datasets.py --eye-only

Setup:
  pip install kaggle roboflow requests
  Set KAGGLE credentials: kaggle.json in ~/.kaggle/
  Set ROBOFLOW_API_KEY environment variable (free at roboflow.com)
"""

import os, sys, argparse, zipfile, shutil, json
from pathlib import Path

ROBOFLOW_API_KEY = __import__("os").environ.get("ROBOFLOW_API_KEY", "")  # set in your environment; never hardcode
BASE = Path(".")
FACE_DIR   = BASE / "facefuel_datasets_v4"
TONGUE_DIR = BASE / "tongue_datasets_v4"
EYE_DIR    = BASE / "eye_datasets_v4"

for d in [FACE_DIR, TONGUE_DIR, EYE_DIR]:
    d.mkdir(exist_ok=True)


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def dl_kaggle(slug, out_dir, desc):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Kaggle] {desc}")
    try:
        import kaggle
        kaggle.api.dataset_download_files(slug, path=str(out_dir), unzip=True)
        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        print(f"  ✅ {len(imgs):,} images")
        return True
    except ImportError:
        print("  ❌ pip install kaggle")
    except Exception as e:
        print(f"  ⚠ {e}")
        print(f"  Manual: https://www.kaggle.com/datasets/{slug}")
    return False


def dl_rf(workspace, project, version, out_dir, desc, fmt="yolov8"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Roboflow] {desc}")
    if not ROBOFLOW_API_KEY:
        print(f"  ⚠ No API key → https://universe.roboflow.com/{workspace}/{project}")
        return False
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        rf.workspace(workspace).project(project).version(version).download(
            fmt, location=str(out_dir))
        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        print(f"  ✅ {len(imgs):,} images")
        return True
    except ImportError:
        print("  ❌ pip install roboflow")
    except Exception as e:
        print(f"  ⚠ {e}")
        print(f"  Manual: https://universe.roboflow.com/{workspace}/{project}")
    return False


def dl_http(url, out_path, desc):
    import requests
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[HTTP] {desc}")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
        if str(out_path).endswith(".zip"):
            extract_dir = out_path.parent / out_path.stem
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(out_path) as z:
                z.extractall(extract_dir)
        print(f"  ✅ Saved: {out_path}")
        return True
    except Exception as e:
        print(f"  ⚠ {e}")
    return False


# ═══════════════════════════════════════════════════════════════
# FACE DATASETS
# ═══════════════════════════════════════════════════════════════

def download_face():
    sep("FACE EXPANSION — 11 → 20 classes")

    # ── Existing classes + more images ──────────────────────────

    # ISIC 2019 — Large dermatology dataset (acne, melanoma, etc.)
    dl_kaggle("kmader/skin-lesion-analysis-toward-melanoma-detection",
              FACE_DIR/"isic_skin_lesions",
              "ISIC skin lesions — 25,000+ dermoscopy images (acne, melanoma, nevi)")

    # HAM10000 — 10,000 dermatoscopic images, 7 classes
    dl_kaggle("kmader/isic-2018-task-3",
              FACE_DIR/"ham10000",
              "HAM10000 — 10,015 images: acne, rosacea, seborrheic keratosis, BCC")

    # General skin disease augmented
    dl_kaggle("syedalinaqvi/augmented-skin-conditions-image-dataset",
              FACE_DIR/"augmented_skin_conditions",
              "Augmented skin conditions — 9 conditions, 4,000+ images")

    # DermNet-style face skin diseases
    dl_kaggle("shubhamgoel27/dermnet",
              FACE_DIR/"dermnet",
              "DermNet — 23 skin disease categories, 19,500 images (massive)")

    # Skin defects (acne, redness, eye bags)
    dl_kaggle("trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes",
              FACE_DIR/"skin_defects_training",
              "Skin defects: acne, redness, eye bags — YOLO format")

    # Face skin disease classification
    dl_kaggle("amellia/face-skin-disease",
              FACE_DIR/"face_skin_disease_amellia",
              "Face skin disease: rosacea, vitiligo, eczema — classified images")

    # Skin disease image dataset (comprehensive)
    dl_kaggle("ismailpromus/skin-diseases-image-dataset",
              FACE_DIR/"skin_diseases_comprehensive",
              "Skin diseases: 10 classes including psoriasis, vitiligo, eczema")

    # ── New classes for face ─────────────────────────────────────

    # Rosacea — omega-3 deficiency, gut dysbiosis indicator
    dl_rf("skin-diseases-jzde4", "face-skin-disease", 8,
          FACE_DIR/"rf_face_skin_disease_v8",
          "Face skin disease v8 — rosacea, acne, pimples YOLO annotated")

    # Facial skin diseases (rosacea, eczema, herpes)
    dl_rf("facial-skin-dataset", "face-skin-diseases", 1,
          FACE_DIR/"rf_facial_skin_diseases",
          "Facial skin diseases: panu, rosacea, eksim, herpes")

    # Angular cheilitis / perioral — B2, iron, B12 indicators
    dl_rf("skin-diseases-jzde4", "skin-diseases-i30ay", 1,
          FACE_DIR/"rf_skin_diseases_general",
          "Skin diseases general — includes perioral dermatitis")

    # Vitiligo — autoimmune, B12, copper
    dl_kaggle("hafiznouman786/vitiligo-classified-dataset",
              FACE_DIR/"vitiligo",
              "Vitiligo dataset — autoimmune / B12 / copper deficiency indicator")

    # Eczema / atopic dermatitis — omega-3, zinc, vitamin D
    dl_kaggle("anshtanwar/skin-diseases-classification",
              FACE_DIR/"skin_diseases_classification",
              "Skin diseases classification — eczema, psoriasis, rosacea, vitiligo")

    # Lupus butterfly rash — autoimmune indicator
    dl_rf("lupus-rash", "lupus-butterfly-rash", 1,
          FACE_DIR/"rf_lupus_rash",
          "Lupus butterfly rash — autoimmune indicator")

    # Skin problems detection (enlarged pores, oily skin, freckles)
    dl_rf("asd-f39jz", "skin-problems-detection", 8,
          FACE_DIR/"rf_skin_problems_v8",
          "Skin problems v8 — enlarged pores, oily skin, papule, pustule")

    # Perifollicular hemorrhage (scurvy / vitamin C)
    # Use DermNet scraping approach + ISIC
    dl_kaggle("subirbiswas19/skin-disease-dataset",
              FACE_DIR/"skin_disease_multiclass",
              "Skin disease multiclass — chickenpox, measles, rash patterns")

    print(f"\n  Face datasets saved to: {FACE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
# TONGUE DATASETS
# ═══════════════════════════════════════════════════════════════

def download_tongue():
    sep("TONGUE EXPANSION — 12 → 18 classes")

    # ── Oral ulcer — B12, iron, immune ──────────────────────────
    dl_kaggle("bavithravairam/oral-ulcer",
              TONGUE_DIR/"oral_ulcer",
              "Oral ulcer dataset — B12/iron/immune deficiency indicator")

    # Oral diseases comprehensive
    dl_kaggle("salmansajid05/oral-diseases",
              TONGUE_DIR/"oral_diseases_salmansajid",
              "Oral diseases — ulcer, thrush, lichen planus, leukoplakia")

    # Oral cancer / lesions (includes lichen planus, leukoplakia)
    dl_kaggle("zaidpathan/oral-cancer-lips-and-tongue-images",
              TONGUE_DIR/"oral_cancer_lips_tongue",
              "Oral cancer lips and tongue — lichen planus, leukoplakia, ulcers")

    # Mouth disease classification
    dl_kaggle("shivam17/mouth-disease-classification",
              TONGUE_DIR/"mouth_disease_classification",
              "Mouth disease: gingivitis, thrush, ulcers — immune indicators")

    # Dental/oral health dataset (includes tongue conditions)
    dl_kaggle("truthisneverlinear/tooth-dataset",
              TONGUE_DIR/"tooth_dataset",
              "Tooth and oral health dataset — includes gum/tongue observations")

    # ── Roboflow tongue datasets ─────────────────────────────────

    # Tongue condition detection
    dl_rf("tongue-detection", "tongue-condition", 1,
          TONGUE_DIR/"rf_tongue_condition",
          "Tongue condition — fissured, geographic, coated")

    # Oral disease detection YOLO
    dl_rf("oral-disease-detection", "oral-disease", 2,
          TONGUE_DIR/"rf_oral_disease",
          "Oral disease YOLO — ulcer, thrush, lichen, leukoplakia")

    # Mouth/tongue analysis
    dl_rf("mouth-analysis", "mouth-tongue", 1,
          TONGUE_DIR/"rf_mouth_tongue",
          "Mouth tongue analysis dataset")

    # Stomatitis / angular cheilitis (also visible near mouth)
    dl_rf("skin-conditions", "angular-cheilitis", 1,
          TONGUE_DIR/"rf_angular_cheilitis",
          "Angular cheilitis — B2/iron/B12 deficiency indicator")

    # ── PLoS ONE tongue dataset (Chang et al. 2024) ──────────────
    # 764 annotated tongue images with fissures, tooth marks, thick/yellow coating
    # Published with paper: doi.org/10.1371/journal.pone.0296070
    print("\n[Manual] PLoS ONE tongue feature dataset (Chang et al. 2024)")
    print("  Paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296070")
    print("  Contact authors for dataset access (clinical tongue images)")
    print("  Alternative: Request via Supplementary Materials section")

    # ── Ataturk University tongue lesion dataset ─────────────────
    print("\n[Manual] Ataturk University tongue lesion dataset (2024, BMC Medical Imaging)")
    print("  Paper: https://doi.org/10.1186/s12880-024-01234-3")
    print("  Dataset: 623 patients, 5 classes (fissured, geographic, coated,")
    print("           median rhomboid glossitis, normal)")
    print("  Contact: omiloglu@hotmail.com to request dataset access")

    print(f"\n  Tongue datasets saved to: {TONGUE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
# EYE DATASETS
# ═══════════════════════════════════════════════════════════════

def download_eye():
    sep("EYE EXPANSION — 3 → 8 classes")

    # ── Pterygium — vitamin A, UV damage ────────────────────────
    dl_kaggle("andrewmvd/ocular-disease-recognition-odir5k",
              EYE_DIR/"odir5k",
              "ODIR-5K — 5,000 binocular images: normal, diabetes, glaucoma, cataract, AMD, hypertension, myopia")

    # Eye disease image dataset (Bangladesh hospital — 5,335 images)
    # Includes: Pterygium, Retinal Detachment, Glaucoma, Myopia
    print("\n[Mendeley] Eye Disease Image Dataset (Rashid 2024)")
    print("  URL: https://doi.org/10.17632/s9bfhswzjb.1")
    print("  Classes: Pterygium, Glaucoma, DR, Retinal Detachment, Myopia, Healthy")
    print("  Download from Mendeley Data — click Download All")
    print("  Save as: eye_datasets_v4/mendeley_eye_disease/")

    # Uveitis, conjunctivitis, cataract, eyelid drooping dataset
    print("\n[Mendeley] Uveitis/Conjunctivitis/Cataract/Eyelid dataset")
    print("  URL: https://data.mendeley.com/datasets/n9zp473wfw/1")
    print("  Classes: Uveitis, Conjunctivitis, Cataract, Eyelid Drooping, Normal")
    print("  Save as: eye_datasets_v4/mendeley_uveitis_conjunctivitis/")

    # Kaggle eye disease dataset
    dl_kaggle("kondwani/eye-disease-dataset",
              EYE_DIR/"kaggle_eye_disease",
              "Eye disease dataset — cataracts, conjunctivitis, glaucoma, normal")

    # Ocular diseases Roboflow
    dl_rf("projects-dmmza", "ocular-diseases", 1,
          EYE_DIR/"rf_ocular_diseases",
          "Ocular diseases YOLO — detection format")

    # Eye disease YOLO (Roboflow)
    dl_rf("teguh-rijanandi", "eye-disease", 1,
          EYE_DIR/"rf_eye_disease_teguh",
          "Eye disease YOLO detection dataset")

    # Dry eye / conjunctivitis
    dl_rf("eye-condition-detection", "dry-eye", 1,
          EYE_DIR/"rf_dry_eye",
          "Dry eye detection — omega-3/vitamin A deficiency")

    # Pinguecula (yellowish scleral deposits)
    dl_rf("eye-conditions", "pinguecula-pterygium", 1,
          EYE_DIR/"rf_pinguecula_pterygium",
          "Pinguecula and pterygium — vitamin A / sun exposure indicator")

    # Conjunctivitis (immune / vitamin A / C)
    dl_kaggle("datasnaek/eye-type",
              EYE_DIR/"eye_type_conjunctivitis",
              "Eye type dataset — conjunctivitis, normal, cataract")

    # Additional sclera / eye conditions
    dl_rf("eye-v0bqz", "eye-diseases-2", 1,
          EYE_DIR/"rf_eye_diseases_2",
          "Eye diseases v2 — multiple conditions YOLO")

    print(f"\n  Eye datasets saved to: {EYE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{'='*65}")
    print("EXPANSION COMPLETE — Next steps:")
    print(f"{'='*65}")

    print("""
FACE (target: 20 classes):
  Existing 11 + NEW:
    rosacea             → omega-3 / gut dysbiosis / inflammation
    angular_cheilitis   → B2 (riboflavin) / iron / B12
    perioral_dermatitis → gut dysbiosis / stress / hormones
    vitiligo            → autoimmune / B12 / copper deficiency
    eczema_patch        → omega-3 / zinc / vitamin D
    perifollicular_hem  → vitamin C deficiency (scurvy)
    butterfly_rash      → lupus / autoimmune indicator
    enlarged_pores      → vitamin A deficiency
    oily_skin           → hormonal imbalance / zinc

TONGUE (target: 18 classes):
  Existing 12 + NEW:
    oral_ulcer          → B12 / iron / immune deficiency
    angular_stomatitis  → B2 / iron / B12
    median_rhomboid_gl  → B12 / iron / immune
    lichen_planus       → immune / stress / hepatitis C
    leukoplakia         → immune / tobacco / vitamin A
    hairy_leukoplakia   → immune deficiency

EYE (target: 8 classes):
  Existing 3 + NEW:
    pterygium           → vitamin A / UV / dehydration
    pinguecula          → vitamin A / dehydration
    conjunctivitis      → immune / vitamin A / C
    dry_eye             → omega-3 / vitamin A
    eyelid_drooping     → thyroid / neuromuscular

TOTAL DEFICIENCY DIMENSIONS: 16 → 22

NEXT STEP:
  python face_merge_v2.py    ← merge all face data
  python tongue_merge_v2.py  ← merge all tongue data
  python eye_merge_v2.py     ← merge all eye data
  python retrain_all.py      ← retrain all 3 YOLO11m models
""")

    print("MANUAL DOWNLOADS REQUIRED:")
    print("  1. Mendeley eye disease: https://doi.org/10.17632/s9bfhswzjb.1")
    print("  2. Mendeley uveitis: https://data.mendeley.com/datasets/n9zp473wfw/1")
    print("  3. Ataturk tongue: email omiloglu@hotmail.com")
    print("  4. PLoS ONE tongue: contact Chang et al. (2024)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-only",   action="store_true")
    parser.add_argument("--tongue-only", action="store_true")
    parser.add_argument("--eye-only",    action="store_true")
    args = parser.parse_args()

    do_all = not (args.face_only or args.tongue_only or args.eye_only)

    print("="*65)
    print("FaceFuel v4 — Dataset Expansion")
    print("="*65)
    if ROBOFLOW_API_KEY:
        print(f"  Roboflow API key: SET")
    else:
        print("  Roboflow API key: NOT SET")
        print("  → set ROBOFLOW_API_KEY=your_key or Roboflow datasets will skip")
    print()

    if do_all or args.face_only:   download_face()
    if do_all or args.tongue_only: download_tongue()
    if do_all or args.eye_only:    download_eye()

    print_summary()