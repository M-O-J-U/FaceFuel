"""
FaceFuel v4 — Dataset Expansion Downloader v3
===============================================
Dead datasets removed. Replacements verified and patched in.

REMOVED (confirmed dead/unavailable):
  syedalinaqvi/augmented-skin-conditions-image-dataset
  anshtanwar/skin-diseases-classification
  shivam17/mouth-disease-classification
  sujaykapadnis/mouth-disease
  datasnaek/eye-type

REPLACED WITH (all verified live):
  kadhiravanucet/multimodal-augmented-skin-lesion-dataset  (face)
  javedrashid/mouth-and-oral-diseases-mod                  (tongue)
  data.mendeley.com/datasets/mhjyrn35p4/2                  (tongue, manual)
  gunavenkatdoddi/eye-diseases-classification              (eye, already downloaded)

Run:
  set ROBOFLOW_API_KEY=your_key
  python pipeline/01_download_datasets.py
  python pipeline/01_download_datasets.py --face-only
  python pipeline/01_download_datasets.py --tongue-only
  python pipeline/01_download_datasets.py --eye-only
"""
import os as _os, pathlib as _pl; _os.chdir(_pl.Path(__file__).resolve().parent.parent)  # paths below are repo-root relative

import os, sys, argparse, zipfile
from pathlib import Path

RF_KEY     = os.environ.get("ROBOFLOW_API_KEY", "")
BASE       = Path(".")
FACE_DIR   = BASE / "facefuel_datasets_v4"
TONGUE_DIR = BASE / "tongue_datasets_v4"
EYE_DIR    = BASE / "eye_datasets_v4"
for d in [FACE_DIR, TONGUE_DIR, EYE_DIR]:
    d.mkdir(exist_ok=True)

results = {"ok": [], "fail": [], "skip": [], "manual": []}


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}\n")


def dl_kaggle(slug, out_dir, desc):
    out_dir = Path(out_dir)
    # Skip if already downloaded
    if out_dir.exists():
        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        if imgs:
            print(f"[Kaggle] SKIP (exists) — {slug}  ({len(imgs):,} imgs already)\n")
            results["ok"].append(f"[Kaggle] {slug}  ({len(imgs):,} imgs, cached)")
            return True
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Kaggle] {desc}")
    print(f"         {slug}")
    try:
        import kaggle
        kaggle.api.dataset_download_files(slug, path=str(out_dir),
                                          unzip=True, quiet=False)
        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png")) \
             + list(out_dir.rglob("*.jpeg"))
        print(f"  ✅ {len(imgs):,} images → {out_dir}\n")
        results["ok"].append(f"[Kaggle] {slug}  ({len(imgs):,} imgs)")
        return True
    except ImportError:
        print("  ❌ pip install kaggle\n")
    except Exception as e:
        print(f"  ⚠ {str(e)[:120]}\n")
        print(f"  Manual: https://www.kaggle.com/datasets/{slug}\n")
    results["fail"].append(f"[Kaggle] {slug}")
    return False


def dl_rf(workspace, project, desc, fmt="yolov8",
          out_dir=None, versions=(1, 2, 3, 4, 5)):
    if not RF_KEY:
        print(f"[Roboflow] SKIP (no API key) — {workspace}/{project}\n")
        results["skip"].append(f"[RF] {workspace}/{project}")
        return False
    if out_dir is None:
        out_dir = FACE_DIR / f"rf_{project.replace('-','_')}"
    out_dir = Path(out_dir)
    if out_dir.exists():
        imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        if imgs:
            print(f"[RF] SKIP (exists) — {workspace}/{project}  ({len(imgs):,} imgs)\n")
            results["ok"].append(f"[RF] {workspace}/{project}  ({len(imgs):,} imgs, cached)")
            return True
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Roboflow] {desc}")
    print(f"           {workspace}/{project}")
    try:
        from roboflow import Roboflow
        rf   = Roboflow(api_key=RF_KEY)
        proj = rf.workspace(workspace).project(project)
        for v in versions:
            try:
                proj.version(v).download(fmt, location=str(out_dir))
                imgs = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
                print(f"  ✅ version {v} — {len(imgs):,} images\n")
                results["ok"].append(f"[RF] {workspace}/{project} v{v}  ({len(imgs):,} imgs)")
                return True
            except Exception:
                continue
        raise RuntimeError("No version succeeded")
    except ImportError:
        print("  ❌ pip install roboflow\n")
    except Exception as e:
        print(f"  ⚠ {str(e)[:120]}")
        print(f"  Manual: https://universe.roboflow.com/{workspace}/{project}\n")
    results["fail"].append(f"[RF] {workspace}/{project}")
    return False


def manual_note(desc, url, save_to):
    print(f"[Manual] {desc}")
    print(f"  URL:  {url}")
    print(f"  Save: {save_to}\n")
    results["manual"].append(f"{desc} → {save_to}")


# ═══════════════════════════════════════════════════════════════
#  FACE  (11 → 20 classes)
# ═══════════════════════════════════════════════════════════════
def download_face():
    sep("FACE (11 → 20 classes)")

    # ── Already downloaded — skip automatically ───────────────
    dl_kaggle("shubhamgoel27/dermnet",
              FACE_DIR/"dermnet",
              "DermNet 23-class 19,559 imgs")

    dl_kaggle("ismailpromus/skin-diseases-image-dataset",
              FACE_DIR/"skin_diseases_comprehensive",
              "Skin diseases 10-class 27,153 imgs")

    dl_kaggle("amellia/face-skin-disease",
              FACE_DIR/"face_skin_amellia",
              "Face skin: rosacea, vitiligo, eczema")

    dl_kaggle("subirbiswas19/skin-disease-dataset",
              FACE_DIR/"skin_disease_multiclass",
              "Skin disease multiclass 1,157 imgs")

    dl_kaggle("trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes",
              FACE_DIR/"skin_defects_training",
              "Skin defects YOLO: acne, redness, eye bags")

    dl_kaggle("kmader/skin-cancer-mnist-ham10000",
              FACE_DIR/"ham10000",
              "HAM10000 — 10,015 clinical dermoscopy images")

    dl_kaggle("nodoubttome/skin-cancer9-classesisic",
              FACE_DIR/"skin_cancer_isic9",
              "Skin cancer ISIC 9 classes — 2,357 imgs")

    dl_kaggle("rifatahommed/isic-2018-skin-lesion",
              FACE_DIR/"isic_2018",
              "ISIC 2018 skin lesion — 5,188 imgs")

    # ── NEW REPLACEMENT — augmented-skin-conditions replacement ─
    # Replaces: syedalinaqvi (dead) + anshtanwar (dead)
    dl_kaggle("kadhiravanucet/multimodal-augmented-skin-lesion-dataset",
              FACE_DIR/"multimodal_augmented_skin",
              "✅ NEW: Multimodal augmented HAM10000 skin lesions (replaces 2 dead sets)")

    # Vitiligo (may still 403 — keep trying)
    dl_kaggle("hafiznouman786/vitiligo-classified-dataset",
              FACE_DIR/"vitiligo",
              "Vitiligo classified dataset")

    # ── Roboflow face (set ROBOFLOW_API_KEY to enable) ─────────
    dl_rf("queueiq", "yolov8-skin-disease-detection-rrupp",
          "Face: eczema/lupus/vitiligo/tinea/hives YOLO (1,255 imgs)",
          out_dir=FACE_DIR/"rf_queueiq_skin")

    dl_rf("yolov8-azfy6", "facial-skin-diseases-xsubv",
          "Facial skin diseases classification",
          fmt="folder", out_dir=FACE_DIR/"rf_facial_skin_diseases")

    dl_rf("skin-diseases-detection", "vitiligo-deetcction-ejfy4",
          "Vitiligo detection YOLO — 313 imgs",
          out_dir=FACE_DIR/"rf_vitiligo")

    dl_rf("skin-diseases-jzde4", "face-skin-disease",
          "Face skin disease: acne/vitiligo/psoriasis",
          versions=(8,7,6,5,4,3,2,1),
          out_dir=FACE_DIR/"rf_face_skin_disease")

    dl_rf("techno-india-university", "skin-disease-detection-s7zik-qgwhq",
          "Skin disease detection — 1,649 imgs",
          out_dir=FACE_DIR/"rf_techno_india_skin")

    dl_rf("shaheer-hamid-eqqnk", "skin-disease-detection-vecgj-cxqwx",
          "Skin disease detection — 1,431 imgs",
          out_dir=FACE_DIR/"rf_shaheer_skin")

    print(f"  Face → {FACE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
#  TONGUE  (12 → 18 classes)
# ═══════════════════════════════════════════════════════════════
def download_tongue():
    sep("TONGUE (12 → 18 classes)")

    # ── Already downloaded ────────────────────────────────────
    dl_kaggle("bavithravairam/oral-ulcer",
              TONGUE_DIR/"oral_ulcer",
              "Oral ulcer — 4,416 imgs")

    dl_kaggle("salmansajid05/oral-diseases",
              TONGUE_DIR/"oral_diseases",
              "Oral diseases — 13,862 imgs")

    # ── NEW REPLACEMENT for 3 dead tongue sets ────────────────
    # Replaces: shivam17 (dead), sujaykapadnis (dead)
    dl_kaggle("javedrashid/mouth-and-oral-diseases-mod",
              TONGUE_DIR/"mouth_oral_diseases_mod",
              "✅ NEW: Mouth and Oral Diseases MOD (replaces 2 dead sets)")

    # ── Still try (may work from browser cookies) ─────────────
    dl_kaggle("zaidpathan/oral-cancer-lips-and-tongue-images",
              TONGUE_DIR/"oral_cancer_tongue",
              "Oral cancer lips+tongue: leukoplakia, lichen planus")

    dl_kaggle("gpiosenka/oral-diseases-in-humans",
              TONGUE_DIR/"oral_diseases_humans",
              "Oral diseases in humans — comprehensive")

    dl_kaggle("tarandeep97/oral-ulcers-images",
              TONGUE_DIR/"oral_ulcers_v2",
              "Oral ulcers images v2")

    dl_kaggle("truthisneverlinear/tooth-dataset",
              TONGUE_DIR/"tooth_oral",
              "Tooth and oral health dataset")

    # ── Roboflow tongue ───────────────────────────────────────
    dl_rf("tongue-detection", "tongue-condition",
          "Tongue condition: fissured, geographic, coated",
          out_dir=TONGUE_DIR/"rf_tongue_condition")

    dl_rf("oral-disease-detection", "oral-disease",
          "Oral disease YOLO: ulcer, thrush, lichen, leukoplakia",
          out_dir=TONGUE_DIR/"rf_oral_disease")

    dl_rf("skin-conditions", "angular-cheilitis",
          "Angular cheilitis — B2/iron/B12 indicator",
          out_dir=TONGUE_DIR/"rf_angular_cheilitis")

    # ── Mendeley oral images — clinical from Karnataka hospitals ─
    # 323 clinical lesion images (165 benign + 158 malignant)
    # REPLACES: datasnaek/eye-type was wrong category anyway
    manual_note(
        "✅ NEW: Mendeley Oral Images Dataset (Karnataka hospitals, clinical)",
        "https://data.mendeley.com/datasets/mhjyrn35p4/2",
        "tongue_datasets_v4/mendeley_oral_lesions/"
    )
    manual_note(
        "Ataturk University 623-patient tongue dataset",
        "Email: omiloglu@hotmail.com  Paper: https://doi.org/10.1186/s12880-024-01234-3",
        "tongue_datasets_v4/ataturk_tongue_623/"
    )

    print(f"  Tongue → {TONGUE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
#  EYE  (3 → 8 classes)
# ═══════════════════════════════════════════════════════════════
def download_eye():
    sep("EYE (3 → 8 classes)")

    # ── Already downloaded ────────────────────────────────────
    dl_kaggle("andrewmvd/ocular-disease-recognition-odir5k",
              EYE_DIR/"odir5k",
              "ODIR-5K — 14,392 imgs")

    dl_kaggle("kondwani/eye-disease-dataset",
              EYE_DIR/"eye_disease_kondwani",
              "Eye disease: conjunctivitis, cataract, normal — 766 imgs")

    dl_kaggle("gunavenkatdoddi/eye-diseases-classification",
              EYE_DIR/"eye_diseases_classification",
              "Eye diseases classification — 4,217 imgs ✅ already downloaded")

    dl_kaggle("jr2ngb/cataractdataset",
              EYE_DIR/"cataract",
              "Cataract vs normal — 1,202 imgs")

    dl_kaggle("nafin59/monkeypox-skin-lesion-dataset",
              EYE_DIR/"eye_periorbital",
              "Periorbital lesions — 6,027 imgs")

    # ── datasnaek (dead) — NOT replaced here, Mendeley covers it ─
    # The Mendeley oral dataset goes to TONGUE, not eye.
    # Eye-type was: conjunctivitis, cataract, normal
    # gunavenkatdoddi already covers these classes — sufficient.

    # ── mariaherrerot (may still work) ────────────────────────
    dl_kaggle("mariaherrerot/eyediseases",
              EYE_DIR/"eye_diseases_maria",
              "Eye diseases: glaucoma, DR, cataract, normal")

    # ── Roboflow eye ──────────────────────────────────────────
    dl_rf("eyes-fvx7c", "eye-disease-k39qi",
          "Eye disease — confirmed slug 2024",
          out_dir=EYE_DIR/"rf_eye_disease_k39qi")

    dl_rf("king-baldwin-iv", "eye-disease-v8",
          "Eye disease v8 classification",
          fmt="folder", out_dir=EYE_DIR/"rf_eye_disease_v8")

    dl_rf("kittipot-pratchayasuntorn", "dry-eye",
          "Dry eye detection — 139 imgs",
          out_dir=EYE_DIR/"rf_dry_eye")

    # ── Mendeley eye (HIGH PRIORITY — manual) ─────────────────
    manual_note(
        "Mendeley Eye Disease (5,335 imgs) — Pterygium, Glaucoma, DR, Myopia",
        "https://doi.org/10.17632/s9bfhswzjb.1",
        "eye_datasets_v4/mendeley_eye_disease/"
    )
    manual_note(
        "Mendeley Uveitis/Conjunctivitis/Eyelid Drooping",
        "https://data.mendeley.com/datasets/n9zp473wfw/1",
        "eye_datasets_v4/mendeley_uveitis/"
    )

    print(f"  Eye → {EYE_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════
def print_summary():
    sep("DOWNLOAD SUMMARY")
    print(f"  ✅ Succeeded  : {len(results['ok'])}")
    for r in results["ok"]:    print(f"     {r}")
    print(f"\n  ❌ Failed     : {len(results['fail'])}")
    for r in results["fail"]:  print(f"     {r}")
    print(f"\n  ⏭  Skipped RF : {len(results['skip'])}")
    for r in results["skip"]:  print(f"     {r}")
    print(f"\n  📋 Manual req : {len(results['manual'])}")
    for r in results["manual"]:print(f"     {r}")

    print(f"""
{'='*65}
  CHANGES IN v3
{'='*65}
  ❌ REMOVED (confirmed dead):
     syedalinaqvi/augmented-skin-conditions-image-dataset
     anshtanwar/skin-diseases-classification
     shivam17/mouth-disease-classification
     sujaykapadnis/mouth-disease
     datasnaek/eye-type

  ✅ REPLACED WITH (all verified live):
     kadhiravanucet/multimodal-augmented-skin-lesion-dataset → FACE
     javedrashid/mouth-and-oral-diseases-mod                 → TONGUE
     data.mendeley.com/datasets/mhjyrn35p4/2 (manual)       → TONGUE
     gunavenkatdoddi/eye-diseases-classification (already DL) → EYE

  MANUAL STILL NEEDED:
     https://doi.org/10.17632/s9bfhswzjb.1  → eye_datasets_v4/mendeley_eye_disease/
     https://data.mendeley.com/datasets/n9zp473wfw/1 → eye_datasets_v4/mendeley_uveitis/
     https://data.mendeley.com/datasets/mhjyrn35p4/2 → tongue_datasets_v4/mendeley_oral_lesions/

  NEXT:
     python scripts/scan_project.py
     python pipeline/03_merge_face.py
     python pipeline/03_merge_tongue.py
     python pipeline/03_merge_eye.py
     python pipeline/04_train_yolo.py
{'='*65}
""")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-only",   action="store_true")
    parser.add_argument("--tongue-only", action="store_true")
    parser.add_argument("--eye-only",    action="store_true")
    args = parser.parse_args()
    do_all = not (args.face_only or args.tongue_only or args.eye_only)

    print("="*65)
    print("  FaceFuel v4 — Dataset Expansion v3 (verified + replacements)")
    print("="*65)
    print(f"  Roboflow : {'SET ✅' if RF_KEY else 'NOT SET — RF datasets will skip'}")
    try:
        import kaggle; kaggle.api.authenticate()
        print(f"  Kaggle   : SET ✅\n")
    except Exception as e:
        print(f"  Kaggle   : {e}\n")

    if do_all or args.face_only:   download_face()
    if do_all or args.tongue_only: download_tongue()
    if do_all or args.eye_only:    download_eye()
    print_summary()