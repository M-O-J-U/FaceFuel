# CLAUDE.md — FaceFuel Project Guide

This file is project memory for Claude Code. Read it fully before making changes.
It describes what FaceFuel is, its current technical state, known issues, and the
work that still needs doing. Treat file paths and numbers here as the
source of truth over any older comments found inside individual scripts.

> **State as of 2026-10-04 (v4.1 audit):** the inference layer was rebuilt as the
> `facefuel/` package served by `server.py` (the old `server_v4.py`,
> `step10_inference.py`, `Phase7_tongue_inference.py`, `eye_inference.py` are archived in
> `legacy/`). Known Issues 1–5 below are resolved or mitigated — see §8 for what changed
> and why, and `docs/RESEARCH_NOTES.md` §1 for the research-relevant findings.

---

## 1. What This Project Is

FaceFuel is an independent research project building a **tri-modal computer
vision system** (face + tongue + eye) that screens for nutritional
deficiencies and lifestyle health signals from ordinary smartphone photos —
no blood tests required. It is explicitly a **wellness-awareness tool, not a
medical diagnostic device**, and every user-facing output must carry that
disclaimer.

**Author:** Abdul Moiz Muhammad ("MOJU") — independent researcher,
COMSATS University Islamabad, Wah Cantt, Pakistan.
ORCID: 0009-0006-2795-5271 · GitHub: github.com/M-O-J-U

**Published research (Zenodo, all CC BY-NC-ND 4.0):**
| Paper | Title | DOI | Status |
|---|---|---|---|
| 1 | FaceFuel: Face Pipeline | 10.5281/zenodo.19394708 | Live; desk-rejected by IEEE JBHI |
| 2 | Bimodal Face+Tongue Fusion | 10.5281/zenodo.19411317 | Live; desk-rejected by IEEE JBHI |
| 3 | TriModal Face+Tongue+Eye | 10.5281/zenodo.19468059 | Live; not yet submitted to a journal |

**JBHI desk-rejection reasons (Papers 1 & 2):** biomedical/informatics
contribution not clear enough, methods rigor below threshold, and —
the core issue — insufficient clinical evaluation/evidence (no blood-panel
ground truth, hand-built Bayesian priors). This is the central unsolved
problem of the whole project. Scaling datasets in v4 did **not** resolve it;
only real clinically-labeled data will.

---

## 2. System Architecture

Each modality (face, tongue, eye) runs the same five-stage pattern:

1. **Localization** — MediaPipe face alignment (face), or crop derived from
   the aligned face / a dedicated photo (tongue, eye).
2. **Detection** — a YOLO11m object detector finds visual features
   (e.g. dark circles, tongue coating type, scleral icterus).
3. **Embedding** — DINOv2 (ViT-S/14) extracts a feature vector per
   anatomical sub-region, concatenated into one composite vector.
4. **Severity regression** — a per-class MLP with Monte Carlo Dropout
   outputs a severity score and an uncertainty estimate per detected
   feature.
5. **Fusion** — a weighted product-of-experts Bayesian engine combines
   face, tongue, and eye posteriors into one probability distribution
   over deficiency categories. Weights: face = 0.40, tongue = 0.35,
   eye = 0.25. A modality with no evidence for a category contributes
   zero, not a guess — categories are expanded to the full dimension
   space before fusion, not assumed present.

Served via `server.py` (FastAPI, CUDA, tested on an RTX 4070 Super 12GB):
- `POST /analyze` — one selfie → face + eye (no tongue photo required)
- `POST /analyze/tongue` — tongue photo only
- `POST /analyze/combined` — selfie + tongue → all three modalities
- `GET /health`, `GET /api/info`, `GET /` (web frontend, `static/index.html`)

v4.1 detail that differs from the description above: per-modality scoring is no
longer a hand-set CPT Bayesian engine. Each deficiency's evidence is the strongest
severity among DETECTED features mapping to it (`facefuel/schema.py` `*_FEAT_DEF`),
normalised per modality, then fused with the same PoE weights. The severity MLP
grades detector-confirmed features only (`facefuel/evidence.py: REQUIRE_DETECTOR`).

---

## 3. Current Model State (v4)

Weight locations are resolved in ONE place, `facefuel/paths.py` (env var →
`weights/<canonical name>` → the legacy paths listed below).
`python scripts/collect_weights.py` copies them into `weights/`.

### Face — YOLO11m
- **Weights:** `runs\detect\runs\detect\runs\face\face_yolo11m_v4\weights\best.pt`
  (note the doubled `runs/detect/` prefix — a YOLO `project`/`name` config
  bug from training; see Known Issues)
- mAP50 = 0.560 overall. This number is misleading on its own — several
  classes have very small validation splits (e.g. acne had 2 val images),
  dragging the average down while visually strong classes like eczema
  (0.973), rosacea (0.817), dark_circle (0.773) are genuinely solid.
- **Classes with real training data (10 configured, 6 actually trained):**
  dark_circle, wrinkle, redness, dark_spot, rosacea, eczema
- **Classes with ZERO recovered training images** (present in the schema,
  not functional): acne, blackhead, vitiligo, butterfly_rash. The merge
  pipeline could not locate usable source images for these despite being
  in several candidate datasets. This is a known, unresolved gap.
- **Severity MLP:** `facefuel_models\face_severity_mlp_v4.pt`, F1 = 0.901
  across the 6 active classes (dark_circle 0.969, dark_spot 0.945,
  wrinkle 0.917, eczema 0.896, rosacea 0.834, redness 0.814).

### Tongue — YOLO11m
- **Weights:** `runs\detect\training_runs\tongue_v4\weights\best.pt`
- mAP50 = 0.871 (best tongue result across all versions).
- **18 classes configured, 14 have data:** tongue_body, white_coating,
  yellow_coating, thick_coating, red_tongue, pale_tongue, fissured,
  geographic, smooth_glossy, crenated, oral_ulcer, lichen_planus,
  leukoplakia, hairy_leukoplakia.
- **Zero-data classes:** no_coating, purple_tongue, angular_stomatitis,
  median_rhomboid_glossitis. The Ataturk University clinical dataset
  (contact: omiloglu@hotmail.com, paper DOI 10.1186/s12880-024-01234-3,
  623 patients) was identified as the best source for
  median_rhomboid_glossitis specifically but was never obtained — still
  worth pursuing.
- **Severity MLP:** `facefuel_models\tongue_severity_mlp_v4.pt`, F1 = 0.761.
  Strong: leukoplakia 0.902, yellow_coating 0.911, oral_ulcer 0.898.
  Weak: lichen_planus 0.250 (only ~82 samples), fissured 0.558.

### Eye — YOLO11m
- **Weights:** `runs\detect\training_runs\eye_v2\weights\best.pt`
- mAP50 = 0.990 (0.993 best epoch per results.csv) — the strongest model in the entire project, across all
  versions and modalities.
- **6 classes, all functional:** conjunctival_pallor, scleral_icterus,
  xanthelasma, pterygium, conjunctivitis, eyelid_drooping.
  (`pinguecula` and `dry_eye` were planned but dropped — zero usable
  training images were ever found; the one large "Mendeley eye disease"
  dataset downloaded for this turned out to be fundus/retinal photography,
  not visible-light selfie images, and was correctly excluded during
  merging except for its small Pterygium subset.)
- **Severity MLP:** `facefuel_models\eye_severity_mlp_v4.pt`, F1 = 0.985,
  every class ≥ 0.943.
- `scleral_icterus` has a three-layer false-positive guard: a raised
  per-class confidence threshold (0.65 vs 0.30 default), a LAB
  colour gate (mean B channel > 145 and L channel > 140, i.e. genuinely
  yellow and not just dark/shadowed), and a rule that the severity MLP
  alone can never report it — YOLO must confirm. This was necessary
  because early eye models fired scleral_icterus on every image
  including clearly healthy white sclera.

### Deficiency framework — 22 dimensions
```
iron_deficiency, b12_deficiency, vitamin_d_deficiency, zinc_deficiency,
omega3_deficiency, vitamin_a_deficiency, vitamin_c_deficiency,
poor_sleep_quality, hormonal_imbalance, dehydration, high_stress,
liver_stress, gut_dysbiosis, hypothyroid, folate_deficiency,
cholesterol_imbalance, riboflavin_deficiency, autoimmune_risk,
eye_inflammation, copper_deficiency, skin_inflammation, oral_health_risk
```
Face covers the first 11 (general dimensions); tongue extends coverage
to 16 (adds liver_stress, gut_dysbiosis, hypothyroid, folate_deficiency,
cholesterol_imbalance); eye covers all 22, contributing the only evidence
for several autoimmune/inflammation-adjacent categories. See
`ALL_DEFS`, `FACE_DEFS`, `TONGUE_DEFS`, `EYE_DEFS` and the
`*_FEAT_DEF` mapping dicts in `server_v4.py` for the exact mapping
from detected visual feature → deficiency category.

---

## 4. Known Issues To Be Aware Of / Fix

> Status 2026-10-04: #1 fixed at source (pipeline/04_train_yolo.py passes an ABSOLUTE
> `project`) plus the central resolver; #2 fixed (facefuel/models.py load_severity_mlp —
> the old face/tongue scripts never actually loaded the v4 MLPs); #3 obsolete (wrapper
> archived); #4 the numeric leak lived in v3 server.py, not server_v4.py, but provenance
> was missing — every category now has status/sources/assessed_by (tests/test_fusion.py);
> #5 done (legacy/, local/). Original text kept below for history.

1. **Doubled training output path.** `retrain_all.py` passed a `project`
   value that already contained `runs/detect`, and Ultralytics prepends
   `runs/detect/` again automatically, producing nested paths like
   `runs\detect\runs\detect\runs\face\face_yolo11m_v4\...`. Tongue and eye
   runs used a flatter `training_runs/` project name and only got doubled
   once (`runs\detect\training_runs\...`). Don't "fix" this by moving
   files without also updating every hardcoded path in `server_v4.py`,
   `eye_inference.py`, `step10_inference.py`, `Phase7_tongue_inference.py`.
   A better fix: retrain with a clean, non-nested `project=` value, or
   add a small path-resolution helper that all inference scripts import
   from one place instead of hardcoding strings in five files.

2. **MLP checkpoint format changed in v4.** Old (v3) checkpoints were a
   raw `state_dict`. New (v4) checkpoints are a dict:
   `{"state_dict", "classes", "active_idx", "inactive_idx", "in_dim", "n_active"}`
   — because several classes had zero training data and had to be
   excluded from the trained head count. Any inference code must branch
   on `isinstance(ckpt, dict) and "state_dict" in ckpt` and must map
   active-head outputs back to the full class list using `active_idx`
   (see `eye_inference.py`'s `run_eye_inference` for the reference
   implementation — face and tongue inference should follow the same
   pattern and should be double-checked, since they were patched via
   regex rather than rewritten clean).

3. **`tongue_inference.py` is a thin wrapper**, not the real
   implementation: `from Phase7_tongue_inference import *`. It was
   created via a PowerShell redirect that initially wrote UTF-16 with
   null bytes (Python can't parse that) — already fixed, but confirm
   it's UTF-8 if it's ever regenerated. The actual tongue logic lives in
   `Phase7_tongue_inference.py`. Consider renaming this properly during
   cleanup rather than keeping the historical `PhaseN_` naming.

4. **Unverified: tongue-exclusive source labeling.** Deficiencies that
   only tongue data can reveal (liver_stress, gut_dysbiosis, hypothyroid,
   folate_deficiency) were observed in `/analyze` (face+eye only, no
   tongue photo) outputs showing a non-trivial probability and a
   `source: "tongue"` tag — which is wrong when no tongue was analyzed;
   it's a leaking non-zero prior. This needs to be confirmed as fixed
   or not in the current `server_v4.py` `build_response()` /
   `fuse_posteriors()` logic, and fixed if not: when a modality's
   posterior vector is all zero, its exclusive categories should not
   appear in the top insights or should be explicitly marked as
   "not assessed" rather than carrying a stray probability.

5. **Legacy script sprawl.** The repo root currently mixes several
   generations of the pipeline: `step1_`…`step10_` (face v1),
   `Phase1_`…`Phase7_` (tongue v1, different numbering scheme than the
   face steps — confusing), `1_eye_dataset.py`…`6_eye_severity.py`
   (eye v1), plus v2 merge scripts, v3/v4 expansion scripts,
   `ablation_fusion.py`, `ablation_study.py`, `tongue_fix_v4.py`,
   `dataset.py`, and multiple `.bak` backup files created by earlier
   automated patches. Large binaries also sit in the project root:
   `FaceFuel promo.mp4` (~140MB), `yolo11m.pt`, `yolo26n.pt`,
   `yolov8m.pt` (pretrained base weights, re-downloadable), a handful of
   personal test photos (`Myself.jpeg`, `Myself2.jpeg`, etc.), and
   `facefuel_structure.txt` (a multi-MB directory scan dump). None of
   this should ship to a public GitHub repo as-is.

---

## 5. Current Directory Map (v4.1)

```
server.py                       <- the server (FastAPI API + frontend). Run: python server.py
facefuel/                       <- runtime package; the ONLY code server.py imports
  paths.py                        every model-file path (env-overridable)
  schema.py                       class lists, *_INACTIVE, ALL_DEFS, *_FEAT_DEF, FOOD_RECS, ADVICE
  models.py                       shared DINOv2, embed_regions (train/infer parity), v4 MLP loader, YOLO
  face.py / eye.py / tongue.py    modality pipelines
  evidence.py / fusion.py         detector-confirmed evidence, PoE fusion, provenance
static/index.html               <- web frontend (vanilla HTML/CSS/JS, no build step)
pipeline/                       <- training, runnable from anywhere, in order:
  01_download_datasets.py (was expand_datasets_v3)   02_diagnose_datasets.py
  03_merge_{face,tongue,eye}.py (were *_merge_v2)    04_train_yolo.py (was retrain_all)
  05_extract_features.py (replaces 5_*_features_v4)  06_train_severity_mlp.py (was 6_severity_mlp_v4)
scripts/                        <- collect_weights.py, domain_shift_probe.py, scan_project.py
tests/                          <- pytest: fusion/evidence (no GPU) + API smoke (needs weights)
docs/                           <- PROJECT_SUMMARY, RESEARCH_NOTES, DEPLOYMENT, PUBLISH_CHECKLIST,
                                   domain_shift_probe.md, figures/
weights/                        <- canonical copies of the 7 inference files (git-ignored)
legacy/                         <- v1 face (step1-10), v1 tongue (Phase1-7), v1 eye (1_-6_),
                                   v3 server, ablations, original v4 server + patch scripts
local/                          <- git-ignored: personal/test photos, promo video, base YOLO
                                   weights, .bak files, structure dump, old frontend copies
paper_results/                  <- .tex/.bib/figures per paper; generators/write_paper*.py

datasets (unchanged, git-ignored): facefuel_datasets*/ tongue_datasets*/ eye_datasets*/
merged YOLO sets: facefuel_{face,tongue,eye}_v4/   features: facefuel_features/
MLPs: facefuel_models/   YOLO runs: runs/ (legacy doubled paths), training_runs/ (new runs)
```

Environment: Python 3.14 at C:\Python314 has the full stack (torch 2.11 cu128,
ultralytics 8.4.60, transformers 5.12, mediapipe 0.10.35). Set
PYTHONIOENCODING=utf-8 when piping output (emoji prints crash on cp1252).

## 6. Work To Do

Work through these in whatever order makes sense technically; none of
them are time-boxed and none should be treated as a fixed schedule.

### A. Codebase cleanup and consolidation
- Fix the doubled `runs/detect/...` path problem at the source
  (retraining config) or add one central path-resolution module that
  every inference script imports, so weight paths are declared once.
- Investigate and fix the tongue-exclusive source-labeling issue
  described above (#4 in Known Issues) if it's still present.
- Audit `step10_inference.py` and `Phase7_tongue_inference.py` for the
  same MLP-checkpoint-dict handling that `eye_inference.py` already has
  cleanly implemented — make sure the active/inactive class remapping is
  correct in both, not just patched to not crash.
- Decide what to do with the zero-data classes (acne, blackhead,
  vitiligo, butterfly_rash for face; no_coating, purple_tongue,
  angular_stomatitis, median_rhomboid for tongue). Either find workable
  datasets for them, or cleanly remove them from the active class lists
  and the deficiency-mapping dicts so the system doesn't advertise
  capabilities it doesn't have.
- Identify and archive or delete superseded/legacy files: the v1
  `step1`–`step9` face scripts, the v1 `Phase1`–`Phase6` tongue scripts,
  the v1 eye scripts (`1_eye_dataset.py` through `6_eye_severity.py` —
  note `7_` and `8_` numbers are reused inconsistently, check carefully
  before deleting), `ablation_fusion.py`, `ablation_study.py`,
  `tongue_fix_v4.py`, `dataset.py`, and any `.py.bak` files, *only after*
  confirming nothing in the current `server_v4.py` import chain depends
  on them. Keep whatever is still load-bearing.
- Rationalize naming: the mix of `step10_`, `Phase7_`, `5_`, `6_`,
  `_v2`, `_v3`, `_v4` suffixes across different modalities is confusing.
  Propose and apply a consistent naming scheme.

### B. Frontend
- Build a modern, interactive, high-end, top-notch, smooth transitions, animations, top affects like transparency, mobile-friendly web frontend to replace
  the placeholder `static/` directory. It should call `/analyze`,
  `/analyze/tongue`, and `/analyze/combined`, and present: the returned
  aligned face / tongue crop thumbnails, per-modality detected features
  with severity/confidence, the ranked deficiency table with confidence
  bands and food/advice recommendations, a clear "not medical advice"
  disclaimer, and graceful handling of the `no_face_detected` response.
- Make sure CORS, host/port, and static file serving in `server_v4.py`
  are consistent with whatever frontend build/deploy approach is chosen.

### C. Strengthening the research itself
- Research (web search is authorized) additional publicly available,
  appropriately licensed datasets that could fill the zero-data classes
  listed above, prioritizing anything with real clinical labels over
  more classification-folder scraped images.
- Research better-calibrated or learned fusion approaches as an
  alternative/supplement to the current hand-set product-of-experts
  weights (0.40/0.35/0.25) — this is one of JBHI's named objections
  ("methods rigor... not learned").
- Research recent comparable work in non-invasive multimodal visual
  health screening to cite, compare against, and position the next
  paper submission relative to.
- Research realistic paths to closing the clinical-evidence gap short
  of a full prospective clinical trial — e.g. existing public datasets
  that pair visual photos with lab-confirmed diagnoses (beyond the
  Eyes-Defy-Anemia dataset already used), weak-supervision or synthetic
  augmentation techniques that reviewers would find credible, or a
  minimal viable validation protocol that could be run with volunteers.
- Keep all of this consistent with the project's actual scope — do not
  overstate diagnostic claims anywhere, including in any new text
  written for papers or the README.

### D. Deployment prep
- Prepare `server_v4.py` and its dependencies for deployment to a
  public GPU-backed host (containerize if useful). Document the exact
  steps needed (env vars, model weight locations, GPU requirements).

### E. Documentation for portfolio and GitHub
- Write a comprehensive project-summary markdown file covering the full
  story across all three papers/versions: motivation, architecture,
  dataset evolution, results at each stage, the JBHI rejection and what
  it means, and the current strategic plan — written for inclusion in a
  personal portfolio site.
- Write an attractive, well-structured `README.md` for the public GitHub
  repo: overview, architecture description, results tables (the mAP/F1
  numbers in this file), setup/run instructions, API examples for the
  three endpoints, citation block for the three Zenodo DOIs, license,
  and a prominent "this is a research prototype, not medical advice"
  disclaimer.
- Produce a clear publish/do-not-publish file list for the GitHub
  release: a `.gitignore` plus an explicit checklist of what ships
  (source code, configs, `requirements.txt`, the two paper figures,
  a small number of illustrative sample images if needed, the README
  and project-summary) versus what must stay local (full raw datasets,
  the pretrained base weight files which are trivially re-downloadable,
  the promo video if it's large, personal test photos, the full
  `facefuel_structure.txt` dump, and any `.bak` files).

---

## 7. Ground Rules

- Treat the current mAP/F1 numbers in this file as accurate as of the
  last training run. If you retrain anything, update this file with
  the new numbers and note what changed.
- Never remove or weaken the "not medical advice" disclaimer anywhere
  it currently appears, and include it in any new user-facing surface
  (frontend, README) you add.
- Don't silently drop a class from the schema. Class names now live in ONE
  place, `facefuel/schema.py` (`*_CLASSES`, `*_INACTIVE`, `*_FEAT_DEF`), and must
  match the data.yaml files and YOLO heads (the loaders assert this at start-up).
  To disable a class, add it to `*_INACTIVE`; don't delete it from the list, or
  YOLO class indices shift. A partial rename causes exactly the kind of silent
  bug this project has already hit twice.
- When in doubt about whether a file is still load-bearing, check what
  `server.py` imports (directly or transitively). That is only the `facefuel/`
  package; nothing in `legacy/` or `pipeline/` is imported by the server.
- Run `python -m pytest tests -q` after any change to facefuel/ or server.py.

---

## 8. Changelog — v4.1 audit (2026-10-04)

Findings (all verified by running code; details in docs/RESEARCH_NOTES.md §1):
- Before the audit all three endpoints returned HTTP 500. `/analyze` died on numpy
  float32 JSON serialisation; tongue failed because Phase7's MLP path string
  contained a backslash-t (TAB) escape.
- `step10_inference.py` still ran the v2 face YOLO and v2 MLP; Phase7 globbed for the
  v3 tongue YOLO. The regex patch scripts (now legacy/v4_original/) silently failed.
- The eye stage received the LAB-normalised face, but its LAB gate and the v4 training
  data are raw images; it now gets the raw aligned face (512 px).
- The v4 MLPs were trained with no healthy negatives, so they saturate on real selfies
  (dark_circle > 0.35 on 56% of test faces). Mitigation: detector confirmation required;
  MLP-only signals are returned as `unconfirmed_signals`. See scripts/domain_shift_probe.py.
- The tongue detector localises tongue_body in only 22% of val images, so the whole
  photo is analysed when it is not found (matching how MLP features were extracted),
  with a user warning.
- TONGUE_FEAT_DEF mapped geographic to "vitamin_b_deficiency" (not a framework
  dimension, so it was silently dropped); it now maps to b12_deficiency, and
  schema._validate() prevents repeats.
- MLP F1 was selected and reported on the same unseeded split. The split is now seeded;
  a held-out test split is still TODO.

Changes: facefuel/ package; server.py rewrite (same endpoints, unified response schema
with status/sources/assessed_by/evidence per category, EXIF-aware decoding, inference
lock, 503 on missing models, deterministic MC dropout); zero-data classes marked
inactive centrally (kept in the lists for YOLO index alignment, cross-checked against
the checkpoint's inactive_idx); new frontend; pipeline/ renames; requirements, Dockerfile
(not yet built — Docker Desktop was not running), docs. 21 tests pass.

Open decisions for the author: GitHub history cleanup (the public repo contains personal
photos — see docs/PUBLISH_CHECKLIST.md), code licence, weight redistribution, and
retraining with negatives and a held-out test split.

