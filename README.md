# FaceFuel v3 — TriModal Visual Health Intelligence

> **Commercial licensing available.** Contact: mojuaries111@gmail.com

[![DOI Paper 1](https://zenodo.org/badge/DOI/10.5281/zenodo.19394708.svg)](https://doi.org/10.5281/zenodo.19394708)
[![DOI Paper 2](https://zenodo.org/badge/DOI/10.5281/zenodo.19411317.svg)](https://doi.org/10.5281/zenodo.19411317)
[![DOI Paper 3](https://zenodo.org/badge/DOI/10.5281/zenodo.19468059.svg)](https://doi.org/10.5281/zenodo.19468059)

---

**FaceFuel analyzes a selfie photograph to estimate 16 nutritional and lifestyle deficiency probabilities — no blood tests, no clinic visit.**

It is the first system to combine face skin, ocular biomarkers, and tongue surface in a single probabilistic Bayesian framework. Eye analysis runs automatically from the same selfie. Tongue analysis is optional and adds 4 exclusive deficiency dimensions.

Check out the promo @:
https://youtu.be/_RNtt8QNQOs?si=Nus4Sj01vJsvH1kC
---

## Performance

| Modality | Model | mAP@0.5 | Severity F1 |
|---|---|---|---|
| Face skin | YOLO11m | **0.872** | 0.677 |
| Tongue | YOLOv8m v3 | **0.836** | 0.800 |
| Eye | YOLO11m | **0.913** | 0.991 |

**Speed:** 96–116ms (face+eye) · 180–235ms (face+eye+tongue) · RTX 4070 Super

---

## What It Detects

**16 deficiency dimensions across 3 visual modalities:**

| Category | Source |
|---|---|
| Iron deficiency | Face + Eye |
| B12 deficiency | Face + Tongue |
| Vitamin D deficiency | Face |
| Zinc deficiency | Face + Tongue |
| Omega-3 deficiency | Face + Eye |
| Vitamin A / C deficiency | Face |
| Poor sleep quality | Face |
| High stress | Face |
| Hormonal imbalance | Face |
| Dehydration | Face |
| **Liver stress** | Tongue + Eye |
| **Gut dysbiosis** | Tongue only |
| **Hypothyroid tendency** | Tongue only |
| **Folate deficiency** | Tongue only |
| **Cholesterol imbalance** | Eye only |

Bold = modality-exclusive (not detectable from skin alone)

---

## Architecture

```
Selfie ──► MediaPipe alignment ──► Face YOLO11m + Color analysis
       └──► Eye crop (automatic) ──► Eye YOLO11m + LAB color gate

[Optional] Tongue photo ──► Tongue YOLO11m

All three ──► DINOv2 ViT-S/14 feature extraction
         ──► Per-feature severity MLP + Monte Carlo Dropout
         ──► Bayesian inference engine
         ──► 3-way Product-of-Experts fusion
         ──► 16-dimensional deficiency posterior
```

**Key design decisions:**
- Eye analysis from the same selfie — zero extra photos for the user
- LAB color gating on scleral icterus prevents false positives on healthy eyes
- MC Dropout uncertainty flows directly into Bayesian evidence weighting
- Log-space product-of-experts prevents numerical underflow in fusion

---

## Published Research

| Paper | Title | DOI |
|---|---|---|
| 1 | FaceFuel: Multi-Stage Face Pipeline | [10.5281/zenodo.19394708](https://doi.org/10.5281/zenodo.19394708) |
| 2 | Face+Tongue Bimodal Fusion | [10.5281/zenodo.19411317](https://doi.org/10.5281/zenodo.19411317) |
| 3 | TriModal: Face, Tongue and Eye | [10.5281/zenodo.19468059](https://doi.org/10.5281/zenodo.19468059) |

All papers peer-reviewed and published on Zenodo. IEEE JBHI submissions under review.

---

## Installation

```bash
git clone https://github.com/M-O-J-U/FaceFuel.git
cd FaceFuel
pip install -r requirements.txt
```

Download the MediaPipe face landmarker model:
```bash
# Download face_landmarker.task from:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Model weights (`.pt` files) are not included due to file size. Contact for access.

---

## Running the Server

```bash
python server.py
```

Open `http://localhost:8000` — upload a selfie to analyze.

### API Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/analyze` | POST | selfie image | face + eye analysis, 16 deficiencies |
| `/analyze/combined` | POST | selfie + tongue | all 3 modalities fused |
| `/analyze/tongue` | POST | tongue image | tongue-only analysis |
| `/health` | GET | — | server status |

---

## Training Pipeline

```bash
# Face pipeline
python step1_diagnose.py
python step2_smart_merge.py
python step6_train_yolo.py
python step7_preprocessing.py
python step8_dinov2_features.py
python step9_bayesian_engine.py
python step10_inference.py

# Tongue pipeline
python Phase2_merge.py
python Phase2b_tongue_train.py
python Phase3_pseudolabel.py
python Phase4_tongue_retrain.py
python Phase5_tongue_features.py
python Phase6_tongue_severity.py

# Eye pipeline
python 1_eye_dataset.py
python 2_audit_eye.py
python 3_eye_merge.py
python 4_eye_train.py
python 5_eye_features.py
python 6_eye_severity.py
```

---

## Datasets Used

Training data assembled from public sources:

**Face:** Roboflow Universe (acne, dark circles, wrinkles, pigmentation), Kaggle, Open Images
→ 5,721 images · 61,832 bounding boxes · 11 skin feature classes

**Tongue:** Roboflow Universe (tongue segmentation, TCM tongue datasets), Kaggle
→ 9,125 images · 12 feature classes · 4-phase pseudo-labeling pipeline

**Eye:** IEEE DataPort Eyes-Defy-Anemia (202 images with real Hb measurements),
Kaggle palpebral conjunctiva, Roboflow eye-conjunctiva-detector + jaundice datasets,
Roboflow xanthelasma
→ 2,497 images · 3 feature classes · clinical Hb-based labeling

---

## Environment

```
Python 3.11.9
PyTorch 2.5.1 + CUDA 12.1
Ultralytics 8.4.26
MediaPipe 0.10.33
FastAPI + uvicorn
NVIDIA GeForce RTX 4070 Super (12GB)
```

---

## File Structure

```
FaceFuel/
├── server.py                    # FastAPI v3 server
├── step10_inference.py          # Face inference pipeline
├── step1-9_*.py                 # Face training scripts
├── Phase2-7_tongue_*.py         # Tongue training pipeline
├── 1_eye_dataset.py – 6_eye_severity.py  # Eye pipeline
├── eye_inference.py             # Eye inference module
├── write_paper.py               # Paper 1 LaTeX generator
├── write_paper2.py              # Paper 2 LaTeX generator
├── write_paper3.py              # Paper 3 LaTeX generator
├── ablation_study.py            # Paper 1 ablation experiments
├── ablation_fusion.py           # Paper 2 fusion ablation
├── static/
│   └── index.html               # Web frontend
├── requirements.txt
└── README.md
```

---

## Commercial Licensing

This project is available for commercial licensing.

**What's included in a license:**
- Complete source code (all training + inference pipelines)
- 3 trained YOLO11m/YOLOv8m detection models
- 3 severity MLP models with uncertainty quantification
- 3 peer-reviewed research papers
- Full documentation and training reproducibility

**Suitable for:** nutrition supplement brands, telehealth platforms, corporate wellness apps, health & fitness applications.

**Contact:** mojuaries111@gmail.com

---

## Citation

```bibtex
@article{muhammad2026trimodal,
  author = {Muhammad, Abdul Moiz},
  title  = {TriModal: Face, Tongue, and Eye as Complementary Visual
            Channels for Non-Invasive Nutritional and Lifestyle
            Deficiency Screening},
  year   = {2026},
  doi    = {10.5281/zenodo.19468059}
}
```

---

## License

Code: MIT License
Research papers: CC BY-NC-ND 4.0

Trained model weights retain their respective dataset licenses.
For commercial use of the complete system, contact mojuaries111@gmail.com

---

*Built independently · Abdul Moiz Muhammad · COMSATS University Islamabad · Pakistan · 2026*
