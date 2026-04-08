"""
Generates: paper_results/facefuel_paper3.tex + paper_results/references3.bib
Run: python write_paper3.py
"""
from pathlib import Path

OUT = Path("paper_results")
OUT.mkdir(exist_ok=True)

TEX = r"""\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{cite}
\renewcommand\IEEEkeywordsname{Keywords}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\begin{document}

\title{TriModal: Face, Tongue, and Eye as Complementary Visual Channels\\
for Non-Invasive Nutritional and Lifestyle Deficiency Screening}

\author{Abdul~Moiz~Muhammad\\
COMSATS University Islamabad (Independent Research)\\
Wah Cantt, Punjab, Pakistan\\
mojuaries111@gmail.com}

\maketitle

\begin{abstract}
Non-invasive health screening from photographs has been explored independently
for facial skin, tongue surface, and ocular features, but no prior work has
unified all three visual channels in a single probabilistic framework.
This paper introduces TriModal FaceFuel, a tri-modal system that analyzes a
selfie photograph and an optional tongue photograph to produce calibrated
posterior estimates over 16 nutritional and lifestyle deficiency categories.

The system extends prior face and tongue pipelines
\cite{muhammad2026facefuel,muhammad2026bimodal} with a new eye analysis module
trained on 2{,}497 clinically labeled images across three eye feature classes:
conjunctival pallor (linked to iron and B12 deficiency), scleral icterus
(linked to liver stress), and xanthelasma (linked to cholesterol imbalance).
The eye module introduces one new deficiency dimension, cholesterol imbalance,
and runs automatically from the same selfie used for face analysis.

A three-way weighted product-of-experts posterior fusion combines face
(weight $\alpha=0.40$), tongue ($\beta=0.35$), and eye ($\gamma=0.25$)
posteriors into a unified 16-dimensional output. The eye module achieves
mAP@0.5\,=\,0.913, the highest of any single modality in the system.
A LAB color gate (B\,$>$\,145, L\,$>$\,140) suppresses false positive
scleral icterus detections by requiring genuine yellow pigmentation
before reporting the finding. Face detection is upgraded from YOLOv8m
to YOLO11m, improving face mAP from 0.790 to 0.872 (+10.4\%).
The full tri-modal pipeline runs in under 235\,ms on consumer GPU hardware.
\end{abstract}

\begin{IEEEkeywords}
tri-modal health screening, conjunctival pallor, scleral icterus,
xanthelasma, YOLO11m, DINOv2, Bayesian fusion, product-of-experts,
facial analysis, tongue diagnosis, ocular biomarkers
\end{IEEEkeywords}

%% ─────────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

\IEEEPARstart{T}{he} human body signals nutritional and systemic health
status through multiple visual channels simultaneously. Periorbital
darkening and pallor indicate iron or B12 deficiency from the face.
Pale conjunctiva in the lower eyelid indicates hemoglobin deficiency
from the eye. Smooth, atrophic tongue indicates folate or B12 depletion.
Xanthelasma deposits near the eyelids indicate lipid dysregulation.
A clinician examining a patient typically reads all of these channels
in a single visual assessment, yet no computational system has
unified them before this work.

Prior publications in this research program addressed face alone
\cite{muhammad2026facefuel} and face-tongue bimodal fusion
\cite{muhammad2026bimodal}. This paper adds the eye as a third
independent visual channel and introduces a three-way posterior
fusion framework.

The practical contribution is significant: the eye analysis adds zero
extra photographs to the user workflow. The eye region is extracted
automatically from the aligned face image produced by Stage 1 of the
existing face pipeline. Users take one selfie and optionally one tongue
photograph. The system returns a 16-dimensional deficiency posterior
combining evidence from up to three visual modalities.

\subsection*{Contributions}
\begin{itemize}
  \item An eye analysis module (YOLO11m, mAP@0.5\,=\,0.913) trained
    on 2{,}497 clinically labeled images including 202 with paired
    hemoglobin measurements.
  \item A LAB color gate for scleral icterus that prevents false
    positives on healthy eyes by verifying yellow pigmentation
    (B\,$>$\,145) in the conjunctival region.
  \item A three-way product-of-experts fusion framework handling
    asymmetric coverage (11 / 15 / 16 categories) across face,
    tongue, and eye.
  \item One new eye-exclusive deficiency dimension:
    cholesterol imbalance (detectable from xanthelasma).
  \item Upgrade of the face detector from YOLOv8m to YOLO11m:
    mAP improves from 0.790 to 0.872 (+10.4\%).
\end{itemize}

%% ─────────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\subsection{Ocular Biomarkers}

Conjunctival pallor as a non-invasive anemia indicator has been
validated clinically and computationally. \cite{asare2023} compared
machine learning algorithms on conjunctiva, palm, and fingernail images,
with CNN achieving 99.12\% accuracy. \cite{benson2025} applied Vision
Transformers to conjunctiva-sclera images, achieving 98.47\% accuracy
with attention map explainability. Scleral icterus (yellow sclera) as
an indicator of elevated bilirubin and liver dysfunction is a standard
clinical sign visible when serum bilirubin exceeds 2.5\,mg/dL
\cite{wiki_jaundice}. Xanthelasma and arcus senilis as visual
indicators of hyperlipidemia are documented in \cite{statpearls_arcus}.

\subsection{Prior Work in This Series}

The face pipeline \cite{muhammad2026facefuel} established a five-stage
architecture (MediaPipe alignment, YOLOv8m detection, region-aware DINOv2,
MC Dropout severity MLP, Bayesian inference) achieving mean F1\,=\,0.677
across 11 deficiency categories. The bimodal paper \cite{muhammad2026bimodal}
added tongue analysis (mAP\,=\,0.836, F1\,=\,0.800) and a two-modality
product-of-experts fusion, extending coverage to 15 categories.
This paper adds the eye as a third channel and unifies all three in
a single framework.

%% ─────────────────────────────────────────────────────────────
\section{Eye Analysis Module}
\label{sec:eye}

\subsection{Dataset Assembly}

Eye training data was assembled from five sources:
(1) Eyes-Defy-Anemia \cite{eyesdefyanemia}: 202 conjunctiva images
(India 95, Italy 107) with paired Hb measurements — labeled by WHO
threshold: Hb\,$<$\,11.0\,g/dL $\to$ conjunctival\_pallor,
Hb\,$\geq$\,12.0\,g/dL $\to$ normal;
(2) Palpebral conjunctiva (Kaggle): 183 anemia images;
(3) Eye-Conjunctiva-Detector (Roboflow): 218 annotated images;
(4) Eye-Disease-YOLO (Roboflow): 191 jaundice/normal images;
(5) Xanthelasma (Roboflow): 1{,}043 images.

The merged dataset contains 2{,}497 images.
Table \ref{tab:eye_dataset} shows the class distribution.

\begin{table}[h]
\centering
\caption{Eye module dataset class distribution.}
\label{tab:eye_dataset}
\begin{tabular}{lrr}
\toprule
\textbf{Class} & \textbf{Images} & \textbf{Labeling} \\
\midrule
Conjunctival pallor  & 615  & Clinical Hb + source labels \\
Scleral icterus      &  95  & Jaundice class labels \\
Xanthelasma          & 1{,}043 & Folder-based labels \\
Normal / background  & 744  & Hb $\geq$12.0 or Normal class \\
\midrule
\textbf{Total}       & 2{,}497 & \\
\bottomrule
\end{tabular}
\end{table}

\subsection{YOLO11m Eye Detector}

A YOLO11m \cite{ultralytics2024} detector was trained for 120 epochs
with early stopping (patience\,=\,35, triggered at epoch 67).
Color-preserving augmentation was applied: HSV hue jitter minimized
($h=0.01$) to avoid corrupting the color signals that define
conjunctival pallor and scleral icterus. Table \ref{tab:eye_detection}
reports per-class results. The overall mAP@0.5\,=\,0.913 is the
highest of any module in the system.

\begin{table}[h]
\centering
\caption{YOLO11m eye detection results (validation set).}
\label{tab:eye_detection}
\begin{tabular}{lccc}
\toprule
\textbf{Class} & \textbf{AP@0.5} & \textbf{Precision} & \textbf{Recall} \\
\midrule
Conjunctival pallor & 0.859 & 0.603 & 0.989 \\
Scleral icterus     & 0.886 & 0.722 & 1.000 \\
Xanthelasma         & 0.995 & 0.921 & 1.000 \\
\midrule
\textbf{All}        & \textbf{0.913} & 0.749 & 0.996 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Eye Pipeline Architecture}

Figure \ref{fig:eye_pipeline} shows the five-stage eye analysis pipeline.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{eye_pipeline}
\caption{The FaceFuel eye analysis pipeline. The eye region is extracted
from the same $256\times256$ aligned face image used by the face pipeline.
DINOv2 ViT-S/14 extracts 384-dimensional embeddings from three anatomical
zones (sclera left, sclera right, periorbital) producing a 1152-dimensional
composite vector. The severity MLP with Monte Carlo Dropout ($T=25$) provides
uncertainty-weighted evidence to the Bayesian inference engine. LAB color
gating prevents false positive scleral icterus detections.}
\label{fig:eye_pipeline}
\end{figure*}

The eye region is extracted from the aligned face image (rows 15--50\%,
columns 5--95\%) without any additional photograph. Three DINOv2 regions
are defined: sclera left (columns 0--45\%), sclera right (55--100\%),
and periorbital (rows 0--35\%), producing a
$3 \times 384 = 1152$-dimensional feature vector.

\subsection{LAB Color Gate for Scleral Icterus}
\label{sec:lab_gate}

Scleral icterus produces elevated B channel (yellow direction) in LAB
color space. Normal sclera: B\,$\approx$\,128 (neutral). Icteric
sclera: B\,$>$\,145. The color gate blocks scleral icterus reports
unless:
\begin{equation}
B_{\text{mean}} > 145 \;\text{ and }\; L_{\text{mean}} > 140
\label{eq:lab_gate}
\end{equation}

Three suppression layers ensure robust false positive prevention:
(1) per-class confidence threshold raised to 0.65 for scleral icterus;
(2) LAB color gate applied post-detection;
(3) MLP severity alone is insufficient — YOLO confirmation required.
Empirical validation confirmed zero false positives on five healthy
eye images from the test set.

\subsection{Eye Severity MLP}

The 1152-dimensional vector is processed by a shared-backbone MLP
(LayerNorm, 1152$\to$256$\to$128, GELU, dropout 0.30) with
per-feature heads (128$\to$64$\to$1). Training: 120 epochs,
AdamW, OneCycleLR, BCEWithLogitsLoss with positive class weighting.
Mean F1\,=\,0.991 — the highest severity regression F1 in the system.

%% ─────────────────────────────────────────────────────────────
\section{TriModal Fusion}
\label{sec:fusion}

\subsection{Asymmetric Coverage Expansion}

Face covers 11 deficiency categories, tongue 15, eye 16.
Each posterior is expanded to the full 16-dimensional space with
uncovered dimensions set to zero before fusion.

\subsection{Three-Way Product-of-Experts}

For deficiency $k$ with posteriors $f_k$, $t_k$, $e_k$:
\begin{equation}
p_{\text{tri}}(k) \propto \begin{cases}
f_k^{\alpha}\,t_k^{\beta}\,e_k^{\gamma} & \text{all three active}\\
\text{two-way PoE (weight-normalized)} & \text{two active}\\
p \times 0.85 & \text{tongue or eye only}\\
p & \text{face only}
\end{cases}
\label{eq:trimodal}
\end{equation}
with $\alpha=0.40$, $\beta=0.35$, $\gamma=0.25$,
normalized to sum to one.

Figure \ref{fig:fusion} illustrates the complete fusion framework.

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{trimodal_fusion}
\caption{TriModal product-of-experts posterior fusion. Face (11-dim),
tongue (15-dim), and eye (16-dim) posteriors are each expanded to the
full 16-dimensional deficiency space and combined multiplicatively.
Tongue-exclusive dimensions (liver stress, gut dysbiosis, hypothyroid,
folate deficiency) and the eye-exclusive dimension (cholesterol imbalance)
receive evidence only from their respective modalities.}
\label{fig:fusion}
\end{figure*}

Eye weight ($\gamma=0.25$) is set lower than face and tongue because
the eye dataset is smaller and the selfie eye crop is lower resolution
than a dedicated eye photograph. The discount factor (0.85) applied to
single-modality tongue or eye signals reflects the absence of
corroborating evidence.

%% ─────────────────────────────────────────────────────────────
\section{Experiments}
\label{sec:experiments}

\subsection{Model Upgrade Summary}

Table \ref{tab:yolo_comparison} reports YOLO performance across all
three modalities, including the YOLOv8m to YOLO11m upgrade for the
face detector.

\begin{table}[h]
\centering
\caption{YOLO detector performance across all three modalities.}
\label{tab:yolo_comparison}
\begin{tabular}{llcc}
\toprule
\textbf{Modality} & \textbf{Model} & \textbf{mAP@0.5} & $\Delta$ \\
\midrule
Face  & YOLOv8m $\to$ YOLO11m & $0.790 \to 0.872$ & $+0.082$ \\
Tongue & YOLOv8m v3 (retained)  & 0.836              & — \\
Eye   & YOLO11m (new module)   & 0.913              & new \\
\bottomrule
\end{tabular}
\end{table}

The tongue YOLO11m fine-tune achieved 0.807, below the YOLOv8m v3
result of 0.836, due to class imbalance in rare classes (crenated:
42 instances; pale tongue: 4 instances). The v3 weights were retained.

\subsection{Qualitative Evaluation}

Table \ref{tab:qualitative} reports results on four eye test cases
designed to verify expected system behavior.

\begin{table}[h]
\centering
\caption{Qualitative eye module evaluation on test images.}
\label{tab:qualitative}
\begin{tabular}{p{2.1cm}p{1.8cm}p{2.5cm}}
\toprule
\textbf{Image} & \textbf{Detection} & \textbf{Top-1 deficiency} \\
\midrule
Yellow sclera (bilateral) & Scleral icterus 42\% & Liver stress 18.4\% \\
Healthy eye               & None                  & Flat posterior \\
Xanthelasma deposits      & Xanthelasma $>$0.90   & Cholesterol imbalance \\
Pale conjunctiva          & Pallor $>$0.70        & Iron deficiency \\
\bottomrule
\end{tabular}
\end{table}

The yellow sclera case correctly identifies liver stress as the primary
concern. The healthy eye produces no features and a flat posterior,
confirming that the three-layer scleral icterus suppression works in
practice. This is a critical result: a system that fires liver stress
warnings on every user regardless of sclera color would be clinically
and commercially unusable.

\subsection{Deficiency Coverage}

Table \ref{tab:coverage} summarizes which modalities contribute evidence
for each of the 16 deficiency dimensions.

\begin{table}[h]
\centering
\caption{16-deficiency coverage by modality in TriModal FaceFuel.}
\label{tab:coverage}
\begin{tabular}{ll}
\toprule
\textbf{Deficiency} & \textbf{Modality} \\
\midrule
Iron deficiency          & Face + Eye (conjunctival pallor) \\
B12 deficiency           & Face + Tongue (smooth glossy) \\
Vitamin D deficiency     & Face \\
Zinc deficiency          & Face + Tongue (geographic, crenated) \\
Omega-3 deficiency       & Face + Eye (xanthelasma) \\
Vitamin A, C deficiency  & Face \\
Poor sleep, stress       & Face (periorbital features) \\
Hormonal imbalance       & Face \\
Dehydration              & Face \\
Liver stress             & Tongue (yellow coating) + Eye (scleral icterus) \\
Gut dysbiosis            & Tongue (white/thick coating) \\
Hypothyroid              & Tongue (crenated) \\
Folate deficiency        & Tongue (smooth glossy, red tongue) \\
Cholesterol imbalance    & \textbf{Eye only} (xanthelasma) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{System Timing}

The complete tri-modal pipeline processes face\,+\,eye in 96--116\,ms
and face\,+\,eye\,+\,tongue in 180--235\,ms on an NVIDIA RTX\,4070
Super. The eye module adds only 29--36\,ms over face-alone analysis,
a modest overhead for the additional diagnostic dimension it provides.

%% ─────────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}

\subsection{Why mAP = 0.913 Despite Small Dataset}

The eye module achieves the highest mAP of any module in the system
despite having fewer training images (2{,}497 vs 5{,}721 face and
9{,}125 tongue) because the three targeted conditions are visually
highly distinctive. Conjunctival pallor is a color condition that
differs consistently from normal pink conjunctiva. Scleral icterus
produces a yellow hue on normally white tissue. Xanthelasma produces
raised yellowish plaques that differ strongly from normal periorbital
skin. Visually distinctive conditions are easier to detect than subtle
ones.

\subsection{Clinical Grounding of Eye Features}

Conjunctival pallor trained on the Eyes-Defy-Anemia dataset is the most
directly clinically grounded component of the entire FaceFuel system,
because the labels come from real hemoglobin measurements rather than
visual annotations. The Hb\,$<$\,11.0\,g/dL threshold for anemia labeling
is consistent with WHO definitions. This means the conjunctival pallor
detector was effectively trained to predict a blood test result from an image.

\subsection{Limitations}

The scleral icterus class has only 95 training images. The LAB color
gate thresholds were determined empirically and have not been validated
against a clinical series with known bilirubin values. Eye crops from
selfies are lower resolution than dedicated eye photographs, which may
limit sensitivity for subtle conjunctival pallor cases. Finally, the
fusion weights ($\alpha$, $\beta$, $\gamma$) were set based on dataset
quality differences rather than optimized against clinical ground truth.

%% ─────────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

This paper presented TriModal FaceFuel, the first system to unify face,
tongue, and eye visual biomarkers in a single probabilistic framework
for non-invasive nutritional health screening. The eye module achieves
mAP@0.5\,=\,0.913 and introduces cholesterol imbalance as a 16th
deficiency dimension detectable from a selfie photograph. A three-layer
suppression mechanism for scleral icterus produces zero false positives
on healthy eyes in our evaluation.

The practical design principle throughout this series of papers has been
to minimize user friction while maximizing diagnostic coverage. The
single-selfie face-plus-eye architecture achieves exactly this: one
photograph, two independent visual channels, up to 16 deficiency
dimensions in under 120\,ms. Tongue analysis remains an optional
enhancement that deepens coverage without disrupting the core user
experience.

Across three papers \cite{muhammad2026facefuel,muhammad2026bimodal},
this research program has built a complete multi-modal visual health
screening system from publicly available datasets without institutional
clinical support, demonstrating that meaningful nutritional deficiency
screening is achievable from consumer smartphone photographs.

%% ─────────────────────────────────────────────────────────────
\section*{Acknowledgments}

This research was conducted independently. The author thanks the maintainers
of Roboflow Universe, Kaggle, and IEEE DataPort for making eye condition
datasets publicly available.
The author used Claude (Anthropic) as a coding and drafting assistance
tool during this work.

\bibliographystyle{IEEEtran}
\bibliography{references3}

\end{document}
"""

BIB = r"""@article{muhammad2026facefuel,
  author  = {Muhammad, Abdul Moiz},
  title   = {FaceFuel: A Multi-Stage Heterogeneous Fusion Pipeline
             for Non-Invasive Nutritional Deficiency Screening
             from Facial Imagery},
  year    = {2026},
  note    = {Zenodo. DOI: [insert Paper 1 DOI]}
}
@article{muhammad2026bimodal,
  author  = {Muhammad, Abdul Moiz},
  title   = {Multi-Modal Visual Health Assessment Through
             Product-of-Experts Posterior Fusion of Facial
             and Lingual Biomarkers},
  year    = {2026},
  note    = {Zenodo. DOI: [insert Paper 2 DOI]}
}
@article{park2021,
  author  = {Park, Eunchul and Hwang, Jaesung and Lee, Jooheung and others},
  title   = {Non-invasive anemia detection via conjunctival pallor analysis},
  journal = {npj Digital Medicine},
  volume  = {4}, pages = {1--9}, year = {2021}
}
@article{asare2023,
  author  = {Asare, Joseph W. and Appiahene, Peter and
             Donkoh, Emmanuel T. and Dimauro, Giovanni},
  title   = {Iron deficiency anemia detection using machine learning:
             conjunctiva, palm and fingernails comparative study},
  journal = {Engineering Reports},
  volume  = {5}, pages = {e12667}, year = {2023}
}
@article{benson2025,
  author  = {Benson, Abena E. and others},
  title   = {Non-invasive anemia detection from conjunctiva and sclera
             images using vision transformer with attention map
             explainability},
  journal = {Scientific Reports}, year = {2025}
}
@misc{wiki_jaundice,
  author  = {{Wikipedia contributors}},
  title   = {Jaundice --- {Wikipedia}{,} The Free Encyclopedia},
  year    = {2026},
  url     = {https://en.wikipedia.org/wiki/Jaundice}
}
@misc{statpearls_arcus,
  title   = {Arcus Senilis},
  author  = {{StatPearls}},
  year    = {2025},
  url     = {https://www.ncbi.nlm.nih.gov/books/NBK554370/}
}
@misc{eyewiki_arcus,
  title   = {Arcus Senilis},
  author  = {{EyeWiki}},
  year    = {2026},
  url     = {https://eyewiki.org/Arcus_Senilis}
}
@article{hinton2002,
  author  = {Hinton, Geoffrey E.},
  title   = {Training products of experts by minimizing
             contrastive divergence},
  journal = {Neural Computation},
  volume  = {14}, number = {8}, pages = {1771--1800}, year = {2002}
}
@article{oquab2024,
  author  = {Oquab, Maxime and Darcet, Timoth{\'e}e and
             Moutakanni, Th{\'e}o and others},
  title   = {{DINOv2}: Learning robust visual features without supervision},
  journal = {Transactions on Machine Learning Research}, year = {2024}
}
@misc{ultralytics2024,
  author  = {{Ultralytics}},
  title   = {{YOLO11}},
  year    = {2024},
  url     = {https://docs.ultralytics.com/models/yolo11/}
}
@article{mdpi2025skin,
  author  = {Ridwan, Ahmad and others},
  title   = {Enhancing Dermatological Diagnosis: How Effective Is
             YOLO11 Compared to Leading CNN Models?},
  journal = {Dermatology Research and Practice}, year = {2025}
}
@misc{eyesdefyanemia,
  author  = {{IEEE DataPort}},
  title   = {Eyes-Defy-Anemia Dataset},
  year    = {2023},
  url     = {https://ieee-dataport.org/documents/eyes-defy-anemia}
}
@misc{roboflow2023,
  author  = {{Roboflow Inc.}},
  title   = {Roboflow Universe},
  year    = {2023},
  url     = {https://universe.roboflow.com}
}
@inproceedings{karpathy2015,
  author    = {Karpathy, Andrej and Fei-Fei, Li},
  title     = {Deep visual-semantic alignments for image captioning},
  booktitle = {CVPR},
  pages     = {3128--3137}, year = {2015}
}
"""

tex_path = OUT / "facefuel_paper3.tex"
bib_path = OUT / "references3.bib"
tex_path.write_text(TEX.strip(), encoding="utf-8")
bib_path.write_text(BIB.strip(), encoding="utf-8")

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  FaceFuel Paper 3 (TriModal) — Generated                     ║
╠═══════════════════════════════════════════════════════════════╣
║  paper_results/facefuel_paper3.tex                           ║
║  paper_results/references3.bib                               ║
╠═══════════════════════════════════════════════════════════════╣
║  Figures in LaTeX (upload these exact filenames):            ║
║    eye_pipeline.png     → \\includegraphics (Figure 1)        ║
║    trimodal_fusion.png  → \\includegraphics (Figure 2)        ║
╠═══════════════════════════════════════════════════════════════╣
║  Before submission:                                          ║
║    1. Insert Paper 1 and Paper 2 DOIs in references3.bib     ║
║    2. Upload both .png figures to Overleaf                   ║
║    3. Compile — figures appear at correct positions          ║
║    4. Submit Zenodo first → then IEEE JBHI                   ║
╚═══════════════════════════════════════════════════════════════╝
""")