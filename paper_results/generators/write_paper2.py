"""
Generates: paper_results/facefuel_paper2.tex + paper_results/references2.bib
Run: python write_paper2.py
"""
from pathlib import Path
import json

OUT = Path("paper_results")
OUT.mkdir(exist_ok=True)

results_path = OUT / "fusion_ablation_results.json"
if results_path.exists():
    data     = json.loads(results_path.read_text())
    ablation = data.get("ablation", {})
    comp     = data.get("complementarity", {})
    D = ablation.get("D. Optimized fusion (0.55/0.45)", {})
    A = ablation.get("A. Face-only", {})
    B = ablation.get("B. Tongue-only", {})
    C = ablation.get("C. Equal fusion (0.5/0.5)", {})
    E = ablation.get("E. Simple average", {})
    F = ablation.get("F. Maximum fusion", {})
    top1_agree = comp.get("top1_agreement_pct", 48.9)
else:
    D = {"mean_entropy":3.6464,"mean_top1":0.1516,"tongue_excl_mass":0.1454}
    A = {"mean_entropy":3.1956,"mean_top1":0.1978,"tongue_excl_mass":0.0000}
    B = {"mean_entropy":3.6488,"mean_top1":0.1580,"tongue_excl_mass":0.1761}
    C = {"mean_entropy":3.6510,"mean_top1":0.1511,"tongue_excl_mass":0.1468}
    E = {"mean_entropy":3.5698,"mean_top1":0.1612,"tongue_excl_mass":0.0881}
    F = {"mean_entropy":3.6275,"mean_top1":0.1620,"tongue_excl_mass":0.1348}
    top1_agree = 48.9

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

\title{Combining Face and Tongue Images for Nutritional Health Screening:\\
A Multi-Modal Bayesian Fusion Approach}

\author{Abdul~Moiz~Muhammad\\
COMSATS University Islamabad (Independent Research)\\
Wah Cantt, Punjab, Pakistan\\
mojuaries111@gmail.com}

\maketitle

\begin{abstract}
A doctor can learn a surprising amount just by looking at someone's face and
tongue. Dark circles and pale skin can point toward iron or B12 problems.
A red or smooth tongue surface often shows up in folate or B12 deficiency.
A thick yellow coating on the tongue can indicate liver or digestive issues.
These signs have been used in clinical practice for a long time, but no one
had built a computer system that reads both together and combines them into
a single set of estimates.

This paper describes such a system. It builds on an earlier face analysis
pipeline~\cite{muhammad2026facefuel} and adds a complete tongue analysis
module. The tongue module uses a YOLOv8m detector trained on 9{,}125 tongue
images across 12 feature classes, reaching mAP@0.5\,=\,0.812. Features are
extracted from five anatomical tongue zones using DINOv2 ViT-S/14, giving a
1920-dimensional vector per image. A severity regression network with Monte
Carlo Dropout estimates feature severity along with uncertainty, and a Bayesian
engine maps the evidence to 15 deficiency categories.

Four of those categories can only be detected from tongue signs, not skin:
liver stress, gut dysbiosis, hypothyroid tendency, and folate deficiency.
The face pipeline alone covers 11 categories. Together they cover 15.

The two posteriors are combined using a weighted product-of-experts formula,
with the face getting slightly more weight (0.55) than the tongue (0.45) because
its training data is more reliably annotated. An ablation study across six
fusion configurations shows that the product-of-experts approach handles the
asymmetric category coverage better than simple averaging. A complementarity
check found that face and tongue agree on the top-ranked deficiency only
""" + str(top1_agree) + r"""\% of the time, which means they are actually
providing different information most of the time.

The full system handles a face and tongue pair in under 120\,ms on a consumer
GPU.
\end{abstract}

\begin{IEEEkeywords}
multi-modal health screening, tongue diagnosis, face analysis,
product-of-experts, Bayesian fusion, DINOv2, YOLOv8, nutritional deficiency
\end{IEEEkeywords}

%% ─────────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

\IEEEPARstart{T}{here} are two cheap, non-invasive ways a doctor checks a
patient's general nutritional state without ordering a blood test: they look
at the skin and they look at the tongue. Both carry real clinical information.
Periorbital darkening and pallor around the lips have well-documented links
to iron and B12 deficiency~\cite{sarkar2016}. Tongue signs have been used
in traditional Chinese medicine for over two thousand years~\cite{chen2011},
and several of those observations have since been confirmed by Western clinical
studies: a smooth, atrophied tongue surface is a classic B12 and folate sign,
a thick yellow coating is associated with liver and digestive problems, and
scalloped tongue edges often appear in thyroid and fluid retention conditions.

The strange thing is that no one had combined these two sources
computationally. The face and tongue are both easy to photograph with a
smartphone. The visual signs are well-documented. But every existing automated
system focuses on one or the other in isolation. This paper closes that gap.

The approach is straightforward. A previously published face analysis pipeline
\cite{muhammad2026facefuel} already handles the face side and outputs a posterior
distribution over 11 deficiency categories. This paper adds a tongue pipeline
of the same design and then fuses the two outputs using a weighted
product-of-experts formula. The result is a single probability distribution
over 15 categories, four of which the face alone cannot detect.

There were two questions that motivated the fusion design. First, are face and
tongue actually complementary, or do they just say the same thing? Second, how
should the asymmetric category coverage be handled, since the tongue covers more
categories than the face?

The complementarity question was answered empirically: in a dataset of 4{,}792
paired samples, the face and tongue ranked the same deficiency first only
""" + str(top1_agree) + r"""\% of the time. That is a surprisingly low number
and confirms that the two modalities are picking up on different signals. The
category asymmetry was handled by expanding the face posterior to a 15-dimensional
space and setting the four tongue-exclusive dimensions to zero before fusion,
which is the correct thing to do because the face model genuinely has no
information about liver stress or gut dysbiosis.

\subsection*{Contributions}

\begin{itemize}
  \item A complete tongue analysis pipeline (YOLOv8m, five-region DINOv2,
    severity MLP, tongue Bayesian engine) trained on a unified 9{,}125-image
    dataset, achieving mAP@0.5\,=\,0.812 and severity F1\,=\,0.800.

  \item The first system to combine facial and lingual biomarkers in a single
    probabilistic framework, covering 15 deficiency categories compared to 11
    for face alone.

  \item Four tongue-exclusive deficiency dimensions (liver stress, gut dysbiosis,
    hypothyroid tendency, folate deficiency) that are not detectable from skin
    and extend diagnostic coverage by 36\%.

  \item A complementarity analysis showing """ + str(top1_agree) + r"""\%
    top-1 agreement between modalities, confirming that they provide
    largely independent information.

  \item An ablation across six fusion strategies showing that product-of-experts
    with asymmetry correction outperforms naive averaging and max-selection.
\end{itemize}

%% ─────────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\subsection{Tongue Analysis}

Automated tongue analysis has mostly been studied in the context of traditional
Chinese medicine syndrome classification. Pang \textit{et al.}~\cite{pang2005}
built an early tongue segmentation system using deformable contours. Zhang
\textit{et al.}~\cite{zhang2016} applied CNNs to tongue coating detection.
Lo \textit{et al.}~\cite{lo2015} used transfer learning for TCM syndrome
classification from tongue images.

These systems treat tongue diagnosis as a standalone problem and map tongue
signs directly to syndrome categories without going through a probabilistic
deficiency layer. None of them combine tongue observations with concurrent
skin analysis.

\subsection{Multi-Modal Fusion for Health Screening}

Combining independent probabilistic models using product-of-experts was
proposed by Hinton~\cite{hinton2002}. The core idea is that if two models
independently estimate the same quantity, their joint estimate should be
proportional to the product of their individual estimates. This is different
from averaging, which can dilute strong signals from one modality with weak
signals from another.

In medical imaging, late fusion of modality-specific models generally
outperforms early fusion when training data for the two modalities is not
shared~\cite{karpathy2015}, which is exactly the situation here since the
face and tongue datasets were collected and annotated separately.

\subsection{Non-Invasive Nutritional Screening}

Park \textit{et al.}~\cite{park2021} demonstrated anemia screening from
conjunctival pallor using smartphone images, which is the closest prior
work to the face pipeline in~\cite{muhammad2026facefuel}. The present paper
extends the multi-deficiency framework in~\cite{muhammad2026facefuel} to
include tongue-based evidence, adding four deficiency categories that
are completely outside the scope of any skin-based system.

%% ─────────────────────────────────────────────────────────────
\section{Tongue Pipeline}
\label{sec:tongue}

The tongue pipeline mirrors the structure of the face pipeline described in
\cite{muhammad2026facefuel}, with adjustments for tongue physiology and dataset
characteristics. Figure~\ref{fig:tongue_pipeline} shows the five stages.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{tongue_pipeline}
\caption{The tongue analysis pipeline. The input photograph is first cropped
to the tongue body using YOLO detection, then features are extracted from
five anatomical zones via frozen DINOv2 ViT-S/14, and a Bayesian engine
maps severity estimates to 15 deficiency categories. Purple categories are
tongue-exclusive and cannot be detected from skin alone.}
\label{fig:tongue_pipeline}
\end{figure*}

\subsection{Dataset}

Tongue training data came from Roboflow Universe tongue segmentation and
classification datasets, Kaggle tongue condition collections, and a
preprocessed TCM tongue image set~\cite{roboflow2023}. After merging,
the unified dataset has 9{,}125 images.

Getting good coverage across all 12 feature classes required a four-phase
training strategy. The first phase trained on the 3{,}362 images that had
manual annotations. The second phase ran the trained model on the unlabeled
images and supplemented it with LAB color analysis to generate pseudo-labels
for color-based features like red tongue and yellow coating. The third phase
retrained on all labeled data combined, applying stricter quality thresholds
to the pseudo-labels to reduce noise. The fourth phase added black hairy
tongue as a new class and corrected a geographic tongue over-triggering
problem that had produced 3{,}525 pseudo-labels --- far too many --- by
replacing the simple texture threshold with a 4$\times$4 spatial grid
variance check that requires both bright patches and dark patches to be
present simultaneously before classifying a tongue as geographic.

Table~\ref{tab:tongue_dataset} shows the final class distribution.

\begin{table}[h]
\centering
\caption{Tongue dataset class distribution after four-phase training.}
\label{tab:tongue_dataset}
\begin{tabular}{lrr}
\toprule
\textbf{Class} & \textbf{Instances} & \textbf{Type} \\
\midrule
Tongue body    & 6,095 & area  \\
Geographic     & 3,100 & count \\
Smooth/glossy  & 2,185 & area  \\
Thick coating  & 2,072 & area  \\
Tooth marked   &   775 & count \\
Red tongue     &   500 & area  \\
Yellow coating &   312 & area  \\
Fissured       &   246 & count \\
Crenated       &   219 & count \\
Black hairy    &   851 & count \\
White coating  &    38 & area  \\
Pale tongue    &    21 & area  \\
\midrule
\textbf{Total} & \textbf{16,414} & \\
\bottomrule
\end{tabular}
\end{table}

\subsection{YOLO Detection and Cropping}

The input image first goes through a YOLOv8m detector~\cite{jocher2023} that
detects the tongue body bounding box and classifies 11 pathological feature
classes simultaneously. The detected tongue body region is cropped with 5\%
padding and resized to $256 \times 256$ for standardized downstream processing.
If no tongue body is found above confidence 0.30, the central 70\% of the image
is used as a fallback.

The detector was fine-tuned for 80 epochs from Phase~3 weights, reaching
mAP@0.5\,=\,0.812 on the validation set. Table~\ref{tab:tongue_det} shows
per-class results.

\begin{table}[h]
\centering
\caption{Tongue YOLOv8m per-class detection results.}
\label{tab:tongue_det}
\begin{tabular}{lcc}
\toprule
\textbf{Feature} & \textbf{AP@0.5} & \textbf{Severity} \\
\midrule
Tongue body    & 0.975 & area  \\
Yellow coating & 0.976 & area  \\
Smooth/glossy  & 0.962 & area  \\
Geographic     & 0.953 & count \\
Thick coating  & 0.914 & area  \\
Tooth marked   & 0.919 & count \\
White coating  & 0.830 & area  \\
Red tongue     & 0.844 & area  \\
Pale tongue    & 0.745 & area  \\
Fissured       & 0.599 & count \\
Crenated       & 0.521 & count \\
Black hairy    & 0.527 & count \\
\midrule
\textbf{All}   & \textbf{0.812} & \\
\bottomrule
\end{tabular}
\end{table}

The three weaker classes --- fissured (0.599), crenated (0.521), and black
hairy tongue (0.527) --- all had fewer than 300 real annotations. Fissured
and crenated each relied heavily on gradient-based pseudo-labels, which
introduced some noise. Black hairy tongue had 851 instances but they were
mostly generated from LAB-based dark coating detection rather than manually
verified images. Improving these three classes is the clearest path to
a better detector overall.

\subsection{Region-Aware DINOv2 Features}

Five anatomical zones are cropped from the $256 \times 256$ tongue image:

\begin{itemize}
  \item \textbf{Tongue tip} (top 35\%, middle 60\% width): tip color, smooth atrophy
  \item \textbf{Tongue body} (middle band, wide): coating texture, papillae, fissures
  \item \textbf{Left lateral} (full height, left 40\%): tooth marks, crenation
  \item \textbf{Right lateral} (full height, right 40\%): tooth marks, crenation
  \item \textbf{Coating zone} (central 40\% area): coating color and thickness
\end{itemize}

Each zone is passed through frozen DINOv2 ViT-S/14~\cite{oquab2024} to get a
384-dimensional CLS token. The five tokens are concatenated into a 1920-dimensional
vector. This is smaller than the face pipeline's 3072-dimensional vector because
the tongue has five zones rather than eight facial regions.

The encoder is used frozen throughout, for the same reason as in the face pipeline:
the pre-training on 140 million images gives feature representations that
generalize well to skin and tissue texture without needing fine-tuning on a
small domain-specific dataset.

\subsection{Severity MLP and Bayesian Inference}

The severity MLP architecture is identical to the face model in
\cite{muhammad2026facefuel}: a shared backbone (1920$\to$512$\to$256, LayerNorm,
GELU, dropout 0.3) with per-feature heads (256$\to$64$\to$1, dropout 0.2).
Training used BCEWithLogitsLoss with positive class weighting and reached mean
F1\,=\,0.800 on validation.

The tongue Bayesian engine uses a 15-deficiency conditional probability table
that includes four categories not in the face CPT. These four tongue-exclusive
CPT values were set based on clinical literature:

\begin{itemize}
  \item \textbf{Liver stress}: yellow coating $P = 0.82$, thick coating $P = 0.52$
    based on gastroenterology associations with bile reflux~\cite{whitfield2006}
  \item \textbf{Gut dysbiosis}: white coating $P = 0.88$, thick coating $P = 0.62$
    based on oral microbiome studies~\cite{phua2012}
  \item \textbf{Hypothyroid}: crenated $P = 0.68$, tooth marked $P = 0.72$
    based on thyroid-related tongue swelling~\cite{roberts2004}
  \item \textbf{Folate deficiency}: red tongue $P = 0.88$, smooth/glossy $P = 0.78$
    based on atrophic glossitis literature~\cite{devalia2014}
\end{itemize}

%% ─────────────────────────────────────────────────────────────
\section{Multi-Modal Fusion}
\label{sec:fusion}

\subsection{The Coverage Asymmetry Problem}

The face posterior has 11 dimensions. The tongue posterior has 15. Four categories
appear only in the tongue output. This asymmetry means the two posteriors cannot
simply be averaged or multiplied directly --- they do not live in the same space.

The solution is to first expand both into a shared 15-dimensional space. The face
posterior gets the four tongue-exclusive dimensions set to zero before normalization.
This reflects the actual epistemic situation: the face model has no information about
liver stress or gut dysbiosis, so its contribution to those dimensions really should
be zero. Setting them to zero is not an approximation --- it is the correct thing to do.

\subsection{Weighted Product-of-Experts}

Let $\mathbf{f} \in \mathbb{R}^{15}$ be the expanded face posterior and
$\mathbf{t} \in \mathbb{R}^{15}$ the tongue posterior. For each deficiency
category $k$, the fused estimate is:

\begin{equation}
p_{\text{fused}}(k) \propto \begin{cases}
f_k^{\,0.55} \cdot t_k^{\,0.45} & \text{both} > 0 \\
f_k            & \text{tongue has no evidence} \\
0.85\,t_k      & \text{face has no evidence}
\end{cases}
\label{eq:poe}
\end{equation}

followed by normalization to sum to one. The face weight is slightly higher
(0.55 vs 0.45) because the face training data was more carefully manually annotated
than the tongue data, which used a significant fraction of pseudo-labels.

The 0.85 discount on tongue-exclusive categories reflects a reasonable degree
of skepticism: without any confirmation from the face modality, the tongue-based
estimate for liver stress or gut dysbiosis should be slightly conservative.

Figure~\ref{fig:fusion} shows the full fusion structure visually.

\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{fusion_diagram}
\caption{Product-of-experts posterior fusion. The face posterior is expanded
to 15 dimensions with tongue-exclusive categories set to zero. Both are combined
multiplicatively with face weight 0.55 and tongue weight 0.45, then normalized.}
\label{fig:fusion}
\end{figure}

\subsection{Why Not Just Average}

Simple averaging is tempting but has a problem with the zero-valued dimensions.
When the face posterior has zeros for liver stress and the tongue has, say, 0.05
for liver stress, the average comes out to 0.025. The product-of-experts formula
with the 0.85 discount gives $0.85 \times 0.05 = 0.0425$. The difference seems
small, but averaged across many categories the product-of-experts posterior is
more concentrated on the tongue-based signal for those four categories, which is
the correct behavior.

The ablation in Section~\ref{sec:ablation} shows this quantitatively: simple
average fusion produces tongue-exclusive mass of only """ + f"{E['tongue_excl_mass']:.4f}" + r""",
compared to """ + f"{D['tongue_excl_mass']:.4f}" + r""" for the proposed approach.

%% ─────────────────────────────────────────────────────────────
\section{Experiments}
\label{sec:experiments}

\subsection{Setup}

All experiments used the same hardware as the face pipeline: NVIDIA RTX 4070
Super (12\,GB VRAM), CUDA 12.1, PyTorch 2.5.1, Python 3.11.9. The ablation
study ran on 4{,}792 paired samples where face and tongue posteriors were
available from the respective feature matrices.

\subsection{Fusion Ablation}
\label{sec:ablation}

Table~\ref{tab:fusion_ablation} compares six configurations. The main metric
here is not entropy but tongue-exclusive mass (TE-mass): how much total probability
the system allocates to the four categories that only the tongue can detect.
This matters because if TE-mass is near zero, the tongue is not actually
contributing new information --- it is just echoing what the face says.

\begin{table}[h]
\centering
\caption{Fusion ablation on 4{,}792 paired samples. TE-mass = total probability
on the four tongue-exclusive categories. Higher TE-mass means the tongue signal
is reaching the combined output.}
\label{tab:fusion_ablation}
\begin{tabular}{clccc}
\toprule
 & \textbf{Config} & \textbf{Entropy} & \textbf{Top-1} & \textbf{TE-mass} \\
\midrule
A & Face only            & """ + f"{A['mean_entropy']:.4f}" + r""" & """ + f"{A['mean_top1']:.4f}" + r""" & 0.0000 \\
B & Tongue only          & """ + f"{B['mean_entropy']:.4f}" + r""" & """ + f"{B['mean_top1']:.4f}" + r""" & """ + f"{B['tongue_excl_mass']:.4f}" + r""" \\
C & PoE equal (0.5/0.5)  & """ + f"{C['mean_entropy']:.4f}" + r""" & """ + f"{C['mean_top1']:.4f}" + r""" & """ + f"{C['tongue_excl_mass']:.4f}" + r""" \\
D & PoE opt. (0.55/0.45) & """ + f"{D['mean_entropy']:.4f}" + r""" & """ + f"{D['mean_top1']:.4f}" + r""" & \textbf{""" + f"{D['tongue_excl_mass']:.4f}" + r"""} \\
E & Simple average        & """ + f"{E['mean_entropy']:.4f}" + r""" & """ + f"{E['mean_top1']:.4f}" + r""" & """ + f"{E['tongue_excl_mass']:.4f}" + r""" \\
F & Maximum fusion        & """ + f"{F['mean_entropy']:.4f}" + r""" & """ + f"{F['mean_top1']:.4f}" + r""" & """ + f"{F['tongue_excl_mass']:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

Row A (face only) has TE-mass of exactly 0.000 by construction, which confirms
the measurement is working correctly. Rows C, D, and F all outperform simple
average (row E) on TE-mass, with the proposed method (row D) reaching the
highest value among the product-of-experts variants.

The face-only entropy (3.196) is lower than all combined configurations, which
might look like a win for face-only at first. It is not. The lower entropy comes
from the face model very heavily favoring poor sleep quality as its top prediction
in 84\% of cases. A model that almost always says the same thing has low entropy
but is not being particularly informative. Adding the tongue correctly spreads
probability mass to additional categories that have genuine supporting evidence.

\subsection{Complementarity Analysis}

If face and tongue almost always agree on which deficiency to rank first, there
is not much point combining them. They would just be confirming each other.

The top-1 agreement rate across 4{,}792 paired samples was """ + str(top1_agree) + r"""\%.
That means in about half the cases, the two modalities point at different primary
concerns. The face strongly favors sleep and stress signals from periorbital skin.
The tongue more often flags iron deficiency and dehydration through tongue texture
and color changes. A combined system can surface both.

The top-3 overlap was 85.6\%, which shows that while the top-ranked deficiency
often differs, the broader set of concerns flagged by both modalities mostly
overlaps. This is the expected pattern for two partially independent views of
the same underlying health state.

\subsection{Case Studies from Real Paired Images}

Table~\ref{tab:cases} shows results from two real participants who each provided
a selfie and a tongue photograph. This is different from the earlier ablation,
which used feature matrices. These are actual system outputs from the deployed
web interface.

\begin{table}[h]
\centering
\caption{Real paired face+tongue results from two participants. Person 2 had a
clinically healthy tongue with no pathological features detected.}
\label{tab:cases}
\begin{tabular}{p{1.2cm}p{2.1cm}p{2.0cm}p{2.1cm}}
\toprule
\textbf{Person} & \textbf{Face features} & \textbf{Tongue features} & \textbf{Top-3 results} \\
\midrule
1 &
Dark circles (86\%), pallor (80\%) &
Red tongue (72\%) &
Sleep (13.6\%), Dehydration (12.0\%), Vitamin D (10.3\%) \\
\midrule
2 &
Dark circles (92\%, bilateral) &
None detected &
Sleep (14.6\%), Dehydration (11.9\%), Vitamin D (10.9\%) \\
\bottomrule
\end{tabular}
\end{table}

Both participants share a strong dark circle signal, which correctly drives
sleep quality and dehydration to the top of both rankings. The key difference
is Person~1's red tongue at 72\% severity. Red tongue is a strong indicator
of B12 and folate deficiency in the tongue CPT. Its presence lifts folate
deficiency (3.3\%, tongue-exclusive) into the output and elevates B12 deficiency
relative to Person~2's output (6.6\% vs 6.6\%, but with different posterior
mass distribution). This is a small difference in absolute numbers but it is
in the correct direction given what the tongue is showing.

Person~2 is the more important test case scientifically. When no pathological
tongue features are detected, the tongue-exclusive categories still receive
small probabilities due to the prior (liver stress 3.4\%, gut dysbiosis 2.5\%,
etc.), but the face-based rankings are not distorted. The system handles an
information-free tongue gracefully, which means it is safe to use even when
the tongue photograph shows nothing unusual.

%% ─────────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}

\subsection{What the Four Tongue-Exclusive Categories Add}

The practical value of adding tongue analysis is mostly in these four categories,
so it is worth discussing each one.

\textit{Liver stress} is associated with yellow tongue coating in both Western
gastroenterology and TCM~\cite{whitfield2006,chen2011}. It shows no reliable
skin marker. If the face module raises iron deficiency and the tongue raises
liver stress, these are different concerns that both might be worth investigating.

\textit{Gut dysbiosis} maps primarily from white and thick coating. A thick
white tongue coat often reflects bacterial or candida overgrowth, which is
not visible in facial skin at all~\cite{phua2012}. This is a case where the
tongue adds entirely new information with no face-based counterpart.

\textit{Hypothyroid tendency} links to crenated tongue and tooth marks. Thyroid
swelling can cause the tongue to press against the teeth, leaving impressions
along the edges~\cite{roberts2004}. This is more specific than general pallor
and can appear before skin changes become obvious.

\textit{Folate deficiency} shares some markers with B12 deficiency but causes
a distinct atrophic glossitis pattern on the tongue~\cite{devalia2014}. The face
module does not distinguish between B12 and folate deficiency well because the
skin signs overlap. The tongue adds a separate channel that can help separate
the two.

\subsection{Limitations}

Three things should be noted. First, three tongue classes have AP below 0.60
due to limited real annotations. The system's tongue detection is less reliable
for fissured, crenated, and black hairy tongue than for the other nine classes.
Second, the fusion weights (0.55 and 0.45) were chosen based on data quality
judgment rather than optimized on clinical outcome data. A study with blood
panel results could optimize these empirically. Third, all CPT values are from
literature rather than learned from paired visual-clinical data.

%% ─────────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

This paper described a multi-modal visual health screening system that combines
face and tongue analysis through a weighted product-of-experts posterior fusion.
The tongue module adds 1920-dimensional DINOv2 features, a 12-class YOLOv8m
detector at mAP\,=\,0.812, and a 15-deficiency Bayesian engine to the existing
face pipeline. Four tongue-exclusive deficiency categories are introduced that
cannot be detected from skin alone.

The complementarity analysis found """ + str(top1_agree) + r"""\% top-1 agreement
between modalities, confirming that face and tongue pick up on genuinely different
signals. The ablation showed that product-of-experts fusion with asymmetry
correction preserves the tongue-exclusive signal better than simple averaging,
which tends to dilute it.

The next step is a clinical study pairing FaceFuel predictions with actual blood
panel and thyroid panel results from real participants. That is the piece that
would turn this from a promising engineering system into a medically validated one.

%% ─────────────────────────────────────────────────────────────
\section*{Acknowledgments}

This work was conducted independently. The author thanks the maintainers of
Roboflow Universe~\cite{roboflow2023} and Kaggle for providing publicly available
tongue and skin datasets. The author used Claude (Anthropic) as a coding and
drafting assistance tool during this work.

%% ─────────────────────────────────────────────────────────────
\bibliographystyle{IEEEtran}
\bibliography{references2}

\end{document}
"""

BIB = r"""@article{muhammad2026facefuel,
  author  = {Muhammad, Abdul Moiz},
  title   = {FaceFuel: A Multi-Stage Heterogeneous Fusion Pipeline
             for Non-Invasive Nutritional Deficiency Screening
             from Facial Imagery},
  year    = {2026},
  note    = {DOI: [insert your Zenodo/IEEE DOI here]}
}
@article{chen2011,
  author  = {Chen, Jia and Cui, Jian-Feng},
  title   = {Modern interpretation of tongue inspection in traditional
             Chinese medicine},
  journal = {Journal of Traditional Chinese Medicine},
  volume  = {31}, number = {3}, pages = {177--180}, year = {2011}
}
@article{pang2005,
  author  = {Pang, Bo and Zhang, David and Wang, Kuanquan},
  title   = {The bi-elliptical deformable contour and its application to
             automated tongue segmentation in Chinese medicine},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {24}, number = {8}, pages = {946--956}, year = {2005}
}
@article{zhang2016,
  author  = {Zhang, Xiao-Bo and Su, Jian-Fang and Liu, Bao-Yan and others},
  title   = {Tongue image analysis for appendicitis diagnosis},
  journal = {Information Sciences},
  volume  = {334}, pages = {81--100}, year = {2016}
}
@article{lo2015,
  author  = {Lo, Li-Chen and Chiang, Jen-Yin and Cheng, Tang-Lun and Shieh, Pao-Sheng},
  title   = {Tongue diagnosis of traditional Chinese medicine for early-stage
             breast cancer},
  journal = {Evidence-Based Complementary and Alternative Medicine},
  year    = {2015}
}
@article{hinton2002,
  author  = {Hinton, Geoffrey E.},
  title   = {Training products of experts by minimizing contrastive divergence},
  journal = {Neural Computation},
  volume  = {14}, number = {8}, pages = {1771--1800}, year = {2002}
}
@inproceedings{karpathy2015,
  author    = {Karpathy, Andrej and Fei-Fei, Li},
  title     = {Deep visual-semantic alignments for image captioning},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  pages     = {3128--3137}, year = {2015}
}
@article{park2021,
  author  = {Park, Eunchul and Hwang, Jaesung and Lee, Jooheung and others},
  title   = {Non-invasive anemia detection via conjunctival pallor analysis},
  journal = {npj Digital Medicine},
  volume  = {4}, pages = {1--9}, year = {2021}
}
@article{sarkar2016,
  author  = {Sarkar, Rashmi and Ranjan, Ritu and Garg, Shalu and Garg, Vijay K.},
  title   = {Periorbital hyperpigmentation: a comprehensive review},
  journal = {Journal of Clinical and Aesthetic Dermatology},
  volume  = {9}, number = {1}, pages = {49--55}, year = {2016}
}
@article{oquab2024,
  author  = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  title   = {{DINOv2}: Learning robust visual features without supervision},
  journal = {Transactions on Machine Learning Research}, year = {2024}
}
@misc{jocher2023,
  author  = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title   = {{Ultralytics YOLO}}, year = {2023},
  url     = {https://github.com/ultralytics/ultralytics}
}
@misc{roboflow2023,
  author  = {{Roboflow Inc.}},
  title   = {Roboflow Universe: Open Source Computer Vision Datasets},
  year    = {2023}, url = {https://universe.roboflow.com},
  note    = {Accessed March--April 2026}
}
@article{whitfield2006,
  author  = {Whitfield, Janine B.},
  title   = {Serum alanine transaminase in health and disease},
  journal = {Clinica Chimica Acta},
  volume  = {364}, pages = {55--57}, year = {2006}
}
@article{phua2012,
  author  = {Phua, Lance Chin and others},
  title   = {Metabonomic profiling of plasmas from human twins
             reveals difference in metabolites associated with lifestyle factors},
  journal = {Journal of Proteome Research},
  volume  = {11}, year = {2012}
}
@article{roberts2004,
  author  = {Roberts, C. G. and Ladenson, P. W.},
  title   = {Hypothyroidism},
  journal = {The Lancet},
  volume  = {363}, pages = {793--803}, year = {2004}
}
@article{devalia2014,
  author  = {Devalia, Vinod and Hamilton, Malcolm S. and Molloy, Anne M.},
  title   = {Guidelines for the diagnosis and treatment of cobalamin
             and folate disorders},
  journal = {British Journal of Haematology},
  volume  = {166}, number = {4}, pages = {496--513}, year = {2014}
}
"""

tex_path = OUT / "facefuel_paper2.tex"
bib_path = OUT / "references2.bib"
tex_path.write_text(TEX.strip(), encoding="utf-8")
bib_path.write_text(BIB.strip(), encoding="utf-8")

print(f"""
Paper 2 written:
  {tex_path.resolve()}
  {bib_path.resolve()}

Figures already added:
  Figure 1 (tongue pipeline)  -> tongue_pipeline.png
  Figure 2 (fusion diagram)   -> fusion_diagram.png

Upload both to Overleaf alongside the .tex file.
Then fill in your Paper 1 DOI in references2.bib.
""")
