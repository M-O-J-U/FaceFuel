# FaceFuel — Research Notes (October 2026)

Working notes for strengthening the next submission. Written after a full code
audit of the v4 system, so it starts with what the audit found, then covers data,
fusion methodology, related work, and a realistic route to clinical evidence.

Nothing here changes the scope of the project: FaceFuel is a **wellness-awareness
screening prototype**, not a diagnostic device, and no text derived from these
notes should claim otherwise.

---

## 1. What the v4 audit found (these belong in the next paper's "Limitations")

### 1.1 The v4 server never ran v4 face/tongue models
Before this audit, `server_v4.py` imported `step10_inference.py` (still the **v2**
face YOLO and v2 MLP) and `Phase7_tongue_inference.py` (whose v4 MLP path contained
a `\t` escape and could never load). All three endpoints returned HTTP 500. Any
qualitative "v4 outputs" observed before October 2026 came from the v2/v3 code paths.
The rebuilt `facefuel/` package is the first end-to-end v4 implementation.

### 1.2 The severity MLPs have never seen a healthy example
`5_*_features_v4.py` only extracted features from images that carry at least one
label, so every training row has ≥1 positive class. The MLPs learned *"which of these
conditions is this?"*, never *"is anything here?"*. `scripts/domain_shift_probe.py`
quantifies it (full table: [`domain_shift_probe.md`](domain_shift_probe.md)):

| | In-domain validation images | 9 real face photos (out-of-domain) |
|---|---|---|
| face `dark_circle` > 0.35 when absent | 0 % of negatives | **56 %** of photos |
| eye `conjunctivitis` > 0.35 when absent | 2 % of negatives | **44 %** of photos |
| eye `conjunctival_pallor` > 0.35 when absent | 0 % of negatives | **44 %** of photos |

In-domain separation is excellent, which is consistent with the published F1. On
selfies the outputs saturate. The detectors also over-fire out of domain (e.g. eye
YOLO `eyelid_drooping` at 0.46–0.94 on most faces). **Mitigation now in code:** MLP findings
must be confirmed by the detector (`facefuel/evidence.py: REQUIRE_DETECTOR`); MLP-only
signals are returned as `unconfirmed_signals` and never used as evidence. **Real fix:**
add negatives (§2.3) and retrain; re-run the probe as a before/after figure.

### 1.3 Reported MLP F1 is optimistic
`6_severity_mlp_v4.py` selected the best epoch on the same 15 % validation split it
then reported, with an unseeded split. The F1 values (0.901 / 0.761 / 0.985) are
therefore best-epoch validation scores, not held-out test scores, and are not exactly
reproducible. The split is now seeded; a proper fix is a train/val/test split
(ideally grouped by source dataset — see 1.4) and reporting test F1 with bootstrap CIs.

### 1.4 Likely source-dataset leakage and shortcut risk
Merged sets mix many sources (file prefixes `sdc_`, `dn_`, `xan_`, `muv_`, `tc_`…), and
several classes come almost entirely from one source (e.g. xanthelasma from `xan_`).
A model can learn *which dataset* an image came from rather than the condition. The
random image-level split does not detect this. Recommended: **leave-one-source-out**
evaluation per class, and report it next to the random split.

### 1.5 Tongue localisation
The tongue detector finds `tongue_body` in only 22 % of tongue validation images,
because most source datasets label the lesion but not the tongue. When no tongue is
found the server now analyses the whole photo (matching how MLP features were
extracted) and warns the user. Adding `tongue_body` boxes (or a segmentation model
such as TongueSAM-style SAM fine-tunes) is cheap and would help a lot.

### 1.6a Face labels were shifted during the v4 merge (found 2026-10-04)
Tracing every face image back to its source folder by exact file hash shows that,
for classification-folder sources (DermNet, skin-disease collections, augmented
dermoscopy, face-skin set), the merge wrote **shifted class ids**:

| Intended class (per `03_merge_face.py`) | Label actually written |
|---|---|
| `dark_spot` (melanoma, nevi, BCC, keratoses, warts — 13k+ images) | `eczema` |
| `redness` (vascular tumours, vasculitis, hives, exanthems) | `rosacea` |
| `acne` (DermNet acne & rosacea, facial acne) | `wrinkle` |
| `eczema`, `vitiligo`, `butterfly_rash`, `rosacea` (≈ 12k images) | **no label** — the image was kept as a YOLO background |

Only the `orig_` source (v1–v3 data with real bounding boxes) was labelled correctly.
Consequences: the v4 face per-class scores (eczema 0.973, rosacea 0.817…) describe
the wrong conditions, the "zero-data" classes acne / vitiligo / butterfly_rash in fact
had ≈ 1.3k / 1.1k / 0.5k images, and the detector was taught that 12k photos of
skin disease contain nothing. `pipeline/03b_build_clean_v5.py` relabels these by hash.

### 1.6b Train/validation leakage inflated every v4 score
Exact duplicate files appear in both splits: **56 % of eye validation images, 34 % of
tongue and 32 % of face** had a byte-identical copy in training — before counting
flipped/rotated augmentations, which `03b` also groups. Re-scored on a fair test split
(only v4-validation images that were never in v4 training), the v4 eye detector drops
from the reported **mAP50 0.993 to 0.829** (precision 0.755; drooping eyelid 0.747).
All published v4 numbers should be treated as optimistic until re-measured on the v5
test splits (`docs/model_comparison.md`).

### 1.7 Outcome of the v5 retraining (2026-10-05) — what changed in v4.2
`pipeline/03b_build_clean_v5.py` rebuilt all three datasets: it removed exact and
flip/rotation duplicates (38 % of eye, 37 % of tongue and 23 % of face images were
duplicates), relabelled 29,175 face images by source-folder hash, added 745 verified
normal-eye photos as negatives, and created a **held-out test split** made only of
v4-validation images that were never in v4 training. Old and new models were scored on
that split (`scripts/compare_models.py` → [`model_comparison.md`](model_comparison.md)):

| | Detector mAP50 old → new | Severity MLP mean F1 old → new | Adopted |
|---|---|---|---|
| Eye | 0.829 → **0.991** | 0.698 → **0.901**; normal eyes flagged 94 % → **6 %** | v5 detector + v5 MLP |
| Tongue | **0.841** → 0.804 | 0.691 → **0.701** | v4 detector kept + v5 MLP |
| Face | 0.139 → **0.672** | 0.213 → **0.778** | v5 detector + v5 MLP |

- The face gain mainly reflects **fixing the labels**: scored on corrected labels, the
  v4 face models were mostly wrong (e.g. dark spot 0.083 AP, because v4 had learned to
  call dark spots "eczema").
- Acne, vitiligo and butterfly rash are now trained (test AP 0.870 / 0.854 / 0.430);
  only blackhead lacks data.
- The tongue detector got worse when retrained on the deduplicated set (white coating
  0.71 → 0.53, lichen planus 0.67 → 0.38, both small classes), so v4 stays in service.
  Losing duplicates/augmentations cut the effective training data for rare classes.
- Still open: no healthy **face** photos (the v5 face MLP alone fires dark circle on 78 %
  of real faces, so the detector-confirmation rule stays), no tongue negatives, the
  test split is small for some classes (scleral icterus n = 3), and these are still
  image-level scores on curated photos, not clinical accuracy.

### 1.6 Per-modality scoring changed in v4.1
v1–v3 used hand-set conditional probability tables (CPTs) and priors per modality —
the "hand-built Bayesian priors" JBHI objected to. v4 classes no longer match those
CPT rows, so v4.1 uses one transparent rule for all three modalities: each
deficiency's evidence is the strongest severity among detected features that map to
it (`schema.FEAT_DEF`), normalised per modality, then fused with the same
product-of-experts weights. It is still expert-specified, but it has far fewer
invented numbers (a mapping table instead of 11×11 + 11×15 probability tables plus
priors), and a modality with no findings now reports nothing instead of its prior.
State this change explicitly in the next paper.

---

## 2. Filling the zero-data classes

Current inactive classes (kept in the schema so YOLO class ids stay aligned, but
excluded from inference and coverage): face `blackhead`; tongue `no_coating`,
`purple_tongue`, `angular_stomatitis`, `median_rhomboid`. (Acne, vitiligo and
butterfly_rash were recovered in v4.2 from data already on disk — §1.7. The rows below
for those classes now describe *better* data, not missing data.) **Verify every licence before training on or redistributing data.**

| Class | Candidate source | Labels | Licence (as found) | Notes |
|---|---|---|---|---|
| purple_tongue, no_coating (peeled) | **TCM-Tongue** (2025), 6,719 images, 20 categories incl. *zishe* (purple) and *botaishe* (peeling coating) — [arXiv 2507.18288](https://arxiv.org/abs/2507.18288), [GitHub](https://github.com/m28805746-max/Intelligent-tongue-diagnosis-detection-dataset) | Bounding boxes (COCO/YOLO/VOC), practitioner-verified | arXiv page shows CC BY 4.0; GitHub states none — confirm with authors | **Best single fix for tongue.** Also has a *healthy tongue* class, which provides the negatives the tongue MLP lacks (§1.2). Map classes carefully (TCM "red/crimson" vs our `red_tongue`). |
| acne (+ lesion counts) | **ACNE04** (Wu et al., ICCV 2019), 1,457 images, 18,983 dermatologist boxes — [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.html); cleaned re-annotation [acne04v2](https://github.com/AIpourlapeau/acne04v2) | Boxes + severity grade | Not stated — research-use; ask authors | Boxes known to be loose/merged; prefer acne04v2. Fits `count_to_severity` directly. |
| acne, eczema, vitiligo (weak labels) | **SCIN** (Google + Stanford, 2024), >10k consented, self-contributed photos — [GitHub](https://github.com/google-research-datasets/scin), [blog](https://research.google/blog/scin-a-new-resource-for-representative-dermatology-images/) | Image-level dermatologist differentials, Fitzpatrick + Monk skin tone | CC BY 4.0 (per dataset paper) | Smartphone photos of real people — **the closest match to FaceFuel's input domain** and diverse in skin tone. Image-level only: good for MLP training and for out-of-domain evaluation, not YOLO boxes. |
| vitiligo | Roboflow Universe vitiligo detection sets (several CC BY 4.0); Mendeley "Dermatological Images for Vitiligo" (10,459 images incl. 4,127 healthy skin) — [Mendeley](https://data.mendeley.com/datasets/wbvnkmbxbf) | Boxes (Roboflow) / image-level (Mendeley) | CC BY 4.0 (Roboflow, per listing); Mendeley — check | Healthy-skin images double as negatives. Many are body, not face, close-ups. |
| butterfly_rash | Fitzpatrick17k lupus classes (atlas images) — [repo](https://github.com/mattgroh/fitzpatrick17k) | Image-level | Non-commercial (verify) | Very few facial malar-rash images exist publicly. Consider **dropping this class** rather than training on a handful of atlas photos. |
| angular_stomatitis, median_rhomboid | Ataturk University cohort (623 patients, DOI 10.1186/s12880-024-01234-3; contact omiloglu@hotmail.com) | Clinical | Request access | Still the only clinical source found; no public alternative located. |
| blackhead | Roboflow acne-type detection sets with comedone classes | Boxes | Varies | Low priority: weak link to any nutritional dimension. |

Recommendation: make TCM-Tongue + SCIN the v5 data additions. TCM-Tongue fills two
tongue classes and adds healthy negatives; SCIN brings real smartphone photos across
skin tones. Formally retire `butterfly_rash` and `blackhead` unless good data appears.
Each is a one-line change in `facefuel/schema.py` plus a retrain.

### 2.3 Negatives (most important data change)
Whatever else is added, add **no-condition images** for every modality and train the
MLPs with all-zero label rows. Put them in the dataset as background images (empty
label files); `pipeline/05_extract_features.py` turns those into negatives, and YOLO
uses them as background. The v5 eye set already includes 745 verified normal-eye photos
(`pipeline/03b_build_clean_v5.py`). Further sources: TCM-Tongue healthy tongues; healthy-skin sets above;
non-anaemic conjunctivas from the anaemia datasets in §4.2. A generic face dataset
(e.g. FFHQ, CC BY-NC-SA 4.0) could serve as weak "presumed-healthy" face negatives,
but should be described as such.

---

## 3. Fusion: from hand-set weights to learned, calibrated fusion

JBHI's methods objection was partly that the 0.40/0.35/0.25 weights are not learned.
Options, roughly in order of effort:

1. **Calibrate each modality first.** Temperature or Platt scaling per modality (and
   per class) on a held-out split, reporting reliability diagrams and ECE. A product
   of uncalibrated experts is hard to defend; a product of calibrated ones is the
   textbook naïve-Bayes combination.
2. **Learn the PoE weights.** Treat the weights as parameters and fit them by
   maximising held-out log-likelihood against whatever outcome labels exist — the
   same functional form, now estimated rather than chosen. Report the fitted weights
   with bootstrap CIs, and compare against equal weights in an ablation.
3. **Logistic-regression stacking with missing-modality indicators.** Combine
   per-modality logits with a logistic regression; missing modalities enter as
   binary side information. This handles face-only / tongue-only / tri-modal inputs
   in one model, and it generalises PoE, which is the special case under conditional
   independence. See the late-fusion work in [van Hout et al., ICASSP 2014](https://pure.uva.nl/ws/files/2652792/167625_vanHoutICASSP2014.pdf)
   and [probabilistic ensembling for multimodal detection](https://arxiv.org/pdf/2104.02904);
   for medical data with block-missing modalities, [MSB stacking](https://arxiv.org/pdf/2605.25050).
4. **Uncertainty-aware outputs.** Conformal prediction sets give coverage guarantees
   that a reviewer can check, and recent work extends them to missing modalities
   ([Conformal Fusion Under Missing Modalities](https://arxiv.org/pdf/2608.07183)).
   "These areas are plausible at 90 % coverage" is a more honest output than a ranked
   list of normalised scores.

The catch: methods 2–4 need **outcome labels at the deficiency level** (lab values),
not visual-feature labels. That is the clinical-evidence gap again (§4). Until then,
the honest framing is: calibrated per-modality detectors (doable now), with fusion
weights as a documented, ablated design choice.

---

## 4. Closing the clinical-evidence gap without a full trial

### 4.1 Reframe the claim so it can be tested
"Detects iron deficiency from a selfie" cannot be validated with available data.
"The conjunctival-pallor signal correlates with laboratory haemoglobin" can. Validate
**links in the chain** (visual feature ↔ lab value) where public paired data exists,
and present the deficiency layer as hypothesis-generating.

### 4.2 Public datasets that pair photos with lab values
| Dataset | Pairs | Use for FaceFuel |
|---|---|---|
| **Girija et al. — Non-invasive anaemia screening** ([IEEE DataPort, DOI 10.21227/n7km-nr64](https://ieee-dataport.org/documents/dataset-non-invasive-anemia-screening-using-conjunctiva-palm-and-nail-bed-images-0)): 1,485 participants, conjunctiva + palm + nail-bed photos, **lab Hb 6.0–18.7 g/dL**, 166 anaemic (WHO Hb < 12) | Photo ↔ Hb | **External validation of the eye module's `conjunctival_pallor`** against lab Hb: AUC for anaemia, Spearman ρ with Hb, calibration. The single most valuable next experiment. |
| **Eyes-defy-anemia** ([IEEE DataPort](https://ieee-dataport.org/documents/eyes-defy-anemia)), 218 conjunctiva photos, Italy + India, with Hb | Photo ↔ Hb | Already used in training — do **not** reuse for validation; keep it on the training side and validate on Girija. |
| **CP-AnemiC** ([UTS](https://opus.lib.uts.edu.au/handle/10453/177704)), conjunctival pallor in children (Hb labels — verify) | Photo ↔ Hb | Paediatric external test set (report separately). |
| Peru anaemia dataset ([RCSI](https://revistas.unsm.edu.pe/index.php/rcsi/article/view/955)): 909 nail photos + palm/fingertip videos with Hb | Photo ↔ Hb | Out of FaceFuel's modalities; cite as related. |
| Scleral jaundice cohorts — BiliScreen (70 people, r = 0.89 with serum bilirubin), [Ajou smartphone study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8466674/) (130 patients) | Photo ↔ bilirubin | Not public; cite as the evidence standard for `scleral_icterus`, and request data from the authors. |

### 4.3 A minimal volunteer validation protocol (feasible at COMSATS)
- **Design:** prospective, cross-sectional diagnostic-accuracy study, reported to
  [TRIPOD+AI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/) and STARD 2015.
  Get institutional ethics approval and written consent first.
- **Participants:** adult volunteers (students and staff), with no exclusion by skin tone.
- **Index test:** FaceFuel on a standardised selfie and tongue photo. Take 3 phones ×
  2 lighting conditions per person to measure robustness, and freeze the model
  version before collection begins.
- **Reference standard, same day:** CBC (Hb), serum ferritin, B12, 25(OH)D and a lipid
  panel. This is a common, inexpensive panel at Pakistani diagnostic labs.
- **Sample size:** anaemia is common in Pakistani women of reproductive age (national
  surveys report around 40 %), so 150–200 participants should give an AUC estimate
  with useful precision for the anaemia endpoint. Compute this formally (e.g. Riley et al.
  sample-size criteria) before recruiting.
- **Pre-registered primary endpoint:** AUC of conjunctival-pallor severity for anaemia
  (Hb < 12 g/dL women, < 13 men). Secondary endpoints: ferritin, B12, vitamin D vs
  the corresponding fused scores; calibration; subgroup results by sex and skin tone.
- **Also report the null results.** For a screening-awareness tool, a credible
  negative result on vitamin D is publishable and builds trust.

Even n ≈ 150 with real lab values would directly answer the "no clinical evidence"
desk-rejection. Combined with the public-data validation in §4.2, it supports a
much stronger paper than further dataset scaling, which v4 showed does not address the
objection.

### 4.4 Weak supervision and synthetic data, used carefully
Reviewers accept these for **augmenting visual-feature training** (e.g. diffusion
augmentation of rare classes, evaluated on real-only test sets — see
[latent diffusion augmentation in dermatology](https://arxiv.org/pdf/2308.12453)),
but not as substitutes for outcome labels. Never evaluate on synthetic images, and
report real-only test performance.

---

## 5. Related work to position against

| Work | Modality ↔ ground truth | Relevance |
|---|---|---|
| Mannino et al., *Nature Communications* 2018 — smartphone fingernail photos vs CBC Hb (n = 337; ±2.4 g/dL, sensitivity up to 97 %) — [news](https://www.news.gatech.edu/2018/12/10/no-bleeding-required-anemia-detection-smartphone) | Nail ↔ Hb | The canonical "selfie vs blood test" validation design to emulate |
| Mariakakis et al., BiliScreen, *IMWUT* 2017 — [project](https://ubicomplab.cs.washington.edu/publications/biliscreen) | Sclera ↔ serum bilirubin | Evidence bar for `scleral_icterus` |
| AITongue — prospective multicentre tongue-image gastric-cancer models, *eClinicalMedicine* 2023 (937 cases / 1,911 controls, 10 centres; external AUC 0.83–0.88) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9941057) | Tongue ↔ endoscopy/pathology | Shows tongue imaging can pass clinical review when validated prospectively and externally |
| TCM-Tongue dataset, 2025 — [arXiv](https://arxiv.org/abs/2507.18288) | Tongue ↔ practitioner labels | Data source and detection benchmark |
| Lin et al., *European Heart Journal* 2020 — facial photos for coronary artery disease (AUC ≈ 0.73–0.78) | Face ↔ angiography | Face-to-systemic-disease precedent, with a clinical reference standard |
| AI facial analysis in healthcare review, *Patterns* 2025 — [Cell](https://www.cell.com/patterns/pdf/S2666-3899(25)00023-6.pdf) | Review | Survey to cite for scope and ethics |
| SCIN, 2024 — [GitHub](https://github.com/google-research-datasets/scin) | Consented smartphone skin photos | Representative-data standard, skin-tone reporting |

**Positioning:** FaceFuel's distinct contribution is **tri-modal fusion with explicit
provenance and missing-modality handling** for wellness screening, not single-
condition accuracy. Single-modality, single-endpoint studies such as Mannino and
BiliScreen set the evidence bar. The next paper should bring at least one FaceFuel
modality up to that bar (conjunctival pallor ↔ Hb on Girija et al.) and present fusion
as the framework that extends it.

---

## 6. Suggested next-paper plan (ordered by evidence per hour)

1. Run the eye module on Girija et al. and report AUC / ρ / calibration vs lab Hb, as external validation.
2. ~~Add negatives and retrain~~ — **done for eye in v4.2** (normal eyes flagged 94 % → 6 %). Still needed for face (healthy faces) and tongue (TCM-Tongue healthy class).
3. ~~Held-out test split~~ — **done in v4.2**. Still to add: leave-one-source-out evaluation and bootstrap CIs.
4. Calibrate each modality, learn the PoE weights on whatever labelled outcome data exists, and ablate them against the hand-set weights.
5. Add TCM-Tongue classes and retire the classes that have no data.
6. Run the volunteer protocol (§4.3) as the clinical-evidence centrepiece.
7. Target venues that value validation studies (e.g. *JMIR mHealth and uHealth*, *npj Digital Medicine*, *IEEE JBHI* resubmission), and report to TRIPOD+AI.
