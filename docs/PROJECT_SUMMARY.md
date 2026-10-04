# FaceFuel — Project Summary

*Tri-modal computer vision for wellness screening from smartphone photos*
**Abdul Moiz Muhammad (MOJU)**, independent researcher, COMSATS University Islamabad, Wah Cantt, Pakistan
2026 · Three preprints on Zenodo · Code: [github.com/M-O-J-U/FaceFuel](https://github.com/M-O-J-U/FaceFuel) · [Promo video](https://youtu.be/_RNtt8QNQOs)

---

## The question

Most people only get a blood panel when something is already wrong. Yet the face, eyes
and tongue often carry visible hints of what is happening inside the body. Clinicians have long
looked at a pale inner eyelid when they suspect anaemia, yellowed whites of the eyes when they
suspect liver trouble, and a smooth red tongue when they suspect low B12 or folate.

FaceFuel asks a narrow, testable question: **can ordinary smartphone photos surface those
visual hints reliably enough to nudge someone toward a proper check-up?** It is deliberately
a wellness-awareness tool, not a diagnostic device, and every output says so.

## What I built

A five-stage pipeline, repeated for three "visual channels" and then fused:

1. **Locate:** MediaPipe aligns the face from 478 landmarks; the eye band is cut from the aligned face; the tongue comes from a second photo.
2. **Detect:** a YOLO11m detector per channel marks visible signs: 6 face classes, 6 eye classes and 14 tongue classes are active.
3. **Describe:** DINOv2 (ViT-S/14) embeds fixed anatomical regions such as the cheeks, sclera and tongue zones.
4. **Grade:** a small multi-head network grades each detected sign. It is run 20 times with dropout (Monte Carlo dropout), so it reports its own uncertainty.
5. **Fuse:** signs map onto a 22-area nutrition and lifestyle framework, and the three channels are combined with a weighted product of experts (face 0.40, tongue 0.35, eye 0.25). The system tracks where every conclusion came from. An area that none of the channels used can see is reported as *not assessed* rather than guessed.

It ships as a FastAPI service with a mobile-friendly web interface. On a consumer RTX 4070 Super
a full three-channel analysis takes about 0.2 seconds, and all models together use less than 2 GB of GPU memory.

## How it evolved

| Stage | What changed | Key numbers |
|---|---|---|
| **Paper 1 — Face** ([10.5281/zenodo.19394708](https://doi.org/10.5281/zenodo.19394708)) | Selfie → 11 skin features → Bayesian engine over 11 deficiency categories | YOLOv8m mAP@0.5 0.790 on 5,721 images; mean F1 0.677; 58 ms per image. Ablations: region-aware DINOv2 +0.149 F1 over whole-face features. |
| **Paper 2 — Face + tongue** ([10.5281/zenodo.19411317](https://doi.org/10.5281/zenodo.19411317)) | Added a tongue pipeline and product-of-experts fusion; four categories visible only from the tongue | Tongue mAP 0.812 on 9,125 images; face and tongue agree on the top category only 48.9 % of the time, so they carry different information |
| **Paper 3 — Tri-modal** ([10.5281/zenodo.19468059](https://doi.org/10.5281/zenodo.19468059)) | Added an eye channel from the same selfie (pallor, yellow sclera, xanthelasma); face moved to YOLO11m | Eye mAP 0.913; face mAP 0.790 → 0.872; < 235 ms end to end |
| **v4 — Scale** (unpublished) | Re-collected and merged much larger datasets; retrained all three detectors and graders | Eye mAP 0.993, tongue 0.871, face 0.559 on new, larger validation sets; grader F1 0.985 / 0.761 / 0.901 |
| **v4.1 — Engineering audit** (Oct 2026) | Rebuilt the inference layer as one package with shared, tested components; new web interface; deployment tooling | All endpoints working; 21 automated tests; train/inference feature parity verified (cosine ≥ 0.99999) |

## The rejection, and what it taught me

I submitted Papers 1 and 2 to *IEEE Journal of Biomedical and Health Informatics*, and both
were desk-rejected. The reasons were fair: the biomedical contribution was not clear enough,
the methods rigour was below the bar, and, centrally, **there was no clinical evidence**.
No prediction had been checked against a blood test, and the probabilistic engine relied on
priors I had set by hand.

My first response (v4) was to scale up the data. That made the detectors better on their own
benchmarks, but it could not answer the objection, because more labelled *photos* are still not
*clinical ground truth*. A later audit of my own code made the point sharper:

- The v4 server had never run the v4 face and tongue models: every endpoint was failing, and older models sat underneath.
- The grading networks had only been trained on images containing a condition, never a healthy one, so on real selfies they over-report. I measured this and published the probe ([domain_shift_probe.md](domain_shift_probe.md)).
- The reported grader F1 scores were chosen and reported on the same validation split.

I fixed what engineering can fix: one source of truth for every class list and file path,
graders that only grade what the detector confirms, full provenance in every result, and
regression tests. I documented the rest as limitations rather than hiding them. The lesson I
took from it: **in health AI, the hard part is not the model, it is the evidence.**

## Where it goes next

The plan, detailed in [RESEARCH_NOTES.md](RESEARCH_NOTES.md), puts clinical evidence ahead of more model work:

1. **External validation against lab values.** A public dataset of 1,485 people pairs eye photos with measured haemoglobin. It is the right first test of whether the pale-eyelid signal tracks real anaemia.
2. **Add healthy examples** to every channel, retrain, and show the over-reporting fall.
3. **Calibrate and learn the fusion,** replacing the hand-set weights with fitted, reported ones.
4. **A small volunteer study** of roughly 150–200 adults with same-day blood panels (CBC, ferritin, B12, vitamin D, lipids), pre-registered and reported to TRIPOD+AI, including whatever comes out negative.

## What this project demonstrates

- **End-to-end ML engineering:** data collection and merging across dozens of public sources, detector and multi-head regressor training, uncertainty estimation, multi-modal fusion, API and frontend, and containerised deployment.
- **Research practice:** three preprints, ablation studies, and honest handling of a rejection, including finding and documenting flaws in my own pipeline.
- **Responsible framing:** explicit provenance, "not assessed" instead of guesses, and a disclaimer on every surface.

*FaceFuel is a research prototype for wellness awareness. It is not a medical device and has not been clinically validated.*
