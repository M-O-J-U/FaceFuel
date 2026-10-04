"""
Per-feature evidence: combine YOLO detections with severity-MLP output into one
reported finding per visual feature. Shared by all three modalities so the rules
are identical everywhere.
"""
import numpy as np

from .models import count_to_severity, level

# MLP "present" threshold: the decision threshold the v4 MLPs were validated at
# (their reported F1 scores are computed at pred > 0.35).
MLP_PRESENT = 0.35
YOLO_SCALE  = 0.85     # a YOLO confidence counts slightly less than a severity

# The severity MLPs grade features the DETECTOR found; they do not report on their
# own. Reason: the v4 MLPs were trained only on images containing at least one
# labelled condition (no healthy negatives), so on real selfies they saturate —
# e.g. dark_circle ≈ 1.0 on nearly every face (scripts/domain_shift_probe.py).
# MLP-only signals are returned separately as unconfirmed and are not evidence.
REQUIRE_DETECTOR = True


def collect_features(classes, inactive, severity, uncertainty, yolo_conf, yolo_count,
                     small_lesion=(), skip=("tongue_body",), require_yolo=()):
    """
    Returns (features, unconfirmed):
      features    {feature: {severity, level, confidence, yolo_count, detected_by}}
                  for every ACTIVE class with enough evidence;
      unconfirmed {feature: mlp_severity} — MLP-only signals the detector did not
                  confirm (reported for transparency, never used as evidence).
    Inactive classes are never reported. `require_yolo` names classes the MLP may
    never report alone even if REQUIRE_DETECTOR is switched off (scleral_icterus).
    """
    out, unconfirmed = {}, {}
    for i, name in enumerate(classes):
        if name in skip or name in inactive:
            continue
        s, u = float(severity[i]), float(uncertainty[i])
        yc, cnt = float(yolo_conf.get(name, 0.0)), int(yolo_count.get(name, 0))
        mlp_hit = s >= MLP_PRESENT
        if yc == 0.0 and (REQUIRE_DETECTOR or name in require_yolo):
            if mlp_hit:
                unconfirmed[name] = round(s, 3)
            continue
        if not mlp_hit and yc == 0.0:
            continue
        cnt_sev = count_to_severity(cnt) if name in small_lesion else 0.0
        combined = max(s if mlp_hit else 0.0, yc * YOLO_SCALE, cnt_sev)
        by = []
        if mlp_hit: by.append("severity_mlp")
        if yc > 0:  by.append(f"yolo(x{cnt})" if cnt > 1 else "yolo")
        out[name] = {
            "severity":    round(combined, 3),
            "level":       level(combined),
            # MC-dropout agreement for MLP findings; detector confidence otherwise
            "confidence":  round(float(np.exp(-u * 3)) if mlp_hit else yc, 2),
            "yolo_count":  cnt,
            "detected_by": by,
        }
    return out, unconfirmed


def evidence_scores(modality_features: dict, feat_def: dict) -> dict:
    """
    deficiency → evidence score in [0,1]: the strongest severity among the
    detected features that map to it. Only deficiencies with evidence appear.
    """
    scores = {}
    for feat, data in modality_features.items():
        for d in feat_def.get(feat, []):
            scores[d] = max(scores.get(d, 0.0), float(data["severity"]))
    return scores
