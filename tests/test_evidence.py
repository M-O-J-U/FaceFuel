import numpy as np

from facefuel import evidence, schema
from facefuel.evidence import collect_features

FACE = schema.FACE_CLASSES


def sev(**kw):
    s = np.zeros(len(FACE), dtype=np.float32)
    for k, v in kw.items():
        s[FACE.index(k)] = v
    return s


UNC = np.full(len(FACE), 0.05, dtype=np.float32)


def test_mlp_alone_is_unconfirmed_not_evidence():
    feats, unconf = collect_features(FACE, schema.FACE_INACTIVE, sev(dark_circle=0.99), UNC, {}, {})
    assert feats == {} and unconf == {"dark_circle": 0.99}


def test_detector_plus_mlp_reports_mlp_severity():
    feats, _ = collect_features(FACE, schema.FACE_INACTIVE, sev(dark_circle=0.9), UNC,
                                {"dark_circle": 0.5}, {"dark_circle": 1})
    assert feats["dark_circle"]["severity"] == 0.9
    assert feats["dark_circle"]["detected_by"] == ["severity_mlp", "yolo"]


def test_detector_alone_reports_scaled_confidence():
    feats, _ = collect_features(FACE, schema.FACE_INACTIVE, sev(), UNC, {"wrinkle": 0.6}, {"wrinkle": 1})
    assert feats["wrinkle"]["severity"] == round(0.6 * evidence.YOLO_SCALE, 3)


def test_inactive_classes_never_reported():
    feats, unconf = collect_features(FACE, schema.FACE_INACTIVE, sev(blackhead=0.99), UNC,
                                     {"blackhead": 0.9}, {"blackhead": 5})
    assert "blackhead" not in feats and "blackhead" not in unconf


def test_small_lesion_count_raises_severity():
    feats, _ = collect_features(FACE, schema.FACE_INACTIVE, sev(), UNC,
                                {"dark_spot": 0.4}, {"dark_spot": 8}, small_lesion={"dark_spot"})
    assert feats["dark_spot"]["severity"] == 0.78
