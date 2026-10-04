"""
Fast, model-free tests for the schema and fusion logic.
Run:  python -m pytest tests -q
"""
import math

import pytest

from facefuel import fusion, schema

TONGUE_ONLY = {"folate_deficiency"}   # covered by no active face/eye class


def res(modality, **feats):
    return {"modality": modality,
            "features": {f: {"severity": s, "level": "high", "confidence": 1.0,
                             "yolo_count": 1, "detected_by": ["yolo"]} for f, s in feats.items()}}


# ── schema invariants ────────────────────────────────────────────────────────
def test_feature_maps_only_use_known_deficiencies():
    known = set(schema.ALL_DEFS)
    for mod, fmap in schema.FEAT_DEF.items():
        for feat, defs in fmap.items():
            assert set(defs) <= known, (mod, feat)


def test_inactive_classes_do_not_create_coverage():
    # angular_stomatitis (no training data) is the only tongue route to riboflavin,
    # so tongue must not claim to cover it; nothing active maps to vitamin D.
    assert "riboflavin_deficiency" not in schema.coverage("tongue")
    assert "vitamin_d_deficiency" not in set().union(*(schema.coverage(m) for m in schema.CLASSES))


def test_v5_face_classes_are_active():
    # v5 recovered acne / vitiligo / butterfly_rash; only blackhead lacks data
    assert schema.FACE_INACTIVE == {"blackhead"}
    assert {"hormonal_imbalance", "autoimmune_risk"} <= schema.coverage("face")


def test_tongue_only_categories_really_are_tongue_only():
    for d in TONGUE_ONLY:
        assert d in schema.coverage("tongue")
        assert d not in schema.coverage("face") | schema.coverage("eye")


# ── Known Issue #4: no tongue photo → tongue categories never leak ───────────
def test_face_eye_only_never_tags_tongue():
    r = fusion.build_report({"face": res("face", dark_circle=0.9),
                             "eye": res("eye", xanthelasma=0.8)})
    da = r["deficiency_analysis"]
    for d, v in da.items():
        assert "tongue" not in v["sources"] and "tongue" not in v["assessed_by"], d
    for d in TONGUE_ONLY:
        assert da[d]["status"] == "not_assessed"
        assert da[d]["probability"] == 0.0
        assert da[d]["probability_pct"] is None
    assert all(i["issue"] not in TONGUE_ONLY for i in r["top_insights"])


def test_eye_only_category_is_attributed_to_eye_not_face():
    r = fusion.build_report({"face": res("face", dark_circle=0.9),
                             "eye": res("eye", xanthelasma=0.8)})
    chol = r["deficiency_analysis"]["cholesterol_imbalance"]
    assert chol["status"] == "flagged" and chol["sources"] == ["eye"]
    assert chol["evidence"] == ["eye:xanthelasma"]


def test_no_findings_means_nothing_flagged():
    r = fusion.build_report({"face": res("face"), "eye": res("eye")})
    assert r["top_insights"] == []
    assert r["summary"]["flagged"] == 0
    assert all(v["probability"] == 0.0 for v in r["deficiency_analysis"].values())


def test_no_signal_vs_not_assessed():
    r = fusion.build_report({"face": res("face", wrinkle=0.7), "eye": res("eye")})
    da = r["deficiency_analysis"]
    assert da["iron_deficiency"]["status"] == "no_signal"          # face covers it, nothing found
    assert da["vitamin_d_deficiency"]["status"] == "not_assessed"  # nothing maps to it in v4


# ── fusion maths (unchanged product-of-experts) ──────────────────────────────
def test_fused_probabilities_sum_to_one():
    fused = fusion.fuse({"face": {"iron_deficiency": 0.9, "dehydration": 0.4},
                         "eye": {"iron_deficiency": 0.7, "liver_stress": 0.6},
                         "tongue": {"iron_deficiency": 0.5, "gut_dysbiosis": 0.5}})
    assert math.isclose(sum(fused.values()), 1.0, rel_tol=1e-9)
    assert set(fused) == {"iron_deficiency", "dehydration", "liver_stress", "gut_dysbiosis"}


def test_two_modality_weights_renormalise():
    # face (0.40) and eye (0.25) agree on one category only → p = 1 for both posteriors
    fused = fusion.fuse({"face": {"iron_deficiency": 1.0}, "eye": {"iron_deficiency": 1.0}})
    assert fused == pytest.approx({"iron_deficiency": 1.0})


def test_minor_single_modality_is_discounted():
    raw = fusion.fuse({"face": {"iron_deficiency": 1.0}, "eye": {"liver_stress": 1.0}})
    # face weight 0.40 → undiscounted 1.0; eye weight 0.25 → 0.85; then normalised
    assert raw["iron_deficiency"] == pytest.approx(1.0 / 1.85)
    assert raw["liver_stress"] == pytest.approx(0.85 / 1.85)


def test_empty_evidence_fuses_to_empty():
    assert fusion.fuse({"face": {}, "eye": {}}) == {}
