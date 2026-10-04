"""
FaceFuel schema — the one place class names, the deficiency framework and the
feature → deficiency mapping are declared.

Class lists MUST match the order of the YOLO heads / data.yaml files
(facefuel_{face,tongue,eye}_v4/data.yaml). Classes that had no training data are
kept in the lists (so class indices stay aligned with the trained YOLO heads) but
are listed in *_INACTIVE: their detections are discarded and they contribute no
evidence. At load time models.py cross-checks *_INACTIVE against the severity MLP
checkpoint's `inactive_idx` and warns if the two disagree.

Coverage (which deficiencies a modality can say anything about) is DERIVED from
the active classes' mappings below — it is never hand-listed, so it cannot drift.
"""

# ── Detector / severity class lists (order = YOLO class id) ──────────────────
FACE_CLASSES = [
    "dark_circle", "acne", "blackhead", "wrinkle", "redness",
    "dark_spot", "rosacea", "vitiligo", "eczema", "butterfly_rash",
]
TONGUE_CLASSES = [
    "tongue_body", "white_coating", "yellow_coating", "thick_coating",
    "no_coating", "red_tongue", "pale_tongue", "purple_tongue",
    "fissured", "geographic", "smooth_glossy", "crenated",
    "oral_ulcer", "angular_stomatitis", "median_rhomboid",
    "lichen_planus", "leukoplakia", "hairy_leukoplakia",
]
EYE_CLASSES = [
    "conjunctival_pallor", "scleral_icterus", "xanthelasma",
    "pterygium", "conjunctivitis", "eyelid_drooping",
]

# Zero training images were recovered for these (CLAUDE.md §3). Not functional.
FACE_INACTIVE   = {"acne", "blackhead", "vitiligo", "butterfly_rash"}
TONGUE_INACTIVE = {"no_coating", "purple_tongue", "angular_stomatitis", "median_rhomboid"}
EYE_INACTIVE    = set()

CLASSES  = {"face": FACE_CLASSES, "tongue": TONGUE_CLASSES, "eye": EYE_CLASSES}
INACTIVE = {"face": FACE_INACTIVE, "tongue": TONGUE_INACTIVE, "eye": EYE_INACTIVE}

# ── 22-dimension deficiency / wellness framework ─────────────────────────────
ALL_DEFS = [
    "iron_deficiency", "b12_deficiency", "vitamin_d_deficiency",
    "zinc_deficiency", "omega3_deficiency", "vitamin_a_deficiency",
    "vitamin_c_deficiency", "poor_sleep_quality", "hormonal_imbalance",
    "dehydration", "high_stress", "liver_stress", "gut_dysbiosis",
    "hypothyroid", "folate_deficiency", "cholesterol_imbalance",
    "riboflavin_deficiency", "autoimmune_risk", "eye_inflammation",
    "copper_deficiency", "skin_inflammation", "oral_health_risk",
]

# ── Visual feature → deficiency mapping ──────────────────────────────────────
FACE_FEAT_DEF = {
    "dark_circle":    ["iron_deficiency", "b12_deficiency", "poor_sleep_quality"],
    "acne":           ["hormonal_imbalance", "zinc_deficiency", "high_stress"],
    "blackhead":      ["zinc_deficiency", "vitamin_a_deficiency"],
    "wrinkle":        ["vitamin_c_deficiency", "dehydration", "omega3_deficiency"],
    "redness":        ["skin_inflammation", "vitamin_c_deficiency", "high_stress"],
    "dark_spot":      ["vitamin_c_deficiency", "vitamin_a_deficiency"],
    "rosacea":        ["gut_dysbiosis", "omega3_deficiency", "skin_inflammation"],
    "vitiligo":       ["autoimmune_risk", "b12_deficiency", "copper_deficiency"],
    "eczema":         ["omega3_deficiency", "zinc_deficiency", "skin_inflammation"],
    "butterfly_rash": ["autoimmune_risk"],
    # colour-analysis signals (LAB thresholds on the raw aligned face, from v2)
    "pallor":         ["iron_deficiency", "b12_deficiency"],
    "lip_pallor":     ["iron_deficiency"],
}
TONGUE_FEAT_DEF = {
    "white_coating":     ["gut_dysbiosis", "dehydration"],
    "yellow_coating":    ["liver_stress", "gut_dysbiosis"],
    "thick_coating":     ["gut_dysbiosis", "liver_stress"],
    "red_tongue":        ["b12_deficiency", "folate_deficiency", "high_stress"],
    "pale_tongue":       ["iron_deficiency", "b12_deficiency"],
    "fissured":          ["b12_deficiency", "zinc_deficiency", "dehydration"],
    # was "vitamin_b_deficiency", which is not a framework dimension and so was
    # silently dropped; b12 is the closest existing B-vitamin dimension.
    "geographic":        ["zinc_deficiency", "b12_deficiency"],
    "smooth_glossy":     ["b12_deficiency", "folate_deficiency"],
    "crenated":          ["hypothyroid", "dehydration"],
    "oral_ulcer":        ["b12_deficiency", "iron_deficiency", "oral_health_risk"],
    "lichen_planus":     ["autoimmune_risk", "high_stress", "oral_health_risk"],
    "leukoplakia":       ["vitamin_a_deficiency", "oral_health_risk"],
    "hairy_leukoplakia": ["autoimmune_risk", "oral_health_risk"],
    # inactive (no training data) — kept so the mapping is complete if data arrives
    "no_coating":         [],
    "purple_tongue":      ["omega3_deficiency"],
    "angular_stomatitis": ["riboflavin_deficiency", "iron_deficiency", "b12_deficiency"],
    "median_rhomboid":    ["oral_health_risk"],
}
EYE_FEAT_DEF = {
    "conjunctival_pallor": ["iron_deficiency", "b12_deficiency"],
    "scleral_icterus":     ["liver_stress"],
    "xanthelasma":         ["cholesterol_imbalance"],
    "pterygium":           ["vitamin_a_deficiency", "dehydration"],
    "conjunctivitis":      ["vitamin_a_deficiency", "vitamin_c_deficiency", "eye_inflammation"],
    "eyelid_drooping":     ["hypothyroid"],
}
FEAT_DEF = {"face": FACE_FEAT_DEF, "tongue": TONGUE_FEAT_DEF, "eye": EYE_FEAT_DEF}

# Product-of-experts weights (CLAUDE.md §2). Hand-set; see docs/RESEARCH_NOTES.md.
MODALITY_WEIGHTS = {"face": 0.40, "tongue": 0.35, "eye": 0.25}

FOOD_RECS = {
    "iron_deficiency":       ["spinach", "lentils", "red meat", "tofu", "pumpkin seeds"],
    "b12_deficiency":        ["eggs", "dairy", "salmon", "beef liver", "fortified cereals"],
    "vitamin_d_deficiency":  ["fatty fish", "egg yolks", "fortified milk", "mushrooms"],
    "zinc_deficiency":       ["oysters", "beef", "chickpeas", "cashews", "pumpkin seeds"],
    "omega3_deficiency":     ["salmon", "walnuts", "flaxseed", "chia seeds", "mackerel"],
    "vitamin_a_deficiency":  ["sweet potato", "carrots", "kale", "egg yolks", "liver"],
    "vitamin_c_deficiency":  ["citrus fruits", "bell peppers", "broccoli", "kiwi"],
    "poor_sleep_quality":    ["magnesium-rich foods", "tart cherry juice", "less caffeine after 2pm"],
    "hormonal_imbalance":    ["healthy fats", "fiber", "less added sugar"],
    "dehydration":           ["water through the day", "cucumber", "watermelon"],
    "high_stress":           ["magnesium-rich foods", "B-vitamin-rich foods", "regular exercise"],
    "liver_stress":          ["less alcohol", "leafy greens", "beets"],
    "gut_dysbiosis":         ["fermented foods", "fiber", "less added sugar"],
    "hypothyroid":           ["iodine-rich foods", "selenium-rich foods (brazil nuts)", "zinc-rich foods"],
    "folate_deficiency":     ["leafy greens", "lentils", "asparagus", "fortified cereals"],
    "cholesterol_imbalance": ["oats", "beans", "avocado", "olive oil", "salmon"],
    "riboflavin_deficiency": ["dairy", "eggs", "lean meat", "almonds", "spinach"],
    "autoimmune_risk":       ["anti-inflammatory diet", "omega-3 sources", "vitamin D sources"],
    "eye_inflammation":      ["omega-3 sources", "vitamin A sources"],
    "copper_deficiency":     ["shellfish", "nuts", "seeds", "dark chocolate", "legumes"],
    "skin_inflammation":     ["omega-3 sources", "zinc-rich foods", "vitamin E sources"],
    "oral_health_risk":      ["vitamin C sources", "less added sugar"],
}
ADVICE = {
    "iron_deficiency":       "Pair iron-rich foods with vitamin C to boost absorption. A blood test (ferritin, CBC) is the only way to confirm.",
    "b12_deficiency":        "B12 comes mainly from animal foods; vegans usually need a supplement. Confirm with a blood test.",
    "vitamin_d_deficiency":  "15–30 min of daylight most days helps. A 25(OH)D blood test confirms status.",
    "zinc_deficiency":       "Soaking legumes reduces phytates that block zinc absorption.",
    "omega3_deficiency":     "Aim for two servings of oily fish per week, or plant sources daily.",
    "vitamin_a_deficiency":  "Vitamin A is fat-soluble — pair with healthy fats.",
    "vitamin_c_deficiency":  "Cooking destroys vitamin C; eat some raw fruit or vegetables daily.",
    "poor_sleep_quality":    "Consistent sleep and wake times matter more than total hours. Aim for 7–9 h.",
    "hormonal_imbalance":    "Hormonal questions need a clinician's evaluation; diet supports but does not replace care.",
    "dehydration":           "Thirst is a late signal — aim for pale-yellow urine through the day.",
    "high_stress":           "Chronic stress affects sleep, skin and appetite; diet and stress management both help.",
    "liver_stress":          "Yellowing of the eyes or skin should be checked by a doctor promptly.",
    "gut_dysbiosis":         "Fiber and fermented foods support a diverse gut microbiome.",
    "hypothyroid":           "Thyroid function can only be assessed with a blood test (TSH). See a doctor if concerned.",
    "folate_deficiency":     "Especially important before and during pregnancy.",
    "cholesterol_imbalance": "Xanthelasma can accompany raised lipids — a lipid panel is the way to check.",
    "riboflavin_deficiency": "More common in people who avoid both dairy and meat.",
    "autoimmune_risk":       "Visual signs here are non-specific; persistent symptoms deserve a medical review.",
    "eye_inflammation":      "Persistent redness, pain or vision change needs an eye-care professional.",
    "copper_deficiency":     "Rare — usually linked to excess zinc supplementation.",
    "skin_inflammation":     "Gentle skincare and an anti-inflammatory diet can help; a dermatologist can advise.",
    "oral_health_risk":      "White or red patches that persist beyond two weeks should be seen by a dentist or doctor.",
}


def active_classes(modality: str) -> list:
    return [c for c in CLASSES[modality] if c not in INACTIVE[modality]]


def coverage(modality: str) -> set:
    """Deficiencies the modality can produce evidence for, via its ACTIVE classes."""
    fmap, inactive = FEAT_DEF[modality], INACTIVE[modality]
    return {d for feat, defs in fmap.items() if feat not in inactive for d in defs}


def _validate():
    known = set(ALL_DEFS)
    for mod, fmap in FEAT_DEF.items():
        for feat, defs in fmap.items():
            bad = set(defs) - known
            assert not bad, f"{mod}.{feat} maps to unknown deficiencies {bad}"
        missing = [c for c in CLASSES[mod] if c not in fmap and c != "tongue_body"]
        assert not missing, f"{mod} classes without a mapping entry: {missing}"
        assert INACTIVE[mod] <= set(CLASSES[mod]), f"{mod} inactive set has unknown classes"
    assert set(FOOD_RECS) == known and set(ADVICE) == known, "FOOD_RECS/ADVICE out of sync"


_validate()
