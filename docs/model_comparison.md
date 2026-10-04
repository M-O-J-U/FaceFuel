# FaceFuel — old (v4) vs new (v5) models on the clean test split

## eye

**Detector (YOLO11m) on v5 test**

| | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| old | 0.829 | 0.823 | 0.755 | 0.911 |
| **new** | **0.991** | 0.991 | 0.952 | 0.960 |

| class | old AP50 | new AP50 |
|---|---|---|
| conjunctival_pallor | 0.949 | 0.995 |
| scleral_icterus | 0.309 | 0.995 |
| xanthelasma | 0.995 | 0.995 |
| pterygium | 0.995 | 0.995 |
| conjunctivitis | 0.982 | 0.988 |
| eyelid_drooping | 0.747 | 0.980 |

**Severity MLP on v5 test** (309 images, 96 with no condition)

| class | n | old F1 | new F1 | old AUROC | new AUROC |
|---|---|---|---|---|---|
| conjunctival_pallor | 11 | 0.645 | 0.952 | 0.957 | 0.977 |
| scleral_icterus | 3 | 0.286 | 0.600 | 0.987 | 0.992 |
| xanthelasma | 14 | 0.636 | 0.966 | 1.000 | 1.000 |
| pterygium | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| conjunctivitis | 91 | 0.916 | 0.950 | 0.986 | 0.995 |
| eyelid_drooping | 82 | 0.703 | 0.941 | 0.922 | 0.989 |
| **mean** | | **0.698** | **0.901** | | |

No-condition images flagged by at least one class: old 93.8% → new 6.2%

**Out-of-domain: share of 9 real face photos on which each class fires (>0.35)**

| class | old | new |
|---|---|---|
| conjunctival_pallor | 44% | 0% |
| scleral_icterus | 0% | 33% |
| xanthelasma | 0% | 0% |
| pterygium | 11% | 11% |
| conjunctivitis | 44% | 11% |
| eyelid_drooping | 44% | 11% |

## tongue

**Detector (YOLO11m) on v5 test**

| | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| old | 0.841 | 0.831 | 0.721 | 0.823 |
| **new** | **0.804** | 0.796 | 0.678 | 0.822 |

| class | old AP50 | new AP50 |
|---|---|---|
| tongue_body | 0.945 | 0.940 |
| white_coating | 0.710 | 0.528 |
| yellow_coating | 0.856 | 0.824 |
| thick_coating | 0.900 | 0.857 |
| red_tongue | 0.791 | 0.770 |
| pale_tongue | 0.995 | 0.995 |
| fissured | 0.700 | 0.728 |
| geographic | 0.958 | 0.954 |
| smooth_glossy | 0.973 | 0.974 |
| crenated | 0.370 | 0.455 |
| oral_ulcer | 0.977 | 0.968 |
| lichen_planus | 0.671 | 0.384 |
| leukoplakia | 0.971 | 0.965 |
| hairy_leukoplakia | 0.955 | 0.915 |

**Severity MLP on v5 test** (1343 images, 0 with no condition)

| class | n | old F1 | new F1 | old AUROC | new AUROC |
|---|---|---|---|---|---|
| tongue_body | 220 | 0.812 | 0.817 | 0.980 | 0.980 |
| white_coating | 18 | 0.333 | 0.429 | 0.828 | 0.894 |
| yellow_coating | 27 | 0.814 | 0.786 | 0.990 | 0.992 |
| thick_coating | 104 | 0.816 | 0.830 | 0.987 | 0.988 |
| red_tongue | 37 | 0.610 | 0.675 | 0.972 | 0.964 |
| pale_tongue | 3 | 0.667 | 0.333 | 0.957 | 0.941 |
| fissured | 106 | 0.628 | 0.694 | 0.965 | 0.972 |
| geographic | 216 | 0.871 | 0.894 | 0.989 | 0.990 |
| smooth_glossy | 202 | 0.934 | 0.908 | 0.995 | 0.993 |
| crenated | 44 | 0.386 | 0.538 | 0.951 | 0.950 |
| oral_ulcer | 442 | 0.841 | 0.859 | 0.960 | 0.965 |
| lichen_planus | 12 | 0.320 | 0.364 | 0.916 | 0.901 |
| leukoplakia | 429 | 0.840 | 0.860 | 0.963 | 0.965 |
| hairy_leukoplakia | 30 | 0.806 | 0.831 | 0.993 | 0.996 |
| **mean** | | **0.691** | **0.701** | | |

No-condition images flagged by at least one class: old nan% → new nan%

## face

**Detector (YOLO11m) on v5 test**

| | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| old | 0.139 | 0.058 | 0.205 | 0.254 |
| **new** | **0.672** | 0.547 | 0.660 | 0.621 |

| class | old AP50 | new AP50 |
|---|---|---|
| dark_circle | 0.741 | 0.714 |
| acne | 0.000 | 0.870 |
| wrinkle | 0.222 | 0.248 |
| redness | 0.108 | 0.715 |
| dark_spot | 0.083 | 0.888 |
| rosacea | 0.037 | 0.602 |
| vitiligo | 0.000 | 0.854 |
| eczema | 0.064 | 0.728 |
| butterfly_rash | 0.000 | 0.430 |

**Severity MLP on v5 test** (2913 images, 0 with no condition)

| class | n | old F1 | new F1 | old AUROC | new AUROC |
|---|---|---|---|---|---|
| dark_circle | 166 | 0.879 | 0.898 | 0.994 | 0.997 |
| acne | 67 | 0.000 | 0.750 | nan | 0.984 |
| wrinkle | 127 | 0.596 | 0.903 | 0.972 | 0.993 |
| redness | 349 | 0.182 | 0.730 | 0.665 | 0.958 |
| dark_spot | 1620 | 0.064 | 0.952 | 0.183 | 0.981 |
| rosacea | 11 | 0.023 | 0.900 | 0.817 | 0.964 |
| vitiligo | 163 | 0.000 | 0.761 | nan | 0.973 |
| eczema | 382 | 0.171 | 0.681 | 0.335 | 0.953 |
| butterfly_rash | 58 | 0.000 | 0.425 | nan | 0.943 |
| **mean** | | **0.213** | **0.778** | | |

No-condition images flagged by at least one class: old nan% → new nan%

**Out-of-domain: share of 9 real face photos on which each class fires (>0.35)**

| class | old | new |
|---|---|---|
| dark_circle | 56% | 78% |
| acne | nan% | 11% |
| wrinkle | 0% | 0% |
| redness | 11% | 11% |
| dark_spot | 44% | 11% |
| rosacea | 0% | 0% |
| vitiligo | nan% | 0% |
| eczema | 11% | 22% |
| butterfly_rash | nan% | 0% |
