# FaceFuel v4 — severity-MLP domain-shift probe

### face — in-domain validation (150 images)

| class | positives | mean MLP (pos) | negatives | mean MLP (neg) | neg > 0.35 |
|---|---|---|---|---|---|
| dark_circle | 8 | 1.00 | 142 | 0.00 | 0% |
| wrinkle | 14 | 0.82 | 136 | 0.00 | 0% |
| redness | 2 | 1.00 | 148 | 0.01 | 1% |
| dark_spot | 5 | 1.00 | 145 | 0.00 | 0% |
| rosacea | 15 | 0.93 | 135 | 0.07 | 5% |
| eczema | 107 | 0.93 | 43 | 0.05 | 5% |

### eye — in-domain validation (150 images)

| class | positives | mean MLP (pos) | negatives | mean MLP (neg) | neg > 0.35 |
|---|---|---|---|---|---|
| conjunctival_pallor | 21 | 1.00 | 129 | 0.00 | 0% |
| scleral_icterus | 3 | 1.00 | 147 | 0.00 | 0% |
| xanthelasma | 64 | 1.00 | 86 | 0.00 | 0% |
| pterygium | 9 | 1.00 | 141 | 0.00 | 0% |
| conjunctivitis | 31 | 0.93 | 119 | 0.02 | 2% |
| eyelid_drooping | 22 | 0.93 | 128 | 0.02 | 2% |

### tongue — in-domain validation (150 images)

| class | positives | mean MLP (pos) | negatives | mean MLP (neg) | neg > 0.35 |
|---|---|---|---|---|---|
| white_coating | 1 | 0.00 | 149 | 0.00 | 0% |
| yellow_coating | 3 | 0.35 | 147 | 0.01 | 1% |
| thick_coating | 9 | 0.89 | 141 | 0.04 | 4% |
| red_tongue | 4 | 0.99 | 146 | 0.02 | 2% |
| pale_tongue | 0 | – | 150 | 0.00 | 0% |
| fissured | 16 | 0.84 | 134 | 0.07 | 8% |
| geographic | 13 | 0.80 | 137 | 0.05 | 6% |
| smooth_glossy | 9 | 0.96 | 141 | 0.03 | 4% |
| crenated | 17 | 0.98 | 133 | 0.03 | 3% |
| oral_ulcer | 47 | 0.84 | 103 | 0.05 | 6% |
| lichen_planus | 2 | 0.14 | 148 | 0.00 | 0% |
| leukoplakia | 23 | 0.85 | 127 | 0.09 | 10% |
| hairy_leukoplakia | 1 | 0.58 | 149 | 0.01 | 1% |

Tongue body localised in 33/150 validation images (22%).


### Out-of-domain: 15 photos in `test`

MLP sigmoid per active class; detector hits in brackets.

| photo | face:dark_circle | face:wrinkle | face:redness | face:dark_spot | face:rosacea | face:eczema | eye:conjunctival_pallor | eye:scleral_icterus | eye:xanthelasma | eye:pterygium | eye:conjunctivitis | eye:eyelid_drooping |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 545454545454.jpg | no face | | | | | | | | | | | |
| 8787877897.jpg | no face | | | | | | | | | | | |
| Acne-Treatment-1-1024x724.jpg | 0.00 [0.54] | 0.10 | 0.91 | 0.00 | 0.13 | 0.57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.58 | 0.59 [0.81] |
| adasdasasas.jpg | no face | | | | | | | | | | | |
| asdasdas.webp | 0.21 | 0.00 | 0.00 | 0.94 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.55 | 0.67 [0.56] |
| download (1).jpg | no face | | | | | | | | | | | |
| download.jpg | 1.00 | 0.00 | 0.00 | 0.01 | 0.04 | 0.00 | 0.49 | 0.18 | 0.01 | 0.09 | 0.33 | 0.11 [0.88] |
| images (1).jpg | 1.00 | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.57 | 0.01 | 0.00 | 0.00 | 0.00 [0.48] | 0.81 [0.46] |
| images (2).jpg | 0.70 | 0.00 | 0.00 | 0.46 | 0.00 | 0.01 [0.33] | 0.27 | 0.14 | 0.04 | 0.46 | 0.09 | 0.08 [0.72] |
| images.jpg | 0.14 | 0.00 | 0.00 | 0.99 [0.62] | 0.00 | 0.00 | 0.69 | 0.00 | 0.12 | 0.06 | 0.00 | 0.01 [0.94] |
| rdfsdcvcvdwe.jpg | no face | | | | | | | | | | | |
| red eye.webp | 1.00 [0.67] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 1.00 [0.54] | 0.00 |
| werfsefeqwe.jpg | no face | | | | | | | | | | | |
| yellow eye 2.jpg | 0.00 [0.57] | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| yellow eye.webp | 1.00 [0.45] | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.44 | 0.25 | 0.02 | 0.01 | 0.06 | 0.53 [0.91] |
| **fraction > 0.35** | **56%** | **0%** | **11%** | **44%** | **0%** | **11%** | **44%** | **0%** | **0%** | **11%** | **44%** | **44%** |
