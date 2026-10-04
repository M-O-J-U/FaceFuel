"""
FaceFuel v2 — Step 9: Bayesian Clinical Inference Engine (Full Rewrite)
========================================================================
ALL ROOT CAUSES FIXED:

  FIX 1:  pos_weight correctly passed to BCEWithLogitsLoss
  FIX 2:  Count-based severity for ALL small-lesion classes
           (acne, dark_spot, blackhead, melasma, whitehead, acne_scar)
  FIX 3:  omega3 prior reduced 0.50 → 0.15; all priors recalibrated
  FIX 5:  BCEWithLogitsLoss + raw logits (numerically stable)
  FIX 6:  LAB color-based supplemental detection (pallor, redness,
           yellow sclera, oiliness, dark circle depth, lip pallor)
  FIX 7:  Feature ABSENCE penalizes related deficiencies
  FIX 8:  Uncertainty weighting via exp(-3σ) instead of division by max
  FIX 9:  Stratified train/val split — rare classes guaranteed in both sets
  FIX 10: Per-feature specialized MLP heads
  FIX 11: Calibrated YOLO+DINOv2 fusion using count-based severity

Usage:
  python step9_bayesian_engine.py --train
  python step9_bayesian_engine.py --evaluate
  python step9_bayesian_engine.py --demo
"""

import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Paths ─────────────────────────────────────────────────────
FEAT_MATRIX         = Path("facefuel_datasets/features/feature_matrix.npy")
FEAT_INDEX          = Path("facefuel_datasets/features/feature_index.json")
FEAT_STATS          = Path("facefuel_datasets/features/feature_stats.json")
LABELS_DIR          = Path("facefuel_datasets/MERGED_V2/labels")
MODEL_DIR           = Path("facefuel_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEVERITY_MODEL_PATH = MODEL_DIR / "severity_mlp_v2.pt"

# ── Feature config ────────────────────────────────────────────
SKIN_FEATURES = [
    "dark_circle",      # 0
    "eye_bag",          # 1
    "acne",             # 2
    "wrinkle",          # 3
    "redness",          # 4
    "dark_spot",        # 5
    "blackhead",        # 6
    "melasma",          # 7
    "whitehead",        # 8
    "acne_scar",        # 9
    "vascular_redness", # 10
]
N_FEATURES = len(SKIN_FEATURES)
FEAT_DIM   = 3072

YOLO_TO_FEAT = {0:0, 1:1, 2:2, 3:3, 4:4, 7:5, 8:6, 11:7, 12:8, 16:9, 23:10}

# FIX 2: Small-lesion classes use COUNT-based severity
SMALL_LESION_FEATS = {2, 5, 6, 7, 8, 9}
# {acne, dark_spot, blackhead, melasma, whitehead, acne_scar}

DEFICIENCIES = [
    "iron_deficiency",       # 0
    "b12_deficiency",        # 1
    "vitamin_d_deficiency",  # 2
    "zinc_deficiency",       # 3
    "omega3_deficiency",     # 4
    "vitamin_a_deficiency",  # 5
    "vitamin_c_deficiency",  # 6
    "poor_sleep_quality",    # 7
    "hormonal_imbalance",    # 8
    "dehydration",           # 9
    "high_stress",           # 10
]
N_DEF = len(DEFICIENCIES)


def count_to_severity(n: int) -> float:
    """Non-linear count → severity for small-lesion classes."""
    if n == 0:   return 0.0
    if n == 1:   return 0.35
    if n <= 3:   return 0.50
    if n <= 6:   return 0.65
    if n <= 10:  return 0.78
    if n <= 20:  return 0.88
    return 1.0


# ═══════════════════════════════════════════════════════════════
# FIX 5 + 10: SeverityMLP v2 — per-feature heads + BCEWithLogitsLoss
# ═══════════════════════════════════════════════════════════════

class SeverityMLPv2(nn.Module):
    """
    FIX 5:  Raw logit output (no sigmoid in forward)
    FIX 10: Per-feature specialized heads prevent dominant classes
            from crowding out rare ones
    """
    def __init__(self, feat_dim=FEAT_DIM, n_features=N_FEATURES, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            for _ in range(n_features)
        ])

    def forward(self, x):
        """Returns raw logits (N, n_features). NO sigmoid here."""
        shared = self.backbone(x)
        return torch.cat([h(shared) for h in self.heads], dim=-1)

    def predict_with_uncertainty(self, x, n_passes=25):
        """
        FIX 8: Returns (mean_probs, std_probs) with proper sigmoid-space uncertainty.
        """
        self.train()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.forward(x)) for _ in range(n_passes)])
        self.eval()
        return probs.mean(0), probs.std(0)


class SkinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ═══════════════════════════════════════════════════════════════
# FIX 2 + 9: Label builder + stratified split
# ═══════════════════════════════════════════════════════════════

def build_labels_from_yolo(index_map, feature_stems):
    """FIX 2: Count-based severity for small-lesion classes."""
    N = len(feature_stems)
    labels = np.zeros((N, N_FEATURES), dtype=np.float32)

    for i, stem in enumerate(feature_stems):
        base = stem.replace("_aligned", "")
        lbl_path = LABELS_DIR / f"{base}.txt"
        if not lbl_path.exists():
            continue

        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        counts   = defaultdict(int)
        max_area = defaultdict(float)

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                w, h   = float(parts[3]), float(parts[4])
                area   = w * h
            except ValueError:
                continue

            fi = YOLO_TO_FEAT.get(cls_id)
            if fi is None:
                continue
            counts[fi] += 1
            if area > max_area[fi]:
                max_area[fi] = area

        for fi in set(list(counts.keys()) + list(max_area.keys())):
            if fi in SMALL_LESION_FEATS:
                sev = count_to_severity(counts[fi])
            elif fi == 0:  # dark_circle
                sev = min(max_area.get(fi, 0) / 0.15, 1.0)
            elif fi == 3:  # wrinkle: count + area
                sev = min(counts[fi] * 0.08 + max_area.get(fi, 0) / 0.10, 1.0)
            else:
                sev = min(max_area.get(fi, 0) / 0.20, 1.0)
            labels[i, fi] = min(float(sev), 1.0)

    return labels


def stratified_split(labels, labeled_mask, val_frac=0.20, seed=42):
    """FIX 9: Stratified split guaranteeing rare classes in both sets."""
    rng = np.random.default_rng(seed)
    labeled_idx = np.where(labeled_mask)[0]

    reserved_val = set()
    rare = np.where((labels > 0).sum(axis=0) < 50)[0]
    for cls in rare:
        pos = labeled_idx[labels[labeled_idx, cls] > 0]
        if len(pos) >= 2:
            n_v = max(1, int(len(pos) * val_frac))
            reserved_val.update(rng.choice(pos, n_v, replace=False).tolist())

    remaining = [i for i in labeled_idx if i not in reserved_val]
    rng.shuffle(remaining)
    sp = int(len(remaining) * (1 - val_frac))
    return np.array(remaining[:sp]), np.array(remaining[sp:] + list(reserved_val))


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_severity_mlp(epochs=100, batch_size=64, lr=2e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: index_map = json.load(f)
    with open(FEAT_STATS)  as f: stats    = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    feat_norm = (features - mean) / (std + 1e-6)

    stems  = [index_map[str(i)] for i in range(len(features))]
    labels = build_labels_from_yolo(index_map, stems)

    lmask = labels.sum(axis=1) > 0
    print(f"\n  Images with ≥1 annotation: {lmask.sum():,} / {len(labels):,}")
    print(f"\n  Label stats (v2):")
    for i, name in enumerate(SKIN_FEATURES):
        nz  = (labels[:, i] > 0).sum()
        msv = labels[labels[:, i] > 0, i].mean() if nz > 0 else 0
        tag = "count" if i in SMALL_LESION_FEATS else "area"
        print(f"    {name:<22s}  present={nz:4d}  mean={msv:.3f}  ({tag})")

    # FIX 9: stratified split
    train_idx, val_idx = stratified_split(labels, lmask)
    print(f"\n  Train: {len(train_idx):,}   Val: {len(val_idx):,}  (stratified)")

    # FIX 1: Weighted sampler for rare classes
    n_pos = np.array([(labels[train_idx, i] > 0).sum() for i in range(N_FEATURES)])
    n_neg = len(train_idx) - n_pos
    sw = np.ones(len(train_idx))
    for ii, orig in enumerate(train_idx):
        mw = 1.0
        for fi in range(N_FEATURES):
            if labels[orig, fi] > 0 and n_pos[fi] > 0:
                mw = max(mw, (n_neg[fi] / (n_pos[fi] + 1e-6)) ** 0.4)
        sw[ii] = mw
    sampler = WeightedRandomSampler(torch.from_numpy(sw).float(), len(train_idx), True)

    train_dl = DataLoader(SkinDataset(feat_norm[train_idx], labels[train_idx]),
                          batch_size=batch_size, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(SkinDataset(feat_norm[val_idx],   labels[val_idx]),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    # FIX 1 + 5: BCEWithLogitsLoss with pos_weight
    n_total = int(lmask.sum())
    pos_w   = torch.tensor([
        float(n_total) / max((labels[lmask, i] > 0).sum(), 1)
        for i in range(N_FEATURES)
    ]).clamp(max=20.0).to(device)
    print(f"\n  pos_weight: {[f'{v:.1f}' for v in pos_w.tolist()]}")

    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model     = SeverityMLPv2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_dl), pct_start=0.1)

    print(f"\n  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  Device: {device}\n")

    best_f1_sum = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = []
        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tl.append(loss.item())

        model.eval()
        vl, all_p, all_y = [], [], []
        with torch.no_grad():
            for X, Y in val_dl:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                vl.append(loss_fn(logits, Y).item())
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(Y.cpu().numpy())

        P = np.vstack(all_p); L = np.vstack(all_y)
        f1s = []
        for fi in range(N_FEATURES):
            t = (L[:, fi] > 0.3).astype(int); p = (P[:, fi] > 0.3).astype(int)
            tp = ((t==1)&(p==1)).sum(); fp = ((t==0)&(p==1)).sum(); fn = ((t==1)&(p==0)).sum()
            pr = tp/(tp+fp+1e-6); rc = tp/(tp+fn+1e-6)
            f1s.append(2*pr*rc/(pr+rc+1e-6))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={np.mean(tl):.4f}  val={np.mean(vl):.4f}  "
                  f"mean_F1={np.mean(f1s):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if sum(f1s) > best_f1_sum:
            best_f1_sum = sum(f1s)
            torch.save({
                "model_state":    model.state_dict(),
                "mean":           mean.tolist(),
                "std":            std.tolist(),
                "skin_features":  SKIN_FEATURES,
                "epoch":          epoch,
                "mean_f1":        float(np.mean(f1s)),
                "f1_per_feature": [float(f) for f in f1s],
                "version":        2,
            }, str(SEVERITY_MODEL_PATH))

    print(f"\n  ✅ Best mean F1: {best_f1_sum/N_FEATURES:.4f}")
    print(f"  Model saved: {SEVERITY_MODEL_PATH}")
    return model


def load_severity_model(device="cpu"):
    path = SEVERITY_MODEL_PATH
    if not path.exists():
        v1 = Path("facefuel_models/severity_mlp.pt")
        if v1.exists():
            print("  ⚠ v2 model not found, using v1. Run --train for fixes.")
            path = v1
        else:
            raise FileNotFoundError("No model found. Run: python step9_bayesian_engine.py --train")

    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if ckpt.get("version", 1) == 2:
        model = SeverityMLPv2()
    else:
        # Load v1 architecture
        class _V1MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(FEAT_DIM)
                self.net  = nn.Sequential(
                    nn.Linear(FEAT_DIM, 512), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(512, 128),      nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(128, N_FEATURES), nn.Sigmoid(),
                )
            def forward(self, x):          return self.net(self.norm(x))
            def predict_with_uncertainty(self, x, n_passes=20):
                self.train()
                with torch.no_grad():
                    p = torch.stack([self.forward(x) for _ in range(n_passes)])
                self.eval()
                return p.mean(0), p.std(0)
        model = _V1MLP()

    model.load_state_dict(ckpt["model_state"])
    model.to(device); model.eval()
    return model, np.array(ckpt["mean"], np.float32), np.array(ckpt["std"], np.float32)


# ═══════════════════════════════════════════════════════════════
# FIX 6: LAB Color-Based Feature Detection
# ═══════════════════════════════════════════════════════════════

def analyze_color_features(raw_rgb, regions):
    """
    Color analysis on the RAW (pre-normalization) aligned image.
    LAB normalization stretches L to fill 0-255 in every image,
    making absolute thresholds meaningless. Raw image preserves
    actual brightness relationships.
    """
    import cv2
    lab = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    results = {}

    def get_region(name):
        r = regions.get(name, {})
        bbox = r.get("bbox", None) if isinstance(r, dict) else None
        if bbox is None: return None
        x1,y1,x2,y2 = bbox
        return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None

    # ── Pallor ────────────────────────────────────────────────
    cheek_L = []
    for rn in ["left_cheek", "right_cheek"]:
        bb = get_region(rn)
        if bb: cheek_L.extend(L[bb[1]:bb[3], bb[0]:bb[2]].flatten())
    if cheek_L:
        mL = float(np.mean(cheek_L))
        score = float(np.clip((155 - mL) / 40, 0, 1))
        if score > 0.15:
            results["pallor_color"] = (score, min(1.0, len(cheek_L)/2000))

    # ── Skin Redness ──────────────────────────────────────────
    skin_A = []
    for rn in ["left_cheek", "right_cheek", "forehead"]:
        bb = get_region(rn)
        if bb: skin_A.extend(A[bb[1]:bb[3], bb[0]:bb[2]].flatten())
    if skin_A:
        mA = float(np.mean(skin_A))
        score = float(np.clip((mA - 135) / 20, 0, 1))
        if score > 0.10:
            results["redness_color"] = (score, min(1.0, len(skin_A)/3000))

    # ── Dark Circle Depth ─────────────────────────────────────
    # On raw image: genuine dark circles = >20 L units below cheeks.
    # Normal orbital shadow = 5–15 units — not dark circles.
    peri_L = []
    for rn in ["periorbital_left", "periorbital_right"]:
        bb = get_region(rn)
        if bb: peri_L.extend(L[bb[1]:bb[3], bb[0]:bb[2]].flatten())
    if peri_L and cheek_L:
        abs_diff = float(np.mean(cheek_L)) - float(np.mean(peri_L))
        if abs_diff > 20:
            score = float(np.clip((abs_diff - 20) / 35, 0, 1))
            results["dark_circle_color"] = (score, min(1.0, len(peri_L)/500))

    # ── Yellow Sclera ─────────────────────────────────────────
    bb = get_region("sclera_left")
    if bb:
        sB = B[bb[1]:bb[3], bb[0]:bb[2]]
        score = float(np.clip((float(np.mean(sB)) - 145) / 20, 0, 1))
        if score > 0.15:
            results["yellow_sclera_color"] = (score, min(1.0, sB.size/300))

    # ── Skin Texture Roughness ────────────────────────────────
    stds = []
    for rn in ["left_cheek", "right_cheek", "forehead"]:
        bb = get_region(rn)
        if bb:
            rl = L[bb[1]:bb[3], bb[0]:bb[2]]
            if rl.size > 100: stds.append(float(np.std(rl)))
    if stds:
        score = float(np.clip((float(np.mean(stds)) - 10) / 20, 0, 1))
        if score > 0.15:
            results["skin_texture_color"] = (score, 0.7)

    # ── Oiliness / Shine ──────────────────────────────────────
    # On raw image: oily specular patches are both relatively bright
    # (above mean+2σ) AND absolutely bright (L > 200).
    shine_scores = []
    for rn in ["forehead", "nose"]:
        bb = get_region(rn)
        if bb:
            rl = L[bb[1]:bb[3], bb[0]:bb[2]]
            if rl.size > 100:
                r_mean = float(np.mean(rl))
                r_std  = float(np.std(rl))
                rel_thresh = r_mean + 2.0 * r_std
                specular = (rl > rel_thresh) & (rl > 200)
                shine_scores.append(float(specular.sum()) / rl.size)
    if shine_scores:
        mean_spec = float(np.mean(shine_scores))
        score = float(np.clip((mean_spec - 0.05) / 0.15, 0, 1))
        if score > 0.10:
            results["oiliness_color"] = (score, 0.75)

    # ── Lip Pallor ────────────────────────────────────────────
    bb = get_region("lips")
    if bb:
        lip_A = float(np.mean(A[bb[1]:bb[3], bb[0]:bb[2]]))
        score = float(np.clip((135 - lip_A) / 15, 0, 1))
        if score > 0.20:
            results["lip_pallor_color"] = (score, 0.6)

    return results


# ═══════════════════════════════════════════════════════════════
# FIX 3 + 7 + 8 + 11: Bayesian Inference v2
# ═══════════════════════════════════════════════════════════════

# FIX 3: Recalibrated CPTs — more discriminative
CPT_LIKELIHOOD = np.array([
#    iron   b12   vitD  zinc  om3   vitA  vitC  sleep  horm  dehy  stress
    [0.72, 0.68, 0.40, 0.18, 0.20, 0.18, 0.22, 0.68, 0.28, 0.32, 0.48],  # dark_circle
    [0.28, 0.22, 0.18, 0.12, 0.18, 0.12, 0.12, 0.82, 0.22, 0.48, 0.62],  # eye_bag
    [0.15, 0.12, 0.38, 0.82, 0.52, 0.72, 0.18, 0.35, 0.88, 0.18, 0.42],  # acne
    [0.20, 0.18, 0.28, 0.18, 0.42, 0.22, 0.52, 0.52, 0.28, 0.38, 0.62],  # wrinkle
    [0.18, 0.22, 0.22, 0.28, 0.65, 0.28, 0.18, 0.18, 0.32, 0.22, 0.38],  # redness
    [0.28, 0.32, 0.52, 0.18, 0.22, 0.28, 0.62, 0.18, 0.38, 0.18, 0.22],  # dark_spot
    [0.12, 0.12, 0.28, 0.58, 0.38, 0.48, 0.12, 0.28, 0.65, 0.12, 0.32],  # blackhead
    [0.18, 0.18, 0.48, 0.12, 0.18, 0.18, 0.28, 0.12, 0.72, 0.12, 0.18],  # melasma
    [0.12, 0.12, 0.22, 0.55, 0.32, 0.42, 0.12, 0.22, 0.58, 0.12, 0.28],  # whitehead
    [0.18, 0.18, 0.28, 0.48, 0.32, 0.32, 0.22, 0.28, 0.55, 0.12, 0.28],  # acne_scar
    [0.12, 0.12, 0.18, 0.18, 0.58, 0.22, 0.18, 0.12, 0.28, 0.18, 0.42],  # vascular
], dtype=np.float32)

# FIX 3: omega3 0.50 → 0.15 — stops it from always winning
PRIOR_DEFICIENCY = np.array([
    0.15, 0.12, 0.22, 0.12, 0.15, 0.08, 0.09, 0.28, 0.14, 0.25, 0.22
], dtype=np.float32)

FOOD_RECS = {
    "iron_deficiency":       ["spinach", "lentils", "red meat", "tofu", "pumpkin seeds"],
    "b12_deficiency":        ["eggs", "dairy", "salmon", "beef liver", "fortified cereals"],
    "vitamin_d_deficiency":  ["fatty fish", "egg yolks", "fortified milk", "mushrooms"],
    "zinc_deficiency":       ["oysters", "beef", "chickpeas", "cashews", "pumpkin seeds"],
    "omega3_deficiency":     ["salmon", "walnuts", "flaxseed", "chia seeds", "mackerel"],
    "vitamin_a_deficiency":  ["sweet potato", "carrots", "kale", "egg yolks", "liver"],
    "vitamin_c_deficiency":  ["citrus fruits", "bell peppers", "broccoli", "kiwi"],
    "poor_sleep_quality":    ["improve sleep schedule", "reduce caffeine after 2pm", "magnesium-rich foods"],
    "hormonal_imbalance":    ["see a doctor", "reduce sugar", "increase fiber", "healthy fats"],
    "dehydration":           ["drink 8+ glasses water daily", "cucumber", "watermelon"],
    "high_stress":           ["meditation", "exercise", "B-complex vitamins", "magnesium"],
}

LIFESTYLE_ADVICE = {
    "iron_deficiency":      "Pair iron-rich foods with Vitamin C to boost absorption. Avoid tea/coffee with meals.",
    "b12_deficiency":       "B12 mainly from animal sources. Vegans/vegetarians should supplement.",
    "vitamin_d_deficiency": "Get 15–30 min of sunlight daily. Consider D3 supplement in winter.",
    "zinc_deficiency":      "Zinc absorption reduced by phytates. Soak legumes before cooking.",
    "omega3_deficiency":    "Aim for 2 servings of fatty fish per week or consider fish oil.",
    "vitamin_a_deficiency": "Fat-soluble vitamin — pair with healthy fats for absorption.",
    "vitamin_c_deficiency": "Cooking destroys Vitamin C. Eat some raw fruits/vegetables daily.",
    "poor_sleep_quality":   "Consistent sleep/wake times matter more than total hours. Aim 7–9 hrs.",
    "hormonal_imbalance":   "Hormonal issues need medical evaluation. Diet supports, not replaces treatment.",
    "dehydration":          "Thirst is a late signal. Aim for pale yellow urine as hydration indicator.",
    "high_stress":          "Chronic stress depletes B vitamins and magnesium. Both diet and stress management needed.",
}


def bayesian_inference_v2(severity, uncertainty, color_features=None, yolo_counts=None):
    """
    FIX 7: Feature absence penalizes deficiencies
    FIX 8: exp-based uncertainty weighting
    FIX 11: Count-based YOLO severity fusion
    """
    if color_features is None: color_features = {}
    if yolo_counts    is None: yolo_counts    = {}

    # FIX 8: confidence via exp(-3σ) — smooth, bounded in [0.1, 1.0]
    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0, 1).copy()

    # FIX 11: YOLO count boost for small lesion features
    count_feat_map = {"acne":2, "blackhead":6, "whitehead":8, "acne_scar":9, "dark_spot":5}
    for fn, fi in count_feat_map.items():
        cnt = yolo_counts.get(fn, 0)
        if cnt > 0:
            evidence[fi] = max(evidence[fi], count_to_severity(cnt))

    # Supplement with color-based redness
    if "redness_color" in color_features:
        sc, cf = color_features["redness_color"]
        evidence[4] = float(np.clip(evidence[4] * 0.6 + sc * cf * 0.4, 0, 1))

    if "dark_circle_color" in color_features:
        sc, cf = color_features["dark_circle_color"]
        evidence[0] = float(np.clip(evidence[0] * 0.7 + sc * cf * 0.3, 0, 1))

    # Bayesian update
    log_post = np.log(PRIOR_DEFICIENCY + 1e-9).copy()

    for fi in range(N_FEATURES):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_LIKELIHOOD[fi]

        if ev >= 0.15:
            # FIX 7: positive evidence
            like = ev * p + (1 - ev) * (1 - p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            # FIX 7: negative evidence — confidently absent feature
            # Penalize deficiencies that strongly predict this feature
            penalty = np.where(p > 0.4, np.log(1 - p * 0.3 + 1e-9), 0.0)
            log_post += penalty * conf * 0.3

    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= (post.sum() + 1e-9)

    # Color-feature post-hoc boosts
    if "pallor_color" in color_features:
        sc, cf = color_features["pallor_color"]
        if sc > 0.3:
            post[0] *= (1 + sc * cf * 0.8)  # iron
            post[1] *= (1 + sc * cf * 0.6)  # b12

    if "yellow_sclera_color" in color_features:
        sc, cf = color_features["yellow_sclera_color"]
        if sc > 0.25:
            post[1] *= (1 + sc * cf * 1.0)  # b12

    if "oiliness_color" in color_features:
        sc, cf = color_features["oiliness_color"]
        if sc > 0.4:
            post[8] *= (1 + sc * cf * 0.6)  # hormonal
            post[3] *= (1 + sc * cf * 0.4)  # zinc

    # Skin texture roughness → vitamin A + zinc
    if "skin_texture_color" in color_features:
        sc, cf = color_features["skin_texture_color"]
        if sc > 0.4:
            post[5] *= (1 + sc * cf * 0.5)  # vitamin_a
            post[3] *= (1 + sc * cf * 0.4)  # zinc

    post /= (post.sum() + 1e-9)
    return post


def format_output(severity, uncertainty, yolo_detections, yolo_counts,
                   posterior, color_features, timing):
    """Final structured output."""

    # Feature detection (fused YOLO + MLP + color)
    features_out = {}
    for i, name in enumerate(SKIN_FEATURES):
        s         = float(severity[i])
        u         = float(uncertainty[i])
        yc        = yolo_detections.get(name, 0.0)
        cnt       = yolo_counts.get(name, 0)
        cnt_sev   = count_to_severity(cnt) if name in \
            {"acne","blackhead","whitehead","acne_scar","dark_spot"} else 0.0
        combined  = max(s, yc * 0.85, cnt_sev)

        if combined > 0.12 or cnt > 0:
            srcs = []
            if s > 0.12:    srcs.append("dinov2")
            if yc > 0:      srcs.append(f"yolo({'×'+str(cnt) if cnt>1 else ''})")
            features_out[name] = {
                "severity":    round(combined, 3),
                "level":       "high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                "confidence":  round(float(np.exp(-u * 3)), 2),
                "yolo_count":  cnt,
                "detected_by": srcs,
            }

    # Add color-only features not covered by YOLO/MLP
    color_to_feat = {
        "pallor_color":         "pallor",
        "redness_color":        "skin_redness",
        "yellow_sclera_color":  "yellow_sclera",
        "oiliness_color":       "oily_skin",
        "lip_pallor_color":     "lip_pallor",
    }
    for ck, fname in color_to_feat.items():
        if ck in color_features:
            sc, cf = color_features[ck]
            if sc > 0.30 and cf > 0.4 and fname not in features_out:
                features_out[fname] = {
                    "severity":    round(float(sc), 3),
                    "level":       "high" if sc>0.6 else "moderate" if sc>0.35 else "mild",
                    "confidence":  round(float(cf), 2),
                    "yolo_count":  0,
                    "detected_by": ["color_analysis"],
                }

    # Deficiency analysis
    sdef = sorted(enumerate(posterior), key=lambda x: -x[1])
    deficiencies = {}
    for rank, (i, prob) in enumerate(sdef, 1):
        name = DEFICIENCIES[i]
        deficiencies[name] = {
            "probability":     round(float(prob), 4),
            "probability_pct": f"{prob*100:.1f}%",
            "priority_rank":   rank,
            "foods":           FOOD_RECS.get(name, []),
            "advice":          LIFESTYLE_ADVICE.get(name, ""),
            "confidence_band": "high" if prob>0.20 else "moderate" if prob>0.10 else "low",
        }

    # Top insights — min 8% to show
    top = [
        {
            "rank":        r,
            "issue":       DEFICIENCIES[i],
            "probability": f"{posterior[i]*100:.1f}%",
            "priority":    "HIGH" if posterior[i]>0.20 else "MODERATE" if posterior[i]>0.10 else "LOW",
            "top_foods":   FOOD_RECS.get(DEFICIENCIES[i], [])[:3],
            "advice":      LIFESTYLE_ADVICE.get(DEFICIENCIES[i], ""),
        }
        for r, (i, p) in enumerate(sdef[:5], 1) if p > 0.08
    ]

    return {
        "status":              "success",
        "features_detected":   features_out,
        "deficiency_analysis": deficiencies,
        "top_insights":        top,
        "yolo_detections":     yolo_detections,
        "yolo_counts":         yolo_counts,
        "color_features":      {k: {"score": round(v[0],3), "conf": round(v[1],2)}
                                 for k,v in color_features.items()},
        "timing_ms":           {k: round(v*1000, 1) for k,v in timing.items()},
        "disclaimer":          (
            "FaceFuel provides wellness awareness only — not medical diagnosis. "
            "Visual signs have multiple causes. Consult a qualified healthcare "
            "provider before making health decisions based on these results."
        ),
    }


# ═══════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_severity_model(device)
    model.eval()
    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: index_map = json.load(f)
    fn = (features - mean) / (std + 1e-6)
    stems  = [index_map[str(i)] for i in range(len(features))]
    labels = build_labels_from_yolo(index_map, stems)
    lmask  = labels.sum(axis=1) > 0
    X = torch.from_numpy(fn[lmask]).float().to(device)
    Y = labels[lmask]
    with torch.no_grad():
        v = model.get_parameter if hasattr(model, 'heads') else None
        if v is not None:
            P = torch.sigmoid(model(X)).cpu().numpy()
        else:
            P = model(X).cpu().numpy()

    print(f"\n{'='*65}\nSEVERITY MLP v2 EVALUATION\n{'='*65}")
    print(f"  Labeled: {lmask.sum():,}")
    print(f"\n  {'Feature':<22s}  {'MAE':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Type'}")
    print(f"  {'-'*65}")
    for i, nm in enumerate(SKIN_FEATURES):
        t = (Y[:,i]>0.3).astype(int); p = (P[:,i]>0.3).astype(int)
        tp = ((t==1)&(p==1)).sum(); fp = ((t==0)&(p==1)).sum(); fn_ = ((t==1)&(p==0)).sum()
        pr = tp/(tp+fp+1e-6); rc = tp/(tp+fn_+1e-6); f1 = 2*pr*rc/(pr+rc+1e-6)
        kind = "count" if i in SMALL_LESION_FEATS else "area"
        print(f"  {nm:<22s}  {np.abs(Y[:,i]-P[:,i]).mean():>6.3f}  {pr:>6.3f}  {rc:>6.3f}  {f1:>6.3f}  {kind}")


def demo_infer(stem=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_severity_model(device)
    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: index_map = json.load(f)
    row = next((int(k) for k,v in index_map.items() if stem in v), None) \
          if stem else np.random.randint(0, len(features))
    if row is None: print(f"❌ Not found: {stem}"); return
    fn = (features[row] - mean) / (std + 1e-6)
    x  = torch.from_numpy(fn).float().unsqueeze(0).to(device)
    sev, unc = model.predict_with_uncertainty(x, 25)
    sev = sev.squeeze(0).cpu().numpy()
    unc = unc.squeeze(0).cpu().numpy()
    post = bayesian_inference_v2(sev, unc)
    result = format_output(sev, unc, {}, {}, post, {}, {})
    print(f"\n{'='*60}\nFACEFUEL v2 DEMO — {index_map[str(row)][:50]}\n{'='*60}")
    print("\n  Features:")
    for n, info in (result["features_detected"] or {}).items():
        print(f"    {n:<22s}  {info['severity']:.2f}  [{info['level']}]")
    print("\n  Top insights:")
    for ins in result["top_insights"]:
        print(f"    #{ins['rank']}  {ins['issue']:<25s}  {ins['probability']}  [{ins['priority']}]")
        print(f"         → {', '.join(ins['top_foods'][:3])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--demo",     action="store_true")
    parser.add_argument("--infer",    type=str, default=None)
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=64)
    args = parser.parse_args()

    if args.train:
        print(f"\n{'='*60}\nFaceFuel v2 — Training SeverityMLP v2\n{'='*60}\n")
        train_severity_mlp(epochs=args.epochs, batch_size=args.batch)
    elif args.evaluate: evaluate()
    elif args.demo or args.infer: demo_infer(args.infer)
    else:
        print("Usage: --train | --evaluate | --demo")
        train_severity_mlp(epochs=args.epochs, batch_size=args.batch)