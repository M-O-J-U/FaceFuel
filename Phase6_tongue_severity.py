"""
FaceFuel — Phase 6: Tongue Severity MLP + Bayesian Engine
==========================================================
Mirrors the face pipeline (step9) but for tongue features.

Architecture:
  Input : 1920-dim DINOv2 feature vector (5 regions × 384)
  Backbone: 1920 → 512 → 256 (shared)
  Heads:  11 per-feature heads → 256 → 64 → 1 (logit)
  Loss:   BCEWithLogitsLoss with pos_weight
  Output: 11 severity scores in [0,1]

Tongue features → Nutrient deficiencies (CPT):
  fissured      → B3, dehydration, stress
  crenated      → Zinc, thyroid, fluid retention
  pale_tongue   → Iron, B12, anemia
  red_tongue    → B12/folate deficiency, inflammation
  yellow_coating → Liver stress, digestive issues
  white_coating  → Candida, gut dysbiosis
  thick_coating  → Digestive stagnation
  geographic     → Zinc, B-complex
  smooth_glossy  → Iron, B12, folate (atrophied papillae)
  tooth_marked   → Zinc, fluid retention, hypothyroid

Run:
  python tongue_severity.py --train
  python tongue_severity.py --evaluate
"""

import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Paths ─────────────────────────────────────────────────────
BASE        = Path("tongue_datasets")
FEAT_MATRIX = BASE / "features" / "tongue_feature_matrix.npy"
FEAT_INDEX  = BASE / "features" / "tongue_feature_index.json"
FEAT_STATS  = BASE / "features" / "tongue_feature_stats.json"
LABELS_DIR  = BASE / "TONGUE_COMBINED"
MODEL_DIR   = Path("facefuel_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = MODEL_DIR / "tongue_severity_mlp.pt"

# ── Feature config ────────────────────────────────────────────
TONGUE_FEATURES = [
    "tongue_body",    # 0 — always present (segmentation)
    "fissured",       # 1
    "crenated",       # 2
    "pale_tongue",    # 3
    "red_tongue",     # 4
    "yellow_coating", # 5
    "white_coating",  # 6
    "thick_coating",  # 7
    "geographic",     # 8
    "smooth_glossy",  # 9
    "tooth_marked",   # 10
]
N_FEATURES = len(TONGUE_FEATURES)
FEAT_DIM   = 1920

# Count-based classes (small/multiple lesions)
COUNT_FEATS = {1, 2, 8, 10, 11}  # fissured, crenated, geographic, tooth_marked, black_hairy

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
    "liver_stress",          # 11
    "gut_dysbiosis",         # 12
    "hypothyroid",           # 13
    "folate_deficiency",     # 14
]
N_DEF = len(DEFICIENCIES)

# ── Tongue CPT ────────────────────────────────────────────────
# P(feature | deficiency) — clinical literature values
# Rows = tongue features (11), Cols = deficiencies (15)
CPT_TONGUE = np.array([
#    iron  b12   vitD  zinc  om3   vitA  vitC  sleep horm  dehy  stress liver gut   thyrd folat
    [0.20, 0.18, 0.10, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],  # tongue_body (always present)
    [0.18, 0.22, 0.15, 0.28, 0.18, 0.20, 0.15, 0.25, 0.18, 0.45, 0.48, 0.22, 0.18, 0.20, 0.18],  # fissured → B3/dehydration/stress
    [0.15, 0.18, 0.20, 0.72, 0.18, 0.22, 0.15, 0.20, 0.25, 0.28, 0.22, 0.18, 0.18, 0.68, 0.18],  # crenated → zinc/thyroid
    [0.82, 0.78, 0.22, 0.18, 0.20, 0.18, 0.18, 0.18, 0.20, 0.15, 0.15, 0.18, 0.15, 0.18, 0.42],  # pale_tongue → iron/B12/anemia
    [0.28, 0.85, 0.18, 0.22, 0.25, 0.18, 0.22, 0.18, 0.22, 0.18, 0.22, 0.28, 0.22, 0.18, 0.88],  # red_tongue → B12/folate
    [0.15, 0.18, 0.18, 0.22, 0.18, 0.18, 0.18, 0.20, 0.22, 0.18, 0.20, 0.82, 0.42, 0.18, 0.18],  # yellow_coating → liver
    [0.12, 0.15, 0.18, 0.22, 0.15, 0.18, 0.15, 0.18, 0.22, 0.18, 0.22, 0.35, 0.88, 0.18, 0.15],  # white_coating → candida/gut
    [0.15, 0.18, 0.18, 0.22, 0.18, 0.18, 0.18, 0.22, 0.22, 0.18, 0.25, 0.52, 0.62, 0.18, 0.18],  # thick_coating → digestive
    [0.18, 0.20, 0.18, 0.72, 0.20, 0.22, 0.18, 0.18, 0.20, 0.18, 0.20, 0.18, 0.18, 0.18, 0.22],  # geographic → zinc/B-complex
    [0.72, 0.82, 0.18, 0.22, 0.20, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.78],  # smooth_glossy → iron/B12/folate
    [0.18, 0.20, 0.18, 0.68, 0.18, 0.20, 0.18, 0.22, 0.25, 0.35, 0.22, 0.18, 0.20, 0.72, 0.18],  # tooth_marked → zinc/fluid/thyroid
], dtype=np.float32)

PRIOR_DEFICIENCY = np.array([
    0.15, 0.12, 0.22, 0.12, 0.15, 0.08, 0.09, 0.28,
    0.14, 0.25, 0.22, 0.10, 0.08, 0.08, 0.10
], dtype=np.float32)

FOOD_RECS_TONGUE = {
    "iron_deficiency":      ["spinach", "lentils", "red meat", "tofu", "pumpkin seeds"],
    "b12_deficiency":       ["eggs", "dairy", "salmon", "beef liver", "fortified cereals"],
    "vitamin_d_deficiency": ["fatty fish", "egg yolks", "fortified milk", "mushrooms"],
    "zinc_deficiency":      ["oysters", "beef", "chickpeas", "cashews", "pumpkin seeds"],
    "omega3_deficiency":    ["salmon", "walnuts", "flaxseed", "chia seeds", "mackerel"],
    "vitamin_a_deficiency": ["sweet potato", "carrots", "kale", "egg yolks", "liver"],
    "vitamin_c_deficiency": ["citrus fruits", "bell peppers", "broccoli", "kiwi"],
    "poor_sleep_quality":   ["improve sleep schedule", "reduce caffeine after 2pm", "magnesium"],
    "hormonal_imbalance":   ["see a doctor", "reduce sugar", "healthy fats", "fiber"],
    "dehydration":          ["drink 8+ glasses water daily", "cucumber", "watermelon"],
    "high_stress":          ["meditation", "exercise", "B-complex vitamins", "magnesium"],
    "liver_stress":          ["reduce alcohol", "milk thistle tea", "leafy greens", "beets"],
    "gut_dysbiosis":         ["probiotics", "fermented foods", "reduce sugar", "fiber"],
    "hypothyroid":           ["consult doctor", "iodine-rich foods", "selenium", "zinc"],
    "folate_deficiency":     ["leafy greens", "lentils", "asparagus", "fortified cereals"],
}

FOOD_RECS_TONGUE["black_hairy_tongue_marker"] = ["probiotics", "reduce antibiotics use", "quit smoking", "improve oral hygiene"]


def count_to_severity(n: int) -> float:
    if n == 0:  return 0.0
    if n == 1:  return 0.40
    if n <= 3:  return 0.55
    if n <= 6:  return 0.70
    if n <= 10: return 0.82
    return 1.0


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class TongueSeverityMLP(nn.Module):
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
                nn.Linear(64, 1)
            )
            for _ in range(n_features)
        ])

    def forward(self, x):
        s = self.backbone(x)
        return torch.cat([h(s) for h in self.heads], dim=-1)

    def predict_with_uncertainty(self, x, n_passes=25):
        self.train()
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(self.forward(x))
                                  for _ in range(n_passes)])
        self.eval()
        return probs.mean(0), probs.std(0)


# ═══════════════════════════════════════════════════════════════
# LABEL BUILDER
# ═══════════════════════════════════════════════════════════════

def build_labels(index_map: dict) -> np.ndarray:
    """Build severity labels from YOLO annotation files."""
    N      = len(index_map)
    labels = np.zeros((N, N_FEATURES), dtype=np.float32)

    # Build lookup: stem → label file path
    lbl_lookup = {}
    for split in ["train", "val"]:
        lbl_dir = LABELS_DIR / split / "labels"
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                # Strip pseudo prefix
                stem = lbl_file.stem.lstrip("p_") if lbl_file.stem.startswith("p_") \
                       else lbl_file.stem
                lbl_lookup[stem]             = lbl_file
                lbl_lookup[lbl_file.stem]    = lbl_file   # also raw

    for i, stem in index_map.items():
        lbl_path = lbl_lookup.get(stem) or lbl_lookup.get(f"p_{stem}")
        if not lbl_path or not lbl_path.exists():
            continue

        lines   = lbl_path.read_text(errors="ignore").strip().splitlines()
        counts  = defaultdict(int)
        max_conf= defaultdict(float)

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
            if 0 <= cls_id < N_FEATURES:
                counts[cls_id]   += 1
                if area > max_conf[cls_id]:
                    max_conf[cls_id] = area

        for fi in set(list(counts.keys()) + list(max_conf.keys())):
            if fi in COUNT_FEATS:
                sev = count_to_severity(counts[fi])
            elif fi == 0:  # tongue_body
                sev = min(max_conf.get(fi, 0) / 0.70, 1.0)
            else:
                sev = min(max_conf.get(fi, 0) / 0.60, 1.0)
            labels[int(i), fi] = min(float(sev), 1.0)

    return labels


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class TongueDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def stratified_split(labels, val_frac=0.20, seed=42):
    rng = np.random.default_rng(seed)
    N   = len(labels)
    idx = np.arange(N)
    rng.shuffle(idx)
    sp  = int(N * (1 - val_frac))
    return idx[:sp], idx[sp:]


def train(epochs=100, batch_size=64, lr=2e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Phase 6 — Training Tongue Severity MLP")
    print(f"  Device: {device}  Epochs: {epochs}")
    print(f"{'='*60}")

    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: idx_map = json.load(f)
    with open(FEAT_STATS)  as f: stats  = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    feat_norm = (features - mean) / (std + 1e-6)

    labels = build_labels({int(k): v for k, v in idx_map.items()})
    lmask  = labels.sum(axis=1) > 0

    print(f"\n  Labeled: {lmask.sum():,} / {len(labels):,}")
    print(f"\n  Label stats:")
    for i, name in enumerate(TONGUE_FEATURES):
        nz  = (labels[:, i] > 0).sum()
        msv = labels[labels[:, i] > 0, i].mean() if nz > 0 else 0
        tag = "count" if i in COUNT_FEATS else "area"
        print(f"    {name:<18s}  present={nz:4d}  mean={msv:.3f}  ({tag})")

    train_idx, val_idx = stratified_split(labels[lmask].sum(axis=1) > 0)
    lmask_idx = np.where(lmask)[0]
    train_idx = lmask_idx[train_idx]
    val_idx   = lmask_idx[val_idx]
    print(f"\n  Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    # pos_weight
    n_total = int(lmask.sum())
    pos_w   = torch.tensor([
        float(n_total) / max((labels[lmask, i] > 0).sum(), 1)
        for i in range(N_FEATURES)
    ]).clamp(max=25.0).to(device)
    print(f"  pos_weight: {[f'{v:.1f}' for v in pos_w.tolist()]}")

    # Weighted sampler
    n_pos = np.array([(labels[train_idx, i] > 0).sum() for i in range(N_FEATURES)])
    n_neg = len(train_idx) - n_pos
    sw    = np.ones(len(train_idx))
    for ii, orig in enumerate(train_idx):
        mw = 1.0
        for fi in range(N_FEATURES):
            if labels[orig, fi] > 0 and n_pos[fi] > 0:
                mw = max(mw, (n_neg[fi] / (n_pos[fi] + 1e-6)) ** 0.4)
        sw[ii] = mw
    sampler = WeightedRandomSampler(torch.from_numpy(sw).float(), len(train_idx), True)

    train_dl = DataLoader(TongueDataset(feat_norm[train_idx], labels[train_idx]),
                          batch_size=batch_size, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(TongueDataset(feat_norm[val_idx],   labels[val_idx]),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model     = TongueSeverityMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_dl), pct_start=0.1)

    best_f1 = 0.0

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
            t  = (L[:, fi] > 0.3).astype(int)
            p  = (P[:, fi] > 0.3).astype(int)
            tp = ((t==1)&(p==1)).sum()
            fp = ((t==0)&(p==1)).sum()
            fn = ((t==1)&(p==0)).sum()
            pr = tp/(tp+fp+1e-6); rc = tp/(tp+fn+1e-6)
            f1s.append(2*pr*rc/(pr+rc+1e-6))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={np.mean(tl):.4f}  val={np.mean(vl):.4f}  "
                  f"mean_F1={np.mean(f1s):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if sum(f1s) > best_f1:
            best_f1 = sum(f1s)
            torch.save({
                "model_state":       model.state_dict(),
                "mean":              mean.tolist(),
                "std":               std.tolist(),
                "tongue_features":   TONGUE_FEATURES,
                "deficiencies":      DEFICIENCIES,
                "epoch":             epoch,
                "mean_f1":           float(np.mean(f1s)),
                "f1_per_feature":    [float(f) for f in f1s],
                "version":           1,
            }, str(MODEL_PATH))

    print(f"\n  ✅ Best mean F1: {best_f1/N_FEATURES:.4f}")
    print(f"  Model saved: {MODEL_PATH}")
    print(f"\n  Next: python tongue_inference.py --test")


# ═══════════════════════════════════════════════════════════════
# BAYESIAN INFERENCE
# ═══════════════════════════════════════════════════════════════

def tongue_bayesian_inference(severity, uncertainty,
                               yolo_counts=None, color_features=None):
    """Bayesian posterior over deficiencies from tongue features."""
    if yolo_counts    is None: yolo_counts    = {}
    if color_features is None: color_features = {}

    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0, 1).copy()

    # Count-based override for structural features
    for fn, fi in [("fissured",2),("crenated",3),("geographic",9),("tooth_marked",11)]:
        n = yolo_counts.get(fn, 0)
        if n > 0:
            evidence[fi] = max(evidence[fi], count_to_severity(n))

    log_post = np.log(PRIOR_DEFICIENCY + 1e-9).copy()

    for fi in range(N_FEATURES):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_TONGUE[fi]
        if ev >= 0.15:
            like = ev * p + (1 - ev) * (1 - p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            log_post += np.where(p > 0.4, np.log(1 - p*0.3 + 1e-9), 0.0) * conf * 0.3

    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= (post.sum() + 1e-9)
    return post


# ═══════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model  = TongueSeverityMLP()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: idx_map = json.load(f)
    with open(FEAT_STATS)  as f: stats  = json.load(f)
    mean     = np.array(stats["mean"], dtype=np.float32)
    std      = np.array(stats["std"],  dtype=np.float32)
    fn       = (features - mean) / (std + 1e-6)
    labels   = build_labels({int(k): v for k, v in idx_map.items()})
    lmask    = labels.sum(axis=1) > 0
    X        = torch.from_numpy(fn[lmask]).float().to(device)
    Y        = labels[lmask]

    with torch.no_grad():
        P = torch.sigmoid(model(X)).cpu().numpy()

    print(f"\n{'='*65}\nTONGUE SEVERITY MLP EVALUATION\n{'='*65}")
    print(f"  {'Feature':<20s}  {'MAE':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Type'}")
    print(f"  {'-'*60}")
    for i, nm in enumerate(TONGUE_FEATURES):
        t  = (Y[:,i] > 0.3).astype(int)
        p  = (P[:,i] > 0.3).astype(int)
        tp = ((t==1)&(p==1)).sum(); fp = ((t==0)&(p==1)).sum()
        fn_= ((t==1)&(p==0)).sum()
        pr = tp/(tp+fp+1e-6); rc = tp/(tp+fn_+1e-6)
        f1 = 2*pr*rc/(pr+rc+1e-6)
        kind = "count" if i in COUNT_FEATS else "area"
        print(f"  {nm:<20s}  {np.abs(Y[:,i]-P[:,i]).mean():>6.3f}  "
              f"{pr:>6.3f}  {rc:>6.3f}  {f1:>6.3f}  {kind}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=64)
    args = parser.parse_args()

    if args.train or (not args.evaluate):
        train(epochs=args.epochs, batch_size=args.batch)
    if args.evaluate:
        evaluate()