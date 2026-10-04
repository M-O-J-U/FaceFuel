"""
FaceFuel — Eye Module: Severity MLP + Bayesian Engine
======================================================
Trains per-feature severity MLP on eye DINOv2 features.

Eye features → Deficiency mapping:
  conjunctival_pallor → iron_deficiency, b12_deficiency, anemia
  scleral_icterus     → liver_stress (already in tongue), bilirubin
  xanthelasma         → cholesterol_imbalance (NEW), omega3_deficiency

New deficiency dimensions added:
  cholesterol_imbalance  (xanthelasma + arcus_senilis proxy)

Run:
  python eye_severity.py --train
  python eye_severity.py --evaluate
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE        = Path("eye_datasets")
FEAT_MATRIX = BASE / "features" / "eye_feature_matrix.npy"
FEAT_INDEX  = BASE / "features" / "eye_feature_index.json"
FEAT_STATS  = BASE / "features" / "eye_feature_stats.json"
LABELS_DIR  = BASE / "EYE_COMBINED"
MODEL_DIR   = Path("facefuel_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH  = MODEL_DIR / "eye_severity_mlp.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EYE_FEATURES = [
    "conjunctival_pallor",   # 0
    "scleral_icterus",       # 1
    "xanthelasma",           # 2
]
N_FEATURES = 3
FEAT_DIM   = 1152

# Eye deficiency CPT (15 original + 1 new = 16 total)
# cholesterol_imbalance is the new eye-exclusive dimension
EYE_DEFICIENCIES = [
    "iron_deficiency",        # 0  — conjunctival pallor (strong signal)
    "b12_deficiency",         # 1  — conjunctival pallor (moderate)
    "vitamin_d_deficiency",   # 2
    "zinc_deficiency",        # 3
    "omega3_deficiency",      # 4  — xanthelasma (moderate)
    "vitamin_a_deficiency",   # 5
    "vitamin_c_deficiency",   # 6
    "poor_sleep_quality",     # 7
    "hormonal_imbalance",     # 8
    "dehydration",            # 9
    "high_stress",            # 10
    "liver_stress",           # 11 — scleral icterus (strong signal)
    "gut_dysbiosis",          # 12
    "hypothyroid",            # 13
    "folate_deficiency",      # 14
    "cholesterol_imbalance",  # 15 — EYE-EXCLUSIVE (xanthelasma)
]
N_DEF = len(EYE_DEFICIENCIES)

# CPT: P(feature | deficiency)
# Rows = eye features (3), Cols = deficiencies (16)
CPT_EYE = np.array([
#   iron  b12  vitD zinc  om3  vitA vitC sleep horm dehy strs livr gut  thyr folat chol
   [0.82, 0.72, 0.15, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.20, 0.18, 0.25, 0.15, 0.18, 0.42, 0.12],  # pallor→iron/B12/folate
   [0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15, 0.88, 0.18, 0.15, 0.12, 0.15],  # icterus→liver
   [0.12, 0.12, 0.12, 0.12, 0.52, 0.12, 0.12, 0.12, 0.22, 0.12, 0.18, 0.15, 0.12, 0.18, 0.12, 0.95],  # xanthelasma→cholesterol/omega3
], dtype=np.float32)

PRIOR_EYE = np.array([
    0.15, 0.12, 0.22, 0.12, 0.15, 0.08, 0.09,
    0.28, 0.14, 0.25, 0.22, 0.10, 0.08, 0.08, 0.10,
    0.12,   # cholesterol_imbalance prior
], dtype=np.float32)

FOOD_RECS_EYE = {
    "conjunctival_pallor": ["spinach", "lentils", "red meat", "eggs", "dairy"],
    "scleral_icterus":     ["reduce alcohol", "leafy greens", "beets", "milk thistle"],
    "xanthelasma":         ["salmon", "walnuts", "flaxseed", "reduce saturated fats"],
    "cholesterol_imbalance": ["oats", "beans", "avocado", "olive oil", "salmon"],
}


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class EyeSeverityMLP(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_features=N_FEATURES, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128,64), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(64,1))
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
    N      = len(index_map)
    labels = np.zeros((N, N_FEATURES), dtype=np.float32)

    lbl_lookup = {}
    for split in ["train", "val"]:
        lbl_dir = LABELS_DIR / split / "labels"
        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                lbl_lookup[lf.stem] = lf

    for i, stem in index_map.items():
        lbl_path = lbl_lookup.get(stem)
        if not lbl_path or not lbl_path.exists():
            continue
        for line in lbl_path.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5: continue
            try:
                cls_id = int(parts[0])
                w, h   = float(parts[3]), float(parts[4])
                if 0 <= cls_id < N_FEATURES:
                    sev = min(w*h/0.50, 1.0)
                    labels[int(i), cls_id] = max(labels[int(i), cls_id], sev)
            except ValueError:
                continue
    return labels


# ═══════════════════════════════════════════════════════════════
# BAYESIAN INFERENCE
# ═══════════════════════════════════════════════════════════════

def eye_bayesian_inference(severity: np.ndarray,
                            uncertainty: np.ndarray,
                            yolo_dets: dict = None) -> np.ndarray:
    """Bayesian posterior over 16 deficiencies from eye features."""
    if yolo_dets is None: yolo_dets = {}

    confidence = np.exp(-uncertainty * 3.0).clip(0.1, 1.0)
    evidence   = (severity * confidence).clip(0, 1).copy()

    # YOLO detection override for high-confidence cases
    feat_map = {"conjunctival_pallor": 0, "scleral_icterus": 1, "xanthelasma": 2}
    for fname, fi in feat_map.items():
        yolo_conf = yolo_dets.get(fname, 0.0)
        if yolo_conf > 0.5:
            evidence[fi] = max(evidence[fi], yolo_conf * 0.9)

    log_post = np.log(PRIOR_EYE + 1e-9).copy()

    for fi in range(N_FEATURES):
        ev   = float(evidence[fi])
        conf = float(confidence[fi])
        p    = CPT_EYE[fi]
        if ev >= 0.15:
            like = ev*p + (1-ev)*(1-p)
            log_post += np.log(like + 1e-9) * ev * conf
        elif ev < 0.05 and conf > 0.5:
            log_post += np.where(p>0.4, np.log(1-p*0.3+1e-9), 0.0)*conf*0.3

    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / (post.sum() + 1e-9)


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class EyeDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def train(epochs=120, batch_size=64, lr=2e-4):
    print(f"\n{'='*60}")
    print(f"Eye Severity MLP — Training")
    print(f"  Device: {DEVICE}  Epochs: {epochs}")
    print(f"{'='*60}")

    features = np.load(str(FEAT_MATRIX))
    with open(FEAT_INDEX) as f: idx_map = json.load(f)
    with open(FEAT_STATS)  as f: stats  = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    fn   = (features - mean) / (std + 1e-6)

    labels = build_labels({int(k): v for k, v in idx_map.items()})
    lmask  = labels.sum(axis=1) > 0

    print(f"\n  Labeled: {lmask.sum():,} / {len(labels):,}")
    print(f"  Label stats:")
    for i, name in enumerate(EYE_FEATURES):
        nz  = (labels[:,i] > 0).sum()
        msv = labels[labels[:,i]>0,i].mean() if nz>0 else 0
        print(f"    {name:<22s}  present={nz:4d}  mean={msv:.3f}")

    # Split
    rng  = np.random.default_rng(42)
    idx  = np.where(lmask)[0]; rng.shuffle(idx)
    sp   = int(len(idx)*0.8)
    ti, vi = idx[:sp], idx[sp:]
    print(f"  Train: {len(ti):,}  Val: {len(vi):,}")

    # pos_weight
    n_total = int(lmask.sum())
    pos_w   = torch.tensor([
        float(n_total)/max((labels[lmask,i]>0).sum(),1)
        for i in range(N_FEATURES)
    ]).clamp(max=25.0).to(DEVICE)
    print(f"  pos_weight: {[f'{v:.1f}' for v in pos_w.tolist()]}")

    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model     = EyeSeverityMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(len(ti) // batch_size, 1)
    # Add 1 to total_steps to prevent boundary error on last epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch + 1, pct_start=0.1)

    train_dl = DataLoader(EyeDataset(fn[ti], labels[ti]),
                          batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(EyeDataset(fn[vi], labels[vi]),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    best_f1 = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        tl = []
        for X, Y in train_dl:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
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
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                logits = model(X)
                vl.append(loss_fn(logits, Y).item())
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(Y.cpu().numpy())

        P = np.vstack(all_p); L = np.vstack(all_y)
        f1s = []
        for fi in range(N_FEATURES):
            t=( L[:,fi]>0.3).astype(int); p=(P[:,fi]>0.3).astype(int)
            tp=((t==1)&(p==1)).sum(); fp=((t==0)&(p==1)).sum(); fn_=((t==1)&(p==0)).sum()
            pr=tp/(tp+fp+1e-6); rc=tp/(tp+fn_+1e-6)
            f1s.append(2*pr*rc/(pr+rc+1e-6))

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={np.mean(tl):.4f}  val={np.mean(vl):.4f}  "
                  f"mean_F1={np.mean(f1s):.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if sum(f1s) > best_f1:
            best_f1 = sum(f1s)
            torch.save({
                "model_state":   model.state_dict(),
                "mean":          mean.tolist(),
                "std":           std.tolist(),
                "eye_features":  EYE_FEATURES,
                "deficiencies":  EYE_DEFICIENCIES,
                "cpt":           CPT_EYE.tolist(),
                "prior":         PRIOR_EYE.tolist(),
                "epoch":         epoch,
                "mean_f1":       float(np.mean(f1s)),
                "f1_per_feature":[float(f) for f in f1s],
                "version":       1,
            }, str(MODEL_PATH))

    print(f"\n  ✅ Best mean F1: {best_f1/N_FEATURES:.4f}")
    print(f"  Saved: {MODEL_PATH}")
    print(f"\n  Next: integrate into server.py")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--epochs",   type=int, default=120)
    parser.add_argument("--batch",    type=int, default=64)
    args = parser.parse_args()

    if args.train or (not args.evaluate):
        train(epochs=args.epochs, batch_size=args.batch)
