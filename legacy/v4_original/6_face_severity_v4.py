"""
FaceFuel v4 — Face Severity MLP Training
==========================================
Trains severity regression MLP on DINOv2 face features.
Input:  facefuel_features/face_features_v4.npz
Output: facefuel_models/face_severity_mlp_v4.pt

Run: python 6_face_severity_v4.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml

FEAT_FILE  = "facefuel_features/face_features_v4.npz"
OUT_PATH   = "facefuel_models/face_severity_mlp_v4.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 120
BATCH      = 64
LR         = 3e-4

class SeverityMLPv4(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256),    nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 128),    nn.GELU(), nn.Dropout(0.20),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64), nn.GELU(),
                nn.Dropout(0.20), nn.Linear(64, 1)
            ) for _ in range(n_cls)
        ])
        self.log_vars = nn.Parameter(torch.zeros(n_cls))

    def forward(self, x):
        h = self.backbone(x)
        out = torch.cat([head(h) for head in self.heads], dim=-1)
        return out, self.log_vars


def train():
    data = np.load(FEAT_FILE, allow_pickle=True)
    X    = torch.tensor(data["X"], dtype=torch.float32)
    y    = torch.tensor(data["y"], dtype=torch.float32)
    classes = data["classes"].tolist()
    NC   = len(classes)
    print(f"  Samples: {X.shape[0]}  Features: {X.shape[1]}  Classes: {NC}")
    print(f"  Classes: {classes}")

    # Train/val split
    n_val = max(1, int(len(X) * 0.15))
    idx   = torch.randperm(len(X))
    tr_X, tr_y = X[idx[n_val:]], y[idx[n_val:]]
    va_X, va_y = X[idx[:n_val]], y[idx[:n_val]]

    # Class weights for imbalanced data
    pos_weight = ((tr_y < 0.3).sum(0) / (tr_y >= 0.3).sum(0).clamp(min=1)).to(DEVICE)

    ds = DataLoader(TensorDataset(tr_X, tr_y), batch_size=BATCH,
                    shuffle=True, drop_last=True)

    model = SeverityMLPv4(X.shape[1], NC).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(ds))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1  = 0.0
    best_sd  = None
    patience = 25
    no_imp   = 0

    print(f"\n  Training {EPOCHS} epochs on {DEVICE}...")
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in ds:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, lv = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()

        # Validation F1
        model.eval()
        with torch.no_grad():
            vp, _ = model(va_X.to(DEVICE))
            vp    = torch.sigmoid(vp).cpu().numpy()
            vy    = va_y.numpy()
        pred_bin = (vp > 0.35).astype(np.float32)
        true_bin = (vy  > 0.30).astype(np.float32)
        tp = (pred_bin * true_bin).sum(0)
        fp = (pred_bin * (1-true_bin)).sum(0)
        fn = ((1-pred_bin) * true_bin).sum(0)
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1_per = 2*p*r/(p+r+1e-9)
        f1_mean = float(f1_per.mean())

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_sd = {k: v.clone().cpu() for k,v in model.state_dict().items()}
            no_imp  = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep}")
                break

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}  loss={total_loss/len(ds):.4f}  "
                  f"val_F1={f1_mean:.3f}  best={best_f1:.3f}")
            for i, cls in enumerate(classes):
                print(f"    {cls:<25} F1={f1_per[i]:.3f}")

    # Save best
    Path(OUT_PATH).parent.mkdir(exist_ok=True)
    model.load_state_dict(best_sd)
    torch.save(best_sd, OUT_PATH)
    print(f"\n  ✅ Best F1={best_f1:.3f}  saved → {OUT_PATH}")
    print(f"  Per-class F1:")
    for i, cls in enumerate(classes):
        bar = "█" * int(f1_per[i] * 20)
        print(f"    {cls:<25} {f1_per[i]:.3f}  {bar}")

if __name__ == "__main__":
    train()