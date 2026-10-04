"""
FaceFuel v4 — Severity MLP Training (face + tongue + eye)
==========================================================
Single script trains all three MLPs sequentially.

Run: python pipeline/06_train_severity_mlp.py
     python pipeline/06_train_severity_mlp.py --face-only   (or --tongue-only / --eye-only)

The architecture is imported from facefuel.models so inference always loads
exactly what was trained. Checkpoint format (read by facefuel.models.load_severity_mlp):
  {"state_dict", "classes", "active_idx", "inactive_idx", "in_dim", "n_active"}
Classes with ≤5 positives are excluded from the head count ("inactive").
"""
import argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from facefuel.models import SeverityMLP

warnings.filterwarnings("ignore")

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "facefuel_models"
SEED    = 42
OUT_DIR.mkdir(exist_ok=True)

CONFIGS = {
    "face": {
        "feat_file": ROOT / "facefuel_features/face_features_v4.npz",
        "out_file":  OUT_DIR / "face_severity_mlp_v4.pt",
        "in_dim":    3072,
        "epochs":    120,
        "batch":     64,
        "lr":        3e-4,
        "patience":  25,
    },
    "tongue": {
        "feat_file": ROOT / "facefuel_features/tongue_features_v4.npz",
        "out_file":  OUT_DIR / "tongue_severity_mlp_v4.pt",
        "in_dim":    1152,
        "epochs":    120,
        "batch":     64,
        "lr":        3e-4,
        "patience":  25,
    },
    "eye": {
        "feat_file": ROOT / "facefuel_features/eye_features_v4.npz",
        "out_file":  OUT_DIR / "eye_severity_mlp_v4.pt",
        "in_dim":    1152,
        "epochs":    100,
        "batch":     32,
        "lr":        3e-4,
        "patience":  20,
    },
}


def train_mlp(modality, cfg):
    feat_file = cfg["feat_file"]
    if not Path(feat_file).exists():
        print(f"  ⚠ {feat_file} not found — run pipeline/05_extract_features.py --modality {modality} first")
        return

    print(f"\n{'='*60}")
    print(f"  Training {modality} severity MLP")
    print(f"{'='*60}")

    data    = np.load(feat_file, allow_pickle=True)
    X       = torch.tensor(data["X"], dtype=torch.float32)
    y       = torch.tensor(data["y"], dtype=torch.float32)
    classes = data["classes"].tolist()
    NC      = len(classes)

    # Remove all-zero classes from training to avoid degenerate heads
    active  = [i for i in range(NC) if y[:,i].sum() > 5]
    inactive= [i for i in range(NC) if i not in active]
    if inactive:
        print(f"  ⚠ Skipping {len(inactive)} zero-sample classes: "
              f"{[classes[i] for i in inactive]}")

    y_act = y[:, active]

    print(f"  Samples: {X.shape[0]}  Features: {X.shape[1]}")
    print(f"  Active classes ({len(active)}): {[classes[i] for i in active]}")

    # Train/val split. Seeded for reproducibility (the original v4 run was not).
    # NOTE: the best epoch is selected on this same split and its F1 is what gets
    # reported, so reported F1 is optimistic — there is no held-out test split.
    n_val  = max(1, int(len(X) * 0.15))
    torch.manual_seed(SEED)
    idx    = torch.randperm(len(X))
    tr_X, tr_y = X[idx[n_val:]], y_act[idx[n_val:]]
    va_X, va_y = X[idx[:n_val]],  y_act[idx[:n_val]]

    # Positive weight for imbalanced classes
    pos_w  = ((tr_y < 0.3).sum(0) /
               (tr_y >= 0.3).sum(0).clamp(min=1)).clamp(max=20).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ds    = DataLoader(TensorDataset(tr_X, tr_y),
                       batch_size=cfg["batch"], shuffle=True, drop_last=True)
    model = SeverityMLP(cfg["in_dim"], len(active)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(ds))

    best_f1 = 0.0
    best_sd = None
    no_imp  = 0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in ds:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vp, _ = model(va_X.to(DEVICE))
            vp    = torch.sigmoid(vp).cpu().numpy()
        vy       = va_y.numpy()
        pred_bin = (vp > 0.35).astype(np.float32)
        true_bin = (vy  > 0.30).astype(np.float32)
        tp = (pred_bin * true_bin).sum(0)
        fp = (pred_bin * (1-true_bin)).sum(0)
        fn = ((1-pred_bin) * true_bin).sum(0)
        p  = tp / (tp+fp+1e-9)
        r  = tp / (tp+fn+1e-9)
        f1_per  = 2*p*r/(p+r+1e-9)
        f1_mean = float(f1_per.mean())

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_sd = {k: v.clone().cpu() for k,v in model.state_dict().items()}
            no_imp  = 0
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                print(f"  Early stop at epoch {ep}  best_F1={best_f1:.3f}")
                break

        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d}  loss={total_loss/len(ds):.4f}"
                  f"  F1={f1_mean:.3f}  best={best_f1:.3f}")

    # Save — store active class indices alongside weights
    model.load_state_dict(best_sd)
    save_dict = {
        "state_dict":    best_sd,
        "classes":       classes,
        "active_idx":    active,
        "inactive_idx":  inactive,
        "in_dim":        cfg["in_dim"],
        "n_active":      len(active),
    }
    torch.save(save_dict, cfg["out_file"])
    print(f"\n  ✅ best F1={best_f1:.3f}  → {cfg['out_file']}")
    print(f"  Per-class F1:")
    for j, ai in enumerate(active):
        bar = "█" * int(f1_per[j]*20)
        print(f"    {classes[ai]:<25} {f1_per[j]:.3f}  {bar}")
    if inactive:
        for ai in inactive:
            print(f"    {classes[ai]:<25} ----  (no training data)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-only",   action="store_true")
    ap.add_argument("--tongue-only", action="store_true")
    ap.add_argument("--eye-only",    action="store_true")
    args = ap.parse_args()
    do_all = not any([args.face_only, args.tongue_only, args.eye_only])

    if do_all or args.face_only:
        train_mlp("face",   CONFIGS["face"])
    if do_all or args.tongue_only:
        train_mlp("tongue", CONFIGS["tongue"])
    if do_all or args.eye_only:
        train_mlp("eye",    CONFIGS["eye"])

    print(f"\n{'='*60}  MLP TRAINING DONE")
    for mod in ["face","tongue","eye"]:
        p = CONFIGS[mod]["out_file"]
        tag = f"{Path(p).stat().st_size//1024} KB" if Path(p).exists() else "MISSING"
        print(f"  {mod:<8} {p}  [{tag}]")