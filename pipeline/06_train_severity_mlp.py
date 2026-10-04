"""
FaceFuel — Stage 6: severity-MLP training (face + tongue + eye)
================================================================
Run: python pipeline/06_train_severity_mlp.py --data v5
     python pipeline/06_train_severity_mlp.py --data v5 --eye-only   (or --face-only / --tongue-only)

Inputs (from pipeline/05_extract_features.py):
  facefuel_features/<modality>_features_<data>_{train,val,test}.npz
- trains on `train`, picks the best epoch on `val` (mean F1 at the inference threshold),
- reports the untouched `test` split (per-class F1 / precision / recall / AUROC and the
  rate at which negative images get flagged) — the v4 MLPs only ever reported the
  epoch-selection split.

The architecture is imported from facefuel.models so inference loads exactly what was
trained. Checkpoint (read by facefuel.models.load_severity_mlp):
  {"state_dict", "classes", "active_idx", "inactive_idx", "in_dim", "n_active", "test_metrics"}
Classes with ≤5 training positives are excluded from the head count ("inactive").
Output: facefuel_models/<modality>_severity_mlp_<data>.pt
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from facefuel.models import SeverityMLP          # noqa: E402
from mlp_metrics import THRESH, negatives_flag_rate, per_class   # noqa: E402

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
FEAT_DIR = ROOT / "facefuel_features"
OUT_DIR = ROOT / "facefuel_models"
HP = {   # epochs, batch, lr, patience
    "face":   (120, 64, 3e-4, 25),
    "tongue": (120, 64, 3e-4, 25),
    "eye":    (100, 32, 3e-4, 20),
}


def load(modality, data, split):
    f = FEAT_DIR / f"{modality}_features_{data}_{split}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} — run pipeline/05_extract_features.py --data {data} --modality {modality}")
    d = np.load(f, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32), d["classes"].tolist()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X).to(DEVICE))[0]).cpu().numpy()


def train_mlp(modality, data):
    epochs, batch, lr, patience = HP[modality]
    Xtr, ytr, classes = load(modality, data, "train")
    Xva, yva, _ = load(modality, data, "val")
    nc = len(classes)
    active = [i for i in range(nc) if ytr[:, i].sum() > 5]
    inactive = [i for i in range(nc) if i not in active]
    print(f"\n{'=' * 64}\n  {modality}: train {len(Xtr)} (negatives {(ytr.sum(1) == 0).sum()})  "
          f"val {len(Xva)}  dim {Xtr.shape[1]}")
    print(f"  active ({len(active)}): {[classes[i] for i in active]}")
    if inactive:
        print(f"  inactive (≤5 positives): {[classes[i] for i in inactive]}")

    torch.manual_seed(SEED); np.random.seed(SEED)
    ytr_a, yva_a = ytr[:, active], yva[:, active]
    pos = ytr_a.sum(0)
    pos_w = torch.tensor(((len(ytr_a) - pos) / np.clip(pos, 1, None)).clip(max=20), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_a)),
                    batch_size=batch, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(SEED))
    model = SeverityMLP(Xtr.shape[1], len(active)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(dl))

    best, best_sd, stale = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb)[0], yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        m = per_class(yva_a, predict(model, Xva))
        f1 = float(np.mean([r[1] for r in m if r[0] > 0]))
        if f1 > best:
            best, best_sd, stale = f1, {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if ep == 1 or ep % 20 == 0:
            print(f"  ep {ep:3d}  loss={tot / len(dl):.4f}  val F1={f1:.3f}  best={best:.3f}")
        if stale >= patience:
            print(f"  early stop at epoch {ep}")
            break
    model.load_state_dict(best_sd)

    test = {}
    test_file = FEAT_DIR / f"{modality}_features_{data}_test.npz"
    if test_file.exists():
        Xte, yte, _ = load(modality, data, "test")
        pte = predict(model, Xte)
        rows = per_class(yte[:, active], pte)
        test = {"n_images": int(len(Xte)), "n_negatives": int((yte.sum(1) == 0).sum()),
                "threshold": THRESH,
                "negatives_flagged": negatives_flag_rate(yte[:, active], pte),
                "per_class": {classes[a]: dict(zip(["n_pos", "f1", "precision", "recall", "auroc", "fpr"], r))
                              for a, r in zip(active, rows)}}
        scored = [r for r in rows if r[0] > 0]
        test["mean_f1"] = float(np.mean([r[1] for r in scored])) if scored else float("nan")
        print(f"\n  TEST ({len(Xte)} images, {test['n_negatives']} negatives)  mean F1={test['mean_f1']:.3f}"
              f"  negatives flagged={test['negatives_flagged']:.1%}")
        for a, r in zip(active, rows):
            print(f"    {classes[a]:<22} n={r[0]:4d}  F1={r[1]:.3f}  P={r[2]:.3f}  R={r[3]:.3f}  AUROC={r[4]:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{modality}_severity_mlp_{data}.pt"
    torch.save({"state_dict": best_sd, "classes": classes, "active_idx": active,
                "inactive_idx": inactive, "in_dim": int(Xtr.shape[1]), "n_active": len(active),
                "val_mean_f1": best, "test_metrics": test}, out)
    (OUT_DIR / f"{modality}_severity_mlp_{data}_metrics.json").write_text(json.dumps(test, indent=2))
    print(f"  ✅ {out}  (val F1 {best:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for m in HP:
        ap.add_argument(f"--{m}-only", action="store_true")
    ap.add_argument("--data", default="v5")
    a = ap.parse_args()
    chosen = [m for m in HP if getattr(a, f"{m}_only")] or list(HP)
    for m in chosen:
        train_mlp(m, a.data)
