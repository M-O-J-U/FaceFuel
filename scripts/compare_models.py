"""
Old vs new models on the clean v5 TEST split (images neither model was trained on).

  python scripts/compare_models.py --modality eye
  python scripts/compare_models.py --modality all --selfies local/test --out docs/model_comparison.md

Detectors: mAP50 / mAP50-95 / precision / recall on facefuel_<m>_v5 test.
Severity MLPs: per-class F1 / AUROC at the inference threshold, and how often images
with no condition get flagged — on facefuel_features/<m>_features_v5_test.npz.
Selfies (optional): fraction of real face photos on which each MLP class fires —
the out-of-domain over-reporting measured in docs/domain_shift_probe.md.

Decision rule used for FaceFuel v4.2: a new model replaces the old one only if it is
better on the test split overall (detector mAP50; MLP mean F1) without a large
regression on any class that has ≥ 10 test examples.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import numpy as np
import torch

from facefuel import schema
from facefuel.models import SeverityMLP, default_device
from mlp_metrics import negatives_flag_rate, per_class

PAIRS = {   # modality: (old detector, new detector, old MLP, new MLP)
    "face":   ("weights/face_yolo11m_v4.pt",   "training_runs/face_v5/weights/best.pt",
               "weights/face_severity_mlp_v4.pt", "facefuel_models/face_severity_mlp_v5.pt"),
    "tongue": ("weights/tongue_yolo11m_v4.pt", "training_runs/tongue_v5/weights/best.pt",
               "weights/tongue_severity_mlp_v4.pt", "facefuel_models/tongue_severity_mlp_v5.pt"),
    "eye":    ("weights/eye_yolo11m_v2.pt",    "training_runs/eye_v5/weights/best.pt",
               "weights/eye_severity_mlp_v4.pt", "facefuel_models/eye_severity_mlp_v5.pt"),
}


def detector(weights, data_yaml, device):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", device=device, imgsz=640,
                               verbose=False, plots=False)
    per = {m.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return {"map50": float(m.box.map50), "map": float(m.box.map), "p": float(m.box.mp),
            "r": float(m.box.mr), "per_class": per}


def load_mlp(path, device):
    ck = torch.load(str(path), map_location=device, weights_only=True)
    mlp = SeverityMLP(ck["in_dim"], len(ck["active_idx"])).to(device).eval()
    mlp.load_state_dict(ck["state_dict"])
    return mlp, ck["active_idx"], ck["classes"]


@torch.no_grad()
def mlp_scores(path, X, device):
    mlp, active, classes = load_mlp(path, device)
    p = torch.sigmoid(mlp(torch.from_numpy(X).float().to(device))[0]).cpu().numpy()
    full = np.zeros((len(X), len(classes)), dtype=np.float32)
    full[:, active] = p
    return full, set(active)


def selfie_rates(mod, paths_, folder, device):
    from facefuel import face, eye
    from facefuel.models import embed_regions
    import cv2
    from PIL import Image
    F = face.load(device)
    crops = []
    for p in sorted(Path(folder).iterdir()):
        try:
            img = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            continue
        al = face.align(img, F)
        if al is None:
            continue
        crops.append(al["aligned"] if mod == "face" else eye.eye_crop(al["aligned_big"]))
    if not crops:
        return None
    X = np.stack([embed_regions(c, mod, device).cpu().numpy() for c in crops])
    out = {}
    for tag, pth in paths_.items():
        s, active = mlp_scores(pth, X, device)
        out[tag] = {schema.CLASSES[mod][i]: float((s[:, i] > 0.35).mean()) for i in sorted(active)}
    return len(crops), out


def compare(mod, selfies, device, lines):
    old_det, new_det, old_mlp, new_mlp = (ROOT / p for p in PAIRS[mod])
    data_yaml = ROOT / f"facefuel_{mod}_v5" / "data.yaml"
    lines.append(f"\n## {mod}\n")
    if new_det.exists():
        o, n = detector(old_det, data_yaml, device), detector(new_det, data_yaml, device)
        lines += ["**Detector (YOLO11m) on v5 test**\n", "| | mAP50 | mAP50-95 | precision | recall |",
                  "|---|---|---|---|---|",
                  f"| old | {o['map50']:.3f} | {o['map']:.3f} | {o['p']:.3f} | {o['r']:.3f} |",
                  f"| **new** | **{n['map50']:.3f}** | {n['map']:.3f} | {n['p']:.3f} | {n['r']:.3f} |", "",
                  "| class | old AP50 | new AP50 |", "|---|---|---|"]
        for c in schema.CLASSES[mod]:
            if c in o["per_class"] or c in n["per_class"]:
                lines.append(f"| {c} | {o['per_class'].get(c, float('nan')):.3f} | {n['per_class'].get(c, float('nan')):.3f} |")
    else:
        lines.append(f"_new detector not trained yet ({new_det.relative_to(ROOT)})_")

    feat = ROOT / "facefuel_features" / f"{mod}_features_v5_test.npz"
    if feat.exists() and new_mlp.exists():
        d = np.load(feat, allow_pickle=True)
        X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
        so, ao = mlp_scores(old_mlp, X, device)
        sn, an = mlp_scores(new_mlp, X, device)
        ro, rn = per_class(y, so), per_class(y, sn)
        lines += ["", f"**Severity MLP on v5 test** ({len(X)} images, {int((y.sum(1) == 0).sum())} with no condition)\n",
                  "| class | n | old F1 | new F1 | old AUROC | new AUROC |", "|---|---|---|---|---|---|"]
        fo, fn = [], []
        for i, c in enumerate(schema.CLASSES[mod]):
            if ro[i][0] == 0 or (i not in ao and i not in an):
                continue
            a, b = ro[i] if i in ao else (0, 0, 0, 0, float("nan"), 0), rn[i] if i in an else (0, 0, 0, 0, float("nan"), 0)
            fo.append(a[1]); fn.append(b[1])
            lines.append(f"| {c} | {ro[i][0]} | {a[1]:.3f} | {b[1]:.3f} | {a[4]:.3f} | {b[4]:.3f} |")
        lines.append(f"| **mean** | | **{np.mean(fo):.3f}** | **{np.mean(fn):.3f}** | | |")
        lines.append(f"\nNo-condition images flagged by at least one class: old "
                     f"{negatives_flag_rate(y, so):.1%} → new {negatives_flag_rate(y, sn):.1%}")
    if selfies and mod in ("face", "eye") and new_mlp.exists():
        r = selfie_rates(mod, {"old": old_mlp, "new": new_mlp}, selfies, device)
        if r:
            n, rates = r
            lines += ["", f"**Out-of-domain: share of {n} real face photos on which each class fires (>0.35)**\n",
                      "| class | old | new |", "|---|---|---|"]
            for c in rates["new"]:
                lines.append(f"| {c} | {rates['old'].get(c, float('nan')):.0%} | {rates['new'][c]:.0%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["face", "tongue", "eye", "all"], default="all")
    ap.add_argument("--selfies", help="folder of real face photos for the out-of-domain check")
    ap.add_argument("--out", help="write the markdown report here")
    a = ap.parse_args()
    dev = default_device()
    lines = ["# FaceFuel — old (v4) vs new (v5) models on the clean test split"]
    for m in (["eye", "tongue", "face"] if a.modality == "all" else [a.modality]):
        compare(m, a.selfies, dev, lines)
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
