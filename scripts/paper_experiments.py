"""
Experiments behind the revised (2026-10) editions of the three FaceFuel papers.
All numbers are computed on the v5 deduplicated splits; `test` is never used for
model selection, threshold choice or temperature fitting.

Run (each step writes <out>/<step>.json and can be run on its own):
  python scripts/paper_experiments.py --step all
  python scripts/paper_experiments.py --step seeds|probe|whole|regions|negatives|calib|yolo|latency|data|fusion

Steps
  seeds      severity MLP retrained with 5 seeds (same recipe as pipeline/06) + bootstrap CI
             of the shipped checkpoint's test mean F1
  probe      linear probe (logistic regression) on the same DINOv2 features
  whole      whole-image DINOv2 CLS (384-d) vs region-aware features (needs GPU + images)
  regions    face leave-one-region-out
  negatives  eye MLP trained without the 745 normal-eye negatives
  calib      ECE/Brier before and after temperature scaling (fitted on val)
  uncert     MC-dropout std as an error detector; risk-coverage
  yolo       image-level detector baseline vs MLP vs detector-confirmed rule (needs GPU)
  latency    per-stage latency of the deployed pipeline (needs GPU)
  data       per-class image/instance counts of the v5 splits
  fusion     sensitivity of the fused ranking to the PoE weights (rule property only)
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
from facefuel import schema                                       # noqa: E402
from facefuel.models import SeverityMLP, REGION_SPECS             # noqa: E402
from mlp_metrics import THRESH, auroc, negatives_flag_rate, per_class   # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = ROOT / "facefuel_features"
MODELS = ROOT / "facefuel_models"
MODS = ("face", "tongue", "eye")
SEEDS = (42, 43, 44, 45, 46)
_spec = importlib.util.spec_from_file_location("train06", ROOT / "pipeline" / "06_train_severity_mlp.py")
_t06 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_t06)
HP = _t06.HP


def load(m, split, kind="v5"):
    d = np.load(FEAT / f"{m}_features_{kind}_{split}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32), d["classes"].tolist(), d["files"].tolist()


def active_of(ytr):
    return [i for i in range(ytr.shape[1]) if ytr[:, i].sum() > 5]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X).to(DEVICE))[0]).cpu().numpy()


@torch.no_grad()
def logits(model, X):
    model.eval()
    return model(torch.from_numpy(X).to(DEVICE))[0].cpu().numpy()


def mean_f1(y, p, thr=THRESH):
    rows = [r for r in per_class(y, p, thr) if r[0] > 0]
    return float(np.mean([r[1] for r in rows])) if rows else float("nan")


def macro_auroc(y, p):
    v = [auroc(y[:, c], p[:, c]) for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.nanmean(v)) if v else float("nan")


def fit_mlp(Xtr, ytr, Xva, yva, m, seed):
    """Same recipe as pipeline/06_train_severity_mlp.py, parameterised by seed."""
    epochs, batch, lr, patience = HP[m]
    seed_all(seed)
    pos = ytr.sum(0)
    pos_w = torch.tensor(((len(ytr) - pos) / np.clip(pos, 1, None)).clip(max=20), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch,
                    shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
    model = SeverityMLP(Xtr.shape[1], ytr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(dl))
    best, best_sd, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb)[0], yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        f1 = mean_f1(yva, predict(model, Xva))
        if f1 > best:
            best, best_sd, stale = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_sd)
    return model


def fit_mlp_fast(Xtr, ytr, Xva, yva, m, seed):
    """fit_mlp with batches indexed on the GPU instead of a DataLoader (same model, loss,
    optimiser, schedule, batch size, epochs and selection rule; ~10x faster)."""
    epochs, batch, lr, patience = HP[m]
    seed_all(seed)
    Xg, yg = torch.from_numpy(Xtr).to(DEVICE), torch.from_numpy(ytr).to(DEVICE)
    pos = ytr.sum(0)
    pos_w = torch.tensor(((len(ytr) - pos) / np.clip(pos, 1, None)).clip(max=20), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    steps = len(Xtr) // batch
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    model = SeverityMLP(Xtr.shape[1], ytr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    best, best_sd, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE, generator=gen)
        for k in range(steps):
            idx = perm[k * batch:(k + 1) * batch]
            loss = loss_fn(model(Xg[idx])[0], yg[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        f1 = mean_f1(yva, predict(model, Xva))
        if f1 > best:
            best, best_sd, stale = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_sd)
    return model


def evaluate(model, Xte, yte):
    p = predict(model, Xte)
    return {"mean_f1": mean_f1(yte, p), "macro_auroc": macro_auroc(yte, p),
            "negatives_flagged": negatives_flag_rate(yte, p)}


def summarize(runs):
    keys = runs[0].keys()
    return {k: {"mean": float(np.nanmean([r[k] for r in runs])), "sd": float(np.nanstd([r[k] for r in runs], ddof=1)),
                "runs": [r[k] for r in runs]} for k in keys}


def splits(m, kind="v5"):
    Xtr, ytr, cls, _ = load(m, "train", kind)
    Xva, yva, _, _ = load(m, "val", kind)
    Xte, yte, _, _ = load(m, "test", kind)
    a = active_of(ytr)
    return Xtr, ytr[:, a], Xva, yva[:, a], Xte, yte[:, a], [cls[i] for i in a]


def shipped(m):
    ck = torch.load(MODELS / f"{m}_severity_mlp_v5.pt", map_location=DEVICE, weights_only=True)
    model = SeverityMLP(ck["in_dim"], ck["n_active"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck["active_idx"]


# ── steps ────────────────────────────────────────────────────────────────────
def step_seeds():
    out = {}
    for m in MODS:
        Xtr, ytr, Xva, yva, Xte, yte, cls = splits(m)
        runs = [evaluate(fit_mlp(Xtr, ytr, Xva, yva, m, s), Xte, yte) for s in SEEDS]
        model, _ = shipped(m)
        p = predict(model, Xte)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(1000):
            idx = rng.integers(0, len(Xte), len(Xte))
            boot.append(mean_f1(yte[idx], p[idx]))
        out[m] = {"seeds": summarize(runs), "shipped": evaluate(model, Xte, yte),
                  "shipped_mean_f1_ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
                  "classes": cls, "n_test": int(len(Xte))}
        print(m, json.dumps(out[m]["seeds"]["mean_f1"]), out[m]["shipped_mean_f1_ci95"])
    return out


def step_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {}
    for m in MODS:
        Xtr, ytr, Xva, yva, Xte, yte, cls = splits(m)
        sc = StandardScaler().fit(Xtr)
        Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        best = None
        for C in (0.001, 0.01, 0.1):
            pva = np.zeros_like(yva); pte = np.zeros_like(yte)
            for c in range(ytr.shape[1]):
                lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr_, ytr[:, c])
                pva[:, c] = lr.predict_proba(Xva_)[:, 1]; pte[:, c] = lr.predict_proba(Xte_)[:, 1]
            # threshold for the probe chosen on val (its scores are not on the MLP's scale)
            thr = max(np.arange(0.2, 0.95, 0.05), key=lambda t: mean_f1(yva, pva, t))
            f = mean_f1(yva, pva, thr)
            if best is None or f > best["val_f1"]:
                best = {"C": C, "thr": float(thr), "val_f1": f, "mean_f1": mean_f1(yte, pte, thr),
                        "macro_auroc": macro_auroc(yte, pte), "negatives_flagged": negatives_flag_rate(yte, pte, thr)}
        out[m] = best
        print(m, best)
    return out


@torch.no_grad()
def extract_whole(m):
    """Whole-image DINOv2 CLS (384-d), same resize/normalisation as the region features."""
    import cv2
    from facefuel.models import _prep_region, get_dino
    dino = get_dino(DEVICE)
    size = REGION_SPECS[m][1]
    for split in ("train", "val", "test"):
        dst = FEAT / f"{m}_features_whole_{split}.npz"
        if dst.exists():
            continue
        _, y, cls, files = load(m, split)
        X = np.zeros((len(files), 384), np.float32)
        for i in range(0, len(files), 64):
            batch = []
            for f in files[i:i + 64]:
                img = cv2.imread(str(ROOT / f"facefuel_{m}_v5" / "images" / split / f))
                batch.append(_prep_region(cv2.resize(img, size)))
            X[i:i + len(batch)] = dino(pixel_values=torch.stack(batch).to(DEVICE)).last_hidden_state[:, 0].cpu().numpy()
        np.savez_compressed(dst, X=X, y=y, classes=np.array(cls), files=np.array(files))
        print("  extracted", dst.name, X.shape)


def step_whole():
    out = {}
    for m in MODS:
        extract_whole(m)
        Xtr, ytr, Xva, yva, Xte, yte, cls = splits(m, "whole")
        runs = [evaluate(fit_mlp(Xtr, ytr, Xva, yva, m, s), Xte, yte) for s in SEEDS[:3]]
        out[m] = summarize(runs)
        print(m, "whole-image", out[m]["mean_f1"])
    return out


def step_regions():
    Xtr, ytr, Xva, yva, Xte, yte, cls = splits("face")
    names = list(REGION_SPECS["face"][0])
    out = {"all_regions": summarize([evaluate(fit_mlp_fast(Xtr, ytr, Xva, yva, "face", s), Xte, yte) for s in SEEDS[:2]])}
    print("all regions", out["all_regions"]["mean_f1"]["mean"])
    for r, name in enumerate(names):
        keep = np.r_[0:r * 384, (r + 1) * 384:len(names) * 384]
        runs = [evaluate(fit_mlp_fast(Xtr[:, keep], ytr, Xva[:, keep], yva, "face", s), Xte[:, keep], yte) for s in SEEDS[:2]]
        out[f"without_{name}"] = summarize(runs)
        print(name, out[f"without_{name}"]["mean_f1"]["mean"])
    return out


def step_negatives():
    Xtr, ytr, Xva, yva, Xte, yte, cls = splits("eye")
    pos = ytr.sum(1) > 0
    with_neg = [evaluate(fit_mlp(Xtr, ytr, Xva, yva, "eye", s), Xte, yte) for s in SEEDS]
    without = [evaluate(fit_mlp(Xtr[pos], ytr[pos], Xva, yva, "eye", s), Xte, yte) for s in SEEDS]
    out = {"n_train_negatives": int((~pos).sum()), "n_test_negatives": int((yte.sum(1) == 0).sum()),
           "with_negatives": summarize(with_neg), "without_negatives": summarize(without)}
    print(json.dumps({k: (v["negatives_flagged"]["mean"], v["mean_f1"]["mean"]) for k, v in out.items() if isinstance(v, dict)}))
    return out


def ece(y, p, bins=15):
    y, p = y.ravel(), p.ravel()
    edges = np.linspace(0, 1, bins + 1)
    e, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if k.any():
            e += k.mean() * abs(p[k].mean() - y[k].mean())
            rel.append([float(p[k].mean()), float(y[k].mean()), int(k.sum())])
    return float(e), rel


def step_calib():
    out = {}
    for m in MODS:
        _, _, Xva, yva, Xte, yte, cls = splits(m)
        model, _ = shipped(m)
        zva, zte = logits(model, Xva), logits(model, Xte)
        Ts = np.exp(np.linspace(np.log(0.25), np.log(8), 200))
        nll = lambda z, y, T: float(np.mean(np.logaddexp(0, z / T) - y * z / T))   # BCE with logits
        T = float(min(Ts, key=lambda t: nll(zva, yva, t)))
        sig = lambda z: 1 / (1 + np.exp(-z))
        p0, p1 = sig(zte), sig(zte / T)
        e0, r0 = ece(yte, p0); e1, r1 = ece(yte, p1)
        out[m] = {"temperature": T, "ece_before": e0, "ece_after": e1,
                  "brier_before": float(np.mean((p0 - yte) ** 2)), "brier_after": float(np.mean((p1 - yte) ** 2)),
                  "f1_at_0.35_after": mean_f1(yte, p1), "reliability_before": r0, "reliability_after": r1}
        print(m, f"T={T:.2f} ECE {e0:.3f}->{e1:.3f}")
    return out


def step_uncert():
    out = {}
    for m in MODS:
        _, _, _, _, Xte, yte, cls = splits(m)
        model, _ = shipped(m)
        x = torch.from_numpy(Xte).to(DEVICE)
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(DEVICE)] if DEVICE == "cuda" else []):
            torch.manual_seed(0); model.train()
            draws = torch.stack([torch.sigmoid(model(x)[0]) for _ in range(20)]).cpu().numpy()
            model.eval()
        mu, sd = draws.mean(0).ravel(), draws.std(0).ravel()
        err = ((mu > THRESH) != (yte.ravel() > 0.5)).astype(float)
        conf_margin = -np.abs(mu - THRESH)          # baseline: closeness to the decision threshold
        rc = {}
        order = np.argsort(sd)                      # keep the most certain first
        for cov in (1.0, 0.9, 0.8, 0.7, 0.5):
            k = order[:int(round(cov * len(order)))]
            rc[str(cov)] = float(err[k].mean())
        out[m] = {"error_rate": float(err.mean()), "auroc_std_detects_error": auroc(err, sd),
                  "auroc_margin_detects_error": auroc(err, conf_margin), "risk_at_coverage": rc,
                  "spearman_std_vs_margin": float(np.corrcoef(np.argsort(np.argsort(sd)), np.argsort(np.argsort(conf_margin)))[0, 1])}
        print(m, out[m])
    return out


def step_yolo():
    import cv2
    from facefuel.models import load_yolo
    from facefuel import face, eye, tongue
    keys = {"face": ("face_yolo", face, schema.FACE_CLASSES),
            "tongue": ("tongue_yolo", tongue, schema.TONGUE_CLASSES),
            "eye": ("eye_yolo", eye, schema.EYE_CLASSES)}
    out = {}
    for m in MODS:
        key, mod, classes = keys[m]
        yolo = load_yolo(key, classes)
        _, yall, cls_all, files = load(m, "test")
        Xte = load(m, "test")[0]
        S = np.zeros((len(files), len(classes)), np.float32)        # max box confidence per class
        for i, f in enumerate(files):
            img = cv2.imread(str(ROOT / f"facefuel_{m}_v5" / "images" / "test" / f))
            r = yolo.predict(img, conf=0.01, verbose=False, device=DEVICE)[0]
            for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                S[i, c] = max(S[i, c], s)
        model, act = shipped(m)
        P = predict(model, Xte)
        rows = {}
        for h, c in enumerate(act):
            name = classes[c]
            if name in schema.INACTIVE[m] or name == "tongue_body":
                continue
            y = yall[:, c]
            if y.sum() == 0:
                continue
            thr = mod.PER_CLASS_CONF.get(name, mod.CONF_THRESHOLD)
            det, mlp = S[:, c] >= thr, P[:, h] > THRESH
            def f1(pred):
                tp = (pred & (y > 0.5)).sum(); fp = (pred & (y <= 0.5)).sum(); fn = (~pred & (y > 0.5)).sum()
                return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
            neg = yall.sum(1) == 0
            rows[name] = {"n_pos": int(y.sum()), "det_thr": thr,
                          "auroc_detector": auroc(y, S[:, c]), "auroc_mlp": auroc(y, P[:, h]),
                          "f1_detector": f1(det), "f1_mlp": f1(mlp), "f1_and": f1(det & mlp), "f1_or": f1(det | mlp),
                          "neg_fpr_detector": float(det[neg].mean()) if neg.any() else None,
                          "neg_fpr_mlp": float(mlp[neg].mean()) if neg.any() else None}
        mean = lambda k: float(np.mean([r[k] for r in rows.values()]))
        out[m] = {"per_class": rows, "n_images": len(files),
                  "mean": {k: mean(k) for k in ("auroc_detector", "auroc_mlp", "f1_detector", "f1_mlp", "f1_and", "f1_or")}}
        print(m, out[m]["mean"])
    return out


def step_latency():
    import cv2
    from facefuel import face, eye, tongue
    from facefuel.models import default_device
    dev = default_device()
    fm, em, tm = face.load(dev), eye.load(dev), tongue.load(dev)
    out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    # face + eye on face-test images where MediaPipe finds a face
    files = sorted((ROOT / "facefuel_face_v5/images/test").glob("*.jpg"))
    rng = random.Random(0); rng.shuffle(files)
    t_face, t_eye, n_try = [], [], 0
    for f in files:
        if len(t_face) >= 60 or n_try > 1500:
            break
        n_try += 1
        img = cv2.imread(str(f))
        t0 = time.perf_counter(); res, al = face.analyze(img, fm); dt = time.perf_counter() - t0
        if res is None:
            continue
        t1 = time.perf_counter(); eye.analyze(al["aligned_big"], em); de = time.perf_counter() - t1
        t_face.append({**res["timing"], "total": dt}); t_eye.append(de)
    tfiles = sorted((ROOT / "facefuel_tongue_v5/images/test").glob("*.jpg"))[:60]
    t_tongue = []
    for f in tfiles:
        img = cv2.imread(str(f))
        t0 = time.perf_counter(); tongue.analyze(img, tm); t_tongue.append(time.perf_counter() - t0)
    med = lambda v: float(np.median(v) * 1000)
    skip = 5   # warm-up
    out["face_stages_ms"] = {k: med([t[k] for t in t_face[skip:]]) for k in t_face[0]}
    out["eye_ms"] = med(t_eye[skip:]); out["tongue_ms"] = med(t_tongue[skip:])
    out["n_face"] = len(t_face) - skip; out["n_tongue"] = len(t_tongue) - skip
    out["face_detect_rate_on_test_crops"] = len(t_face) / n_try
    print(json.dumps(out, indent=1))
    return out


def step_data():
    import csv
    out = {}
    for m in MODS:
        classes = schema.CLASSES[m]
        d = {}
        for split in ("train", "val", "test"):
            lab = ROOT / f"facefuel_{m}_v5" / "labels" / split
            imgs = list((ROOT / f"facefuel_{m}_v5" / "images" / split).iterdir())
            img_c, inst_c, neg = np.zeros(len(classes), int), np.zeros(len(classes), int), 0
            full_c = np.zeros(len(classes), int)      # whole-image boxes (classification-folder sources)
            for im in imgs:
                lf = lab / (im.stem + ".txt")
                lines = [l.split() for l in lf.read_text().splitlines() if l.strip()] if lf.exists() else []
                ids = [int(p[0]) for p in lines]
                for p in lines:
                    if float(p[3]) > 0.98 and float(p[4]) > 0.98:
                        full_c[int(p[0])] += 1
                if not ids:
                    neg += 1
                for c in set(ids):
                    img_c[c] += 1
                for c in ids:
                    inst_c[c] += 1
            d[split] = {"images": len(imgs), "negatives": neg,
                        "per_class_images": dict(zip(classes, img_c.tolist())),
                        "per_class_instances": dict(zip(classes, inst_c.tolist())),
                        "per_class_full_image_boxes": dict(zip(classes, full_c.tolist()))}
        out[m] = d
        print(m, {s: (d[s]["images"], d[s]["negatives"]) for s in d})
    return out


def step_fusion():
    """How much does the fused top-k ranking move when the PoE weights are perturbed?
    Uses the deployed rule on detector-confirmed features of RANDOMLY PAIRED test images
    (there is no paired face/tongue/eye data), so this measures a property of the fusion
    rule, not clinical accuracy."""
    from facefuel import fusion
    from facefuel.evidence import evidence_scores
    res = json.loads((OUT / "yolo_scores.json").read_text()) if (OUT / "yolo_scores.json").exists() else None
    rng = np.random.default_rng(0)
    # synthetic modality evidence: sample detected-feature sets from the test-split label distribution
    feats = {}
    for m in MODS:
        _, y, cls, _ = load(m, "test")
        feats[m] = [{cls[c]: {"severity": float(rng.uniform(0.35, 1.0))} for c in np.flatnonzero(row)
                     if cls[c] not in schema.INACTIVE[m] and cls[c] != "tongue_body"} for row in y]
    base_w = dict(schema.MODALITY_WEIGHTS)
    grid = []
    for wf in np.arange(0.2, 0.61, 0.05):
        for wt in np.arange(0.15, 0.56, 0.05):
            we = 1 - wf - wt
            if we < 0.1:
                continue
            grid.append((round(float(wf), 2), round(float(wt), 2), round(float(we), 2)))
    trials = []
    for _ in range(2000):
        ev = {m: evidence_scores(feats[m][rng.integers(len(feats[m]))], schema.FEAT_DEF[m]) for m in MODS}
        if not all(ev.values()):
            continue
        trials.append(ev)
    def ranking(ev, w):
        schema.MODALITY_WEIGHTS.update(dict(zip(MODS, w)))
        f = fusion.fuse(ev)
        return [d for d, _ in sorted(f.items(), key=lambda kv: -kv[1])]
    base = [ranking(ev, tuple(base_w[m] for m in MODS)) for ev in trials]
    rows = []
    for w in grid:
        top1 = np.mean([ranking(ev, w)[:1] == b[:1] for ev, b in zip(trials, base)])
        top3 = np.mean([set(ranking(ev, w)[:3]) == set(b[:3]) for ev, b in zip(trials, base)])
        rows.append({"w": w, "top1_agreement": float(top1), "top3_set_agreement": float(top3)})
    schema.MODALITY_WEIGHTS.update(base_w)
    near = [r for r in rows if max(abs(a - b) for a, b in zip(r["w"], (0.40, 0.35, 0.25))) <= 0.051]
    out = {"n_trials": len(trials), "grid": rows,
           "within_0.05": {"top1_min": min(r["top1_agreement"] for r in near), "top3_min": min(r["top3_set_agreement"] for r in near)},
           "coverage": {m: sorted(schema.coverage(m)) for m in MODS},
           "union_coverage": len(set().union(*[schema.coverage(m) for m in MODS])),
           "n_dimensions": len(schema.ALL_DEFS)}
    print(out["within_0.05"], out["union_coverage"], {m: len(v) for m, v in out["coverage"].items()})
    return out


STEPS = {"data": step_data, "seeds": step_seeds, "probe": step_probe, "negatives": step_negatives,
         "calib": step_calib, "uncert": step_uncert, "regions": step_regions, "whole": step_whole,
         "yolo": step_yolo, "latency": step_latency, "fusion": step_fusion}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", help="all or comma list of: " + ",".join(STEPS))
    ap.add_argument("--out", default=str(ROOT / "paper_results" / "revision" / "results"))
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    for s in (list(STEPS) if a.step == "all" else a.step.split(",")):
        t0 = time.time()
        print(f"\n=== {s} ===")
        r = STEPS[s]()
        (OUT / f"{s}.json").write_text(json.dumps(r, indent=1))
        print(f"  ({time.time() - t0:.0f}s)")
