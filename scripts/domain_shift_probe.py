"""
Domain-shift probe for the v4 severity MLPs.

1. In-domain: for each modality, mean MLP output on a sample of its held-out
   v5 test images, split by whether the class is present in the label —
   this is the setting the reported F1 scores come from.
2. Tongue localisation rate on tongue validation images (tongue_body found?).
3. Out-of-domain: MLP output and detector hits on a folder of selfies.

The v4 MLPs were trained only on images containing ≥1 labelled condition, so
they never saw a healthy negative; (3) shows what that does on real faces.

Run from the repo root:
  python scripts/domain_shift_probe.py --selfies path/to/folder [--n 150] [--out probe.md]
"""
import argparse
import glob
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from facefuel import eye, face, schema, tongue
from facefuel.evidence import MLP_PRESENT
from facefuel.models import default_device, embed_regions


def read(p):
    try:
        return cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(p))


def in_domain(mod, models, n, rng, lines):
    lbl_dir = ROOT / f"facefuel_{mod}_v5" / "labels" / "test"
    files = sorted(lbl_dir.glob("*.txt"))
    if not files:
        lines.append(f"\n_{mod}: no validation set at {lbl_dir}_\n")
        return
    rows = []
    for lp in rng.sample(files, min(n, len(files))):
        ids = {int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()}
        ip = next(iter(sorted((lbl_dir.parent.parent / "images" / "test").glob(lp.stem + ".*"))), None)
        img = read(ip) if ip else None
        if img is None:
            continue
        s, _ = models["mlp"].predict(embed_regions(img, mod, models["device"]))
        rows.append((ids, s))
    lines += [f"\n### {mod} — in-domain validation ({len(rows)} images)\n",
              "| class | positives | mean MLP (pos) | negatives | mean MLP (neg) | neg > 0.35 |",
              "|---|---|---|---|---|---|"]
    for ci, c in enumerate(schema.CLASSES[mod]):
        if c in models["mlp"].inactive or c == "tongue_body":
            continue
        pos = [s[ci] for ids, s in rows if ci in ids]
        neg = [s[ci] for ids, s in rows if ci not in ids]
        lines.append(f"| {c} | {len(pos)} | {np.mean(pos):.2f} | {len(neg)} | {np.mean(neg):.2f} | "
                     f"{np.mean(np.array(neg) > MLP_PRESENT):.0%} |" if pos else
                     f"| {c} | 0 | – | {len(neg)} | {np.mean(neg):.2f} | {np.mean(np.array(neg) > MLP_PRESENT):.0%} |")
    return rows


def tongue_localisation(models, n, rng, lines):
    imgs = sorted((ROOT / "facefuel_tongue_v5" / "images" / "test").glob("*"))
    if not imgs:
        return
    hits = 0
    sample = rng.sample(imgs, min(n, len(imgs)))
    for p in sample:
        img = read(p)
        if img is not None and tongue.detect(img, models)[0] is not None:
            hits += 1
    lines.append(f"\nTongue body localised in {hits}/{len(sample)} validation images "
                 f"({hits / len(sample):.0%}).\n")


def selfies(folder, F, E, lines):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp", ".avif"})
    fa, ea = schema.active_classes("face"), schema.active_classes("eye")
    lines += [f"\n### Out-of-domain: {len(paths)} photos in `{folder}`\n",
              "MLP sigmoid per active class; detector hits in brackets.\n",
              "| photo | " + " | ".join(f"face:{c}" for c in fa) + " | " +
              " | ".join(f"eye:{c}" for c in ea) + " |",
              "|---|" + "---|" * (len(fa) + len(ea))]
    agg = []
    for p in paths:
        img = read(p)
        al = face.align(img, F) if img is not None else None
        if al is None:
            lines.append(f"| {p.name} | no face |" + " |" * (len(fa) + len(ea) - 1))
            continue
        fs, _ = F["mlp"].predict(embed_regions(al["aligned"], "face", F["device"]))
        fy, _ = face.run_yolo(al["aligned_big"], F)
        crop = eye.eye_crop(al["aligned_big"])
        es, _ = E["mlp"].predict(embed_regions(crop, "eye", E["device"]))
        ey = eye.run_yolo(crop, E)
        cells = [f"{fs[schema.FACE_CLASSES.index(c)]:.2f}" + (f" [{fy[c]:.2f}]" if c in fy else "") for c in fa]
        cells += [f"{es[schema.EYE_CLASSES.index(c)]:.2f}" + (f" [{ey[c]:.2f}]" if c in ey else "") for c in ea]
        lines.append(f"| {p.name} | " + " | ".join(cells) + " |")
        agg.append(np.concatenate([[fs[schema.FACE_CLASSES.index(c)] for c in fa],
                                   [es[schema.EYE_CLASSES.index(c)] for c in ea]]))
    if agg:
        frac = (np.stack(agg) > MLP_PRESENT).mean(0)
        lines.append("| **fraction > 0.35** | " + " | ".join(f"**{f:.0%}**" for f in frac) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfies", help="folder of face photos (out-of-domain test)")
    ap.add_argument("--n", type=int, default=150, help="validation images per modality")
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng, dev = random.Random(a.seed), default_device()
    F, E, T = face.load(dev), eye.load(dev), tongue.load(dev)
    lines = ["# FaceFuel v4 — severity-MLP domain-shift probe"]
    for mod, m in [("face", F), ("eye", E), ("tongue", T)]:
        in_domain(mod, m, a.n, rng, lines)
    tongue_localisation(T, a.n, rng, lines)
    if a.selfies:
        selfies(a.selfies, F, E, lines)
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
