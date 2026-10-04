"""
FaceFuel — Stage 3b: build clean v5 datasets from the v4 merged sets
=====================================================================
Fixes three problems found in the v4 audit (docs/RESEARCH_NOTES.md §1):

1. Face label shift. For classification-folder sources (dn_, sdc_, fa_, aug_, vit_)
   the v4 merge wrote shifted class ids (dark_spot→"eczema", redness→"rosacea",
   acne→"wrinkle") and lost every label whose intended class was rosacea, vitiligo,
   eczema or butterfly_rash. Each such image is traced back to its source folder
   by exact file hash and relabelled with the class 03_merge_face.py intended
   (a full-image box, exactly as the original merge meant to write).
2. Train/val leakage. 32–56 % of v4 validation images had exact copies in train,
   plus flipped/rotated augmentations. Images are grouped by a flip- and
   rotation-invariant perceptual hash; each group is kept once and lives in ONE split.
3. No negatives. Verified visible-light normal-eye photos (uveitis-dataset "Normal",
   Roboflow eye-disease "Normal") are added to the eye set as background images.

Fair test split: `test` = groups that were ONLY in v4 val (never in v4 train), so the
v4 models and the new models are both scored on images neither was trained on.
Everything else is re-split train/val 85/15 by group.

Files are HARD-LINKED (no extra disk space; v4 sets are untouched).
Run:  python pipeline/03b_build_clean_v5.py [--modality face|tongue|eye|all]
Out:  facefuel_{face,tongue,eye}_v5/{images,labels}/{train,val,test} + data.yaml + manifest.csv
"""
import argparse
import ast
import csv
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
VAL_FRAC = 0.15
FULL_BOX = "0.500000 0.500000 0.990000 0.990000"

EYE_NEGATIVES = [
    "eye_datasets_v4/mendeley_uveitis/*/Normal",
    "eye_datasets/rf_eye_disease_yolo/*/Normal",
]
NEG_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def dhash_canonical(p: Path):
    """64-bit difference hash, minimised over 4 rotations × 2 flips (augmentation-invariant)."""
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    best = None
    for flip in (False, True):
        base = cv2.flip(img, 1) if flip else img
        for k in range(4):
            v = np.rot90(base, k)
            s = cv2.resize(np.ascontiguousarray(v), (9, 8), interpolation=cv2.INTER_AREA)
            bits = (s[:, 1:] > s[:, :-1]).flatten()
            h = int("".join("1" if b else "0" for b in bits), 2)
            best = h if best is None or h < best else best
    return best


def face_source_maps():
    """Read the class maps straight out of 03_merge_face.py so the intent can't drift."""
    tree = ast.parse((ROOT / "pipeline" / "03_merge_face.py").read_text(encoding="utf-8"))
    maps = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            name = node.targets[0].id
            if name.endswith("_MAP"):
                maps[name] = ast.literal_eval(node.value)
    vit = {"vitiligo": "vitiligo", "Vitiligo": "vitiligo", "Healthy Skin": None, "non_vitiligo": None}
    return {
        "dn":  ("facefuel_datasets_v4/dermnet", maps["DERMNET_MAP"]),
        "sdc": ("facefuel_datasets_v4/skin_diseases_comprehensive", maps["COMP_MAP"]),
        "fa":  ("facefuel_datasets_v4/face_skin_amellia", maps["AMELLIA_MAP"]),
        "aug": ("facefuel_datasets_v4/augmented_skin_conditions", maps["AUG_MAP"]),
        "vit": ("facefuel_datasets_v4/vitiligo", vit),
    }


def read_labels(lp: Path):
    if not lp.exists():
        return []
    return [l.strip() for l in lp.read_text().splitlines() if len(l.split()) >= 5]


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(mod: str):
    rng = random.Random(SEED)
    v4 = ROOT / f"facefuel_{mod}_v4"
    out = ROOT / f"facefuel_{mod}_v5"
    if out.exists():
        print(f"  {out} exists — delete it first to rebuild"); return
    names = yaml.safe_load(open(v4 / "data.yaml"))["names"]
    c2i = {c: i for i, c in enumerate(names)}
    print(f"\n=== {mod}: {len(names)} classes")

    # 1. collect v4 images with (possibly corrected) labels
    relabel = {}
    src_maps = face_source_maps() if mod == "face" else {}
    if mod == "face":
        for pre, (root, cmap) in src_maps.items():
            for p in (ROOT / root).rglob("*"):
                if p.suffix.lower() in IMG_EXT:
                    cls = cmap.get(p.parent.name)
                    relabel.setdefault(md5(p), (pre, cls))
    items = []           # dict(path, split_v4, labels, md5, group)
    fixed = Counter()
    for split in ("train", "val"):
        for ip in sorted((v4 / "images" / split).iterdir()):
            if ip.suffix.lower() not in IMG_EXT:
                continue
            h = md5(ip)
            labels = read_labels(v4 / "labels" / split / (ip.stem + ".txt"))
            pre = ip.name.split("_")[0]
            if pre in src_maps:
                src = relabel.get(h)
                if src is None or src[1] not in c2i:
                    fixed["dropped (no traceable intended class)"] += 1
                    continue
                new = [f"{c2i[src[1]]} {FULL_BOX}"]
                fixed["relabelled" if new != labels else "unchanged"] += 1
                labels = new
            items.append({"path": ip, "split_v4": split, "labels": labels, "md5": h, "neg": False})
    if fixed:
        print("  face relabel:", dict(fixed))

    # 2. negatives (eye only)
    if mod == "eye":
        negs = []
        for pattern in EYE_NEGATIVES:
            for d in ROOT.glob(pattern):
                negs += [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
        print(f"  + {len(negs)} normal-eye negatives")
        for p in negs:
            items.append({"path": p, "split_v4": None, "labels": [], "md5": md5(p), "neg": True})

    # 3. near-duplicate groups
    for it in items:
        it["group"] = dhash_canonical(it["path"]) or int(it["md5"][:15], 16)
    groups = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    print(f"  {len(items)} images → {len(groups)} unique groups "
          f"({len(items) - len(groups)} duplicates/augmentations removed)")

    # 4. assign each group to one split
    rows, stats = [], Counter()
    for g, members in groups.items():
        splits_v4 = {m["split_v4"] for m in members}
        if all(m["neg"] for m in members):
            r = rng.random()
            split = "train" if r < NEG_SPLIT["train"] else "val" if r < NEG_SPLIT["train"] + NEG_SPLIT["val"] else "test"
        elif splits_v4 == {"val"} or splits_v4 == {"val", None}:
            split = "test"           # never seen by any v4 model
        else:
            split = "val" if rng.random() < VAL_FRAC else "train"
        # representative: most label lines; union classes for full-image (classification) labels
        rep = max(members, key=lambda m: len(m["labels"]))
        labels = list(rep["labels"])
        full = [m for m in members if m["labels"] and all(l.endswith(FULL_BOX) for l in m["labels"])]
        if len(full) > 1:
            cls = sorted({int(l.split()[0]) for m in full for l in m["labels"]})
            labels = [f"{c} {FULL_BOX}" for c in cls]
        stem = f"{rep['path'].stem}"
        if rep["neg"]:
            stem = f"neg_{rep['md5'][:12]}"
        dst = out / "images" / split / f"{stem}{rep['path'].suffix.lower()}"
        link(rep["path"], dst)
        lp = out / "labels" / split / f"{stem}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(labels) + ("\n" if labels else ""))
        stats[(split, "negative" if not labels else "labelled")] += 1
        rows.append([split, dst.name, str(rep["path"].relative_to(ROOT)), len(members),
                     ";".join(names[int(l.split()[0])] for l in labels)])

    yaml.dump({"path": str(out), "train": "images/train", "val": "images/val",
               "test": "images/test", "nc": len(names), "names": names},
              open(out / "data.yaml", "w"), default_flow_style=False)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "source", "group_size", "classes"])
        w.writerows(rows)
    for split in ("train", "val", "test"):
        print(f"  {split:5s} labelled={stats[(split, 'labelled')]:6d}  negatives={stats[(split, 'negative')]:5d}")
    per = defaultdict(Counter)
    for split, _, _, _, cls in rows:
        for c in set(cls.split(";")) - {""}:
            per[c][split] += 1
    for c in names:
        print(f"    {c:<22} train={per[c]['train']:5d}  val={per[c]['val']:4d}  test={per[c]['test']:4d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["face", "tongue", "eye", "all"], default="all")
    a = ap.parse_args()
    for m in (["eye", "tongue", "face"] if a.modality == "all" else [a.modality]):
        build(m)
