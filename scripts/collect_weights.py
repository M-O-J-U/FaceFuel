"""
Gather every model file the server needs into weights/ under canonical names.

  python scripts/collect_weights.py                  # copy from the current (legacy) locations
  python scripts/collect_weights.py --from-training  # copy fresh runs from training_runs/
  python scripts/collect_weights.py --check          # just report what resolves where

Files are COPIED, never moved, so the original training outputs stay intact.
The resulting weights/ folder (~135 MB) is what you upload as a GitHub Release
asset or bake into the Docker image (see docs/DEPLOYMENT.md).
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from facefuel import paths

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
TRAINING_RUNS = {"face_yolo": "face_v4", "tongue_yolo": "tongue_v4", "eye_yolo": "eye_v2"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-training", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dest = paths.WEIGHTS_DIR

    if a.check:
        for k in paths.KEYS:
            try:
                p = paths.resolve(k)
                print(f"  ✅ {k:<16} {p}  ({p.stat().st_size / 2**20:.1f} MB)")
            except FileNotFoundError:
                print(f"  ❌ {k:<16} not found")
        return

    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k in paths.KEYS:
        target = dest / paths.canonical_name(k)
        src = None
        if a.from_training and k in TRAINING_RUNS:
            src = ROOT / "training_runs" / TRAINING_RUNS[k] / "weights" / "best.pt"
        elif paths.legacy_path(k).is_file():
            src = paths.legacy_path(k)
        if src and src.is_file():
            if src.resolve() != target.resolve():
                shutil.copy2(src, target)
            print(f"  copied {src.relative_to(ROOT)} → {target}")
        elif k == "face_landmarker" and not target.exists():
            print(f"  downloading MediaPipe face landmarker → {target}")
            urllib.request.urlretrieve(LANDMARKER_URL, target)
        if target.is_file():
            manifest.append(f"{sha256(target)}  {target.stat().st_size:>10}  {target.name}")
        else:
            print(f"  ❌ {k}: no source found")
    (dest / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"\n  {len(manifest)}/{len(paths.KEYS)} files in {dest}  (MANIFEST.txt has sha256 prefixes)")


if __name__ == "__main__":
    main()
