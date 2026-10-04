"""
FaceFuel v4 — Stage 4: train the three YOLO11m detectors
=========================================================
Run (from anywhere):
  python pipeline/04_train_yolo.py --data v5                 # face, tongue, eye on the clean v5 sets
  python pipeline/04_train_yolo.py --data v5 --eye-only      # or --face-only / --tongue-only
  python pipeline/04_train_yolo.py --validate --split test   # score the weights the server uses

Output: training_runs/<name>/weights/best.pt
`project` is passed as an ABSOLUTE path. Ultralytics nests any relative
`project` under runs/detect/, which is what produced the doubled
runs/detect/runs/detect/... paths of the original v4 runs (CLAUDE.md Known
Issue #1). After training, run `python scripts/collect_weights.py --from-training`
to copy the new weights into weights/ where the server looks first.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from facefuel import paths

DEVICE     = "0"
EPOCHS     = 80
PATIENCE   = 20
IMG_SIZE   = 640
WORKERS    = 1          # ~1 GB RAM per dataloader worker (val uses 2x); 16 GB machines need few
_LOCAL_BASE = ROOT / "local" / "base_weights" / "yolo11m.pt"
PRETRAINED = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "yolo11m.pt"   # else auto-downloaded
PROJECT    = ROOT / "training_runs"

# v4 used 24; on a 12 GB card that fills VRAM and Windows silently spills into system
# memory (~10× slower) or OOMs. 12 fits comfortably (~6 GB).
BATCH = {"face": 12, "tongue": 12, "eye": 12}
WEIGHT_KEY = {"face": "face_yolo", "tongue": "tongue_yolo", "eye": "eye_yolo"}


def runs(data_version: str) -> dict:
    """modality → (data.yaml, run name, batch, weight key in facefuel.paths)"""
    return {m: (ROOT / f"facefuel_{m}_{data_version}/data.yaml", f"{m}_{data_version}",
                BATCH[m], WEIGHT_KEY[m]) for m in BATCH}


def best_path(name: str) -> Path:
    return PROJECT / name / "weights" / "best.pt"


def training_state(name: str) -> str:
    """'none' (never started), 'partial' (interrupted — resumable), or 'done'.
    Ultralytics strips the optimizer and sets epoch = -1 in last.pt when a run finishes."""
    last = PROJECT / name / "weights" / "last.pt"
    if not last.exists():
        return "none"
    import torch
    try:
        ck = torch.load(str(last), map_location="cpu", weights_only=False)
    except Exception:
        return "none"          # unreadable (e.g. power cut mid-write): start fresh
    return "done" if ck.get("epoch", -1) == -1 else "partial"


def train(data_yaml: Path, name: str, batch: int, fresh: bool = False):
    if not data_yaml.exists():
        print(f"  missing: {data_yaml}")
        return None
    state = "none" if fresh else training_state(name)
    if state == "done":
        print(f"  {name}: already finished — skipping ({best_path(name)})")
        return None
    if state == "partial":
        last = PROJECT / name / "weights" / "last.pt"
        print(f"\n{'=' * 65}\n  RESUMING {name} from {last}\n{'=' * 65}\n")
        return YOLO(str(last)).train(resume=True)
    print(f"\n{'=' * 65}\n  Training: {name}\n  data={data_yaml}  batch={batch}  epochs={EPOCHS}"
          f"\n  output: {best_path(name)}\n{'=' * 65}\n")
    t0 = time.time()
    results = YOLO(PRETRAINED).train(
        data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE,
        batch=batch, device=DEVICE, workers=WORKERS,
        project=str(PROJECT), name=name, exist_ok=True,
        cache=False, amp=True,
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, weight_decay=0.0005, warmup_epochs=3, close_mosaic=10,
    )
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"\n  done  {(time.time() - t0) / 60:.1f} min  mAP50={map50:.3f}  {best_path(name)}")
    return results


def validate(weights: Path, data_yaml: Path, label: str, split: str = "val"):
    print(f"\n[{split}] {label}  {weights}")
    m = YOLO(str(weights)).val(data=str(data_yaml), device=DEVICE, imgsz=IMG_SIZE, split=split)
    print(f"  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for mod in BATCH:
        ap.add_argument(f"--{mod}-only", action="store_true")
    ap.add_argument("--data", default="v5", help="dataset version suffix: facefuel_<modality>_<data>")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--weights", help="with --validate: score this file instead of the server's")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--fresh", action="store_true",
                    help="start over even if an interrupted run exists (default: resume it)")
    ap.add_argument("--state", action="store_true", help="print each run's state and exit")
    args = ap.parse_args()
    RUNS = runs(args.data)
    chosen = [m for m in RUNS if getattr(args, f"{m}_only")] or list(RUNS)

    if args.validate:
        for mod in chosen:
            data, _, _, key = RUNS[mod]
            validate(Path(args.weights) if args.weights else paths.resolve(key), data, mod, args.split)
        sys.exit(0)

    if args.state:
        for mod in chosen:
            print(f"{mod}: {training_state(RUNS[mod][1])}")
        sys.exit(0)

    for mod in chosen:
        data, name, batch, _ = RUNS[mod]
        train(data, name, batch, fresh=args.fresh)

    print(f"\n{'=' * 65}  DONE")
    for mod in chosen:
        p = best_path(RUNS[mod][1])
        print(f"  {mod:<8} {p}  [{f'{p.stat().st_size // 2**20} MB' if p.exists() else 'MISSING'}]")
    print("  Next: python scripts/collect_weights.py --from-training\n" + "=" * 65)
