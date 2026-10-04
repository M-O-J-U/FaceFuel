"""
FaceFuel v4 — Stage 4: train the three YOLO11m detectors
=========================================================
Run (from anywhere):
  python pipeline/04_train_yolo.py                 # face, tongue, eye
  python pipeline/04_train_yolo.py --tongue-only   # or --face-only / --eye-only
  python pipeline/04_train_yolo.py --validate      # re-validate the weights the server uses

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
WORKERS    = 4
PRETRAINED = "yolo11m.pt"          # auto-downloaded by Ultralytics if absent
PROJECT    = ROOT / "training_runs"

RUNS = {   # modality: (data.yaml, run name, batch, weight key in facefuel.paths)
    "face":   (ROOT / "facefuel_face_v4/data.yaml",   "face_v4",   24, "face_yolo"),
    "tongue": (ROOT / "facefuel_tongue_v4/data.yaml", "tongue_v4", 24, "tongue_yolo"),
    "eye":    (ROOT / "facefuel_eye_v4/data.yaml",    "eye_v2",    24, "eye_yolo"),
}


def best_path(name: str) -> Path:
    return PROJECT / name / "weights" / "best.pt"


def train(data_yaml: Path, name: str, batch: int):
    if not data_yaml.exists():
        print(f"  missing: {data_yaml}")
        return None
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


def validate(weights: Path, data_yaml: Path, label: str):
    print(f"\n[val] {label}  {weights}")
    m = YOLO(str(weights)).val(data=str(data_yaml), device=DEVICE, imgsz=IMG_SIZE)
    print(f"  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for mod in RUNS:
        ap.add_argument(f"--{mod}-only", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    chosen = [m for m in RUNS if getattr(args, f"{m}_only")] or list(RUNS)

    if args.validate:
        for mod in chosen:
            data, _, _, key = RUNS[mod]
            validate(paths.resolve(key), data, mod)
        sys.exit(0)

    for mod in chosen:
        data, name, batch, _ = RUNS[mod]
        train(data, name, batch)

    print(f"\n{'=' * 65}  DONE")
    for mod in chosen:
        p = best_path(RUNS[mod][1])
        print(f"  {mod:<8} {p}  [{f'{p.stat().st_size // 2**20} MB' if p.exists() else 'MISSING'}]")
    print("  Next: python scripts/collect_weights.py --from-training\n" + "=" * 65)
