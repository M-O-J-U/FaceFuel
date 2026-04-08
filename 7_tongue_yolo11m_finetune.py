"""
FaceFuel — Tongue YOLO11m Fine-tune from v4
============================================
Starts from tongue_v4_fixed weights and fine-tunes with YOLO11m.
Uses batch=20 to push VRAM to ~10GB on RTX 4070 Super.

Run: python tongue_retrain_yolo11m.py
"""
from pathlib import Path
import torch
from ultralytics import YOLO

DEVICE  = "0" if torch.cuda.is_available() else "cpu"
RUN_DIR = Path("runs/tongue")

# ── Find best v4 weights ──────────────────────────────────────
candidates = (
    list(Path(".").rglob("tongue_v4_fixed/weights/best.pt")) +
    list(Path(".").rglob("tongue_v3_improved/weights/best.pt")) +
    list(Path(".").rglob("tongue_v2_retrain/weights/best.pt"))
)
if not candidates:
    print("❌ No tongue weights found. Run tongue training first.")
    exit(1)

start_weights = str(candidates[0])
print(f"Starting from: {start_weights}")

# ── Find data.yaml ────────────────────────────────────────────
yaml = next(Path("tongue_datasets").rglob("TONGUE_COMBINED/data.yaml"), None)
if not yaml:
    print("❌ TONGUE_COMBINED/data.yaml not found.")
    exit(1)

print(f"Data: {yaml}")
print(f"Device: {DEVICE}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Batch: 20  VRAM target: ~10GB")

model = YOLO(start_weights)  # fine-tune FROM v4 weights

results = model.train(
    data          = str(yaml),
    epochs        = 100,
    imgsz         = 640,
    batch         = 20,        # ~10GB VRAM on RTX 4070 Super
    workers       = 0,
    device        = DEVICE,
    project       = str(RUN_DIR),
    name          = "tongue_yolo11m_ft",
    patience      = 35,
    lr0           = 5e-5,      # very low — fine-tuning, not training from scratch
    lrf           = 0.01,
    warmup_epochs = 3,
    box           = 7.5,
    cls           = 3.0,
    hsv_h         = 0.025,
    hsv_s         = 0.6,
    hsv_v         = 0.4,
    fliplr        = 0.5,
    flipud        = 0.0,
    degrees       = 12,
    scale         = 0.3,
    mosaic        = 0.85,
    mixup         = 0.2,
    save          = True,
    exist_ok      = True,
    verbose       = True,
)

map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
best  = next(Path(".").rglob("tongue_yolo11m_ft/weights/best.pt"), None)
print(f"\n✅ Done — mAP@0.5: {map50}")
print(f"   Weights: {best}")
print(f"\n   If mAP > 0.836 (v3 best): use these weights in server.py")
print(f"   If mAP < 0.836: keep using tongue_v4_fixed weights")