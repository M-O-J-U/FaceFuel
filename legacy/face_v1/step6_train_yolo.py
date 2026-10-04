"""
FaceFuel v2 — Step 6: YOLOv8 Detection Training (CUDA-forced)
--------------------------------------------------------------
YOLOv8 is the REGION PROPOSER in FaceFuel v2.
It localises skin features (dark circles, acne, wrinkles, etc.)
on the face. Its detections feed into DINOv2 + Bayesian engine.

Install: pip install ultralytics torch torchvision
"""

import os
import sys
from pathlib import Path

# ── Force CUDA — crash early if GPU not available ─────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # use first GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"         # better CUDA error messages

import torch

print("=" * 65)
print("FaceFuel v2 — GPU Check")
print("=" * 65)

if not torch.cuda.is_available():
    print("❌ CUDA GPU NOT detected.")
    print("   Possible fixes:")
    print("   1. Install CUDA-enabled PyTorch:")
    print("      pip uninstall torch torchvision torchaudio -y")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   2. Make sure your NVIDIA drivers are up to date.")
    print("   3. Run: nvidia-smi  — if this fails, your GPU/driver is the issue.")
    print("\n   If you only have CPU, change FORCE_GPU = True → False below.")
    sys.exit(1)

# GPU info
gpu_name  = torch.cuda.get_device_name(0)
gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1024**3
gpu_count = torch.cuda.device_count()

print(f"  ✅ CUDA available")
print(f"  GPU       : {gpu_name}")
print(f"  VRAM      : {gpu_mem:.1f} GB")
print(f"  GPU count : {gpu_count}")
print(f"  PyTorch   : {torch.__version__}")
print(f"  CUDA ver  : {torch.version.cuda}")

# ── Auto-tune batch size based on VRAM ───────────────────────
if gpu_mem >= 16:
    BATCH = 32
elif gpu_mem >= 8:
    BATCH = 16
elif gpu_mem >= 6:
    BATCH = 8
elif gpu_mem >= 4:
    BATCH = 4
else:
    BATCH = 2
print(f"  Batch sz  : {BATCH}  (auto-tuned for {gpu_mem:.0f}GB VRAM)")

# ── Config ────────────────────────────────────────────────────
DATA_YAML    = "facefuel_datasets/MERGED_V2/data.yaml"
MODEL        = "yolov8m.pt"
EPOCHS       = 100
IMG_SIZE     = 640
PROJECT_NAME = "facefuel_v2"
RUN_NAME     = "yolo_detector_r1"
DEVICE       = "0"               # GPU device ID — "0" = first GPU

# ── Validate data.yaml ────────────────────────────────────────
if not Path(DATA_YAML).exists():
    print(f"\n❌ data.yaml not found: {DATA_YAML}")
    print("   Run step2_smart_merge.py first.")
    sys.exit(1)

img_dir = Path("facefuel_datasets/MERGED_V2/images")
lbl_dir = Path("facefuel_datasets/MERGED_V2/labels")
images  = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
labels  = list(lbl_dir.glob("*.txt"))

print("\n" + "=" * 65)
print("FaceFuel v2 — YOLOv8 Training")
print("=" * 65)
print(f"  Images  : {len(images):,}")
print(f"  Labels  : {len(labels):,}")
print(f"  Model   : {MODEL}")
print(f"  Epochs  : {EPOCHS}")
print(f"  ImgSize : {IMG_SIZE}")
print(f"  Device  : GPU:{DEVICE} ({gpu_name})")
print(f"  Output  : runs/detect/{PROJECT_NAME}/{RUN_NAME}/")

# ── Everything below MUST be inside __main__ on Windows ───────
if __name__ == "__main__":
    try:
        from ultralytics import YOLO

        print("\n  Warming up CUDA...")
        _ = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        print("  ✅ CUDA warm-up done\n")

        model = YOLO(MODEL)

        results = model.train(
            data      = DATA_YAML,
            epochs    = EPOCHS,
            batch     = BATCH,
            imgsz     = IMG_SIZE,
            device    = DEVICE,
            patience  = 15,
            project   = f"runs/detect/{PROJECT_NAME}",
            name      = RUN_NAME,
            exist_ok  = True,
            workers   = 0,           # ← CRITICAL on Windows: prevents multiprocessing crash

            # Skin-specific augmentation
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            flipud       = 0.0,
            fliplr       = 0.5,
            degrees      = 10.0,
            translate    = 0.1,
            scale        = 0.5,
            mosaic       = 1.0,
            mixup        = 0.1,
            erasing      = 0.4,
            close_mosaic = 10,

            # Optimiser
            optimizer     = "AdamW",
            lr0           = 0.001,
            lrf           = 0.01,
            weight_decay  = 0.0005,
            warmup_epochs = 3.0,

            # AMP — ~2x GPU speedup
            amp         = True,

            # Logging
            verbose     = True,
            plots       = True,
            save        = True,
            save_period = 10,
        )

        print("\n" + "=" * 65)
        print("TRAINING COMPLETE")
        print("=" * 65)
        best = Path(f"runs/detect/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
        print(f"  Best weights : {best.resolve()}")
        print(f"  mAP@0.5      : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP@0.5:0.95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"\n  Charts saved : runs/detect/{PROJECT_NAME}/{RUN_NAME}/")
        print("""
NEXT STEPS:
  1. Paste the mAP scores — we'll evaluate and decide on round 2.
  2. Run step4_pseudolabel.py with best.pt to auto-label unlabeled images.
  3. Once mAP@0.5 >= 0.60 move to step7_dinov2_features.py.
""")

    except ImportError:
        print("❌ ultralytics not installed: pip install ultralytics")
    except RuntimeError as e:
        print(f"❌ CUDA RuntimeError: {e}")
        if "out of memory" in str(e).lower():
            print(f"  → Reduce BATCH from {BATCH} to {BATCH//2} and retry.")
    except Exception as e:
        print(f"❌ Error: {e}")