"""
FaceFuel v4 — Update weight paths in all inference scripts
Run once after training:  python update_weights.py
"""
import re
from pathlib import Path

# Actual paths YOLO saved to (prepends runs/detect/)
FACE_PT   = r"runs\detect\runs\detect\runs\face\face_yolo11m_v4\weights\best.pt"
TONGUE_PT = r"runs\detect\training_runs\tongue_v4\weights\best.pt"
EYE_PT    = r"runs\detect\training_runs\eye_v2\weights\best.pt"

def check(path, label):
    p = Path(path)
    if p.exists():
        print(f"  ✅ {label}: {p}  ({p.stat().st_size//1024//1024} MB)")
    else:
        print(f"  ❌ {label}: NOT FOUND at {p}")
    return p.exists()

print("="*60)
print("  Verifying weight files")
print("="*60)
face_ok   = check(FACE_PT,   "Face")
tongue_ok = check(TONGUE_PT, "Tongue")
eye_ok    = check(EYE_PT,    "Eye")

print()
if all([face_ok, tongue_ok, eye_ok]):
    print("  All weights found. Ready for inference.")
else:
    print("  Some weights missing. Check paths above.")

print(f"""
  Copy these paths into your inference scripts:
  
  Face YOLO:
    {FACE_PT}

  Tongue YOLO:
    {TONGUE_PT}

  Eye YOLO:
    {EYE_PT}
""")