"""
FaceFuel v4 — Patch weight paths in existing inference scripts
Run: python patch_inference_paths.py
"""
from pathlib import Path
import re

# New v4 weight paths (verified by update_weights.py)
NEW_PATHS = {
    "face":   r"runs\detect\runs\detect\runs\face\face_yolo11m_v4\weights\best.pt",
    "tongue": r"runs\detect\training_runs\tongue_v4\weights\best.pt",
    "eye":    r"runs\detect\training_runs\eye_v2\weights\best.pt",
}

# Old v3 weight path patterns to replace
OLD_PATTERNS = {
    "face": [
        r"runs/detect/runs/face/face_yolo11m/weights/best\.pt",
        r"runs\\detect\\runs\\face\\face_yolo11m\\weights\\best\.pt",
        r"facefuel_models/face_yolo.*?\.pt",
        r"face_yolo11m.*?best\.pt",
    ],
    "tongue": [
        r"runs/detect/runs/tongue/tongue_v3_improved/weights/best\.pt",
        r"runs\\detect\\runs\\tongue\\tongue_v3_improved\\weights\\best\.pt",
        r"tongue_v3.*?best\.pt",
        r"facefuel_models/tongue.*?\.pt",
    ],
    "eye": [
        r"runs/detect/runs/eye/eye_v1/weights/best\.pt",
        r"runs\\detect\\runs\\eye\\eye_v1\\weights\\best\.pt",
        r"eye_v1.*?best\.pt",
        r"facefuel_models/eye.*?yolo.*?\.pt",
    ],
}

# Files to patch
TARGET_FILES = [
    "step10_inference.py",
    "tongue_inference.py",
    "8_eye_inference.py",
    "eye_inference.py",
    "server.py",
]

print("="*60)
print("  Patching weight paths to v4")
print("="*60)

for fname in TARGET_FILES:
    p = Path(fname)
    if not p.exists():
        print(f"  skip (not found): {fname}")
        continue

    content = p.read_text(encoding="utf-8", errors="ignore")
    original = content
    changed = []

    for modality, patterns in OLD_PATTERNS.items():
        new_path = NEW_PATHS[modality]
        for pat in patterns:
            if re.search(pat, content):
                content = re.sub(pat, new_path.replace("\\", "\\\\"), content)
                changed.append(modality)

    if content != original:
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ {fname}  updated: {list(set(changed))}")
    else:
        print(f"  ── {fname}  no old paths found (may already use v4 or need manual update)")

print(f"""
  Manual check — search these files for old path strings:
    step10_inference.py  → look for 'face_yolo11m' or 'best.pt'
    tongue_inference.py  → look for 'tongue_v3' or 'best.pt'

  New paths to use:
    Face:   {NEW_PATHS["face"]}
    Tongue: {NEW_PATHS["tongue"]}
    Eye:    {NEW_PATHS["eye"]}
""")