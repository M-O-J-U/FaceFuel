"""
Single source of truth for model file locations.

Resolution order for each file:
  1. An explicit environment variable (e.g. FACEFUEL_FACE_YOLO=/models/face.pt)
  2. $FACEFUEL_WEIGHTS_DIR/<canonical name>   (default: <repo>/weights/)
  3. The legacy training-output location (the doubled runs/detect/... paths
     produced by the v4 training runs — see CLAUDE.md, Known Issue #1)

`python scripts/collect_weights.py` copies the legacy files into weights/ under
their canonical names, which is the layout used for deployment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.environ.get("FACEFUEL_WEIGHTS_DIR", ROOT / "weights"))

# key: (env var, canonical filename in WEIGHTS_DIR, legacy path relative to ROOT)
_FILES = {
    # v5 (2026-10-05): retrained on the clean, deduplicated, relabelled v5 sets.
    # Adopted per modality only where it beat v4 on the held-out v5 test split
    # (docs/model_comparison.md); the tongue detector stays v4.
    "face_yolo":       ("FACEFUEL_FACE_YOLO",   "face_yolo11m_v5.pt",
                        "training_runs/face_v5/weights/best.pt"),
    "tongue_yolo":     ("FACEFUEL_TONGUE_YOLO", "tongue_yolo11m_v4.pt",
                        "runs/detect/training_runs/tongue_v4/weights/best.pt"),
    "eye_yolo":        ("FACEFUEL_EYE_YOLO",    "eye_yolo11m_v5.pt",
                        "training_runs/eye_v5/weights/best.pt"),
    "face_mlp":        ("FACEFUEL_FACE_MLP",    "face_severity_mlp_v5.pt",
                        "facefuel_models/face_severity_mlp_v5.pt"),
    "tongue_mlp":      ("FACEFUEL_TONGUE_MLP",  "tongue_severity_mlp_v5.pt",
                        "facefuel_models/tongue_severity_mlp_v5.pt"),
    "eye_mlp":         ("FACEFUEL_EYE_MLP",     "eye_severity_mlp_v5.pt",
                        "facefuel_models/eye_severity_mlp_v5.pt"),
    "face_landmarker": ("FACEFUEL_FACE_LANDMARKER", "face_landmarker.task",
                        "face_landmarker.task"),
}

DINOV2_MODEL = os.environ.get("FACEFUEL_DINOV2", "facebook/dinov2-small")


def candidates(key: str) -> list:
    env, canonical, legacy = _FILES[key]
    out = []
    if os.environ.get(env):
        out.append(Path(os.environ[env]))
    out += [WEIGHTS_DIR / canonical, ROOT / legacy]
    return out


def resolve(key: str) -> Path:
    """Return the first existing location for `key`, or raise with every path tried."""
    tried = candidates(key)
    for p in tried:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Model file '{key}' not found. Tried:\n  " + "\n  ".join(map(str, tried)) +
        f"\nSet {_FILES[key][0]} or place the file in {WEIGHTS_DIR}.")


def canonical_name(key: str) -> str:
    return _FILES[key][1]


def legacy_path(key: str) -> Path:
    return ROOT / _FILES[key][2]


KEYS = tuple(_FILES)
