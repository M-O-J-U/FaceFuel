"""
FaceFuel v2 — Step 7: Preprocessing Pipeline
---------------------------------------------
Runs BEFORE the YOLO detector on every incoming selfie.

Pipeline per image:
  1. MediaPipe face detection        → bounding box + confidence
  2. 468-landmark extraction         → precise facial geometry
  3. Affine alignment                → canonical frontal face (256x256)
  4. CIE LAB color normalization     → skin-tone agnostic color space
  5. 8 semantic region crops         → per-region patches for YOLO + DINOv2

Output per image (saved to facefuel_datasets/preprocessed/):
  {stem}_aligned.jpg       ← aligned face, LAB-normalized, RGB output
  {stem}_regions.json      ← bounding boxes for each of the 8 regions
  {stem}_debug.jpg         ← visualization with landmarks + region boxes

Usage:
  python step7_preprocessing.py --input path/to/image.jpg
  python step7_preprocessing.py --input facefuel_datasets/MERGED_V2/images --batch
  python step7_preprocessing.py --test   ← runs on 5 sample images and shows results
"""

import cv2
import numpy as np
import json
import argparse
import sys
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ── Output directories ────────────────────────────────────────
OUT_DIR       = Path("facefuel_datasets/preprocessed")
OUT_ALIGNED   = OUT_DIR / "aligned"
OUT_REGIONS   = OUT_DIR / "regions"
OUT_DEBUG     = OUT_DIR / "debug"

for d in [OUT_ALIGNED, OUT_REGIONS, OUT_DEBUG]:
    d.mkdir(parents=True, exist_ok=True)

# ── Target size for aligned face ─────────────────────────────
ALIGN_SIZE = 256   # pixels — good balance of detail vs memory

# ── MediaPipe landmark indices for key points ─────────────────
# These are stable indices in the 468-point mesh
LM = {
    "left_eye_inner":   133,
    "left_eye_outer":   33,
    "right_eye_inner":  362,
    "right_eye_outer":  263,
    "left_eye_center":  468,   # iris center (needs refine_landmarks=True)
    "right_eye_center": 473,
    "nose_tip":         4,
    "nose_bottom":      94,
    "mouth_left":       61,
    "mouth_right":      291,
    "mouth_top":        13,
    "mouth_bottom":     14,
    "left_cheek":       234,
    "right_cheek":      454,
    "chin":             152,
    "forehead_center":  10,
    "left_brow_inner":  55,
    "right_brow_inner": 285,
    "left_brow_outer":  46,
    "right_brow_outer": 276,
}

# ── 8 Semantic regions (landmark-based bounding boxes) ────────
# Each region defined as (name, [landmark_indices], padding_frac)
# padding_frac expands the box by that fraction of its size
REGIONS = [
    ("periorbital_left",    # left under-eye + orbital
     [33, 133, 159, 145, 153, 144, 163, 7], 0.35),

    ("periorbital_right",   # right under-eye + orbital
     [362, 263, 386, 374, 380, 373, 390, 249], 0.35),

    ("left_cheek",          # left cheek skin
     [234, 93, 132, 58, 172, 136, 150, 149], 0.2),

    ("right_cheek",         # right cheek skin
     [454, 323, 361, 288, 397, 365, 379, 378], 0.2),

    ("forehead",            # forehead zone
     [10, 338, 297, 332, 284, 251, 389, 356,
      70,  63,  105, 66,  107, 9,   336, 296], 0.15),

    ("nose",
    [6, 197, 195, 5, 4, 1, 19, 94, 2, 164,   # bridge + tip
    129, 209, 49, 48, 64, 98,                  # left alar
    358, 429, 279, 278, 294, 327], 0.3),       # right alar

    ("lips",                # lip border + vermilion
     [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
      291, 375, 321, 405, 314, 17, 84, 181, 91, 146], 0.2),

    ("sclera_left",         # left eye white (for yellowing/redness)
     [33, 7, 163, 144, 145, 153, 154, 155,
      133, 173, 157, 158, 159, 160, 161, 246], 0.1),
]


# ═══════════════════════════════════════════════════════════════
# MEDIAPIPE SETUP
# ═══════════════════════════════════════════════════════════════

def download_model():
    """Download MediaPipe FaceLandmarker model if not already present."""
    import urllib.request
    model_path = Path("face_landmarker.task")
    if not model_path.exists():
        print("  Downloading MediaPipe FaceLandmarker model (~3MB)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/latest/"
               "face_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print(f"  ✅ Model saved: {model_path}")
    return str(model_path)


def build_face_mesh():
    """
    Build MediaPipe FaceLandmarker using the Tasks API (mediapipe 0.10+).
    Downloads the model file automatically on first run.
    """
    model_path = download_model()

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options         = base_options,
        output_face_blendshapes      = False,
        output_facial_transformation_matrixes = False,
        num_faces            = 1,
        min_face_detection_confidence = 0.5,
        min_face_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        running_mode         = VisionTaskRunningMode.IMAGE,
    )
    return FaceLandmarker.create_from_options(options)


# ═══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_and_align(img_bgr: np.ndarray, face_mesh) -> tuple:
    """
    Detect face, extract 478 landmarks (Tasks API), align to canonical pose.

    Returns:
        aligned_rgb  : np.ndarray (ALIGN_SIZE x ALIGN_SIZE x 3) in RGB
        landmarks_2d : np.ndarray (N x 2) pixel coords on aligned image
        M            : 2x3 affine matrix (original → aligned)
        success      : bool
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tasks API uses mp.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )
    result = face_mesh.detect(mp_image)

    if not result.face_landmarks:
        return None, None, None, False

    lm_raw = result.face_landmarks[0]  # first face

    # Convert normalised coords → pixel coords
    pts = np.array([[lm.x * w, lm.y * h] for lm in lm_raw], dtype=np.float32)

    # ── Affine alignment using eye centers ───────────────────
    left_eye  = pts[[33, 133]].mean(axis=0)
    right_eye = pts[[362, 263]].mean(axis=0)

    target_left  = np.array([ALIGN_SIZE * 0.35, ALIGN_SIZE * 0.40])
    target_right = np.array([ALIGN_SIZE * 0.65, ALIGN_SIZE * 0.40])

    src_vec = right_eye - left_eye
    dst_vec = target_right - target_left
    scale   = np.linalg.norm(dst_vec) / (np.linalg.norm(src_vec) + 1e-6)
    angle   = np.degrees(np.arctan2(dst_vec[1], dst_vec[0]) -
                         np.arctan2(src_vec[1], src_vec[0]))

    cx, cy = left_eye
    M = cv2.getRotationMatrix2D((cx, cy), -angle, scale)
    M[0, 2] += target_left[0] - cx
    M[1, 2] += target_left[1] - cy

    aligned = cv2.warpAffine(img_bgr, M, (ALIGN_SIZE, ALIGN_SIZE),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    lm_aligned = (M @ pts_h.T).T

    lm_aligned[:, 0] = np.clip(lm_aligned[:, 0], 0, ALIGN_SIZE - 1)
    lm_aligned[:, 1] = np.clip(lm_aligned[:, 1], 0, ALIGN_SIZE - 1)

    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned_rgb, lm_aligned, M, True


def normalize_lab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert to CIE LAB, normalize each channel, convert back to RGB.

    Why LAB:
      - L channel = lightness (separates from color)
      - A channel = green↔red (redness detection)
      - B channel = blue↔yellow (yellowing/jaundice detection)
      - Perceptually uniform → skin tone variations are linear

    Returns normalized RGB image (same shape, uint8).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Normalize each channel independently (CLAHE-style per-channel)
    for i in range(3):
        ch = lab[:, :, i]
        p2, p98 = np.percentile(ch, (2, 98))
        if p98 > p2:
            lab[:, :, i] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)

    lab_uint8 = lab.astype(np.uint8)
    normalized_rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
    return normalized_rgb


def extract_regions(img_rgb: np.ndarray,
                    landmarks: np.ndarray) -> dict:
    """
    Extract 8 semantic region crops from the aligned face.

    Returns dict: {region_name: {"bbox": [x1,y1,x2,y2], "crop": np.ndarray}}
    """
    h, w = img_rgb.shape[:2]
    regions_out = {}

    for name, lm_indices, pad in REGIONS:
        # Get landmark coords for this region
        pts = landmarks[lm_indices]

        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())

        # Apply padding
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad)
        py = int(bh * pad)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        crop = img_rgb[y1:y2, x1:x2]

        regions_out[name] = {
            "bbox":  [x1, y1, x2, y2],
            "crop":  crop,
            "shape": list(crop.shape[:2]),
        }

    return regions_out


def draw_debug(img_rgb: np.ndarray,
               landmarks: np.ndarray,
               regions: dict) -> np.ndarray:
    """Draw landmarks and region boxes on a debug image."""
    debug = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    # Draw landmarks (small dots)
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(debug, (x, y), 1, (0, 255, 0), -1)

    # Draw region boxes with labels
    colors = [
        (255, 100, 100), (100, 100, 255), (100, 255, 100),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 50),  (150, 200, 50),
    ]
    for i, (name, info) in enumerate(regions.items()):
        x1, y1, x2, y2 = info["bbox"]
        color = colors[i % len(colors)]
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)
        cv2.putText(debug, name.replace("_", " "),
                    (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    return debug


# ═══════════════════════════════════════════════════════════════
# PROCESS SINGLE IMAGE
# ═══════════════════════════════════════════════════════════════

def process_image(img_path: Path,
                  face_mesh,
                  save_debug: bool = True) -> dict | None:
    """
    Full preprocessing pipeline for one image.
    Returns result dict or None if no face detected.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  ⚠ Could not read: {img_path.name}")
        return None

    # 1. Detect + align
    aligned_rgb, landmarks, M, ok = detect_and_align(img_bgr, face_mesh)
    if not ok:
        return None

    # 2. LAB normalize
    normalized_rgb = normalize_lab(aligned_rgb)

    # 3. Extract regions
    regions = extract_regions(normalized_rgb, landmarks)

    stem = img_path.stem

    # 4. Save aligned image
    aligned_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_ALIGNED / f"{stem}_aligned.jpg"), aligned_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 5. Save region metadata (no crops — too much disk space)
    meta = {
        "source":    img_path.name,
        "align_size": ALIGN_SIZE,
        "regions":   {k: {"bbox": v["bbox"], "shape": v["shape"]}
                      for k, v in regions.items()},
    }
    with open(OUT_REGIONS / f"{stem}_regions.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 6. Debug visualization
    if save_debug:
        debug = draw_debug(normalized_rgb, landmarks, regions)
        cv2.imwrite(str(OUT_DEBUG / f"{stem}_debug.jpg"), debug,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    return meta


# ═══════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

def run_batch(input_dir: Path, max_images: int = None):
    """Process all images in a directory."""
    image_exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in input_dir.rglob("*.*")
              if f.suffix.lower() in image_exts]

    if max_images:
        images = images[:max_images]

    print(f"\nProcessing {len(images):,} images from {input_dir}")
    print(f"Output: {OUT_DIR.resolve()}\n")

    face_mesh = build_face_mesh()
    ok = 0
    fail = 0
    t0 = time.time()

    for i, img_path in enumerate(images):
        result = process_image(img_path, face_mesh, save_debug=(i < 20))
        if result:
            ok += 1
        else:
            fail += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(images) - i - 1) / rate
            print(f"  [{i+1:,}/{len(images):,}]  "
                  f"ok={ok}  fail={fail}  "
                  f"{rate:.1f} img/s  "
                  f"ETA {remaining/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"BATCH COMPLETE")
    print(f"{'='*55}")
    print(f"  Processed : {ok:,}  succeeded,  {fail:,} no-face")
    print(f"  Time      : {elapsed/60:.1f} minutes")
    print(f"  Rate      : {ok/elapsed:.1f} images/sec")
    print(f"  Aligned   : {OUT_ALIGNED}")
    print(f"  Regions   : {OUT_REGIONS}")
    print(f"  Debug     : {OUT_DEBUG}  (first 20 images)")
    face_mesh.__exit__(None, None, None)


def run_test():
    """Test on 5 random images from the dataset."""
    img_dir = Path("facefuel_datasets/MERGED_V2/images")
    images = list(img_dir.glob("*.jpg"))[:5] + list(img_dir.glob("*.png"))[:5]
    images = images[:5]

    if not images:
        print("❌ No images found in facefuel_datasets/MERGED_V2/images")
        return

    print(f"\nTest run on {len(images)} images\n")
    face_mesh = build_face_mesh()

    for img_path in images:
        print(f"  Processing: {img_path.name}")
        result = process_image(img_path, face_mesh, save_debug=True)
        if result:
            regions = result["regions"]
            print(f"    ✅ Face detected — {len(regions)} regions extracted")
            for name, info in regions.items():
                print(f"       {name:<22s} bbox={info['bbox']}  "
                      f"size={info['shape']}")
        else:
            print(f"    ❌ No face detected")
        print()

    face_mesh.__exit__(None, None, None)
    print(f"Debug images saved to: {OUT_DEBUG.resolve()}")
    print("Open the _debug.jpg files to visually verify landmark placement.")


def run_single(img_path: Path):
    """Process a single image and print results."""
    face_mesh = build_face_mesh()
    print(f"\nProcessing: {img_path}")
    result = process_image(img_path, face_mesh, save_debug=True)
    if result:
        print(f"✅ Success")
        print(f"   Aligned  : {OUT_ALIGNED / (img_path.stem + '_aligned.jpg')}")
        print(f"   Debug    : {OUT_DEBUG / (img_path.stem + '_debug.jpg')}")
        print(f"   Regions  : {list(result['regions'].keys())}")
    else:
        print("❌ No face detected in image.")
    face_mesh.__exit__(None, None, None)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FaceFuel v2 — Preprocessing Pipeline")
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to image file or directory")
    parser.add_argument("--batch",  action="store_true",
                        help="Batch process all images in --input directory")
    parser.add_argument("--test",   action="store_true",
                        help="Run test on 5 sample images")
    parser.add_argument("--max",    type=int, default=None,
                        help="Max images to process in batch mode")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.input:
        p = Path(args.input)
        if args.batch or p.is_dir():
            run_batch(p, max_images=args.max)
        else:
            run_single(p)
    else:
        print("FaceFuel v2 — Preprocessing Pipeline")
        print("Usage:")
        print("  python step7_preprocessing.py --test")
        print("  python step7_preprocessing.py --input path/to/face.jpg")
        print("  python step7_preprocessing.py --input facefuel_datasets/MERGED_V2/images --batch")
        print("  python step7_preprocessing.py --input facefuel_datasets/MERGED_V2/images --batch --max 500")
        print("\nRunning test mode automatically...\n")
        run_test()