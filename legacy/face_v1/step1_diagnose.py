"""
FaceFuel v2 — Step 1: Diagnose
Reads every data.yaml / classes.txt in each source dataset folder.
Prints the EXACT class ID → name mapping for each dataset.
Run this FIRST before doing any merging or unification.
"""

from pathlib import Path
import yaml
import json

BASE = Path("facefuel_datasets")

# These are the ORIGINAL downloaded source folders only.
# We deliberately exclude merged/processed folders.
SOURCE_FOLDERS = [
    "acne-darkcircles-wrinkles",
    "acne_darkcircles_wrinkles",
    "facial_skin_detection",
    "facial-skin-4fjdg",
    "dark_circles_only",
    "dark-circles-19mqw",
    "skin_issues",
    "unidpro_facial_skin",
    "unidpro_facial",
    "hf_dermatology_bags_acne",
    "hf_dermatology",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def is_valid_yolo_line(line: str) -> bool:
    """Check if a line looks like a real YOLO annotation: int float float float float"""
    parts = line.strip().split()
    if len(parts) < 5:
        return False
    try:
        int(parts[0])          # class id must be integer
        for p in parts[1:5]:
            float(p)           # bbox coords must be floats
        return True
    except ValueError:
        return False

def read_yaml_classes(yaml_path: Path):
    """Parse a data.yaml and return list of class names."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "names" in data:
            names = data["names"]
            if isinstance(names, list):
                return names
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
    except Exception as e:
        print(f"    ⚠ Could not parse {yaml_path.name}: {e}")
    return None

def read_classes_txt(txt_path: Path):
    """Parse a classes.txt (one class name per line)."""
    try:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        classes = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        if classes:
            return classes
    except Exception as e:
        print(f"    ⚠ Could not parse {txt_path.name}: {e}")
    return None

def scan_numeric_ids(labels_dir: Path, sample_size: int = 200):
    """Scan up to sample_size label files and collect all numeric class IDs found."""
    ids_found = set()
    files = list(labels_dir.rglob("*.txt"))[:sample_size]
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if is_valid_yolo_line(line):
                    ids_found.add(int(line.strip().split()[0]))
        except Exception:
            pass
    return sorted(ids_found)

print("=" * 70)
print("FaceFuel v2 — Dataset Diagnostic Report")
print("=" * 70)

all_dataset_maps = {}  # {folder_name: [class0, class1, class2, ...]}

for src_name in SOURCE_FOLDERS:
    src_path = BASE / src_name
    if not src_path.exists():
        continue

    print(f"\n{'─'*60}")
    print(f"📁  {src_name}")

    # Count images and label files
    images = list(src_path.rglob("*.*"))
    img_count  = sum(1 for f in images if f.suffix.lower() in IMAGE_EXTS)
    txt_files  = list(src_path.rglob("*.txt"))

    # Separate real YOLO labels from other txt files
    real_labels = []
    junk_txts   = []
    for tf in txt_files:
        try:
            content = tf.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            lines = content.splitlines()
            valid_lines = sum(1 for l in lines if is_valid_yolo_line(l))
            if valid_lines > 0 and valid_lines / max(len(lines), 1) > 0.5:
                real_labels.append(tf)
            else:
                junk_txts.append(tf)
        except Exception:
            junk_txts.append(tf)

    print(f"   Images       : {img_count}")
    print(f"   Label files  : {len(real_labels)}  (valid YOLO format)")
    print(f"   Junk .txt    : {len(junk_txts)}  (README/classes/notes — will be skipped)")

    # Try to find class names
    class_names = None

    # 1. Look for data.yaml
    for yaml_path in sorted(src_path.rglob("data.yaml")):
        class_names = read_yaml_classes(yaml_path)
        if class_names:
            print(f"   Config found : {yaml_path.relative_to(src_path)}")
            break

    # 2. Look for classes.txt
    if class_names is None:
        for txt_path in sorted(src_path.rglob("classes.txt")):
            class_names = read_classes_txt(txt_path)
            if class_names:
                print(f"   Config found : {txt_path.relative_to(src_path)}")
                break

    # 3. Also try _annotations.coco.json
    if class_names is None:
        for json_path in sorted(src_path.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if "categories" in data:
                    class_names = [cat["name"] for cat in sorted(
                        data["categories"], key=lambda c: c["id"])]
                    print(f"   Config found : {json_path.name} (COCO JSON)")
                    break
            except Exception:
                pass

    if class_names:
        print(f"   Classes ({len(class_names)}) :")
        for i, name in enumerate(class_names):
            print(f"      {i:2d} → {name}")
        all_dataset_maps[src_name] = class_names
    else:
        # Fall back: report numeric IDs actually found in label files
        if real_labels:
            ids = scan_numeric_ids(real_labels[0].parent if real_labels else src_path)
            print(f"   ⚠  No data.yaml found — numeric IDs in labels: {ids}")
            print(f"      → You will need to identify these manually (see MANUAL section below)")
        else:
            print(f"   ⚠  No valid YOLO labels AND no data.yaml found — skipping")

print("\n" + "=" * 70)
print("SUMMARY — datasets with resolved class maps:")
print("=" * 70)
for ds, names in all_dataset_maps.items():
    print(f"  {ds:<35} → {names}")

print("\n" + "=" * 70)
print("MANUAL LOOKUP needed for datasets NOT in the list above.")
print("For each missing one, open its folder and check data.yaml or")
print("classes.txt manually, then add entries to step2_smart_merge.py")
print("=" * 70)
print("\n✅ Diagnostic complete. Now run step2_smart_merge.py")
