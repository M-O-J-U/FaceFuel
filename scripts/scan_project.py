"""
FaceFuel — Project Structure Scanner
=====================================
Scans the entire FaceFuel directory and writes a detailed
structure report to: facefuel_structure.txt

Run: python scan_project.py
     python scan_project.py --path C:\\Users\\mojua\\Desktop\\FaceFuel
     python scan_project.py --depth 4   (default: 5)
"""

import os, sys, argparse, time
from pathlib import Path
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument("--path",  default=".", help="Root directory to scan")
parser.add_argument("--depth", type=int, default=5, help="Max folder depth")
parser.add_argument("--out",   default="facefuel_structure.txt")
args = parser.parse_args()

ROOT    = Path(args.path).resolve()
MAX_D   = args.depth
OUT_TXT = ROOT / args.out

IMG_EXT  = {".jpg",".jpeg",".png",".bmp",".webp",".avif",".tiff"}
PT_EXT   = {".pt",".pth",".onnx"}
CODE_EXT = {".py",".yaml",".yml",".json",".txt",".csv",".md",".bib",".tex"}
DATA_EXT = IMG_EXT | {".zip",".tar",".gz",".csv",".json"}

def fmt_size(n_bytes):
    for unit in ["B","KB","MB","GB","TB"]:
        if n_bytes < 1024: return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"

def count_images(path):
    """Recursively count images in a directory."""
    n = 0
    try:
        for f in Path(path).rglob("*"):
            if f.suffix.lower() in IMG_EXT:
                n += 1
    except PermissionError:
        pass
    return n

def dir_size(path):
    total = 0
    try:
        for f in Path(path).rglob("*"):
            if f.is_file():
                try: total += f.stat().st_size
                except: pass
    except PermissionError:
        pass
    return total

lines = []

def w(s=""):
    lines.append(s)

def scan_dir(path, depth=0, prefix=""):
    if depth > MAX_D:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return

    dirs  = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    for i, d in enumerate(dirs):
        is_last = (i == len(dirs)-1) and not files
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        # gather stats
        img_count = count_images(d)
        sz        = dir_size(d)
        img_str   = f"  [{img_count:,} imgs]" if img_count > 0 else ""
        sz_str    = f"  {fmt_size(sz)}"

        w(f"{prefix}{connector}📁 {d.name}{img_str}{sz_str}")
        scan_dir(d, depth+1, child_prefix)

    # Group files by extension
    ext_groups = defaultdict(list)
    for f in files:
        ext_groups[f.suffix.lower()].append(f)

    # Show .pt files individually (weights)
    for f in ext_groups.get(".pt", []) + ext_groups.get(".pth", []):
        try: sz = fmt_size(f.stat().st_size)
        except: sz = "?"
        connector = "└── " if f == files[-1] else "├── "
        w(f"{prefix}{connector}⚙️  {f.name}  ({sz})")

    # Show .py files individually
    for f in ext_groups.get(".py", []):
        connector = "└── " if f == files[-1] else "├── "
        w(f"{prefix}{connector}🐍 {f.name}")

    # Show yaml/yml/json individually
    for ext in [".yaml",".yml"]:
        for f in ext_groups.get(ext, []):
            connector = "└── " if f == files[-1] else "├── "
            w(f"{prefix}{connector}📄 {f.name}")

    # Group images
    n_img = sum(len(v) for k,v in ext_groups.items() if k in IMG_EXT)
    if n_img > 0:
        connector = "└── " if not [f for f in files
                                    if f.suffix.lower() not in IMG_EXT
                                    and f.suffix.lower() not in {".py",".yaml",".yml"}] \
                    else "├── "
        w(f"{prefix}{connector}🖼  {n_img:,} image files  "
          f"({', '.join(f'{len(v)} {k}' for k,v in ext_groups.items() if k in IMG_EXT)})")

    # Other notable files
    for ext in [".tex",".bib",".md",".txt",".csv"]:
        for f in ext_groups.get(ext, []):
            if f.name == args.out: continue  # skip output file itself
            connector = "└── " if f == files[-1] else "├── "
            try: sz = fmt_size(f.stat().st_size)
            except: sz = "?"
            w(f"{prefix}{connector}📝 {f.name}  ({sz})")


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
w("="*70)
w("  FaceFuel Project Structure Report")
w(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
w(f"  Root:      {ROOT}")
w("="*70)
w()

# ── Top-level summary ──────────────────────────────────────────
top_dirs  = [d for d in ROOT.iterdir() if d.is_dir()]
top_files = [f for f in ROOT.iterdir() if f.is_file()]
total_sz  = dir_size(ROOT)
total_img = count_images(ROOT)

w("SUMMARY")
w("-"*70)
w(f"  Total size:    {fmt_size(total_sz)}")
w(f"  Total images:  {total_img:,}")
w(f"  Top-level dirs:{len(top_dirs)}")
w(f"  Top-level files:{len(top_files)}")
w()

# ── Dataset directories detail ────────────────────────────────
dataset_dirs = ["facefuel_datasets_v4","tongue_datasets_v4","eye_datasets_v4",
                "facefuel_face_v4","facefuel_tongue_v4","facefuel_eye_v4",
                "tongue_dataset","eye_dataset","datasets","data",
                "facefuel_datasets","tongue_datasets","eye_datasets"]

w("DATASET DIRECTORIES")
w("-"*70)
for dname in dataset_dirs:
    dp = ROOT / dname
    if dp.exists():
        imgs = count_images(dp)
        sz   = dir_size(dp)
        subdirs = [d for d in dp.iterdir() if d.is_dir()]
        w(f"  {dname}/")
        w(f"    Images:  {imgs:,}")
        w(f"    Size:    {fmt_size(sz)}")
        w(f"    Subdirs: {len(subdirs)}")
        for sd in sorted(subdirs):
            si = count_images(sd)
            ss = dir_size(sd)
            w(f"      ├── {sd.name:<45} {si:>6,} imgs  {fmt_size(ss)}")
        w()

# ── Model weights ─────────────────────────────────────────────
w("MODEL WEIGHTS (.pt files)")
w("-"*70)
pt_files = list(ROOT.rglob("*.pt"))
if pt_files:
    for pt in sorted(pt_files):
        rel = pt.relative_to(ROOT)
        try: sz = fmt_size(pt.stat().st_size)
        except: sz = "?"
        w(f"  {rel}  ({sz})")
else:
    w("  No .pt files found")
w()

# ── Python scripts ─────────────────────────────────────────────
w("PYTHON SCRIPTS (.py)")
w("-"*70)
py_files = [f for f in ROOT.glob("*.py")]  # top-level only
for f in sorted(py_files):
    try: sz = fmt_size(f.stat().st_size)
    except: sz = "?"
    w(f"  {f.name:<55} {sz}")
w()

# ── Training runs ─────────────────────────────────────────────
w("TRAINING RUNS")
w("-"*70)
runs_dir = ROOT / "runs"
if runs_dir.exists():
    for run in sorted(runs_dir.rglob("results.csv")):
        run_dir = run.parent
        rel     = run_dir.relative_to(ROOT)
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        has_best = "✅" if best_pt.exists() else "❌"
        w(f"  {has_best} {rel}")
        # parse last mAP from results.csv
        try:
            with open(run) as f:
                rows = f.readlines()
            if rows:
                last = rows[-1].strip().split(",")
                headers = rows[0].strip().split(",")
                hmap = {h.strip(): v.strip() for h,v in zip(headers,last)}
                map50 = hmap.get("metrics/mAP50(B)","?")
                map5095 = hmap.get("metrics/mAP50-95(B)","?")
                w(f"       mAP@0.5={map50}  mAP@0.5:95={map5095}")
        except Exception:
            pass
else:
    w("  No runs/ directory found")
w()

# ── Full tree ──────────────────────────────────────────────────
w("="*70)
w("FULL DIRECTORY TREE")
w(f"(max depth: {MAX_D})")
w("="*70)
w(f"📁 {ROOT.name}/")
scan_dir(ROOT, depth=1, prefix="")

w()
w("="*70)
w("END OF REPORT")
w("="*70)

# Write output
output = "\n".join(lines)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"\n✅ Saved to: {OUT_TXT}")
print(f"   Lines:    {len(lines)}")