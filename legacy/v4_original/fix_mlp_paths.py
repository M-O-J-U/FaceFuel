"""
FaceFuel v4 — Fix MLP paths in step10_inference.py and Phase7_tongue_inference.py
Run: python fix_mlp_paths.py
"""
import re, shutil
from pathlib import Path

FIXES = [
    {
        "file": "step10_inference.py",
        "searches": [
            # Any .pt file path in facefuel_models that looks like face MLP
            (r'facefuel_models[/\\]["\']?face_severity_mlp[^"\')\s]*',
             r'facefuel_models\\face_severity_mlp_v4.pt'),
            (r'facefuel_models[/\\]["\']?SeverityMLPv2[^"\')\s]*',
             r'facefuel_models\\face_severity_mlp_v4.pt'),
            # Generic: any .pt file called mlp_path = "facefuel_models/..."
            (r'(mlp_path\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\face_severity_mlp_v4.pt\2'),
            (r'(SEVERITY_MLP\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\face_severity_mlp_v4.pt\2'),
            (r'(severity_mlp_path\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\face_severity_mlp_v4.pt\2'),
        ],
        "load_fix": True,
    },
    {
        "file": "Phase7_tongue_inference.py",
        "searches": [
            (r'facefuel_models[/\\]["\']?tongue_severity_mlp[^"\')\s]*',
             r'facefuel_models\\tongue_severity_mlp_v4.pt'),
            (r'facefuel_models[/\\]["\']?TongueSeverityMLP[^"\')\s]*',
             r'facefuel_models\\tongue_severity_mlp_v4.pt'),
            (r'(mlp_path\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\tongue_severity_mlp_v4.pt\2'),
            (r'(SEVERITY_MLP\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\tongue_severity_mlp_v4.pt\2'),
            (r'(severity_mlp_path\s*=\s*["\'])facefuel_models[/\\][^"\']+(["\'])',
             r'\1facefuel_models\\tongue_severity_mlp_v4.pt\2'),
        ],
        "load_fix": True,
    },
]

LOAD_FIX = '''
def _load_mlp_ckpt(mlp, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        mlp.load_state_dict(ckpt["state_dict"])
    else:
        mlp.load_state_dict(ckpt)
    return mlp
'''

# Also fix YOLO paths while we're here
YOLO_FIXES = [
    {
        "file": "step10_inference.py",
        "old": [
            r'runs[/\\]detect[/\\]runs[/\\]face[/\\]face_yolo11m[/\\]weights[/\\]best\.pt',
            r'"facefuel_models[/\\]face_yolo[^"]*"',
            r"'facefuel_models[/\\]face_yolo[^']*'",
        ],
        "new": r'runs\\detect\\runs\\detect\\runs\\face\\face_yolo11m_v4\\weights\\best.pt',
    },
    {
        "file": "Phase7_tongue_inference.py",
        "old": [
            r'runs[/\\]detect[/\\]runs[/\\]tongue[/\\]tongue_v3_improved[/\\]weights[/\\]best\.pt',
            r'"facefuel_models[/\\]tongue_yolo[^"]*"',
            r"'facefuel_models[/\\]tongue_yolo[^']*'",
        ],
        "new": r'runs\\detect\\training_runs\\tongue_v4\\weights\\best.pt',
    },
]

print("="*60)
print("  Fixing MLP + YOLO paths in inference scripts")
print("="*60)

for fix in FIXES:
    p = Path(fix["file"])
    if not p.exists():
        print(f"  skip: {fix['file']}")
        continue
    shutil.copy2(p, p.with_suffix(".py.bak"))
    content = original = p.read_text(encoding="utf-8", errors="ignore")
    changed = []

    for pattern, replacement in fix["searches"]:
        new_c = re.sub(pattern, replacement, content)
        if new_c != content:
            content = new_c
            changed.append(pattern[:40])

    # Fix load_state_dict for new dict format
    if fix.get("load_fix"):
        old_load = r'(mlp|model)\.load_state_dict\(torch\.load\(([^,)]+),\s*map_location=([^)]+)\)\)'
        new_load = r'_load_mlp_ckpt(\1, \2, \3)'
        new_c = re.sub(old_load, new_load, content)
        if new_c != content:
            content = new_c
            changed.append("load_state_dict → _load_mlp_ckpt")
            if "_load_mlp_ckpt" not in content:
                # inject before first class or def
                m = re.search(r'\nclass |\ndef ', content)
                if m:
                    content = content[:m.start()] + "\n" + LOAD_FIX + content[m.start():]

    if content != original:
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ {fix['file']}: {changed}")
    else:
        print(f"  ── {fix['file']}: no auto-matches found")
        # Show what paths we can find in the file
        for line in original.splitlines():
            if "mlp" in line.lower() and (".pt" in line or "path" in line.lower()):
                print(f"     found: {line.strip()[:100]}")

# YOLO path fixes
print()
for fix in YOLO_FIXES:
    p = Path(fix["file"])
    if not p.exists(): continue
    content = original = p.read_text(encoding="utf-8", errors="ignore")
    for pat in fix["old"]:
        content = re.sub(pat, fix["new"], content)
    if content != original:
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ {fix['file']}: YOLO path updated")
    else:
        print(f"  ── {fix['file']}: YOLO path not auto-matched")
        for line in original.splitlines():
            if "best.pt" in line or "yolo" in line.lower():
                print(f"     found: {line.strip()[:100]}")

print(f"""
  Done. If any file shows '──' (not matched), open it and:
  
  step10_inference.py:
    Change MLP path → facefuel_models\\face_severity_mlp_v4.pt
    Change YOLO path → runs\\detect\\runs\\detect\\runs\\face\\face_yolo11m_v4\\weights\\best.pt
  
  Phase7_tongue_inference.py:
    Change MLP path → facefuel_models\\tongue_severity_mlp_v4.pt
    Change YOLO path → runs\\detect\\training_runs\\tongue_v4\\weights\\best.pt
  
  Then: python server_v4.py
""")