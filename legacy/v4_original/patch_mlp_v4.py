"""
FaceFuel v4 — Patch MLP paths and loading code in face/tongue inference scripts
Run: python patch_mlp_v4.py
"""
import re, shutil
from pathlib import Path

NEW_FACE_MLP   = r"facefuel_models\face_severity_mlp_v4.pt"
NEW_TONGUE_MLP = r"facefuel_models\tongue_severity_mlp_v4.pt"

# Old path patterns
OLD_FACE_MLP_PATS = [
    r"facefuel_models[/\\]face_severity_mlp\.pt",
    r"facefuel_models[/\\]face_severity_mlpv2\.pt",
    r"facefuel_models[/\\]SeverityMLPv2.*?\.pt",
]
OLD_TONGUE_MLP_PATS = [
    r"facefuel_models[/\\]tongue_severity_mlp\.pt",
    r"facefuel_models[/\\]TongueSeverityMLP\.pt",
]

# New loading snippet — handles both old (state_dict only) and new (dict) formats
LOAD_SNIPPET = '''
def _load_mlp_weights(mlp, path, device):
    """Load MLP weights — handles both old state_dict and new dict formats."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        mlp.load_state_dict(ckpt["state_dict"])
    else:
        mlp.load_state_dict(ckpt)
    return mlp
'''

def patch_file(filepath, old_pats, new_path, label):
    p = Path(filepath)
    if not p.exists():
        print(f"  skip (not found): {filepath}")
        return False

    content  = p.read_text(encoding="utf-8", errors="ignore")
    original = content
    changed  = False

    # Replace old paths
    for pat in old_pats:
        if re.search(pat, content):
            content = re.sub(pat, new_path.replace("\\","\\\\"), content)
            changed = True

    # Replace simple torch.load(path).load_state_dict pattern
    # Old: mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    # New: _load_mlp_weights(mlp, mlp_path, device)
    old_load = r'mlp\.load_state_dict\(torch\.load\(([^,)]+),\s*map_location=([^)]+)\)\)'
    new_load = r'_load_mlp_weights(mlp, \1, \2)'
    if re.search(old_load, content):
        content = re.sub(old_load, new_load, content)
        changed = True
        # Inject helper function if not already there
        if "_load_mlp_weights" not in content:
            # Insert before first def that loads models
            insert_before = re.search(r'\ndef (get_models|load_models|get_tongue)', content)
            if insert_before:
                pos = insert_before.start()
                content = content[:pos] + LOAD_SNIPPET + content[pos:]

    if changed and content != original:
        shutil.copy2(p, p.with_suffix(".py.bak"))   # backup
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ {filepath}  patched ({label})")
        return True
    else:
        print(f"  ── {filepath}  no matching patterns found")
        print(f"     → manually set MLP path to: {new_path}")
        return False


print("="*60)
print("  Patching MLP paths to v4")
print("="*60)

patch_file("step10_inference.py",   OLD_FACE_MLP_PATS,   NEW_FACE_MLP,   "face MLP v4")
patch_file("tongue_inference.py",   OLD_TONGUE_MLP_PATS, NEW_TONGUE_MLP, "tongue MLP v4")

print(f"""
  Manual fallback — if auto-patch missed your file:
  
  In step10_inference.py, find the MLP weight path and change to:
    {NEW_FACE_MLP}

  In tongue_inference.py, find the MLP weight path and change to:
    {NEW_TONGUE_MLP}

  Also update MLP loading from:
    mlp.load_state_dict(torch.load(path, map_location=device))
  To:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        mlp.load_state_dict(ckpt["state_dict"])
    else:
        mlp.load_state_dict(ckpt)

  Then test:
    python server_v4.py
""")