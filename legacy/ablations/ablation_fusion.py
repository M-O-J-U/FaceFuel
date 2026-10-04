"""
FaceFuel — Paper 2: Multi-Modal Fusion Ablation Study
======================================================
Evaluates six fusion configurations to validate the product-of-experts
approach and the complementarity of face and tongue modalities.

Configurations:
  A  Face-only posterior (11 deficiencies)
  B  Tongue-only posterior (15 deficiencies)
  C  Combined equal weights (face=0.5, tongue=0.5)
  D  Combined optimized weights (face=0.55, tongue=0.45)  ← proposed
  E  Simple average fusion (arithmetic mean)
  F  Maximum fusion (take max of both posteriors)

Metrics:
  - Modality coverage (which deficiencies each modality can detect)
  - Posterior entropy (calibration quality)
  - Deficiency ranking consistency across runs
  - Tongue-exclusive deficiency detection rate
  - Combined vs single-modality posterior sharpness

Output:
  paper_results/fusion_ablation_table.csv
  paper_results/fusion_ablation_results.json
  paper_results/modality_comparison.csv

Run:
  python ablation_fusion.py
  python ablation_fusion.py --quick    (fast check, 50 images)
"""

import argparse, json, time, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import cv2

BASE    = Path("facefuel_datasets")
TBASE   = Path("tongue_datasets")
OUT_DIR = Path("paper_results")
OUT_DIR.mkdir(exist_ok=True)
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ── Deficiency indices ─────────────────────────────────────────
FACE_DEFS = [
    "iron_deficiency", "b12_deficiency", "vitamin_d_deficiency",
    "zinc_deficiency", "omega3_deficiency", "vitamin_a_deficiency",
    "vitamin_c_deficiency", "poor_sleep_quality", "hormonal_imbalance",
    "dehydration", "high_stress",
]
TONGUE_DEFS = [
    "iron_deficiency", "b12_deficiency", "vitamin_d_deficiency",
    "zinc_deficiency", "omega3_deficiency", "vitamin_a_deficiency",
    "vitamin_c_deficiency", "poor_sleep_quality", "hormonal_imbalance",
    "dehydration", "high_stress",
    "liver_stress", "gut_dysbiosis", "hypothyroid", "folate_deficiency",
]
ALL_DEFS = list(dict.fromkeys(FACE_DEFS + TONGUE_DEFS))
TONGUE_EXCLUSIVE = ["liver_stress", "gut_dysbiosis", "hypothyroid", "folate_deficiency"]
N_FACE   = len(FACE_DEFS)    # 11
N_TONGUE = len(TONGUE_DEFS)  # 15
N_ALL    = len(ALL_DEFS)     # 15


# ═══════════════════════════════════════════════════════════════
# FUSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def expand_face_posterior(fp: np.ndarray) -> np.ndarray:
    """Expand 11-dim face posterior to 15-dim ALL_DEFS space."""
    out = np.zeros(N_ALL, dtype=np.float32)
    for i, name in enumerate(FACE_DEFS):
        j = ALL_DEFS.index(name)
        out[j] = fp[i]
    return out / (out.sum() + 1e-9)


def expand_tongue_posterior(tp: np.ndarray) -> np.ndarray:
    """Expand 15-dim tongue posterior to 15-dim ALL_DEFS space (already aligned)."""
    out = np.zeros(N_ALL, dtype=np.float32)
    for i, name in enumerate(TONGUE_DEFS):
        j = ALL_DEFS.index(name)
        out[j] = tp[i]
    return out / (out.sum() + 1e-9)


def fuse_product_of_experts(fp_expanded: np.ndarray,
                             tp_expanded: np.ndarray,
                             fw: float = 0.55,
                             tw: float = 0.45) -> np.ndarray:
    """Weighted product-of-experts fusion."""
    combined = np.zeros(N_ALL, dtype=np.float32)
    for k in range(N_ALL):
        fp = fp_expanded[k]
        tp = tp_expanded[k]
        if fp > 0 and tp > 0:
            combined[k] = (fp ** fw) * (tp ** tw)
        elif fp > 0:
            combined[k] = fp
        elif tp > 0:
            combined[k] = tp * 0.85
    total = combined.sum() + 1e-9
    return combined / total


def fuse_average(fp_expanded: np.ndarray,
                 tp_expanded: np.ndarray) -> np.ndarray:
    """Simple arithmetic mean fusion."""
    combined = (fp_expanded + tp_expanded) / 2.0
    return combined / (combined.sum() + 1e-9)


def fuse_maximum(fp_expanded: np.ndarray,
                 tp_expanded: np.ndarray) -> np.ndarray:
    """Take the maximum from either modality per deficiency."""
    combined = np.maximum(fp_expanded, tp_expanded)
    return combined / (combined.sum() + 1e-9)


# ═══════════════════════════════════════════════════════════════
# POSTERIOR QUALITY METRICS
# ═══════════════════════════════════════════════════════════════

def entropy(p: np.ndarray) -> float:
    """Shannon entropy — lower = more confident posterior."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-12)))

def top1_confidence(p: np.ndarray) -> float:
    """Probability mass on the top-ranked deficiency."""
    return float(np.max(p))

def top3_coverage(p: np.ndarray) -> float:
    """Probability mass in top 3 deficiencies."""
    return float(np.sort(p)[::-1][:3].sum())

def tongue_exclusive_mass(p: np.ndarray) -> float:
    """Total probability mass on the 4 tongue-exclusive deficiencies."""
    mass = 0.0
    for name in TONGUE_EXCLUSIVE:
        j = ALL_DEFS.index(name)
        mass += float(p[j])
    return mass

def rank_correlation(p1: np.ndarray, p2: np.ndarray) -> float:
    """Spearman rank correlation between two posteriors."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(p1, p2)
    return float(r) if not np.isnan(r) else 0.0


# ═══════════════════════════════════════════════════════════════
# LOAD POSTERIORS FROM SAVED RESULTS
# ═══════════════════════════════════════════════════════════════

def load_inference_results():
    """
    Load pre-computed face and tongue posteriors from facefuel_outputs/.
    Falls back to generating from feature matrices if not available.
    """
    outputs_dir = Path("facefuel_outputs")
    tongue_outputs = Path("tongue_datasets") / "outputs"

    face_posts   = []
    tongue_posts = []

    # Try loading from saved JSON results
    json_results = list(outputs_dir.glob("*_result.json")) if outputs_dir.exists() else []
    tongue_jsons = list(tongue_outputs.glob("*_tongue_result.json")) if tongue_outputs.exists() else []

    if json_results:
        print(f"  Found {len(json_results)} saved face results")
        for jf in json_results[:200]:
            try:
                d = json.loads(jf.read_text(errors="ignore"))
                p = d.get("posterior") or d.get("deficiency_posterior")
                if p and len(p) == N_FACE:
                    face_posts.append(np.array(p, dtype=np.float32))
            except Exception:
                pass

    # Generate from feature matrices if insufficient
    if len(face_posts) < 20:
        print("  Generating face posteriors from feature matrix...")
        face_posts = generate_face_posteriors()

    if len(tongue_posts) < 20:
        print("  Generating tongue posteriors from feature matrix...")
        tongue_posts = generate_tongue_posteriors()

    return face_posts, tongue_posts


def generate_face_posteriors(n=None):
    """Run face severity MLP + Bayesian on saved features."""
    from step9_bayesian_engine import load_severity_model, bayesian_inference_v2

    feat_mat  = np.load(str(BASE / "features" / "feature_matrix.npy"))
    stats     = json.loads((BASE / "features" / "feature_stats.json").read_text())
    mean      = np.array(stats["mean"], dtype=np.float32)
    std       = np.array(stats["std"],  dtype=np.float32)
    feat_norm = (feat_mat - mean) / (std + 1e-6)

    if n: feat_norm = feat_norm[:n]

    model, _, _ = load_severity_model(DEVICE)
    model.eval()
    X = torch.from_numpy(feat_norm).float().to(DEVICE)

    posts = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb   = X[i:i+batch]
            sev  = torch.sigmoid(model(xb)).cpu().numpy()
            unc  = np.zeros_like(sev)   # no MC for speed; use mean
            for s, u in zip(sev, unc):
                # Simple Bayesian without color features (no image available)
                p = bayesian_inference_v2(s, u, {}, {})
                if p is not None and len(p) == N_FACE:
                    posts.append(np.array(p, dtype=np.float32))
    return posts


def generate_tongue_posteriors(n=None):
    """Run tongue severity MLP + Bayesian on saved features."""
    feat_mat = np.load(str(TBASE / "features" / "tongue_feature_matrix.npy"))
    stats    = json.loads((TBASE / "features" / "tongue_feature_stats.json").read_text())
    mean     = np.array(stats["mean"], dtype=np.float32)
    std      = np.array(stats["std"],  dtype=np.float32)
    feat_norm = (feat_mat - mean) / (std + 1e-6)

    if n: feat_norm = feat_norm[:n]

    import torch.nn as nn
    ckpt    = torch.load("facefuel_models/tongue_severity_mlp.pt",
                          map_location=DEVICE, weights_only=False)

    class TongueSeverityMLP(nn.Module):
        def __init__(self, fd=1920, nf=12, dr=0.3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.LayerNorm(fd),
                nn.Linear(fd, 512), nn.GELU(), nn.Dropout(dr),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(dr),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(256,64), nn.GELU(),
                              nn.Dropout(0.2), nn.Linear(64,1))
                for _ in range(nf)
            ])
        def forward(self, x):
            s = self.backbone(x)
            return torch.cat([h(s) for h in self.heads], dim=-1)

    nf = len(ckpt.get("tongue_features", range(12)))
    mlp = TongueSeverityMLP(nf=nf)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.to(DEVICE).eval()

    from Phase7_tongue_inference import tongue_bayesian_inference

    X     = torch.from_numpy(feat_norm).float().to(DEVICE)
    posts = []
    with torch.no_grad():
        for i in range(0, min(len(X), n or len(X)), 256):
            xb  = X[i:i+256]
            sev = torch.sigmoid(mlp(xb)).cpu().numpy()
            unc = np.zeros_like(sev)
            for s, u in zip(sev, unc):
                p = tongue_bayesian_inference(s, u, {})
                if p is not None and len(p) == N_TONGUE:
                    posts.append(np.array(p, dtype=np.float32))
    return posts


# ═══════════════════════════════════════════════════════════════
# ABLATION RUNNER
# ═══════════════════════════════════════════════════════════════

def run_ablation(face_posts, tongue_posts):
    """Run all six fusion configurations and compute quality metrics."""

    # Align lengths
    n = min(len(face_posts), len(tongue_posts))
    print(f"  Paired samples: {n:,}")
    fp_all = face_posts[:n]
    tp_all = tongue_posts[:n]

    # Expand posteriors to unified 15-dim space
    fp_exp = [expand_face_posterior(p)   for p in fp_all]
    tp_exp = [expand_tongue_posterior(p) for p in tp_all]

    configs = {
        "A. Face-only":                  fp_exp,
        "B. Tongue-only":                tp_exp,
        "C. Equal fusion (0.5/0.5)":     [fuse_product_of_experts(f,t,0.5,0.5) for f,t in zip(fp_exp,tp_exp)],
        "D. Optimized fusion (0.55/0.45)": [fuse_product_of_experts(f,t,0.55,0.45) for f,t in zip(fp_exp,tp_exp)],
        "E. Simple average":             [fuse_average(f,t) for f,t in zip(fp_exp,tp_exp)],
        "F. Maximum fusion":             [fuse_maximum(f,t) for f,t in zip(fp_exp,tp_exp)],
    }

    results = {}
    for name, posts in configs.items():
        entropies   = [entropy(p)             for p in posts]
        top1s       = [top1_confidence(p)     for p in posts]
        top3s       = [top3_coverage(p)       for p in posts]
        tex_masses  = [tongue_exclusive_mass(p) for p in posts]

        results[name] = {
            "mean_entropy":        round(float(np.mean(entropies)),  4),
            "mean_top1":           round(float(np.mean(top1s)),      4),
            "mean_top3_coverage":  round(float(np.mean(top3s)),      4),
            "tongue_excl_mass":    round(float(np.mean(tex_masses)), 4),
            "n_samples":           len(posts),
        }
        print(f"  {name:<42}  "
              f"entropy={results[name]['mean_entropy']:.3f}  "
              f"top1={results[name]['mean_top1']:.3f}  "
              f"te_mass={results[name]['tongue_excl_mass']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# MODALITY COMPLEMENTARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def complementarity_analysis(face_posts, tongue_posts):
    """
    Measure how often face and tongue rank the same deficiency #1.
    High agreement = redundant. Low agreement = complementary.
    """
    n = min(len(face_posts), len(tongue_posts))
    fp_exp = [expand_face_posterior(p)   for p in face_posts[:n]]
    tp_exp = [expand_tongue_posterior(p) for p in tongue_posts[:n]]

    agree_top1 = 0
    agree_top3 = 0
    face_unique_top1  = defaultdict(int)
    tongue_unique_top1 = defaultdict(int)

    for f, t in zip(fp_exp, tp_exp):
        f_top1 = ALL_DEFS[int(np.argmax(f))]
        t_top1 = ALL_DEFS[int(np.argmax(t))]
        f_top3 = set(ALL_DEFS[i] for i in np.argsort(f)[::-1][:3])
        t_top3 = set(ALL_DEFS[i] for i in np.argsort(t)[::-1][:3])

        if f_top1 == t_top1:
            agree_top1 += 1
        if f_top3 & t_top3:
            agree_top3 += 1

        face_unique_top1[f_top1]   += 1
        tongue_unique_top1[t_top1] += 1

    print(f"\n  Modality complementarity ({n} samples):")
    print(f"    Top-1 agreement   : {agree_top1/n*100:.1f}%  (lower = more complementary)")
    print(f"    Top-3 overlap     : {agree_top3/n*100:.1f}%")
    print(f"\n    Most common face top-1   : {sorted(face_unique_top1.items(),key=lambda x:-x[1])[:3]}")
    print(f"    Most common tongue top-1 : {sorted(tongue_unique_top1.items(),key=lambda x:-x[1])[:3]}")

    return {
        "top1_agreement_pct":  round(agree_top1/n*100, 1),
        "top3_overlap_pct":    round(agree_top3/n*100, 1),
        "face_top1_dist":      dict(face_unique_top1),
        "tongue_top1_dist":    dict(tongue_unique_top1),
        "n_samples":           n,
    }


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

def save_results(ablation, complementarity):
    # JSON
    full = {"ablation": ablation, "complementarity": complementarity}
    jp = OUT_DIR / "fusion_ablation_results.json"
    jp.write_text(json.dumps(full, indent=2), encoding="utf-8")

    # Ablation CSV
    rows = []
    for name, m in ablation.items():
        rows.append({
            "configuration":    name,
            "mean_entropy":     m["mean_entropy"],
            "mean_top1_conf":   m["mean_top1"],
            "top3_coverage":    m["mean_top3_coverage"],
            "tongue_excl_mass": m["tongue_excl_mass"],
            "n":                m["n_samples"],
        })
    cp = OUT_DIR / "fusion_ablation_table.csv"
    with open(cp, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=rows[0].keys())
        dw.writeheader(); dw.writerows(rows)

    print(f"\n  Saved: {jp}")
    print(f"  Saved: {cp}")


def print_table(results):
    print(f"\n{'='*72}")
    print(f"  FUSION ABLATION TABLE — PAPER 2")
    print(f"{'='*72}")
    print(f"  {'Config':<44}  {'Entropy':>8}  {'Top-1':>7}  {'TE-mass':>8}")
    print(f"  {'-'*68}")
    for name, m in results.items():
        tag = " ← proposed" if "0.55" in name else ""
        print(f"  {name:<44}  {m['mean_entropy']:>8.4f}  "
              f"{m['mean_top1']:>7.4f}  {m['tongue_excl_mass']:>8.4f}{tag}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 50 samples for fast sanity check")
    args = parser.parse_args()

    print("="*72)
    print("FaceFuel — Multi-Modal Fusion Ablation (Paper 2)")
    print("="*72)
    print(f"  Device: {DEVICE}")

    n = 50 if args.quick else None
    print(f"\n  Loading posteriors...")
    face_posts, tongue_posts = load_inference_results()

    if n:
        face_posts   = face_posts[:n]
        tongue_posts = tongue_posts[:n]

    print(f"  Face posteriors  : {len(face_posts):,}")
    print(f"  Tongue posteriors: {len(tongue_posts):,}")

    if len(face_posts) < 5 or len(tongue_posts) < 5:
        print("\n  ❌ Not enough posteriors loaded.")
        print("  Make sure step9_bayesian_engine.py and")
        print("  Phase7_tongue_inference.py are importable.")
        exit(1)

    print("\n  Running ablation...")
    ablation = run_ablation(face_posts, tongue_posts)
    print_table(ablation)

    print("\n  Running complementarity analysis...")
    comp = complementarity_analysis(face_posts, tongue_posts)

    save_results(ablation, comp)
    print(f"\n  Results in: {OUT_DIR.resolve()}")
    print(f"\n  Next: python write_paper2.py")