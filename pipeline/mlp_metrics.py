"""Metrics shared by pipeline/06_train_severity_mlp.py and scripts/compare_models.py."""
import numpy as np

THRESH = 0.35   # decision threshold used at inference (facefuel/evidence.py MLP_PRESENT)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney U); nan if only one class present."""
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def per_class(y: np.ndarray, p: np.ndarray, thresh: float = THRESH) -> list:
    """[(n_pos, f1, precision, recall, auroc, false_positive_rate_on_negatives)] per column."""
    out = []
    for c in range(y.shape[1]):
        t, s = y[:, c] > 0.5, p[:, c]
        pr = s > thresh
        tp, fp, fn = (pr & t).sum(), (pr & ~t).sum(), (~pr & t).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / (~t).sum() if (~t).sum() else float("nan")
        out.append((int(t.sum()), float(f1), float(prec), float(rec), auroc(y[:, c], s), float(fpr)))
    return out


def negatives_flag_rate(y: np.ndarray, p: np.ndarray, thresh: float = THRESH) -> float:
    """Share of all-negative images on which ANY class exceeds the threshold."""
    neg = y.sum(1) == 0
    return float((p[neg] > thresh).any(1).mean()) if neg.any() else float("nan")
