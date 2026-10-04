"""
Cross-modal fusion and response building.

Per modality, detected features become evidence over the deficiency framework
(evidence.evidence_scores), normalised into that modality's posterior. Modalities
are then combined with a weighted product of experts (face 0.40, tongue 0.35,
eye 0.25). A modality with no evidence for a category contributes nothing — it
is excluded from that category's product, never treated as a guess.

Every category in the response carries explicit provenance:
  status       "flagged"      — at least one run modality found evidence
               "no_signal"    — a run modality covers it, but found nothing
               "not_assessed" — no modality that was run can assess it
  sources      modalities that contributed evidence
  assessed_by  modalities that were run AND can assess the category
so e.g. tongue-only categories can never appear tagged "tongue" (or with a
stray probability) when no tongue photo was analysed — CLAUDE.md Known Issue #4.
"""
from . import schema
from .evidence import evidence_scores

SINGLE_MODALITY_DISCOUNT = 0.85   # applied when the only evidence is a minor-weight modality


def modality_evidence(result: dict) -> dict:
    return evidence_scores(result["features"], schema.FEAT_DEF[result["modality"]])


def fuse(evidence: dict) -> dict:
    """
    evidence: {modality: {deficiency: score}} for the modalities that were RUN.
    Returns {deficiency: fused probability} over deficiencies with any evidence
    (sums to 1), or {} when nothing was found.
    """
    posts = {}
    for mod, ev in evidence.items():
        total = sum(ev.values())
        posts[mod] = {d: v / total for d, v in ev.items()} if total > 0 else {}

    fused = {}
    for d in schema.ALL_DEFS:
        present = [(posts[m][d], schema.MODALITY_WEIGHTS[m]) for m in posts if posts[m].get(d, 0) > 0]
        if not present:
            continue
        if len(present) == 1:
            p, w = present[0]
            fused[d] = p * (SINGLE_MODALITY_DISCOUNT if w < 0.40 else 1.0)
        else:
            wsum = sum(w for _, w in present)
            val = 1.0
            for p, w in present:
                val *= p ** (w / wsum)
            fused[d] = val
    total = sum(fused.values())
    return {d: v / total for d, v in fused.items()} if total > 0 else {}


def _priority(p):
    return "HIGH" if p > 0.20 else "MODERATE" if p > 0.10 else "LOW"


def build_report(results: dict, top_n: int = 5) -> dict:
    """results: {modality: result dict from face/eye/tongue.analyze} for modalities run."""
    evidence = {m: modality_evidence(r) for m, r in results.items()}
    fused = fuse(evidence)
    run = list(results)
    cover = {m: schema.coverage(m) for m in run}

    contributing = {d: [] for d in schema.ALL_DEFS}   # (modality, feature) pairs
    for m, r in results.items():
        fmap = schema.FEAT_DEF[m]
        for feat in r["features"]:
            for d in fmap.get(feat, []):
                contributing[d].append(f"{m}:{feat}")

    analysis, order = {}, []
    for d in schema.ALL_DEFS:
        p = fused.get(d, 0.0)
        sources = [m for m in run if evidence[m].get(d, 0) > 0]
        assessed_by = [m for m in run if d in cover[m]]
        status = "flagged" if p > 0 else "no_signal" if assessed_by else "not_assessed"
        analysis[d] = {
            "status":            status,
            "probability":       round(float(p), 4),
            "probability_pct":   f"{p * 100:.1f}%" if status == "flagged" else None,
            "evidence_strength": round(max((evidence[m].get(d, 0.0) for m in run), default=0.0), 3),
            "confidence_band":   ("high" if p > 0.20 else "moderate" if p > 0.10 else "low")
                                 if status == "flagged" else None,
            "sources":           sources,
            "assessed_by":       assessed_by,
            "evidence":          contributing[d],
            "foods":             schema.FOOD_RECS[d],
            "advice":            schema.ADVICE[d],
        }
        order.append((0 if status == "flagged" else 1 if status == "no_signal" else 2, -p, d))

    ranked = {}
    for rank, (_, _, d) in enumerate(sorted(order), 1):
        analysis[d]["priority_rank"] = rank if analysis[d]["status"] == "flagged" else None
        ranked[d] = analysis[d]

    top = [{
        "rank":        i,
        "issue":       d,
        "probability": f"{fused[d] * 100:.1f}%",
        "priority":    _priority(fused[d]),
        "sources":     analysis[d]["sources"],
        "evidence":    analysis[d]["evidence"],
        "top_foods":   schema.FOOD_RECS[d][:3],
        "advice":      schema.ADVICE[d],
    } for i, d in enumerate(sorted(fused, key=lambda k: -fused[k])[:top_n], 1)]

    counts = {s: sum(1 for v in analysis.values() if v["status"] == s)
              for s in ("flagged", "no_signal", "not_assessed")}
    return {"deficiency_analysis": ranked, "top_insights": top, "summary": counts}
