"""Development-only grouped threshold/kNN pipeline; all image access is read-only.

Quality labels remain patch labels. Each acquisition group contributes total weight
one, divided equally among its original groups and then among their patches. The
nearest patch from an acquisition group supplies at most one vote. A tied nearest
distance with contradictory patch labels supplies no vote. No file actions occur.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

from features_v1 import FEATURE_NAMES, RULE_FEATURES, extract_features

VERSION = "2.0.0-dev-grouped"
PROTECTED_CATEGORIES = ["含砾细砂岩", "灰白色细粒长石石英砂岩", "碳质页岩", "煤"]
DEFAULT_POLICY = {
    "k": 5, "keep_votes_required": 5, "exclude_votes_required": 4,
    "min_rule_bad_groups": 3, "radius_quantile": 0.90,
    "minimum_category_groups": 5,
}


def vector(features):
    x = np.array([features[n] for n in FEATURE_NAMES], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Feature values must be finite")
    return x


def weighted_quantile(values, weights, quantile):
    """Inverse weighted empirical CDF, aggregating equal values before quantiles.

    Aggregation makes the statistic invariant to repeating an observation while
    dividing its weight. This deliberately differs from an interpolated quantile.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("Expected matching nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expected finite values and positive weights")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be in [0,1]")
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    cumulative = np.cumsum(mass)
    target = quantile * math.fsum(mass.tolist())
    # A tiny tolerance only resolves floating-point equality at a CDF step.
    index = np.searchsorted(cumulative + 1e-12 * cumulative[-1], target, side="left")
    return float(unique[min(int(index), len(unique)-1)])


def _prepare(items):
    rows = sorted((dict(r) for r in items), key=lambda r: str(r["sample_id"]))
    if len(rows) < 2 or len({str(r["sample_id"]) for r in rows}) != len(rows):
        raise ValueError("Need at least two rows and unique sample IDs")
    group_acquisition, group_category = {}, {}
    for r in rows:
        for field in ("sample_id", "group_id", "category"):
            if not isinstance(r[field], str) or not r[field]:
                raise ValueError("Expected nonempty string: " + field)
        if r["bad"] not in (0, 1):
            raise ValueError("Only confirmed binary quality labels are fitted")
        # Group identity is retained; this is an additional independence boundary.
        r["acquisition_group_id"] = r.get("acquisition_group_id") or r["group_id"]
        if not isinstance(r["acquisition_group_id"], str):
            raise ValueError("Acquisition ID must be a string")
        g, a, c = r["group_id"], r["acquisition_group_id"], r["category"]
        if g in group_acquisition and group_acquisition[g] != a:
            raise ValueError("Original group split between acquisition groups")
        if g in group_category and group_category[g] != c:
            raise ValueError("Conflicting category within an original group")
        group_acquisition[g], group_category[g] = a, c
        if r.get("split_role", "train") != "train" or r.get("role", "train") != "train":
            raise ValueError("Only training-partition records may be fitted")
        vector(r["features"])
    if {r["bad"] for r in rows} != {0, 1}:
        raise ValueError("Fit requires both keep and exclude labels")
    count = Counter(r["group_id"] for r in rows)
    members = defaultdict(set)
    for g, a in group_acquisition.items():
        members[a].add(g)
    original_weights = np.array([1.0/count[r["group_id"]] for r in rows])
    weights = np.array([1.0/(count[r["group_id"]]*len(members[r["acquisition_group_id"]])) for r in rows])
    return rows, original_weights, weights


def _category_evidence(rows, minimum):
    result = {}
    for category in sorted({r["category"] for r in rows}):
        rr = [r for r in rows if r["category"] == category]
        good, bad = [r for r in rr if r["bad"] == 0], [r for r in rr if r["bad"] == 1]
        original = {r["group_id"] for r in rr}
        acquisition = {r["acquisition_group_id"] for r in rr}
        reasons = []
        if category in PROTECTED_CATEGORIES:
            reasons.append("explicitly_protected_category")
        if len(original) < minimum:
            reasons.append("insufficient_original_groups")
        if len(acquisition) < minimum:
            reasons.append("insufficient_independent_acquisition_groups")
        if not good:
            reasons.append("no_confirmed_keep_examples")
        if not bad:
            reasons.append("no_confirmed_exclude_examples")
        result[category] = {
            "samples": len(rr), "original_groups": len(original),
            "acquisition_groups": len(acquisition), "keep": len(good), "exclude": len(bad),
            "keep_original_groups": len({r["group_id"] for r in good}),
            "exclude_original_groups": len({r["group_id"] for r in bad}),
            "automatic_exclusion_requires_human": bool(reasons), "protection_reasons": reasons,
        }
    return result


def _nearest_group_indices(distances, rows, excluded_group=None, excluded_acquisition=None, k=5):
    members = defaultdict(list)
    for i, r in enumerate(rows):
        if r["group_id"] == excluded_group or r["acquisition_group_id"] == excluded_acquisition:
            continue
        members[r["acquisition_group_id"]].append(i)
    nearest = []
    for acquisition, indices in members.items():
        minimum = min(float(distances[i]) for i in indices)
        tied = sorted((i for i in indices if math.isclose(float(distances[i]), minimum, rel_tol=1e-12, abs_tol=1e-12)),
                      key=lambda i: rows[i]["sample_id"])
        nearest.append((minimum, acquisition, tied))
    nearest.sort(key=lambda t: (t[0], t[1], rows[t[2][0]]["sample_id"]))
    return nearest[:k]


def fit_model(items, policy=None):
    """Fit only the supplied TRAINING rows; calibration/validation are external."""
    cfg = dict(DEFAULT_POLICY)
    if policy:
        unknown = set(policy)-set(cfg)
        if unknown:
            raise ValueError("Unknown policy fields: " + repr(sorted(unknown)))
        cfg.update(policy)
    if cfg["k"] != 5 or cfg["keep_votes_required"] != 5 or not 1 <= cfg["exclude_votes_required"] <= 5:
        raise ValueError("This version uses five independent groups and five unanimous keep votes")
    if cfg["min_rule_bad_groups"] < 3 or cfg["minimum_category_groups"] < 5 or not 0 < cfg["radius_quantile"] <= 1:
        raise ValueError("Cannot weaken group-evidence safeguards")
    rows, original_weights, weights = _prepare(items)
    x = np.stack([vector(r["features"]) for r in rows])
    y = np.array([r["bad"] for r in rows], dtype=int)
    center = np.array([weighted_quantile(x[:, j], weights, .5) for j in range(x.shape[1])])
    q25 = np.array([weighted_quantile(x[:, j], weights, .25) for j in range(x.shape[1])])
    q75 = np.array([weighted_quantile(x[:, j], weights, .75) for j in range(x.shape[1])])
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.maximum(q75-q25, np.maximum(std*.25, 1e-6))
    z = np.clip((x-center)/scale, -8, 8)
    radii, radius_weights = [], []
    for i, r in enumerate(rows):
        distances = np.sqrt(np.mean((z-z[i])**2, axis=1))
        nearest = _nearest_group_indices(distances, rows, r["group_id"], r["acquisition_group_id"], cfg["k"])
        if len(nearest) == cfg["k"]:
            radii.append(nearest[-1][0]); radius_weights.append(weights[i])
    radius = weighted_quantile(radii, radius_weights, cfg["radius_quantile"]) if radii else None
    rules = []
    for feature, reason in RULE_FEATURES.items():
        values = x[:, FEATURE_NAMES.index(feature)]
        unique = np.unique(values)
        thresholds = unique[:-1] + (unique[1:]-unique[:-1])/2
        best = None
        for threshold in thresholds:
            hit = values >= threshold
            if np.any(hit & (y == 0)):
                continue
            support = np.flatnonzero(hit & (y == 1))
            original = {rows[i]["group_id"] for i in support}
            acquisition = {rows[i]["acquisition_group_id"] for i in support}
            if min(len(original), len(acquisition)) < cfg["min_rule_bad_groups"]:
                continue
            weighted_support = math.fsum(weights[support].tolist())
            score = (round(weighted_support, 12), len(acquisition), float(threshold))
            if best is None or score > best[0]:
                best = (score, float(threshold), len(support), len(original), len(acquisition), weighted_support)
        if best:
            _, threshold, count, originals, acquisitions, support = best
            rules.append({"feature": feature, "op": ">=", "threshold": threshold, "reason": reason,
                          "development_support_bad": count, "development_support_bad_weight": support,
                          "development_support_bad_original_groups": originals,
                          "development_support_bad_acquisition_groups": acquisitions,
                          "development_false_exclusions": 0})
    original_totals, acquisition_totals = defaultdict(float), defaultdict(float)
    for r, w in zip(rows, weights):
        original_totals[r["group_id"]] += float(w)
        acquisition_totals[r["acquisition_group_id"]] += float(w)
    return {
        "engine_version": VERSION, "method": "hierarchically_group_weighted_rules_with_one_group_one_vote_knn",
        "status": "development_only_not_independent_acceptance", "file_actions": "none_predictions_are_recommendations",
        "feature_names": list(FEATURE_NAMES), "feature_extractor": "unchanged_features_v1.py",
        "center": center.tolist(), "scale": scale.tolist(), "reference_features": x.tolist(),
        "reference_labels": y.tolist(), "reference_sample_ids": [r["sample_id"] for r in rows],
        "reference_group_ids": [r["group_id"] for r in rows],
        "reference_acquisition_group_ids": [r["acquisition_group_id"] for r in rows],
        "reference_categories": [r["category"] for r in rows],
        "reference_weights": weights.tolist(), "reference_original_group_weights": original_weights.tolist(),
        "weight_audit": {"formula": "1/(patches_in_original_group * originals_in_acquisition_group)",
                         "per_original_group_total": dict(sorted(original_totals.items())),
                         "per_acquisition_group_total": dict(sorted(acquisition_totals.items())),
                         "total_weight": float(weights.sum())},
        "fit_original_group_count": len(original_totals), "fit_acquisition_group_count": len(acquisition_totals),
        "mixed_quality_original_group_count": sum(len({r["bad"] for r in rows if r["group_id"] == g}) > 1 for g in original_totals),
        "k": cfg["k"], "keep_votes_required": cfg["keep_votes_required"],
        "exclude_votes_required": cfg["exclude_votes_required"], "max_neighbor_distance": radius,
        "radius_fit": {"method": "leave_acquisition_group_out_fifth_distinct_group_weighted_quantile",
                       "quantile": cfg["radius_quantile"], "supported_training_rows": len(radii),
                       "unavailable_means_review": True}, "rules": rules, "policy": cfg,
        "category_evidence": _category_evidence(rows, cfg["minimum_category_groups"]),
        "protected_categories": list(PROTECTED_CATEGORIES), "unknown_category_requires_human_exclusion": True,
        "tie_policy": "mixed_labels_at_same_group_minimum_distance_abstain",
        "confidence_semantics": "decision_tier_not_calibrated_probability", "neural_network_training": False,
    }


def predict(model, features, category, query_group_id=None, query_acquisition_group_id=None):
    """Return an auditable recommendation, never write/move/delete an image."""
    if model["engine_version"] != VERSION or model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Incompatible model or feature ordering")
    rows = [{"sample_id": s, "group_id": g, "acquisition_group_id": a, "bad": int(y)}
            for s, g, a, y in zip(model["reference_sample_ids"], model["reference_group_ids"],
                                  model["reference_acquisition_group_ids"], model["reference_labels"])]
    original_to_acquisition = {r["group_id"]: r["acquisition_group_id"] for r in rows}
    known_acquisition = original_to_acquisition.get(query_group_id)
    if known_acquisition is not None:
        if query_acquisition_group_id is not None and query_acquisition_group_id != known_acquisition:
            raise ValueError("Query original/acquisition identity conflicts with references")
        query_acquisition_group_id = known_acquisition
    x, refs = vector(features), np.asarray(model["reference_features"], float)
    center, scale = np.asarray(model["center"], float), np.asarray(model["scale"], float)
    z, rz = np.clip((x-center)/scale, -8, 8), np.clip((refs-center)/scale, -8, 8)
    distances = np.sqrt(np.mean((rz-z)**2, axis=1))
    nearest = _nearest_group_indices(distances, rows, query_group_id, query_acquisition_group_id, model["k"])
    neighbors = []
    for distance, acquisition, tied in nearest:
        representative = rows[tied[0]]
        labels = {rows[i]["bad"] for i in tied}
        vote = next(iter(labels)) if len(labels) == 1 else None
        neighbors.append({"sample_id": representative["sample_id"], "group_id": representative["group_id"],
                          "acquisition_group_id": acquisition, "distance": distance, "bad_vote": vote,
                          "tie_abstained": vote is None, "tied_sample_ids": [rows[i]["sample_id"] for i in tied],
                          "tied_original_group_ids": sorted({rows[i]["group_id"] for i in tied}),
                          "reference_patch_fit_weight": model["reference_weights"][tied[0]]})
    valid_votes = [r["bad_vote"] for r in neighbors if r["bad_vote"] is not None]
    bad_votes = sum(valid_votes)
    hits = [r["reason"] for r in model["rules"] if features[r["feature"]] >= r["threshold"]]
    sufficient = len(neighbors) == model["k"] and len(valid_votes) == model["k"]
    fifth_distance = neighbors[-1]["distance"] if len(neighbors) == model["k"] else None
    radius = model["max_neighbor_distance"]
    in_range = bool(sufficient and radius is not None and fifth_distance <= radius)
    raw = "review"
    if in_range and hits and bad_votes >= model["exclude_votes_required"]:
        raw = "likely_exclude"
    elif in_range and not hits and bad_votes == 0:
        raw = "likely_keep"
    evidence = model["category_evidence"].get(category)
    protection_reasons = list(evidence["protection_reasons"]) if evidence else ["category_absent_from_fit"]
    if category in model["protected_categories"] and "explicitly_protected_category" not in protection_reasons:
        protection_reasons.append("explicitly_protected_category")
    gate = bool(raw == "likely_exclude" and protection_reasons)
    effective = "review" if gate else raw
    return {
        "machine_decision": effective, "raw_decision": raw, "raw_machine_decision": raw,
        "rarity_gate_applied": gate, "rarity_protection_reasons": protection_reasons,
        "requires_human_confirmation": bool(effective == "review"),
        "automatic_exclusion_effective": bool(effective == "likely_exclude"),
        "rule_triggers": hits, "neighbor_bad_votes": int(bad_votes), "neighbor_count": len(neighbors),
        "effective_vote_count": len(valid_votes), "neighbors": neighbors,
        "neighbor_sample_ids": [r["sample_id"] for r in neighbors],
        "neighbor_group_ids": [r["group_id"] for r in neighbors],
        "neighbor_acquisition_group_ids": [r["acquisition_group_id"] for r in neighbors],
        "fifth_neighbor_distance": fifth_distance, "within_development_range": in_range,
        "query_group_id": query_group_id, "query_acquisition_group_id": query_acquisition_group_id,
        "confidence_semantics": "decision_tier_not_calibrated_probability", "file_action": "none",
    }
