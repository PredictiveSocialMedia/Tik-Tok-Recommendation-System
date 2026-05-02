"""
Held-out evaluation for the trajectory-aware ranking module.

Implements suggestion #1 from the prof's email:
  "Add proper evaluation metrics (NDCG@k, MRR@k) to the trajectory-aware
  ranking and report numbers on a held-out set."

The trajectory ranker scores videos by their predicted lifecycle regime
(spike / balanced / durable) and per-objective velocity components. This
module turns those features into a single per-objective score, derives
relevance grades from the ground-truth label z-scores, and reports
NDCG@k and MRR@k across the held-out split.

Pure functions, no I/O — the CLI runner in
``scripts/eval_trajectory_held_out.py`` handles the file reads and
prints / saves the metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .evaluator import aggregate, mrr_at_k, ndcg_at_k


# Each objective is most informative for a single regime per the comments in
# baseline_common.OBJECTIVE_RANKING_WEIGHTS:
#   reach       → spike    (viral lift)
#   engagement  → balanced (sustained interaction)
#   conversion  → durable  (evergreen value)
OBJECTIVE_REGIME: Dict[str, str] = {
    "reach": "spike",
    "engagement": "balanced",
    "conversion": "durable",
}

# Relevance grades 0-3 match the scale used by the pairwise ranker
# (LearnedPairwiseReranker / RankerWeightOptimizer).
RELEVANCE_GRADES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)

DEFAULT_K_VALUES: Tuple[int, ...] = (10, 20)
DEFAULT_OBJECTIVES: Tuple[str, ...] = ("reach", "engagement", "conversion")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def trajectory_score_for_objective(
    trajectory_features: Dict[str, Any],
    objective: str,
) -> float:
    """
    Derive a single ranking score from a row's ``trajectory_features``.

    Score = primary regime probability × regime confidence, where the
    primary regime is the one most associated with the objective
    (spike→reach, balanced→engagement, durable→conversion).

    Both factors live in [0, 1] so the resulting score is in [0, 1].
    Returns 0.0 if the trajectory features are missing or malformed.
    """
    if not isinstance(trajectory_features, dict):
        return 0.0
    regime = OBJECTIVE_REGIME.get(objective)
    if regime is None:
        return 0.0
    probs = trajectory_features.get("regime_probabilities")
    if not isinstance(probs, dict):
        return 0.0
    prob = _as_float(probs.get(regime)) or 0.0
    confidence = _as_float(trajectory_features.get("regime_confidence")) or 0.0
    return max(0.0, min(1.0, prob)) * max(0.0, min(1.0, confidence))


def target_z_for_objective(row: Dict[str, Any], objective: str) -> Optional[float]:
    """Pull ``targets_z[objective]`` out of a row, or None when missing."""
    targets_z = row.get("targets_z")
    if not isinstance(targets_z, dict):
        return None
    return _as_float(targets_z.get(objective))


def assign_relevance_grades(
    z_scores: Sequence[float],
    grades: Sequence[float] = RELEVANCE_GRADES,
) -> List[float]:
    """
    Convert continuous z-scores into discrete relevance grades by quantile
    bucketing — top quartile gets the highest grade, bottom quartile the
    lowest.

    Returns a list of grades aligned 1:1 with the input ``z_scores``.
    Empty input returns an empty list. Ties on quantile boundaries are
    awarded the higher grade (i.e. ``>= threshold`` gets the grade above).
    """
    if not z_scores:
        return []
    n_grades = len(grades)
    if n_grades == 0:
        raise ValueError("grades must contain at least one value.")

    sorted_scores = sorted(z_scores)
    thresholds: List[float] = []
    # Compute (n_grades - 1) split points at evenly spaced quantiles.
    for k in range(1, n_grades):
        idx = int(round(k * len(sorted_scores) / n_grades))
        idx = max(0, min(len(sorted_scores) - 1, idx))
        thresholds.append(sorted_scores[idx])

    out: List[float] = []
    for value in z_scores:
        grade_idx = 0
        for threshold in thresholds:
            if value >= threshold:
                grade_idx += 1
        out.append(float(grades[grade_idx]))
    return out


def _candidate_id(row: Dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("video_id") or "")


def evaluate_trajectory_held_out(
    rows: Iterable[Dict[str, Any]],
    objectives: Sequence[str] = DEFAULT_OBJECTIVES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Dict[str, float]]:
    """
    Run the held-out evaluation across the given rows.

    For each objective:
      1. Filter to rows that carry a ``targets_z[objective]`` value.
      2. Score each row by its trajectory regime probability ×
         confidence (see ``trajectory_score_for_objective``).
      3. Convert each row's ``targets_z[objective]`` into a relevance
         grade in {0, 1, 2, 3} via quantile bucketing.
      4. Sort by predicted score (descending) and report NDCG@k and
         MRR@k for each k in ``k_values``.

    Returns a dict mapping ``objective -> {"n_rows", "ndcg@k", "mrr@k", ...}``.
    Objectives with no usable rows are omitted from the output.
    """
    rows_list = [row for row in rows if isinstance(row, dict)]
    metrics: Dict[str, Dict[str, float]] = {}

    for objective in objectives:
        usable: List[Tuple[str, float, float]] = []
        z_values: List[float] = []
        for row in rows_list:
            z = target_z_for_objective(row, objective)
            if z is None:
                continue
            cid = _candidate_id(row)
            if not cid:
                continue
            features = row.get("features")
            trajectory_features = (
                features.get("trajectory_features")
                if isinstance(features, dict)
                else None
            )
            score = trajectory_score_for_objective(
                trajectory_features if isinstance(trajectory_features, dict) else {},
                objective,
            )
            usable.append((cid, score, z))
            z_values.append(z)

        if not usable:
            continue

        grades = assign_relevance_grades(z_values)
        scored: List[Tuple[str, float, float]] = [
            (cid, score, grade) for (cid, score, _), grade in zip(usable, grades)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        recommended_ids = [item[0] for item in scored]
        relevance_dict = {item[0]: item[2] for item in scored}
        relevant_set = {item[0] for item in scored if item[2] > 0}

        objective_metrics: Dict[str, float] = {"n_rows": float(len(scored))}
        for k in k_values:
            objective_metrics[f"ndcg@{k}"] = ndcg_at_k(recommended_ids, relevance_dict, k)
            objective_metrics[f"mrr@{k}"] = mrr_at_k(recommended_ids, relevant_set, k)
        metrics[objective] = objective_metrics

    return metrics


def format_metrics_markdown(
    metrics: Dict[str, Dict[str, float]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> str:
    """
    Render a metrics dict as a markdown table suitable for a PR description.
    """
    if not metrics:
        return "_No metrics — held-out evaluation produced no usable rows._"

    headers = ["Objective", "N"]
    for k in k_values:
        headers.append(f"NDCG@{k}")
    for k in k_values:
        headers.append(f"MRR@{k}")

    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(
        ["---"] + [" ---: "] * (len(headers) - 1)
    ) + "|")

    for objective, payload in metrics.items():
        row = [objective, str(int(payload.get("n_rows", 0)))]
        for k in k_values:
            row.append(f"{payload.get(f'ndcg@{k}', 0.0):.4f}")
        for k in k_values:
            row.append(f"{payload.get(f'mrr@{k}', 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def summarize_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Macro-average NDCG@k / MRR@k across objectives. Useful for one-line
    reporting alongside the per-objective table.
    """
    if not metrics:
        return {}
    rolled: Dict[str, List[float]] = {}
    for payload in metrics.values():
        for key, value in payload.items():
            if key == "n_rows":
                continue
            rolled.setdefault(key, []).append(float(value))
    return {key: aggregate(values) for key, values in rolled.items()}


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_OBJECTIVES",
    "OBJECTIVE_REGIME",
    "RELEVANCE_GRADES",
    "assign_relevance_grades",
    "evaluate_trajectory_held_out",
    "format_metrics_markdown",
    "summarize_metrics",
    "target_z_for_objective",
    "trajectory_score_for_objective",
]
