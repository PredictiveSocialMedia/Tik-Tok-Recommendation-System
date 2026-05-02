"""
A/B testing infrastructure for systematic ranker comparison.

Implements suggestion #8 from the prof's email:
  "Add A/B testing infrastructure to compare ranking approaches
  systematically."

Generalizes the bootstrap-CI comparison pattern (used 1-on-1 in
``ranker_comparison``) to support **N variants at once**. Caller
provides:

  - a list of named variants, each wrapping a scorer callable
  - the held-out rows
  - parallel candidate IDs
  - parallel relevance grades (e.g. quantile buckets of ``targets_z``)
  - k-values and bootstrap-resample count

Returns:

  - per-variant absolute NDCG@k / MRR@k metrics
  - all-pairs paired-bootstrap CIs on the lift between every variant
  - a markdown-renderable comparison report

The framework is **scorer-agnostic** — variants can be heuristics,
trained models (sklearn / lightgbm / torch / anything pickleable), or
literal random baselines. As long as ``variant.score(rows)`` returns
one float per row, the comparison harness works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .evaluator import mrr_at_k, ndcg_at_k

AB_TESTING_VERSION = "ab_testing.v1"

DEFAULT_K_VALUES: Tuple[int, ...] = (10, 20)
DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000

# Relevance grades used by ``assign_relevance_grades`` (matches the scale
# the rest of the recommender uses).
RELEVANCE_GRADES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Variant container
# ---------------------------------------------------------------------------


# A scorer takes a sequence of row dicts and returns one float per row.
# Higher = better. NaN/inf returned values are coerced to 0.0 internally.
ScorerFn = Callable[[Sequence[Dict[str, Any]]], Sequence[float]]


@dataclass
class RankerVariant:
    """A named ranking strategy plus its scorer callback."""

    name: str
    score: ScorerFn
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("RankerVariant.name must be a non-empty string.")
        if not callable(self.score):
            raise TypeError("RankerVariant.score must be callable.")


# ---------------------------------------------------------------------------
# Relevance grading helper (kept self-contained so this module has no
# cross-dependency on suggestion-#7 / suggestion-#1 PRs).
# ---------------------------------------------------------------------------


def assign_relevance_grades(
    z_scores: Sequence[float],
    grades: Sequence[float] = RELEVANCE_GRADES,
) -> List[float]:
    """Quantile-bucket continuous z-scores into discrete relevance grades."""
    if not z_scores:
        return []
    n_grades = len(grades)
    if n_grades == 0:
        raise ValueError("grades must contain at least one value.")
    sorted_scores = sorted(z_scores)
    thresholds: List[float] = []
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


# ---------------------------------------------------------------------------
# Internal: ranking metrics from (id, score, grade) triples
# ---------------------------------------------------------------------------


def _coerce_finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _ranking_metrics_for_scored(
    scored: Sequence[Tuple[str, float, float]],
    k_values: Sequence[int],
) -> Dict[str, float]:
    if not scored:
        return {}
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    recommended_ids = [item[0] for item in ordered]
    relevance_dict = {item[0]: item[2] for item in ordered}
    relevant_set = {item[0] for item in ordered if item[2] > 0}
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg_at_k(recommended_ids, relevance_dict, k)
        out[f"mrr@{k}"] = mrr_at_k(recommended_ids, relevant_set, k)
    return out


# ---------------------------------------------------------------------------
# Internal: paired bootstrap on the lift between two scored sequences
# ---------------------------------------------------------------------------


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _paired_bootstrap_lift(
    scored_a: Sequence[Tuple[str, float, float]],
    scored_b: Sequence[Tuple[str, float, float]],
    k_values: Sequence[int],
    n_resamples: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    n = len(scored_a)
    if n == 0 or n != len(scored_b):
        return {}
    metric_keys: List[str] = []
    for k in k_values:
        metric_keys.append(f"ndcg@{k}")
        metric_keys.append(f"mrr@{k}")

    diffs: Dict[str, List[float]] = {key: [] for key in metric_keys}
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        sample_a = [scored_a[i] for i in idx]
        sample_b = [scored_b[i] for i in idx]
        metrics_a = _ranking_metrics_for_scored(sample_a, k_values)
        metrics_b = _ranking_metrics_for_scored(sample_b, k_values)
        for key in metric_keys:
            diffs[key].append(
                float(metrics_a.get(key, 0.0) - metrics_b.get(key, 0.0))
            )

    out: Dict[str, Dict[str, float]] = {}
    for key, values in diffs.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "lift_mean": float(arr.mean()) if arr.size else 0.0,
            "lift_ci_low": _percentile(arr, 2.5),
            "lift_ci_high": _percentile(arr, 97.5),
            "lift_positive_share": float(np.mean(arr > 0.0)) if arr.size else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ab_test(
    variants: Sequence[RankerVariant],
    rows: Sequence[Dict[str, Any]],
    candidate_ids: Sequence[str],
    relevance_grades: Sequence[float],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run an N-way ranker comparison on ``rows`` (with parallel
    ``candidate_ids`` and ``relevance_grades``) using each variant's
    scorer.

    Each variant's scorer is called once with the full row list. Any
    NaN/inf scores are coerced to 0.0. The return value is a structured
    dict suitable for JSON serialization:

        {
          "version": "ab_testing.v1",
          "n_rows": int,
          "n_variants": int,
          "k_values": [...],
          "n_resamples": int,
          "variants": [
            {"name": ..., "description": ..., "metrics": {...}},
            ...
          ],
          "all_pairs_lift": [
            {
              "variant_a": ..., "variant_b": ...,
              "lifts": {
                "ndcg@10": {"lift_mean": ..., "lift_ci_low": ..., ...},
                ...
              }
            },
            ...
          ],
        }

    All variant scores are computed against the same row order, and
    all-pairs bootstrap CIs use the same paired indices per resample —
    so the comparison is fully apples-to-apples.
    """
    if not variants:
        raise ValueError("At least one variant is required.")
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("Variant names must be unique.")
    if len(rows) != len(candidate_ids):
        raise ValueError("rows and candidate_ids must have the same length.")
    if len(rows) != len(relevance_grades):
        raise ValueError("rows and relevance_grades must have the same length.")

    n = len(rows)
    rng = np.random.default_rng(random_state)

    # Score each variant once over the full row list.
    variant_scored: Dict[str, List[Tuple[str, float, float]]] = {}
    variant_metrics: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        raw_scores = list(variant.score(rows))
        if len(raw_scores) != n:
            raise ValueError(
                f"Variant {variant.name!r} returned {len(raw_scores)} scores "
                f"but {n} rows were given."
            )
        scores = _coerce_finite(raw_scores)
        scored = [
            (str(candidate_ids[i]), float(scores[i]), float(relevance_grades[i]))
            for i in range(n)
        ]
        variant_scored[variant.name] = scored
        variant_metrics[variant.name] = _ranking_metrics_for_scored(scored, k_values)

    # All-pairs paired bootstrap on the lift.
    all_pairs: List[Dict[str, Any]] = []
    names = [v.name for v in variants]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            lifts = _paired_bootstrap_lift(
                variant_scored[name_a],
                variant_scored[name_b],
                k_values=k_values,
                n_resamples=n_resamples,
                rng=rng,
            )
            all_pairs.append(
                {
                    "variant_a": name_a,
                    "variant_b": name_b,
                    "lifts": lifts,
                }
            )

    return {
        "version": AB_TESTING_VERSION,
        "n_rows": n,
        "n_variants": len(variants),
        "k_values": list(k_values),
        "n_resamples": int(n_resamples),
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "metrics": variant_metrics[variant.name],
            }
            for variant in variants
        ],
        "all_pairs_lift": all_pairs,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report_markdown(report: Dict[str, Any]) -> str:
    """
    Render the dict from ``run_ab_test`` as a markdown report:

      - per-variant metrics table
      - all-pairs lift table with 95% CI and P(lift > 0)
    """
    if not report or not report.get("variants"):
        return "_No A/B-test report — variants list was empty._"
    k_values = report.get("k_values") or list(DEFAULT_K_VALUES)
    lines: List[str] = []

    # Per-variant absolute metrics
    headers = ["Variant", "N"]
    for k in k_values:
        headers.append(f"NDCG@{k}")
    for k in k_values:
        headers.append(f"MRR@{k}")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + [" ---: "] * (len(headers) - 1)) + "|")
    n_rows = int(report.get("n_rows", 0))
    for v in report["variants"]:
        cells = [v.get("name", ""), str(n_rows)]
        m = v.get("metrics", {})
        for k in k_values:
            cells.append(f"{m.get(f'ndcg@{k}', 0.0):.4f}")
        for k in k_values:
            cells.append(f"{m.get(f'mrr@{k}', 0.0):.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Pair lift table
    lines.append("")
    lines.append(
        "**All-pairs lift (variant_a minus variant_b) with 95% paired bootstrap CI:**"
    )
    lines.append("")
    lines.append(
        "| Variant A | Variant B | Metric | Lift | 95% CI | P(lift > 0) |"
    )
    lines.append("|---|---|---|---:|:---:|---:|")
    for pair in report.get("all_pairs_lift", []):
        a = pair.get("variant_a", "")
        b = pair.get("variant_b", "")
        for metric_name, payload in pair.get("lifts", {}).items():
            lines.append(
                "| {a} | {b} | {metric} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.2f} |".format(
                    a=a,
                    b=b,
                    metric=metric_name,
                    mean=payload.get("lift_mean", 0.0),
                    lo=payload.get("lift_ci_low", 0.0),
                    hi=payload.get("lift_ci_high", 0.0),
                    p=payload.get("lift_positive_share", 0.0),
                )
            )
    return "\n".join(lines)


__all__ = [
    "AB_TESTING_VERSION",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_K_VALUES",
    "RELEVANCE_GRADES",
    "RankerVariant",
    "ScorerFn",
    "assign_relevance_grades",
    "format_report_markdown",
    "run_ab_test",
]
