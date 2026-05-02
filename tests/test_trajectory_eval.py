from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.trajectory_eval import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_OBJECTIVES,
    OBJECTIVE_REGIME,
    assign_relevance_grades,
    evaluate_trajectory_held_out,
    format_metrics_markdown,
    summarize_metrics,
    target_z_for_objective,
    trajectory_score_for_objective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    row_id: str,
    spike: float = 0.0,
    balanced: float = 0.0,
    durable: float = 0.0,
    confidence: float = 0.5,
    z_reach: Any = 0.0,
    z_engagement: Any = 0.0,
    z_conversion: Any = 0.0,
) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "video_id": row_id.split("::", 1)[0],
        "features": {
            "trajectory_features": {
                "regime_probabilities": {
                    "spike": spike,
                    "balanced": balanced,
                    "durable": durable,
                },
                "regime_confidence": confidence,
            }
        },
        "targets_z": {
            "reach": z_reach,
            "engagement": z_engagement,
            "conversion": z_conversion,
        },
    }


# ---------------------------------------------------------------------------
# trajectory_score_for_objective
# ---------------------------------------------------------------------------


def test_score_uses_objective_specific_regime_probability() -> None:
    features = {
        "regime_probabilities": {"spike": 0.8, "balanced": 0.1, "durable": 0.1},
        "regime_confidence": 1.0,
    }
    assert trajectory_score_for_objective(features, "reach") == pytest.approx(0.8, abs=1e-9)
    assert trajectory_score_for_objective(features, "engagement") == pytest.approx(0.1, abs=1e-9)
    assert trajectory_score_for_objective(features, "conversion") == pytest.approx(0.1, abs=1e-9)


def test_score_scales_with_confidence() -> None:
    features = {
        "regime_probabilities": {"spike": 0.5, "balanced": 0.5, "durable": 0.0},
        "regime_confidence": 0.4,
    }
    assert trajectory_score_for_objective(features, "reach") == pytest.approx(0.2, abs=1e-9)


def test_score_returns_zero_for_unknown_objective() -> None:
    features = {
        "regime_probabilities": {"spike": 0.9, "balanced": 0.05, "durable": 0.05},
        "regime_confidence": 1.0,
    }
    assert trajectory_score_for_objective(features, "mystery") == 0.0


def test_score_returns_zero_when_features_missing_or_malformed() -> None:
    assert trajectory_score_for_objective({}, "reach") == 0.0
    assert trajectory_score_for_objective({"regime_probabilities": "not-a-dict"}, "reach") == 0.0
    assert trajectory_score_for_objective(None, "reach") == 0.0  # type: ignore[arg-type]


def test_score_clamps_out_of_range_inputs() -> None:
    features = {
        "regime_probabilities": {"spike": 1.5, "balanced": 0.0, "durable": 0.0},
        "regime_confidence": 2.0,
    }
    # Both factors should clamp to 1.0
    assert trajectory_score_for_objective(features, "reach") == pytest.approx(1.0, abs=1e-9)


def test_score_treats_negative_inputs_as_zero() -> None:
    features = {
        "regime_probabilities": {"spike": -0.3, "balanced": 0.0, "durable": 0.0},
        "regime_confidence": 1.0,
    }
    assert trajectory_score_for_objective(features, "reach") == 0.0


def test_objective_regime_map_covers_default_objectives() -> None:
    assert set(OBJECTIVE_REGIME.keys()) == set(DEFAULT_OBJECTIVES)
    assert set(OBJECTIVE_REGIME.values()) == {"spike", "balanced", "durable"}


# ---------------------------------------------------------------------------
# target_z_for_objective
# ---------------------------------------------------------------------------


def test_target_z_returns_value_when_present() -> None:
    row = {"targets_z": {"reach": -0.42, "engagement": 1.1, "conversion": 0.0}}
    assert target_z_for_objective(row, "reach") == pytest.approx(-0.42, abs=1e-9)
    assert target_z_for_objective(row, "engagement") == pytest.approx(1.1, abs=1e-9)


def test_target_z_returns_none_for_missing_objective() -> None:
    row = {"targets_z": {"reach": 0.5}}
    assert target_z_for_objective(row, "engagement") is None


def test_target_z_returns_none_when_targets_z_absent() -> None:
    assert target_z_for_objective({}, "reach") is None
    assert target_z_for_objective({"targets_z": "wrong-type"}, "reach") is None


def test_target_z_returns_none_for_null_or_non_finite_values() -> None:
    row = {"targets_z": {"reach": None, "engagement": float("nan"), "conversion": float("inf")}}
    assert target_z_for_objective(row, "reach") is None
    assert target_z_for_objective(row, "engagement") is None
    assert target_z_for_objective(row, "conversion") is None


# ---------------------------------------------------------------------------
# assign_relevance_grades
# ---------------------------------------------------------------------------


def test_assign_grades_buckets_by_quantile() -> None:
    # 8 evenly spaced z-scores → 2 per quartile bucket
    z = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    grades = assign_relevance_grades(z)
    # Lowest quartile → 0, then 1, 2, top quartile → 3
    assert grades == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_assign_grades_returns_empty_for_empty_input() -> None:
    assert assign_relevance_grades([]) == []


def test_assign_grades_preserves_input_order() -> None:
    z = [2.0, -1.0, 0.5]  # Highest first in input
    grades = assign_relevance_grades(z)
    assert grades[0] >= grades[1]  # input[0]=2.0 should outrank input[1]=-1.0


def test_assign_grades_handles_ties() -> None:
    z = [0.0, 0.0, 0.0, 0.0]  # All identical
    grades = assign_relevance_grades(z)
    # All four get the same grade — no information to distinguish them
    assert len(set(grades)) == 1


def test_assign_grades_rejects_empty_grade_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assign_relevance_grades([1.0, 2.0], grades=[])


# ---------------------------------------------------------------------------
# evaluate_trajectory_held_out
# ---------------------------------------------------------------------------


def _separable_rows(objective: str, n: int = 12) -> List[Dict[str, Any]]:
    """
    Build n rows where the predicted regime probability for ``objective``
    is perfectly correlated with the ground-truth z-score, so a correct
    ranker should achieve NDCG = 1.
    """
    rows: List[Dict[str, Any]] = []
    primary_regime = OBJECTIVE_REGIME[objective]
    for i in range(n):
        prob = (i + 1) / (n + 1)  # ascending in (0, 1)
        z_value = float(i)        # ascending
        regime_probs = {"spike": 0.0, "balanced": 0.0, "durable": 0.0}
        regime_probs[primary_regime] = prob
        targets_z = {"reach": 0.0, "engagement": 0.0, "conversion": 0.0}
        targets_z[objective] = z_value
        rows.append(
            {
                "row_id": f"row-{i}",
                "video_id": f"row-{i}",
                "features": {
                    "trajectory_features": {
                        "regime_probabilities": regime_probs,
                        "regime_confidence": 1.0,
                    }
                },
                "targets_z": targets_z,
            }
        )
    return rows


def test_evaluate_returns_perfect_score_on_aligned_data() -> None:
    rows = _separable_rows("engagement", n=12)
    metrics = evaluate_trajectory_held_out(rows, objectives=["engagement"], k_values=(10,))
    assert metrics["engagement"]["ndcg@10"] == pytest.approx(1.0, abs=1e-9)
    # MRR@10: top-1 by predicted score has grade 3 (top quartile), so MRR = 1.0
    assert metrics["engagement"]["mrr@10"] == pytest.approx(1.0, abs=1e-9)


def test_evaluate_drops_below_perfect_when_predictions_misaligned() -> None:
    rows = _separable_rows("engagement", n=12)
    # Reverse the regime_probabilities ordering so high-z rows get low scores
    flipped: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        rev = dict(row)
        rev["features"] = {
            "trajectory_features": {
                "regime_probabilities": {
                    "spike": 0.0,
                    "balanced": (len(rows) - i) / (len(rows) + 1),
                    "durable": 0.0,
                },
                "regime_confidence": 1.0,
            }
        }
        flipped.append(rev)
    metrics = evaluate_trajectory_held_out(flipped, objectives=["engagement"], k_values=(10,))
    assert metrics["engagement"]["ndcg@10"] < 0.9


def test_evaluate_skips_objectives_with_no_usable_rows() -> None:
    # Every row has targets_z["engagement"] = None → engagement gets dropped
    rows = [
        _make_row(row_id="a", balanced=0.5, z_engagement=None),
        _make_row(row_id="b", balanced=0.3, z_engagement=None),
    ]
    metrics = evaluate_trajectory_held_out(rows, objectives=["engagement"])
    assert "engagement" not in metrics


def test_evaluate_skips_rows_with_blank_ids() -> None:
    rows = _separable_rows("reach", n=4)
    rows.append(
        {
            "row_id": "",
            "video_id": "",
            "features": {
                "trajectory_features": {
                    "regime_probabilities": {"spike": 0.99, "balanced": 0.0, "durable": 0.0},
                    "regime_confidence": 1.0,
                }
            },
            "targets_z": {"reach": 99.0},
        }
    )
    metrics = evaluate_trajectory_held_out(rows, objectives=["reach"], k_values=(10,))
    # The blank-id row should be ignored, so n_rows stays 4
    assert metrics["reach"]["n_rows"] == 4


def test_evaluate_handles_empty_input() -> None:
    assert evaluate_trajectory_held_out([], objectives=DEFAULT_OBJECTIVES) == {}


def test_evaluate_records_n_rows_and_all_k_metrics() -> None:
    rows = _separable_rows("conversion", n=20)
    metrics = evaluate_trajectory_held_out(
        rows, objectives=["conversion"], k_values=(5, 10, 20)
    )
    payload = metrics["conversion"]
    assert payload["n_rows"] == 20
    for k in (5, 10, 20):
        assert f"ndcg@{k}" in payload
        assert f"mrr@{k}" in payload


def test_evaluate_processes_each_objective_independently() -> None:
    # One row per objective with non-null target so every objective survives
    rows = [
        _make_row(row_id="a", spike=0.9, balanced=0.05, durable=0.05,
                  z_reach=2.0, z_engagement=0.0, z_conversion=0.0),
        _make_row(row_id="b", spike=0.05, balanced=0.9, durable=0.05,
                  z_reach=0.0, z_engagement=2.0, z_conversion=0.0),
        _make_row(row_id="c", spike=0.05, balanced=0.05, durable=0.9,
                  z_reach=0.0, z_engagement=0.0, z_conversion=2.0),
    ]
    metrics = evaluate_trajectory_held_out(rows, objectives=DEFAULT_OBJECTIVES)
    assert set(metrics.keys()) == set(DEFAULT_OBJECTIVES)


# ---------------------------------------------------------------------------
# format_metrics_markdown
# ---------------------------------------------------------------------------


def test_format_markdown_includes_header_and_rows() -> None:
    metrics = {
        "reach": {"n_rows": 100.0, "ndcg@10": 0.5, "ndcg@20": 0.6, "mrr@10": 0.4, "mrr@20": 0.45},
    }
    text = format_metrics_markdown(metrics, k_values=(10, 20))
    assert "Objective" in text
    assert "NDCG@10" in text
    assert "NDCG@20" in text
    assert "MRR@10" in text
    assert "reach" in text
    assert "100" in text
    assert "0.5000" in text  # rounded to 4 decimals


def test_format_markdown_empty_metrics_returns_placeholder() -> None:
    text = format_metrics_markdown({}, k_values=DEFAULT_K_VALUES)
    assert "No metrics" in text


# ---------------------------------------------------------------------------
# summarize_metrics
# ---------------------------------------------------------------------------


def test_summarize_macro_averages_across_objectives() -> None:
    metrics = {
        "reach": {"n_rows": 10.0, "ndcg@10": 0.4, "mrr@10": 0.3},
        "engagement": {"n_rows": 10.0, "ndcg@10": 0.6, "mrr@10": 0.5},
    }
    summary = summarize_metrics(metrics)
    assert summary["ndcg@10"] == pytest.approx(0.5, abs=1e-9)
    assert summary["mrr@10"] == pytest.approx(0.4, abs=1e-9)
    assert "n_rows" not in summary  # n_rows is not a metric to average


def test_summarize_returns_empty_for_empty_input() -> None:
    assert summarize_metrics({}) == {}
