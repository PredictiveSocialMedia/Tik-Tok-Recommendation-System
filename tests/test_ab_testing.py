from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.ab_testing import (  # noqa: E402
    AB_TESTING_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_K_VALUES,
    RELEVANCE_GRADES,
    RankerVariant,
    assign_relevance_grades,
    format_report_markdown,
    run_ab_test,
)


# ---------------------------------------------------------------------------
# RankerVariant validation
# ---------------------------------------------------------------------------


def test_variant_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RankerVariant(name="", score=lambda rows: [0.0 for _ in rows])


def test_variant_rejects_whitespace_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RankerVariant(name="   ", score=lambda rows: [0.0 for _ in rows])


def test_variant_rejects_non_callable_scorer() -> None:
    with pytest.raises(TypeError, match="callable"):
        RankerVariant(name="bad", score="not-a-function")  # type: ignore[arg-type]


def test_variant_accepts_valid_inputs() -> None:
    v = RankerVariant(
        name="my-ranker",
        score=lambda rows: [0.0] * len(rows),
        description="anything goes",
    )
    assert v.name == "my-ranker"
    assert v.description == "anything goes"


# ---------------------------------------------------------------------------
# assign_relevance_grades
# ---------------------------------------------------------------------------


def test_assign_grades_quantile_bucketing() -> None:
    grades = assign_relevance_grades([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert grades == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_assign_grades_empty_returns_empty() -> None:
    assert assign_relevance_grades([]) == []


def test_assign_grades_rejects_empty_grade_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assign_relevance_grades([1.0], grades=[])


def test_assign_grades_handles_uniform_values() -> None:
    # All identical → all rows get the same grade (no information to split on)
    grades = assign_relevance_grades([0.5, 0.5, 0.5, 0.5])
    assert len(set(grades)) == 1


# ---------------------------------------------------------------------------
# run_ab_test — basic structure
# ---------------------------------------------------------------------------


def _identity_scorer(field: str) -> "callable":
    """Build a scorer that just reads ``row[field]`` for each row."""

    def _scorer(rows: Sequence[Dict[str, Any]]) -> List[float]:
        return [float(row.get(field, 0.0)) for row in rows]

    return _scorer


def _make_rows_with_scores(n: int = 12) -> List[Dict[str, Any]]:
    """Each row carries (good, bad, random) score signals + a true grade."""
    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        # `good` perfectly predicts grade; `bad` is anti-correlated; `noise` is random.
        rows.append(
            {
                "id": f"r{i}",
                "good": float(i),
                "bad": float(-i),
                "noise": float(rng.uniform(-1.0, 1.0)),
                "grade": float(i % 4),
            }
        )
    return rows


def test_run_ab_test_returns_documented_top_level_keys() -> None:
    rows = _make_rows_with_scores(n=12)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="good", score=_identity_scorer("good")),
        RankerVariant(name="bad", score=_identity_scorer("bad")),
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=20
    )
    expected_keys = {
        "version",
        "n_rows",
        "n_variants",
        "k_values",
        "n_resamples",
        "variants",
        "all_pairs_lift",
    }
    assert expected_keys.issubset(report.keys())
    assert report["version"] == AB_TESTING_VERSION
    assert report["n_rows"] == 12
    assert report["n_variants"] == 2


def test_run_ab_test_per_variant_metrics_have_all_k_values() -> None:
    rows = _make_rows_with_scores(n=20)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [RankerVariant(name="good", score=_identity_scorer("good"))]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(5, 10, 20), n_resamples=10
    )
    metrics = report["variants"][0]["metrics"]
    for k in (5, 10, 20):
        assert f"ndcg@{k}" in metrics
        assert f"mrr@{k}" in metrics


def test_run_ab_test_includes_all_pairs() -> None:
    rows = _make_rows_with_scores(n=10)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="a", score=_identity_scorer("good")),
        RankerVariant(name="b", score=_identity_scorer("bad")),
        RankerVariant(name="c", score=_identity_scorer("noise")),
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=10
    )
    # 3 variants → 3 unique pairs
    assert len(report["all_pairs_lift"]) == 3
    pair_keys = {(p["variant_a"], p["variant_b"]) for p in report["all_pairs_lift"]}
    assert pair_keys == {("a", "b"), ("a", "c"), ("b", "c")}


def test_run_ab_test_winning_variant_has_higher_ndcg() -> None:
    rows = _make_rows_with_scores(n=20)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="good", score=_identity_scorer("good")),
        RankerVariant(name="bad", score=_identity_scorer("bad")),
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=50
    )
    metric_by_variant = {v["name"]: v["metrics"] for v in report["variants"]}
    assert metric_by_variant["good"]["ndcg@10"] > metric_by_variant["bad"]["ndcg@10"]


def test_run_ab_test_lift_positive_when_first_variant_better() -> None:
    rows = _make_rows_with_scores(n=20)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="good", score=_identity_scorer("good")),
        RankerVariant(name="bad", score=_identity_scorer("bad")),
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=200, random_state=7
    )
    pair = report["all_pairs_lift"][0]
    assert pair["variant_a"] == "good"
    assert pair["variant_b"] == "bad"
    lift = pair["lifts"]["ndcg@10"]
    assert lift["lift_mean"] > 0.0
    assert lift["lift_positive_share"] > 0.5


# ---------------------------------------------------------------------------
# run_ab_test — error handling
# ---------------------------------------------------------------------------


def test_run_ab_test_rejects_empty_variants() -> None:
    with pytest.raises(ValueError, match="(?i)at least one"):
        run_ab_test([], rows=[], candidate_ids=[], relevance_grades=[])


def test_run_ab_test_rejects_duplicate_variant_names() -> None:
    rows = _make_rows_with_scores(n=4)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="dupe", score=_identity_scorer("good")),
        RankerVariant(name="dupe", score=_identity_scorer("bad")),
    ]
    with pytest.raises(ValueError, match="unique"):
        run_ab_test(variants, rows, ids, grades)


def test_run_ab_test_rejects_id_length_mismatch() -> None:
    rows = _make_rows_with_scores(n=4)
    grades = [0.0] * 4
    ids = ["a", "b"]  # too few
    variants = [RankerVariant(name="x", score=_identity_scorer("good"))]
    with pytest.raises(ValueError, match="same length"):
        run_ab_test(variants, rows, ids, grades)


def test_run_ab_test_rejects_grade_length_mismatch() -> None:
    rows = _make_rows_with_scores(n=4)
    ids = [r["id"] for r in rows]
    grades = [0.0, 1.0]  # too few
    variants = [RankerVariant(name="x", score=_identity_scorer("good"))]
    with pytest.raises(ValueError, match="same length"):
        run_ab_test(variants, rows, ids, grades)


def test_run_ab_test_rejects_scorer_returning_wrong_count() -> None:
    rows = _make_rows_with_scores(n=4)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]

    def _bad_scorer(rows: Sequence[Dict[str, Any]]) -> List[float]:
        # Returns one too few scores
        return [0.0] * (len(rows) - 1)

    variants = [RankerVariant(name="bad", score=_bad_scorer)]
    with pytest.raises(ValueError, match="returned"):
        run_ab_test(variants, rows, ids, grades)


# ---------------------------------------------------------------------------
# run_ab_test — robustness
# ---------------------------------------------------------------------------


def test_run_ab_test_coerces_non_finite_scores_to_zero() -> None:
    rows = _make_rows_with_scores(n=8)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]

    def _nan_scorer(rows: Sequence[Dict[str, Any]]) -> List[float]:
        return [float("nan")] * len(rows)

    variants = [RankerVariant(name="nans", score=_nan_scorer)]
    report = run_ab_test(variants, rows, ids, grades, k_values=(10,), n_resamples=5)
    # No exception, metrics structure is intact
    assert "ndcg@10" in report["variants"][0]["metrics"]


def test_run_ab_test_deterministic_with_fixed_seed() -> None:
    rows = _make_rows_with_scores(n=15)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="a", score=_identity_scorer("good")),
        RankerVariant(name="b", score=_identity_scorer("noise")),
    ]
    report_1 = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=50, random_state=7
    )
    report_2 = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=50, random_state=7
    )
    lift_1 = report_1["all_pairs_lift"][0]["lifts"]["ndcg@10"]["lift_mean"]
    lift_2 = report_2["all_pairs_lift"][0]["lifts"]["ndcg@10"]["lift_mean"]
    assert lift_1 == pytest.approx(lift_2, abs=1e-12)


def test_run_ab_test_lift_zero_when_variants_identical() -> None:
    rows = _make_rows_with_scores(n=12)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="a", score=_identity_scorer("good")),
        RankerVariant(name="b", score=_identity_scorer("good")),  # same scorer
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=100, random_state=11
    )
    lift = report["all_pairs_lift"][0]["lifts"]["ndcg@10"]
    assert abs(lift["lift_mean"]) < 1e-9
    assert abs(lift["lift_ci_low"]) < 1e-9
    assert abs(lift["lift_ci_high"]) < 1e-9


# ---------------------------------------------------------------------------
# format_report_markdown
# ---------------------------------------------------------------------------


def test_format_markdown_includes_metrics_and_lift_sections() -> None:
    rows = _make_rows_with_scores(n=10)
    ids = [r["id"] for r in rows]
    grades = [r["grade"] for r in rows]
    variants = [
        RankerVariant(name="good", score=_identity_scorer("good"), description="optimal"),
        RankerVariant(name="bad", score=_identity_scorer("bad")),
    ]
    report = run_ab_test(
        variants, rows, ids, grades, k_values=(10,), n_resamples=10
    )
    text = format_report_markdown(report)
    assert "Variant" in text
    assert "NDCG@10" in text
    assert "good" in text
    assert "bad" in text
    assert "All-pairs lift" in text
    assert "95% CI" in text


def test_format_markdown_returns_placeholder_for_empty_report() -> None:
    assert "No A/B-test report" in format_report_markdown({})
    assert "No A/B-test report" in format_report_markdown({"variants": []})


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_module_version_and_constants() -> None:
    assert isinstance(AB_TESTING_VERSION, str) and AB_TESTING_VERSION
    assert DEFAULT_BOOTSTRAP_RESAMPLES > 0
    assert len(DEFAULT_K_VALUES) > 0
    assert RELEVANCE_GRADES[0] == 0.0
