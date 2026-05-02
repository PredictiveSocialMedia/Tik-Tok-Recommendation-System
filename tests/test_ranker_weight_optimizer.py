from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.baseline_common import (  # noqa: E402
    DEFAULT_RANKING_WEIGHTS,
    OBJECTIVE_RANKING_WEIGHTS,
)
from src.recommendation.learning.ranker_weight_optimizer import (  # noqa: E402
    COMPONENT_NAMES,
    MIN_PAIRS_PER_OBJECTIVE,
    OPTIMIZER_ID,
    OPTIMIZER_VERSION,
    RankerWeightOptimizer,
    RankingCandidate,
    RankingGroupExample,
    _lambda_loss_and_grad,
    _mean_ndcg,
    _ndcg_at_k,
    _softmax,
)


# ---------------------------------------------------------------------------
# Helpers / synthetic data builders
# ---------------------------------------------------------------------------


def _candidate(
    cid: str,
    label: float,
    *,
    semantic: float = 0.0,
    intent: float = 0.0,
    quality: float = 0.0,
    reference: float = 0.0,
    support: float = 0.0,
    trajectory: float = 0.0,
) -> RankingCandidate:
    return RankingCandidate(
        candidate_id=cid,
        components={
            "semantic_relevance": semantic,
            "intent_alignment": intent,
            "performance_quality": quality,
            "reference_usefulness": reference,
            "support_confidence": support,
            "trajectory_alignment": trajectory,
        },
        relevance_label=label,
    )


def _make_separable_groups(
    objective: str,
    n_groups: int = 12,
    candidates_per_group: int = 5,
    seed: int = 0,
) -> List[RankingGroupExample]:
    """
    Build groups where 'semantic_relevance' perfectly predicts the label and
    every other component is uncorrelated noise. A well-fit optimizer should
    therefore learn to put most of its weight on 'semantic_relevance'.
    """
    rng = np.random.default_rng(seed)
    groups: List[RankingGroupExample] = []
    for g in range(n_groups):
        cands: List[RankingCandidate] = []
        for c in range(candidates_per_group):
            label = float(c)  # 0.0, 1.0, 2.0, ... (distinct → many pairs)
            cands.append(
                _candidate(
                    f"g{g}-c{c}",
                    label=label,
                    semantic=label,  # perfectly aligned with label
                    intent=float(rng.uniform(-1.0, 1.0)),
                    quality=float(rng.uniform(-1.0, 1.0)),
                    reference=float(rng.uniform(-1.0, 1.0)),
                    support=float(rng.uniform(-1.0, 1.0)),
                    trajectory=float(rng.uniform(-1.0, 1.0)),
                )
            )
        groups.append(
            RankingGroupExample(
                query_id=f"q-{g}", objective=objective, candidates=cands
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_component_names_match_default_weight_keys() -> None:
    assert set(COMPONENT_NAMES) == set(DEFAULT_RANKING_WEIGHTS.keys())


def test_component_names_match_objective_weight_keys() -> None:
    for objective, weights in OBJECTIVE_RANKING_WEIGHTS.items():
        assert set(COMPONENT_NAMES) == set(weights.keys()), (
            f"{objective} weights missing or extra components"
        )


def test_optimizer_version_and_id_are_strings() -> None:
    assert isinstance(OPTIMIZER_VERSION, str) and OPTIMIZER_VERSION
    assert isinstance(OPTIMIZER_ID, str) and OPTIMIZER_ID


# ---------------------------------------------------------------------------
# RankingGroupExample.n_pairs
# ---------------------------------------------------------------------------


def test_n_pairs_counts_distinct_label_pairs() -> None:
    group = RankingGroupExample(
        query_id="q",
        objective="engagement",
        candidates=[_candidate("a", 3.0), _candidate("b", 2.0), _candidate("c", 1.0)],
    )
    # All 3 pairs have distinct labels: (a,b), (a,c), (b,c)
    assert group.n_pairs() == 3


def test_n_pairs_zero_when_all_labels_equal() -> None:
    group = RankingGroupExample(
        query_id="q",
        objective="engagement",
        candidates=[_candidate("a", 1.0), _candidate("b", 1.0), _candidate("c", 1.0)],
    )
    assert group.n_pairs() == 0


def test_n_pairs_with_partially_tied_labels() -> None:
    group = RankingGroupExample(
        query_id="q",
        objective="engagement",
        candidates=[_candidate("a", 3.0), _candidate("b", 3.0), _candidate("c", 1.0)],
    )
    # Distinct-label pairs: (a,c) and (b,c). (a,b) is tied so excluded.
    assert group.n_pairs() == 2


def test_n_pairs_zero_for_singleton_or_empty_group() -> None:
    empty = RankingGroupExample(query_id="e", objective="engagement", candidates=[])
    single = RankingGroupExample(
        query_id="s", objective="engagement", candidates=[_candidate("only", 1.0)]
    )
    assert empty.n_pairs() == 0
    assert single.n_pairs() == 0


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one() -> None:
    out = _softmax(np.array([1.0, 2.0, 3.0]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-9)


def test_softmax_handles_large_values_without_overflow() -> None:
    # With raw exp() this would overflow; the implementation must shift by max.
    out = _softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(out).all()
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_equals_one() -> None:
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.array([3.0, 2.0, 1.0])
    assert _ndcg_at_k(scores, labels, k=3) == pytest.approx(1.0, abs=1e-9)


def test_ndcg_zero_for_empty_input() -> None:
    assert _ndcg_at_k(np.array([]), np.array([]), k=10) == 0.0


def test_ndcg_zero_when_all_labels_zero() -> None:
    scores = np.array([0.5, 0.3, 0.1])
    labels = np.array([0.0, 0.0, 0.0])
    assert _ndcg_at_k(scores, labels, k=3) == 0.0


def test_ndcg_truncates_to_k_when_smaller_than_n() -> None:
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    labels = np.array([3.0, 2.0, 1.0, 0.0])
    # Truncating to k=2 still gives a perfect score on the head
    assert _ndcg_at_k(scores, labels, k=2) == pytest.approx(1.0, abs=1e-9)


def test_ndcg_lower_for_reversed_ranking() -> None:
    labels = np.array([3.0, 2.0, 1.0])
    perfect = _ndcg_at_k(np.array([3.0, 2.0, 1.0]), labels, k=3)
    reversed_score = _ndcg_at_k(np.array([1.0, 2.0, 3.0]), labels, k=3)
    assert reversed_score < perfect


# ---------------------------------------------------------------------------
# _mean_ndcg
# ---------------------------------------------------------------------------


def test_mean_ndcg_skips_groups_with_uniform_labels() -> None:
    groups = [
        RankingGroupExample(
            query_id="all-equal",
            objective="engagement",
            candidates=[_candidate("a", 1.0, semantic=1.0), _candidate("b", 1.0, semantic=0.0)],
        ),
        RankingGroupExample(
            query_id="varied",
            objective="engagement",
            candidates=[_candidate("a", 2.0, semantic=2.0), _candidate("b", 1.0, semantic=1.0)],
        ),
    ]
    w = np.zeros(len(COMPONENT_NAMES))
    w[0] = 1.0  # weight only semantic_relevance
    # Only the second group counts; perfect ordering → 1.0
    assert _mean_ndcg(groups, w, k=10) == pytest.approx(1.0, abs=1e-9)


def test_mean_ndcg_skips_singleton_groups() -> None:
    groups = [
        RankingGroupExample(
            query_id="solo",
            objective="engagement",
            candidates=[_candidate("a", 3.0, semantic=1.0)],
        )
    ]
    w = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)
    assert _mean_ndcg(groups, w) == 0.0


def test_mean_ndcg_zero_for_empty_input() -> None:
    assert _mean_ndcg([], np.ones(len(COMPONENT_NAMES))) == 0.0


# ---------------------------------------------------------------------------
# _lambda_loss_and_grad
# ---------------------------------------------------------------------------


def test_lambda_loss_returns_loss_and_gradient_shape() -> None:
    groups = _make_separable_groups("engagement", n_groups=3, seed=1)
    theta = np.zeros(len(COMPONENT_NAMES))
    loss, grad = _lambda_loss_and_grad(theta, groups, sigma=1.0, theta0=None, reg=0.0)
    assert isinstance(loss, float)
    assert grad.shape == (len(COMPONENT_NAMES),)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


def test_lambda_loss_lower_for_aligned_weights() -> None:
    """A theta that emphasises the predictive component should give lower loss
    than a theta that emphasises a noise component, for separable data."""
    groups = _make_separable_groups("engagement", n_groups=8, seed=2)

    aligned = np.full(len(COMPONENT_NAMES), -2.0)
    aligned[0] = 4.0  # heavy weight on semantic_relevance (the predictive one)

    misaligned = np.full(len(COMPONENT_NAMES), -2.0)
    misaligned[1] = 4.0  # heavy weight on intent_alignment (noise)

    loss_aligned, _ = _lambda_loss_and_grad(aligned, groups, reg=0.0)
    loss_misaligned, _ = _lambda_loss_and_grad(misaligned, groups, reg=0.0)
    assert loss_aligned < loss_misaligned


def test_lambda_loss_l2_term_pulls_back_to_warm_start() -> None:
    """With reg dominant and groups empty, the gradient should be reg*(theta-theta0)."""
    theta0 = np.zeros(len(COMPONENT_NAMES))
    theta = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    loss, grad = _lambda_loss_and_grad(theta, [], theta0=theta0, reg=10.0)
    # With no groups, only the L2 term contributes
    expected_loss = 0.5 * 10.0 * float(theta @ theta)
    expected_grad = 10.0 * (theta - theta0)
    assert loss == pytest.approx(expected_loss, abs=1e-9)
    assert np.allclose(grad, expected_grad, atol=1e-9)


def test_lambda_loss_no_reg_when_theta0_is_none() -> None:
    """Even with reg > 0, the L2 term should be skipped when theta0 is None."""
    theta = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    loss, grad = _lambda_loss_and_grad(theta, [], sigma=1.0, theta0=None, reg=10.0)
    assert loss == 0.0
    assert np.allclose(grad, np.zeros_like(theta), atol=1e-9)


def test_lambda_gradient_matches_finite_difference() -> None:
    """Numerical sanity check on the analytical gradient."""
    groups = _make_separable_groups("engagement", n_groups=4, seed=3)
    theta = np.array([0.2, -0.1, 0.05, 0.0, -0.05, 0.15])
    theta0 = np.zeros(len(COMPONENT_NAMES))
    _, grad = _lambda_loss_and_grad(theta, groups, sigma=1.0, theta0=theta0, reg=1.0)

    eps = 1e-5
    numeric = np.zeros_like(theta)
    for k in range(len(theta)):
        plus = theta.copy()
        plus[k] += eps
        minus = theta.copy()
        minus[k] -= eps
        loss_plus, _ = _lambda_loss_and_grad(plus, groups, sigma=1.0, theta0=theta0, reg=1.0)
        loss_minus, _ = _lambda_loss_and_grad(minus, groups, sigma=1.0, theta0=theta0, reg=1.0)
        numeric[k] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(grad, numeric, atol=1e-3)


# ---------------------------------------------------------------------------
# RankerWeightOptimizer.train — fallback paths
# ---------------------------------------------------------------------------


def test_train_falls_back_when_too_few_pairs() -> None:
    # One group with 2 candidates → only 1 pair, far below MIN_PAIRS_PER_OBJECTIVE
    groups = [
        RankingGroupExample(
            query_id="q-only",
            objective="engagement",
            candidates=[_candidate("a", 2.0), _candidate("b", 1.0)],
        )
    ]
    opt = RankerWeightOptimizer()
    results = opt.train(groups, objectives=["engagement"])
    assert results["engagement"] == OBJECTIVE_RANKING_WEIGHTS["engagement"]
    summary = opt.train_summary["engagement"]
    assert summary["status"].startswith("fallback_")
    assert summary["n_pairs"] == 1


def test_train_fallback_returns_default_weights_for_unknown_objective() -> None:
    groups = [
        RankingGroupExample(
            query_id="q",
            objective="mystery",
            candidates=[_candidate("a", 2.0), _candidate("b", 1.0)],
        )
    ]
    opt = RankerWeightOptimizer()
    results = opt.train(groups, objectives=["mystery"])
    assert results["mystery"] == DEFAULT_RANKING_WEIGHTS


def test_train_fallback_summary_records_min_pairs_required() -> None:
    groups = [
        RankingGroupExample(
            query_id="q",
            objective="engagement",
            candidates=[_candidate("a", 2.0), _candidate("b", 1.0)],
        )
    ]
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])
    summary = opt.train_summary["engagement"]
    assert summary["min_pairs_required"] == MIN_PAIRS_PER_OBJECTIVE


def test_train_processes_each_objective_independently() -> None:
    # Two unrelated objectives, both insufficient → both fall back to their own defaults
    groups = [
        RankingGroupExample(
            query_id="q-eng",
            objective="engagement",
            candidates=[_candidate("a", 2.0), _candidate("b", 1.0)],
        ),
        RankingGroupExample(
            query_id="q-reach",
            objective="reach",
            candidates=[_candidate("a", 3.0), _candidate("b", 1.0)],
        ),
    ]
    opt = RankerWeightOptimizer()
    results = opt.train(groups)
    assert results["engagement"] == OBJECTIVE_RANKING_WEIGHTS["engagement"]
    assert results["reach"] == OBJECTIVE_RANKING_WEIGHTS["reach"]


def test_train_marks_optimizer_as_trained() -> None:
    groups = _make_separable_groups("engagement", n_groups=4, seed=4)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])
    assert opt._trained is True  # noqa: SLF001 — internal flag is part of contract


# ---------------------------------------------------------------------------
# RankerWeightOptimizer.train — convergence path (requires scipy)
# ---------------------------------------------------------------------------


def test_train_learns_to_emphasize_dominant_component() -> None:
    pytest.importorskip("scipy")

    groups = _make_separable_groups("engagement", n_groups=20, seed=5)
    opt = RankerWeightOptimizer()
    # Low reg so the optimiser is free to deviate from the warm-start
    results = opt.train(groups, objectives=["engagement"], reg=0.01, max_iter=500)
    weights = results["engagement"]

    # The optimal weight vector should put the largest mass on the predictive component.
    top_component = max(weights, key=weights.get)
    assert top_component == "semantic_relevance"


def test_train_weights_sum_to_one_and_are_non_negative() -> None:
    pytest.importorskip("scipy")

    groups = _make_separable_groups("engagement", n_groups=20, seed=6)
    opt = RankerWeightOptimizer()
    results = opt.train(groups, objectives=["engagement"], reg=0.1)
    weights = results["engagement"]

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(value >= 0.0 for value in weights.values())


def test_train_summary_includes_ndcg_before_and_after() -> None:
    pytest.importorskip("scipy")

    groups = _make_separable_groups("engagement", n_groups=20, seed=7)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"], reg=0.1)
    summary = opt.train_summary["engagement"]
    assert summary["ndcg_before"] is not None
    assert summary["ndcg_after"] is not None
    assert summary["status"] in {"converged", "reverted_ndcg_regression"} or summary["status"].startswith("stopped_iter_")


def test_train_does_not_degrade_ndcg_below_warm_start() -> None:
    pytest.importorskip("scipy")

    groups = _make_separable_groups("engagement", n_groups=20, seed=8)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"], reg=0.1)
    summary = opt.train_summary["engagement"]
    # The training routine reverts whenever ndcg_after drops > 0.005 below before
    assert summary["ndcg_after"] >= summary["ndcg_before"] - 0.005


# ---------------------------------------------------------------------------
# RankerWeightOptimizer.get_weights
# ---------------------------------------------------------------------------


def test_get_weights_returns_none_before_training() -> None:
    opt = RankerWeightOptimizer()
    assert opt.get_weights("engagement") is None


def test_get_weights_returns_none_for_fallback_status() -> None:
    groups = [
        RankingGroupExample(
            query_id="q",
            objective="engagement",
            candidates=[_candidate("a", 2.0), _candidate("b", 1.0)],
        )
    ]
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])
    # Status started with "fallback_", so get_weights should hide the fallback dict
    assert opt.get_weights("engagement") is None


def test_get_weights_returns_none_for_unknown_objective() -> None:
    groups = _make_separable_groups("engagement", n_groups=4, seed=9)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])
    assert opt.get_weights("conversion") is None


def test_get_weights_returns_dict_for_converged_objective() -> None:
    pytest.importorskip("scipy")

    groups = _make_separable_groups("engagement", n_groups=20, seed=10)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"], reg=0.1)

    weights = opt.get_weights("engagement")
    assert weights is not None
    assert set(weights.keys()) == set(COMPONENT_NAMES)


def test_train_summary_property_returns_a_copy() -> None:
    groups = _make_separable_groups("engagement", n_groups=3, seed=11)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])

    snapshot = opt.train_summary
    snapshot["engagement"] = {"hacked": True}  # type: ignore[assignment]
    assert opt.train_summary != snapshot


# ---------------------------------------------------------------------------
# RankerWeightOptimizer save / load round-trip
# ---------------------------------------------------------------------------


def test_save_writes_manifest_with_expected_fields(tmp_path: Path) -> None:
    opt = RankerWeightOptimizer()
    opt.save(tmp_path / "weights")

    manifest = json.loads((tmp_path / "weights" / "manifest.json").read_text("utf-8"))
    expected = {
        "version",
        "optimizer_id",
        "component_names",
        "learned_weights",
        "train_summary",
        "trained",
    }
    assert expected.issubset(manifest.keys())
    assert manifest["version"] == OPTIMIZER_VERSION
    assert manifest["optimizer_id"] == OPTIMIZER_ID
    assert manifest["component_names"] == COMPONENT_NAMES


def test_save_then_load_round_trips_state(tmp_path: Path) -> None:
    groups = _make_separable_groups("engagement", n_groups=3, seed=12)
    opt = RankerWeightOptimizer()
    opt.train(groups, objectives=["engagement"])
    opt.save(tmp_path / "weights")

    loaded = RankerWeightOptimizer.load(tmp_path / "weights")
    assert loaded._trained is opt._trained  # noqa: SLF001
    assert loaded.train_summary == opt.train_summary
    assert loaded._learned_weights == opt._learned_weights  # noqa: SLF001


def test_load_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No lambdarank manifest"):
        RankerWeightOptimizer.load(tmp_path / "does-not-exist")


def test_load_raises_for_version_mismatch(tmp_path: Path) -> None:
    out_dir = tmp_path / "weights"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "ranker_weights.lambdarank.v999",
                "optimizer_id": OPTIMIZER_ID,
                "component_names": COMPONENT_NAMES,
                "learned_weights": {},
                "train_summary": {},
                "trained": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Version mismatch"):
        RankerWeightOptimizer.load(out_dir)


def test_load_raises_for_component_name_mismatch(tmp_path: Path) -> None:
    out_dir = tmp_path / "weights"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": OPTIMIZER_VERSION,
                "optimizer_id": OPTIMIZER_ID,
                "component_names": ["semantic_relevance", "different_component"],
                "learned_weights": {},
                "train_summary": {},
                "trained": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Component name mismatch"):
        RankerWeightOptimizer.load(out_dir)
