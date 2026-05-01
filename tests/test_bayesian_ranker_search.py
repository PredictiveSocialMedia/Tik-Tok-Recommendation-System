from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_bayesian_ranker_search import _split_groups
from src.recommendation.learning.bayesian_ranker_search import (
    BayesianRankerSearch,
    BayesianRankerSearchConfig,
    SearchResult,
)


def test_config_rejects_more_initial_points_than_iterations() -> None:
    with pytest.raises(ValueError, match="n_initial_random"):
        BayesianRankerSearchConfig(n_iterations=2, n_initial_random=3)


def test_encode_decode_round_trips_search_space() -> None:
    cfg = BayesianRankerSearchConfig(
        sigma_min=0.1,
        sigma_max=5.0,
        log_reg_min=-2.0,
        log_reg_max=2.0,
        n_iterations=1,
        n_initial_random=1,
    )
    search = BayesianRankerSearch(cfg)

    encoded = search._encode(2.55, 0.0)
    sigma, reg = search._decode(encoded)

    assert sigma == pytest.approx(2.55, abs=1e-9)
    assert reg == pytest.approx(1.0, abs=1e-9)


def test_search_result_save_and_load(tmp_path: Path) -> None:
    result = SearchResult(
        best_sigma=1.2,
        best_reg=0.5,
        best_ndcg=0.9,
        baseline_ndcg=0.7,
        improvement=0.2,
        n_iterations=2,
        history=[{"iteration": 0, "type": "random", "ndcg": 0.9}],
    )
    path = tmp_path / "nested" / "result.json"

    result.save(str(path))
    loaded = SearchResult.load(str(path))

    assert loaded == result


def test_split_groups_is_deterministic_and_non_empty_on_both_sides() -> None:
    groups = list(range(10))

    train_a, val_a = _split_groups(groups, val_fraction=0.3, seed=13)
    train_b, val_b = _split_groups(groups, val_fraction=0.3, seed=13)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 7
    assert len(val_a) == 3
    assert sorted(train_a + val_a) == groups


def test_run_with_objective_fn_records_baseline_and_best() -> None:
    pytest.importorskip("sklearn")

    cfg = BayesianRankerSearchConfig(
        sigma_min=0.1,
        sigma_max=2.0,
        log_reg_min=-2.0,
        log_reg_max=1.0,
        n_iterations=6,
        n_initial_random=3,
        n_candidates=128,
        seed=7,
    )

    def objective(sigma: float, reg: float) -> float:
        log_reg = __import__("math").log10(reg)
        return 1.0 - ((sigma - 1.7) ** 2) - 0.2 * ((log_reg + 1.0) ** 2)

    result = BayesianRankerSearch(cfg).run([], [], objective_fn=objective)

    assert result.n_iterations == 6
    assert len(result.history) == 6
    assert result.baseline_ndcg == pytest.approx(round(objective(1.0, 5.0), 4), abs=1e-9)
    assert result.best_ndcg >= max(entry["ndcg"] for entry in result.history)
    assert result.improvement == pytest.approx(
        round(result.best_ndcg - result.baseline_ndcg, 4),
        abs=1e-9,
    )
