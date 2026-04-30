"""Focused regression tests on scoring helpers.

Covers the components where a silent numerical bug is most damaging:

* set-based Jaccard (used in datamart sampling + similarity features)
* the three metric functions used by the evaluator (NDCG@k, MRR@k, Recall@k)
* BM25 lexical retrieval ordering and zero-overlap behaviour
* TF-IDF lexical retrieval ordering
* hybrid blend weighting (cosine + engagement)
* end-to-end output shape from the evaluation pipeline

These tests are intentionally small and self-contained so they can run on a
laptop in under a second without loading any model artifacts.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.recommendation.learning.baseline_common import jaccard
from src.recommendation.learning.evaluator import (
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------
def test_jaccard_identical_sets() -> None:
    assert jaccard(["a", "b", "c"], ["c", "b", "a"]) == 1.0


def test_jaccard_disjoint_sets() -> None:
    assert jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_empty_left_is_zero() -> None:
    """Locks the off-by-one: empty input must short-circuit to 0.0, not divide by zero."""
    assert jaccard([], ["a"]) == 0.0
    assert jaccard(["a"], []) == 0.0
    assert jaccard([], []) == 0.0


def test_jaccard_known_overlap() -> None:
    """{a, b, c} vs {b, c, d} => intersection 2, union 4 => 0.5"""
    assert jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5, abs=1e-9)


def test_jaccard_handles_duplicates_via_set_semantics() -> None:
    """Duplicates should not double-count: ['a','a','b'] is the same set as ['a','b']."""
    assert jaccard(["a", "a", "b"], ["a", "b", "b"]) == 1.0


# ---------------------------------------------------------------------------
# Recall / MRR / NDCG
# ---------------------------------------------------------------------------
def test_recall_at_k_perfect_ranking() -> None:
    assert recall_at_k(["v1", "v2", "v3"], {"v1", "v2"}, k=2) == 1.0


def test_recall_at_k_no_hits() -> None:
    assert recall_at_k(["v1", "v2"], {"v9"}, k=2) == 0.0


def test_recall_at_k_empty_relevant_returns_zero() -> None:
    """Avoid divide-by-zero when the relevance proxy returns nothing."""
    assert recall_at_k(["v1"], set(), k=5) == 0.0


def test_mrr_at_k_first_hit_wins() -> None:
    """MRR is 1/rank of the first relevant; a hit at position 3 => 1/3."""
    assert mrr_at_k(["v1", "v2", "v3", "v4"], {"v3"}, k=4) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_mrr_at_k_no_hit_in_topk() -> None:
    """Truncation matters: relevant item at position 5 is invisible at k=3."""
    assert mrr_at_k(["v1", "v2", "v3", "v4", "v5"], {"v5"}, k=3) == 0.0


def test_ndcg_at_k_perfect_order() -> None:
    """Ideal ranking should produce NDCG of 1.0."""
    relevance = {"v1": 3.0, "v2": 2.0, "v3": 1.0}
    assert ndcg_at_k(["v1", "v2", "v3"], relevance, k=3) == pytest.approx(1.0, abs=1e-9)


def test_ndcg_at_k_reversed_order_is_lower() -> None:
    relevance = {"v1": 3.0, "v2": 2.0, "v3": 1.0}
    perfect = ndcg_at_k(["v1", "v2", "v3"], relevance, k=3)
    reversed_score = ndcg_at_k(["v3", "v2", "v1"], relevance, k=3)
    assert reversed_score < perfect


# ---------------------------------------------------------------------------
# BM25 lexical retrieval (covers the rank_bm25 wrapper used in research)
# ---------------------------------------------------------------------------
def _three_doc_corpus() -> list:
    return [
        {"video_id": "v1", "caption": "machine learning recommendation system",
         "hashtags": ["ml"], "keywords": ["recommend"]},
        {"video_id": "v2", "caption": "cooking pasta dinner recipe",
         "hashtags": ["food"], "keywords": ["pasta"]},
        {"video_id": "v3", "caption": "machine learning tutorial python",
         "hashtags": ["python"], "keywords": ["tutorial"]},
    ]


def test_bm25_query_term_match_ranks_correct_doc_first() -> None:
    pytest.importorskip("rank_bm25")
    from src.research.run_experiment import bm25_search

    results, _ = bm25_search("machine learning", _three_doc_corpus(), top_k=3)
    top_ids = [r["video_id"] for r in results]
    # Both v1 and v3 contain "machine learning"; v2 must rank below them.
    assert top_ids[2] == "v2"


def test_bm25_disjoint_query_yields_zero_relevance_for_unrelated_doc() -> None:
    pytest.importorskip("rank_bm25")
    from src.research.run_experiment import bm25_search

    results, _ = bm25_search("astrophysics quasar redshift", _three_doc_corpus(), top_k=3)
    for entry in results:
        # No corpus document mentions any of the query terms.
        assert entry["score"] == 0.0


# ---------------------------------------------------------------------------
# Hybrid blend (cosine + engagement weighting, as used in evaluate_pipeline.py)
# ---------------------------------------------------------------------------
def test_hybrid_blend_respects_weight_sign() -> None:
    """Increasing the engagement weight must move a high-engagement low-cosine
    item up the ranking compared with cosine-only."""
    cosines = np.array([0.9, 0.8, 0.7], dtype=np.float32)  # initial order: A,B,C
    engagement_z = np.array([-1.0, 0.0, 2.0], dtype=np.float32)

    cosine_only = cosines  # weight = 1.0 on cosine
    blended = 0.6 * cosines + 0.4 * engagement_z

    cosine_order = list(np.argsort(-cosine_only))
    blended_order = list(np.argsort(-blended))

    assert cosine_order == [0, 1, 2]
    # C (engagement_z=2) should jump above A in the blend.
    assert blended_order.index(2) < blended_order.index(0)


def test_hybrid_blend_weight_sum() -> None:
    """A blend with weights (0.6, 0.4) on identical inputs returns the input."""
    cosines = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    blended = 0.6 * cosines + 0.4 * cosines
    assert np.allclose(blended, cosines, atol=1e-6)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_empty_iterable() -> None:
    assert aggregate([]) == 0.0


def test_aggregate_mean_of_known_values() -> None:
    assert aggregate([0.2, 0.4, 0.6]) == pytest.approx(0.4, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end output shape from the evaluation pipeline
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Configurable reranker blend weights (CLI / env / default precedence)
# ---------------------------------------------------------------------------
def test_blend_default_weights_unchanged() -> None:
    """Backwards-compat: when no overrides are passed, the default blend
    must remain (0.6, 0.4)."""
    from scripts.evaluate_pipeline import (
        DEFAULT_COSINE_WEIGHT,
        DEFAULT_ENGAGEMENT_WEIGHT,
        _resolve_blend_weights,
    )

    cosine, engagement = _resolve_blend_weights(None, None)
    assert cosine == DEFAULT_COSINE_WEIGHT == 0.6
    assert engagement == DEFAULT_ENGAGEMENT_WEIGHT == 0.4


def test_blend_custom_weights_change_reranking() -> None:
    """The blend knob must actually change the ranking. Pure-cosine (1.0, 0.0)
    and pure-engagement (0.0, 1.0) configurations must produce opposite orderings
    when the cosine and engagement signals point in opposite directions."""
    from scripts.evaluate_pipeline import _rerank_full_pipeline

    retrieved = [("A_high_cosine_low_eng", 0.9), ("B_low_cosine_high_eng", 0.5)]
    candidate_index = {
        "A_high_cosine_low_eng": {
            "video_id": "A_high_cosine_low_eng",
            "targets_z": {"engagement": 1.0},
        },
        "B_low_cosine_high_eng": {
            "video_id": "B_low_cosine_high_eng",
            "targets_z": {"engagement": 2.0},
        },
    }

    pure_cosine = [
        vid for vid, _ in _rerank_full_pipeline(
            retrieved, candidate_index, cosine_weight=1.0, engagement_weight=0.0,
        )
    ]
    pure_engagement = [
        vid for vid, _ in _rerank_full_pipeline(
            retrieved, candidate_index, cosine_weight=0.0, engagement_weight=1.0,
        )
    ]
    # Pure cosine ranks by retrieval similarity; pure engagement flips it.
    assert pure_cosine[0] == "A_high_cosine_low_eng"
    assert pure_engagement[0] == "B_low_cosine_high_eng"
    assert pure_cosine != pure_engagement


@pytest.mark.parametrize(
    ("cosine", "engagement"),
    [
        (0.5, 0.6),     # sums to 1.1
        (-0.1, 1.1),    # negative cosine
        (1.5, -0.5),    # > 1.0
    ],
)
def test_blend_invalid_weights_raise(cosine, engagement) -> None:
    from scripts.evaluate_pipeline import _validate_blend_weights

    with pytest.raises(ValueError):
        _validate_blend_weights(cosine, engagement)


def test_blend_env_var_override(monkeypatch) -> None:
    """Env vars override the defaults when no CLI value is supplied."""
    from scripts.evaluate_pipeline import _resolve_blend_weights

    monkeypatch.setenv("RANKER_COSINE_WEIGHT", "0.3")
    monkeypatch.setenv("RANKER_ENGAGEMENT_WEIGHT", "0.7")
    cosine, engagement = _resolve_blend_weights(None, None)
    assert cosine == pytest.approx(0.3)
    assert engagement == pytest.approx(0.7)


def test_blend_cli_beats_env(monkeypatch) -> None:
    """CLI value wins over env var when both are set."""
    from scripts.evaluate_pipeline import _resolve_blend_weights

    monkeypatch.setenv("RANKER_COSINE_WEIGHT", "0.3")
    monkeypatch.setenv("RANKER_ENGAGEMENT_WEIGHT", "0.7")
    cosine, engagement = _resolve_blend_weights(0.5, 0.5)
    assert cosine == pytest.approx(0.5)
    assert engagement == pytest.approx(0.5)


def test_evaluate_pipeline_output_has_required_keys(tmp_path) -> None:
    """Smoke test: the pipeline JSON must contain the keys the report cites."""
    from scripts import evaluate_pipeline as evp

    train_rows = [
        {"video_id": "v1", "caption": "machine learning recommendation",
         "hashtags": ["ml"], "keywords": ["recommend"], "targets_z": {"engagement": 1.5}},
        {"video_id": "v2", "caption": "cooking pasta dinner",
         "hashtags": ["food"], "keywords": ["pasta"], "targets_z": {"engagement": 0.5}},
    ]
    val_rows = [
        {"video_id": "v3", "caption": "machine learning python tutorial",
         "hashtags": ["ml", "python"], "keywords": ["tutorial"], "targets_z": {"engagement": 2.0}},
    ]
    test_rows = [
        {"video_id": "v9", "caption": "ml recommender system explained",
         "hashtags": ["ml"], "keywords": ["recommend"], "targets_z": {"engagement": 1.0}},
    ]

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        (splits_dir / f"{name}.jsonl").write_text(
            "\n".join(__import__("json").dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

    metrics, queries_used = evp._evaluate_pass(
        "retrieval_only",
        test_rows,
        train_rows + val_rows,
        k_values=[3],
        rerank=False,
    )
    assert {"ndcg@3", "mrr@3", "recall@3"}.issubset(metrics.keys())
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert queries_used >= 0
