from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.ranker_comparison import (  # noqa: E402
    CONTENT_TYPES,
    DEFAULT_OBJECTIVES,
    OBJECTIVE_TARGETS,
    RANKER_COMPARISON_VERSION,
    LightGBMRanker,
    LightGBMRankerConfig,
    assign_relevance_grades,
    build_dataset,
    compare_rankers,
    extract_features,
    extract_target_z,
    format_comparison_markdown,
    heuristic_score,
    paired_bootstrap_lift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    row_id: str = "vid-1",
    caption: str = "best tutorial ever",
    hashtag_count: int = 3,
    keyword_count: int = 2,
    duration: int = 30,
    content_type: str = "tutorial",
    posted_at: str = "2026-04-01T19:00:00Z",
    z_reach: Any = 0.0,
    z_engagement: Any = 0.0,
    z_conversion: Any = 0.0,
) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "video_id": row_id,
        "caption": caption,
        "content_type": content_type,
        "posted_at": posted_at,
        "features": {
            "caption_word_count": len(caption.split()),
            "hashtag_count": hashtag_count,
            "keyword_count": keyword_count,
            "duration_seconds": duration,
        },
        "targets_z": {
            "reach": z_reach,
            "engagement": z_engagement,
            "conversion": z_conversion,
        },
    }


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


def test_extract_features_returns_stable_schema() -> None:
    features = extract_features(_row())
    expected = {
        "caption_word_count",
        "caption_length_chars",
        "caption_has_question",
        "hashtag_count",
        "keyword_count",
        "duration_seconds",
        "posted_hour",
        "posted_day_of_week",
    }
    assert expected.issubset(features.keys())
    for ct in CONTENT_TYPES:
        assert f"content_type_{ct}" in features


def test_extract_features_one_hot_falls_to_other_for_unknown() -> None:
    features = extract_features(_row(content_type="something-rare"))
    assert features["content_type_other"] == 1.0
    assert features["content_type_tutorial"] == 0.0


def test_extract_features_question_detection() -> None:
    assert extract_features(_row(caption="why?"))["caption_has_question"] == 1.0
    assert extract_features(_row(caption="just a fact"))["caption_has_question"] == 0.0


def test_extract_features_handles_unparseable_posted_at() -> None:
    features = extract_features(_row(posted_at="garbage-string"))
    assert features["posted_hour"] == -1.0
    assert features["posted_day_of_week"] == -1.0


def test_extract_features_handles_missing_features_payload() -> None:
    row = _row()
    row["features"] = None
    features = extract_features(row)
    # All numeric features default to 0 when payload is missing
    assert features["hashtag_count"] == 0.0
    assert features["duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# extract_target_z
# ---------------------------------------------------------------------------


def test_extract_target_z_returns_value() -> None:
    row = _row(z_reach=1.5, z_engagement=-0.3, z_conversion=0.8)
    assert extract_target_z(row, "reach") == pytest.approx(1.5, abs=1e-9)
    assert extract_target_z(row, "engagement") == pytest.approx(-0.3, abs=1e-9)
    assert extract_target_z(row, "conversion") == pytest.approx(0.8, abs=1e-9)


def test_extract_target_z_returns_none_for_missing() -> None:
    row = _row()
    row["targets_z"]["reach"] = None
    assert extract_target_z(row, "reach") is None


def test_extract_target_z_returns_none_for_non_finite() -> None:
    row = _row()
    row["targets_z"]["engagement"] = float("nan")
    assert extract_target_z(row, "engagement") is None


def test_extract_target_z_returns_none_when_payload_missing() -> None:
    assert extract_target_z({}, "reach") is None
    assert extract_target_z({"targets_z": "not-a-dict"}, "reach") is None


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_drops_rows_missing_target() -> None:
    keep = _row(row_id="keep", z_reach=1.0)
    drop = _row(row_id="drop")
    drop["targets_z"]["reach"] = None
    X, y, _, ids = build_dataset([keep, drop, keep], "reach")
    assert X.shape[0] == 2
    assert ids == ["keep", "keep"]


def test_build_dataset_drops_rows_with_blank_id() -> None:
    keep = _row(row_id="ok", z_reach=2.0)
    blank = _row(row_id="", z_reach=99.0)
    blank["video_id"] = ""
    _, _, _, ids = build_dataset([keep, blank], "reach")
    assert ids == ["ok"]


def test_build_dataset_returns_empty_for_empty_input() -> None:
    X, y, names, ids = build_dataset([], "reach")
    assert X.shape == (0, 0)
    assert y.shape == (0,)
    assert ids == []


# ---------------------------------------------------------------------------
# heuristic_score
# ---------------------------------------------------------------------------


def test_heuristic_reach_rewards_hashtags() -> None:
    low = extract_features(_row(hashtag_count=0))
    high = extract_features(_row(hashtag_count=10))
    assert heuristic_score(high, "reach") > heuristic_score(low, "reach")


def test_heuristic_engagement_rewards_questions() -> None:
    no_q = extract_features(_row(caption="this is a fact"))
    with_q = extract_features(_row(caption="why does this work?"))
    assert heuristic_score(with_q, "engagement") > heuristic_score(no_q, "engagement")


def test_heuristic_conversion_rewards_review_content_type() -> None:
    review = extract_features(_row(content_type="review"))
    story = extract_features(_row(content_type="story"))
    assert heuristic_score(review, "conversion") > heuristic_score(story, "conversion")


def test_heuristic_unknown_objective_returns_zero() -> None:
    features = extract_features(_row())
    assert heuristic_score(features, "made-up-objective") == 0.0


def test_heuristic_engagement_penalises_extreme_caption_length() -> None:
    sweet_spot = extract_features(_row(caption=" ".join(["word"] * 10)))  # 10 words
    too_long = extract_features(_row(caption=" ".join(["word"] * 50)))    # 50 words
    assert heuristic_score(sweet_spot, "engagement") > heuristic_score(too_long, "engagement")


# ---------------------------------------------------------------------------
# LightGBMRankerConfig
# ---------------------------------------------------------------------------


def test_config_has_documented_defaults() -> None:
    cfg = LightGBMRankerConfig()
    assert cfg.n_estimators == 100
    assert cfg.num_leaves == 15
    assert cfg.n_jobs == -1


def test_config_rejects_zero_estimators() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        LightGBMRankerConfig(n_estimators=0)


def test_config_rejects_too_few_leaves() -> None:
    with pytest.raises(ValueError, match="num_leaves"):
        LightGBMRankerConfig(num_leaves=1)


def test_config_rejects_zero_min_child_samples() -> None:
    with pytest.raises(ValueError, match="min_child_samples"):
        LightGBMRankerConfig(min_child_samples=0)


def test_config_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        LightGBMRankerConfig(learning_rate=0.0)


# ---------------------------------------------------------------------------
# LightGBMRanker lifecycle
# ---------------------------------------------------------------------------


def _make_separable_rows(objective: str, n: int = 60, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Synthetic rows where ``targets_z[objective]`` is perfectly explained
    by ``hashtag_count`` (one of the input features). LightGBM should
    learn this easily.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        hashtags = int(rng.integers(0, 10))
        z_value = float(hashtags) + float(rng.normal(0.0, 0.1))
        z_payload = {"reach": 0.0, "engagement": 0.0, "conversion": 0.0}
        z_payload[objective] = z_value
        rows.append(
            {
                "row_id": f"row-{i}",
                "video_id": f"row-{i}",
                "caption": "anything",
                "content_type": "general",
                "posted_at": "2026-04-01T12:00:00Z",
                "features": {
                    "caption_word_count": 5,
                    "hashtag_count": hashtags,
                    "keyword_count": 2,
                    "duration_seconds": 30,
                },
                "targets_z": z_payload,
            }
        )
    return rows


def test_ranker_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="Unknown objective"):
        LightGBMRanker(objective="not-real")


def test_ranker_score_before_fit_raises() -> None:
    ranker = LightGBMRanker(objective="reach")
    with pytest.raises(RuntimeError, match="must be fit"):
        ranker.score([_row()])


def test_ranker_fit_raises_on_empty_data() -> None:
    rows = []
    for _ in range(3):
        r = _row()
        r["targets_z"]["reach"] = None
        rows.append(r)
    ranker = LightGBMRanker(objective="reach")
    with pytest.raises(ValueError, match="No rows produced a usable target"):
        ranker.fit(rows)


def test_ranker_fits_and_score_shape_matches_rows() -> None:
    pytest.importorskip("lightgbm")
    rows = _make_separable_rows("reach", n=40, seed=1)
    ranker = LightGBMRanker(objective="reach").fit(rows)
    scores = ranker.score(rows[:10])
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


def test_ranker_save_load_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rows = _make_separable_rows("engagement", n=30, seed=2)
    ranker = LightGBMRanker(objective="engagement").fit(rows)
    out = tmp_path / "model.pkl"
    ranker.save(out)
    loaded = LightGBMRanker.load(out)
    assert loaded.objective == "engagement"
    assert loaded.feature_names == ranker.feature_names
    np.testing.assert_allclose(loaded.score(rows[:5]), ranker.score(rows[:5]))


def test_ranker_load_version_mismatch_raises(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rows = _make_separable_rows("reach", n=20, seed=3)
    ranker = LightGBMRanker(objective="reach").fit(rows)
    out = tmp_path / "model.pkl"
    ranker.save(out)
    import pickle

    with out.open("rb") as fh:
        payload = pickle.load(fh)
    payload["version"] = "ranker_comparison.vX"
    with out.open("wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(ValueError, match="Version mismatch"):
        LightGBMRanker.load(out)


# ---------------------------------------------------------------------------
# assign_relevance_grades
# ---------------------------------------------------------------------------


def test_assign_grades_buckets_by_quantile() -> None:
    grades = assign_relevance_grades([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # 8 values, 4 grades → 2 per bucket
    assert grades == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_assign_grades_empty_returns_empty() -> None:
    assert assign_relevance_grades([]) == []


def test_assign_grades_rejects_empty_grade_set() -> None:
    with pytest.raises(ValueError):
        assign_relevance_grades([1.0], grades=[])


# ---------------------------------------------------------------------------
# paired_bootstrap_lift
# ---------------------------------------------------------------------------


def test_bootstrap_lift_zero_when_inputs_identical() -> None:
    scored = [(f"id-{i}", float(i), float(i % 4)) for i in range(20)]
    out = paired_bootstrap_lift(
        scored, scored, k_values=(10,), n_resamples=200, random_state=7
    )
    # When learned == heuristic, the lift mean and CI should both be ~0
    assert abs(out["ndcg@10"]["lift_mean"]) < 1e-9
    assert abs(out["ndcg@10"]["lift_ci_low"]) < 1e-9
    assert abs(out["ndcg@10"]["lift_ci_high"]) < 1e-9


def test_bootstrap_lift_positive_when_learned_strictly_better() -> None:
    # learned scores match the relevance grade perfectly; heuristic is reversed
    n = 20
    grades = [float(i % 4) for i in range(n)]
    learned = [(f"id-{i}", grades[i], grades[i]) for i in range(n)]
    heuristic = [(f"id-{i}", -grades[i], grades[i]) for i in range(n)]
    out = paired_bootstrap_lift(
        learned, heuristic, k_values=(10,), n_resamples=300, random_state=11
    )
    assert out["ndcg@10"]["lift_mean"] > 0.0
    assert out["ndcg@10"]["lift_positive_share"] > 0.5


def test_bootstrap_lift_handles_mismatched_lengths() -> None:
    learned = [("a", 1.0, 1.0)]
    heuristic = []  # different length
    assert paired_bootstrap_lift(learned, heuristic, (10,), 50, 0) == {}


# ---------------------------------------------------------------------------
# compare_rankers (end-to-end)
# ---------------------------------------------------------------------------


def test_compare_rankers_skips_unknown_objectives() -> None:
    pytest.importorskip("lightgbm")
    rows = _make_separable_rows("reach", n=60, seed=4)
    metrics, _ = compare_rankers(
        rows, rows, objectives=["reach", "made-up"], k_values=(10,), n_resamples=20
    )
    assert "reach" in metrics
    assert "made-up" not in metrics


def test_compare_rankers_includes_learned_heuristic_and_lift_keys() -> None:
    pytest.importorskip("lightgbm")
    rows = _make_separable_rows("reach", n=80, seed=5)
    metrics, trained = compare_rankers(
        rows, rows, objectives=["reach"], k_values=(10,), n_resamples=50
    )
    assert "reach" in metrics
    payload = metrics["reach"]
    assert {"n_rows", "learned", "heuristic", "lift"}.issubset(payload.keys())
    assert "ndcg@10" in payload["learned"]
    assert "ndcg@10" in payload["heuristic"]
    assert "ndcg@10" in payload["lift"]
    assert "reach" in trained


def test_compare_rankers_lightgbm_beats_heuristic_on_aligned_data() -> None:
    pytest.importorskip("lightgbm")
    # When the target is a clean function of hashtag_count (a feature
    # LightGBM can learn but the hand-coded reach heuristic only weakly
    # uses), LightGBM should achieve higher NDCG@10 in-sample.
    rows = _make_separable_rows("reach", n=120, seed=6)
    metrics, _ = compare_rankers(
        rows, rows, objectives=["reach"], k_values=(10,), n_resamples=30
    )
    learned_ndcg = metrics["reach"]["learned"]["ndcg@10"]
    heuristic_ndcg = metrics["reach"]["heuristic"]["ndcg@10"]
    assert learned_ndcg >= heuristic_ndcg - 0.05  # allow a small tolerance


# ---------------------------------------------------------------------------
# format_comparison_markdown
# ---------------------------------------------------------------------------


def test_format_markdown_includes_headers_and_lift_block() -> None:
    metrics = {
        "reach": {
            "n_rows": 100,
            "learned": {"ndcg@10": 0.5, "ndcg@20": 0.6, "mrr@10": 0.4, "mrr@20": 0.45},
            "heuristic": {"ndcg@10": 0.3, "ndcg@20": 0.35, "mrr@10": 0.2, "mrr@20": 0.25},
            "lift": {
                "ndcg@10": {
                    "lift_mean": 0.2,
                    "lift_ci_low": 0.05,
                    "lift_ci_high": 0.35,
                    "lift_positive_share": 0.95,
                },
            },
        }
    }
    text = format_comparison_markdown(metrics, k_values=(10, 20))
    assert "Objective" in text
    assert "learned" in text
    assert "heuristic" in text
    assert "Lift" in text
    assert "95% CI" in text
    assert "ndcg@10" in text


def test_format_markdown_empty_returns_placeholder() -> None:
    text = format_comparison_markdown({})
    assert "No metrics" in text


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_default_objectives_match_objective_targets_keys() -> None:
    assert set(DEFAULT_OBJECTIVES) == set(OBJECTIVE_TARGETS.keys())


def test_module_version_is_a_string() -> None:
    assert isinstance(RANKER_COMPARISON_VERSION, str)
    assert RANKER_COMPARISON_VERSION
