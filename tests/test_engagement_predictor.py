from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.engagement_predictor import (  # noqa: E402
    CONTENT_TYPES,
    DEFAULT_TARGETS,
    ENGAGEMENT_PREDICTOR_VERSION,
    SUPPORTED_TARGETS,
    EngagementPredictor,
    EngagementPredictorConfig,
    baseline_metrics,
    build_dataset,
    extract_features,
    extract_target,
    format_metrics_markdown,
    regression_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    caption: str = "tutorial about coding",
    hashtag_count: int = 3,
    keyword_count: int = 2,
    duration: int = 30,
    content_type: str = "tutorial",
    posted_at: str = "2026-04-01T12:30:00Z",
    log_views: float = 9.5,
    engagement_rate: float = 0.04,
    shares_per_1k: float = 1.2,
) -> Dict[str, Any]:
    return {
        "caption": caption,
        "content_type": content_type,
        "posted_at": posted_at,
        "features": {
            "caption_word_count": len(caption.split()),
            "hashtag_count": hashtag_count,
            "keyword_count": keyword_count,
            "duration_seconds": duration,
        },
        "labels": {
            "future_reach_log_delta": log_views,
            "future_engagement_rate": engagement_rate,
            "future_shares_per_1k_views": shares_per_1k,
        },
    }


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


def test_extract_features_returns_stable_schema() -> None:
    features = extract_features(_row())
    expected_numeric = {
        "caption_word_count",
        "caption_length_chars",
        "caption_has_question",
        "hashtag_count",
        "keyword_count",
        "duration_seconds",
        "posted_hour",
        "posted_day_of_week",
    }
    expected_one_hot = {f"content_type_{ct}" for ct in CONTENT_TYPES}
    assert expected_numeric.issubset(features.keys())
    assert expected_one_hot.issubset(features.keys())


def test_extract_features_one_hot_known_content_type() -> None:
    features = extract_features(_row(content_type="tutorial"))
    assert features["content_type_tutorial"] == 1.0
    assert features["content_type_general"] == 0.0
    assert features["content_type_other"] == 0.0


def test_extract_features_one_hot_unknown_content_type_falls_to_other() -> None:
    features = extract_features(_row(content_type="some-rare-thing"))
    assert features["content_type_other"] == 1.0
    assert features["content_type_general"] == 0.0


def test_extract_features_question_mark_signal() -> None:
    with_q = extract_features(_row(caption="why does this work?"))
    without_q = extract_features(_row(caption="how it works"))
    assert with_q["caption_has_question"] == 1.0
    assert without_q["caption_has_question"] == 0.0


def test_extract_features_caption_length_is_character_count() -> None:
    features = extract_features(_row(caption="hello"))
    assert features["caption_length_chars"] == 5.0


def test_extract_features_parses_posted_at_to_hour_and_dow() -> None:
    # 2026-04-01 was a Wednesday (weekday() == 2), 12:30 → hour 12
    features = extract_features(_row(posted_at="2026-04-01T12:30:00Z"))
    assert features["posted_hour"] == 12.0
    assert features["posted_day_of_week"] == 2.0


def test_extract_features_handles_unparseable_posted_at() -> None:
    features = extract_features(_row(posted_at="not-a-timestamp"))
    assert features["posted_hour"] == -1.0
    assert features["posted_day_of_week"] == -1.0


def test_extract_features_handles_missing_features_payload() -> None:
    row = _row()
    row.pop("features")
    features = extract_features(row)
    # All numeric features default to 0.0 when payload is missing
    assert features["caption_word_count"] == 0.0
    assert features["hashtag_count"] == 0.0
    assert features["duration_seconds"] == 0.0


def test_extract_features_handles_non_dict_features_payload() -> None:
    row = _row()
    row["features"] = "wrong-type"
    # Should not raise; should fall back to defaults
    features = extract_features(row)
    assert features["caption_word_count"] == 0.0


# ---------------------------------------------------------------------------
# extract_target
# ---------------------------------------------------------------------------


def test_extract_target_returns_value_for_each_supported_target() -> None:
    row = _row(log_views=9.0, engagement_rate=0.05, shares_per_1k=2.0)
    assert extract_target(row, "log_views") == pytest.approx(9.0, abs=1e-9)
    assert extract_target(row, "engagement_rate") == pytest.approx(0.05, abs=1e-9)
    assert extract_target(row, "shares_per_1k") == pytest.approx(2.0, abs=1e-9)


def test_extract_target_returns_none_for_null_value() -> None:
    row = _row()
    row["labels"]["future_reach_log_delta"] = None
    assert extract_target(row, "log_views") is None


def test_extract_target_returns_none_for_non_finite_value() -> None:
    row = _row()
    row["labels"]["future_engagement_rate"] = float("nan")
    assert extract_target(row, "engagement_rate") is None


def test_extract_target_returns_none_when_labels_missing() -> None:
    assert extract_target({"labels": None}, "log_views") is None
    assert extract_target({}, "log_views") is None


def test_extract_target_rejects_unknown_target_name() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        extract_target(_row(), "made_up_target")


def test_default_targets_are_supported() -> None:
    for target in DEFAULT_TARGETS:
        assert target in SUPPORTED_TARGETS


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_drops_rows_missing_target() -> None:
    good = _row()
    bad = _row()
    bad["labels"]["future_reach_log_delta"] = None
    X, y, _ = build_dataset([good, bad, good], "log_views")
    assert X.shape[0] == 2
    assert len(y) == 2


def test_build_dataset_stable_feature_names() -> None:
    rows = [_row(), _row(), _row()]
    _, _, names = build_dataset(rows, "log_views")
    # Names should be sorted alphabetically and consistent
    assert names == sorted(names)
    assert "caption_word_count" in names
    assert any(name.startswith("content_type_") for name in names)


def test_build_dataset_empty_input_returns_empty_arrays() -> None:
    X, y, names = build_dataset([], "log_views")
    assert X.shape == (0, 0)
    assert y.shape == (0,)
    assert names == []


def test_build_dataset_y_aligns_with_x() -> None:
    rows = [
        _row(log_views=1.0),
        _row(log_views=5.0),
        _row(log_views=9.0),
    ]
    X, y, _ = build_dataset(rows, "log_views")
    assert X.shape[0] == 3
    assert list(y) == [1.0, 5.0, 9.0]


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y.copy())
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spearman"] == pytest.approx(1.0, abs=1e-6)


def test_regression_metrics_zero_for_empty_input() -> None:
    metrics = regression_metrics(np.zeros((0,)), np.zeros((0,)))
    assert metrics["n_rows"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_regression_metrics_constant_prediction_has_zero_r2() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    metrics = regression_metrics(y_true, y_pred)
    # Constant prediction at mean → R² == 0 by definition
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-6)


def test_regression_metrics_records_n_rows() -> None:
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y.copy())
    assert metrics["n_rows"] == 3.0


def test_regression_metrics_spearman_is_one_for_monotonic_pred() -> None:
    # Predictions are monotonic in true labels but on a different scale
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true * 100.0 + 1000.0
    assert regression_metrics(y_true, y_pred)["spearman"] == pytest.approx(1.0, abs=1e-6)


def test_regression_metrics_spearman_is_negative_for_anti_monotonic_pred() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = -y_true
    assert regression_metrics(y_true, y_pred)["spearman"] == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# EngagementPredictorConfig validation
# ---------------------------------------------------------------------------


def test_config_uses_documented_defaults() -> None:
    cfg = EngagementPredictorConfig()
    # Defaults tuned for small (~500 row) splits — see PR description
    assert cfg.n_estimators == 50
    assert cfg.max_depth == 2
    assert cfg.learning_rate == 0.05
    assert cfg.min_samples_leaf == 15
    assert cfg.random_state == 42


def test_config_rejects_zero_estimators() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        EngagementPredictorConfig(n_estimators=0)


def test_config_rejects_zero_max_depth() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        EngagementPredictorConfig(max_depth=0)


def test_config_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        EngagementPredictorConfig(learning_rate=0.0)


def test_config_rejects_zero_min_samples_leaf() -> None:
    with pytest.raises(ValueError, match="min_samples_leaf"):
        EngagementPredictorConfig(min_samples_leaf=0)


# ---------------------------------------------------------------------------
# EngagementPredictor lifecycle
# ---------------------------------------------------------------------------


def _make_synthetic_rows(n: int = 60, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Synthetic dataset where ``log_views`` is a clean linear function of
    ``hashtag_count``. A reasonable model should fit it well.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        hashtags = int(rng.integers(0, 10))
        log_views = 5.0 + 0.3 * hashtags + float(rng.normal(0.0, 0.1))
        row = _row(
            hashtag_count=hashtags,
            log_views=log_views,
            engagement_rate=0.02 + 0.001 * hashtags,
            shares_per_1k=0.5 + 0.05 * hashtags,
        )
        rows.append(row)
    return rows


def test_predictor_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        EngagementPredictor(target="not-a-real-target")


def test_predictor_predict_before_fit_raises() -> None:
    predictor = EngagementPredictor(target="log_views")
    with pytest.raises(RuntimeError, match="must be fit"):
        predictor.predict([_row()])


def test_predictor_evaluate_before_fit_raises_when_rows_have_target() -> None:
    predictor = EngagementPredictor(target="log_views")
    with pytest.raises(RuntimeError, match="must be fit"):
        predictor.evaluate([_row()])


def test_predictor_fit_raises_on_no_usable_rows() -> None:
    rows = []
    for _ in range(3):
        r = _row()
        r["labels"]["future_reach_log_delta"] = None
        rows.append(r)
    predictor = EngagementPredictor(target="log_views")
    with pytest.raises(ValueError, match="No rows produced a usable"):
        predictor.fit(rows)


def test_predictor_fits_and_beats_baseline_on_separable_data() -> None:
    pytest.importorskip("sklearn")
    rows = _make_synthetic_rows(n=80, seed=1)
    predictor = EngagementPredictor(target="log_views")
    predictor.fit(rows)
    metrics = predictor.evaluate(rows)
    # On training data, the fitted GBR should comfortably beat the constant-mean baseline.
    baseline = baseline_metrics(rows, target="log_views")
    assert metrics["mae"] < baseline["mae"]
    assert metrics["r2"] > 0.5


def test_predictor_predict_returns_correct_shape() -> None:
    pytest.importorskip("sklearn")
    train = _make_synthetic_rows(n=40, seed=2)
    predictor = EngagementPredictor(target="log_views").fit(train)
    test = _make_synthetic_rows(n=10, seed=3)
    preds = predictor.predict(test)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


def test_predictor_save_and_load_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rows = _make_synthetic_rows(n=40, seed=4)
    predictor = EngagementPredictor(target="log_views").fit(rows)
    out_path = tmp_path / "engagement_predictor.pkl"
    predictor.save(out_path)
    assert out_path.exists()

    loaded = EngagementPredictor.load(out_path)
    assert loaded.target == "log_views"
    assert loaded.feature_names == predictor.feature_names
    # Predictions should match exactly
    test = _make_synthetic_rows(n=5, seed=5)
    np.testing.assert_allclose(loaded.predict(test), predictor.predict(test))


def test_predictor_load_raises_on_version_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rows = _make_synthetic_rows(n=20, seed=6)
    predictor = EngagementPredictor(target="log_views").fit(rows)
    path = tmp_path / "model.pkl"
    predictor.save(path)

    # Tamper with the saved version string
    import pickle

    with path.open("rb") as fh:
        payload = pickle.load(fh)
    payload["version"] = "engagement_predictor.v999"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    with pytest.raises(ValueError, match="Version mismatch"):
        EngagementPredictor.load(path)


def test_predictor_evaluate_drops_rows_missing_target() -> None:
    pytest.importorskip("sklearn")
    train = _make_synthetic_rows(n=30, seed=7)
    predictor = EngagementPredictor(target="log_views").fit(train)

    eval_rows = _make_synthetic_rows(n=5, seed=8)
    # Knock out one target; the eval should silently skip that row
    eval_rows[0]["labels"]["future_reach_log_delta"] = None
    metrics = predictor.evaluate(eval_rows)
    assert metrics["n_rows"] == 4.0


# ---------------------------------------------------------------------------
# baseline_metrics
# ---------------------------------------------------------------------------


def test_baseline_metrics_in_sample_returns_zero_r2_by_construction() -> None:
    rows = _make_synthetic_rows(n=20, seed=9)
    metrics = baseline_metrics(rows, target="log_views")
    # Baseline predicts the in-sample mean → R² = 0 by construction
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["n_rows"] == 20.0


def test_baseline_metrics_uses_train_mean_when_provided() -> None:
    train = _make_synthetic_rows(n=30, seed=10)
    eval_rows = _make_synthetic_rows(n=10, seed=11)
    fair = baseline_metrics(eval_rows, target="log_views", train_rows=train)
    in_sample = baseline_metrics(eval_rows, target="log_views")
    # Different train and eval distributions usually give different MAEs
    assert fair["n_rows"] == 10.0
    assert in_sample["n_rows"] == 10.0
    # In-sample baseline always has R²==0; fair (train-mean) baseline can be different
    assert in_sample["r2"] == pytest.approx(0.0, abs=1e-6)


def test_baseline_metrics_falls_back_to_eval_mean_when_train_rows_empty() -> None:
    eval_rows = _make_synthetic_rows(n=8, seed=12)
    fair = baseline_metrics(eval_rows, target="log_views", train_rows=[])
    # Empty train → fall back to in-sample mean → R²==0
    assert fair["r2"] == pytest.approx(0.0, abs=1e-6)


def test_baseline_metrics_handles_empty() -> None:
    metrics = baseline_metrics([], target="log_views")
    assert metrics["n_rows"] == 0.0


# ---------------------------------------------------------------------------
# format_metrics_markdown
# ---------------------------------------------------------------------------


def test_format_markdown_includes_headers_and_rows() -> None:
    payload = {
        "log_views": {
            "trained": {"n_rows": 100, "mae": 0.5, "rmse": 0.7, "r2": 0.4, "spearman": 0.6},
            "baseline": {"n_rows": 100, "mae": 1.0, "rmse": 1.2, "r2": 0.0, "spearman": 0.0},
        }
    }
    text = format_metrics_markdown(payload)
    assert "Target" in text
    assert "MAE" in text
    assert "RMSE" in text
    assert "Spearman" in text
    assert "log_views" in text
    assert "trained" in text
    assert "baseline" in text


def test_format_markdown_empty_returns_placeholder() -> None:
    text = format_metrics_markdown({})
    assert "No metrics" in text


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_module_version_is_a_string() -> None:
    assert isinstance(ENGAGEMENT_PREDICTOR_VERSION, str)
    assert ENGAGEMENT_PREDICTOR_VERSION


def test_content_types_includes_other_bucket() -> None:
    assert "other" in CONTENT_TYPES
