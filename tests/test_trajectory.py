from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.trajectory import (  # noqa: E402
    TRAJECTORY_REGIMES,
    TRAJECTORY_VERSION,
    TrajectoryBundle,
    TrajectoryBuildConfig,
    _profile_from_row,
    _regime_from_features,
    _trajectory_feature_vector,
    annotate_rows_with_trajectory_features,
    build_trajectory_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures / row builders
# ---------------------------------------------------------------------------


def _make_row(
    *,
    video_id: str = "vid-1",
    as_of: str = "2026-04-01T00:00:00Z",
    ingested_at: str = "2026-04-01T00:00:00Z",
    early: float = 0.5,
    core: float = 0.5,
    late: float = 0.5,
    stability: float = 0.5,
    series_t0: float = 0.0,
    series_t6: float = 100.0,
    series_t24: float = 200.0,
    series_t96: float = 300.0,
    composite_z: float = 0.4,
    available: bool = True,
) -> Dict[str, Any]:
    """Build a minimal row matching the trajectory module's expected schema."""

    def _objective_payload() -> Dict[str, Any]:
        return {
            "objective_available": available,
            "components": {
                "early_velocity": early,
                "core_velocity": core,
                "late_lift": late,
                "stability": stability,
            },
        }

    def _availability_payload() -> Dict[str, Any]:
        return {
            "objective_available": available,
            "components": {
                "early_velocity": available,
                "core_velocity": available,
                "late_lift": available,
                "stability": available,
            },
        }

    return {
        "video_id": video_id,
        "row_id": f"{video_id}::{as_of}",
        "as_of_time": as_of,
        "ingested_at": ingested_at,
        "labels_trajectory": {
            "reach": {
                **_objective_payload(),
                "series": {
                    "t0": series_t0,
                    "t6": series_t6,
                    "t24": series_t24,
                    "t96": series_t96,
                },
            },
            "engagement": _objective_payload(),
            "conversion": _objective_payload(),
        },
        "targets_trajectory_z": {
            "reach": {"composite_z": composite_z},
            "engagement": {"composite_z": composite_z},
            "conversion": {"composite_z": composite_z},
        },
        "target_availability": {
            "reach": _availability_payload(),
            "engagement": _availability_payload(),
            "conversion": _availability_payload(),
        },
    }


# ---------------------------------------------------------------------------
# TrajectoryBuildConfig validation
# ---------------------------------------------------------------------------


def test_config_uses_documented_defaults() -> None:
    cfg = TrajectoryBuildConfig()
    assert cfg.windows_hours == (6, 24, 96)
    assert cfg.embedding_dim == 16
    assert cfg.feature_version == "trajectory_features.v2"
    assert cfg.encoder_mode == "feature_only"


def test_config_rejects_wrong_window_count() -> None:
    with pytest.raises(ValueError, match="windows_hours must contain exactly three"):
        TrajectoryBuildConfig(windows_hours=(6, 24))  # type: ignore[arg-type]


def test_config_rejects_zero_window_hour() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        TrajectoryBuildConfig(windows_hours=(0, 24, 96))


def test_config_rejects_non_monotonic_windows() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TrajectoryBuildConfig(windows_hours=(24, 6, 96))


def test_config_clamps_embedding_dim_to_minimum_4() -> None:
    cfg = TrajectoryBuildConfig(embedding_dim=2)
    assert cfg.embedding_dim == 4


def test_config_coerces_window_floats_to_ints() -> None:
    cfg = TrajectoryBuildConfig(windows_hours=(6.7, 24.0, 96.9))  # type: ignore[arg-type]
    assert cfg.windows_hours == (6, 24, 96)


def test_config_falls_back_when_feature_version_empty() -> None:
    cfg = TrajectoryBuildConfig(feature_version="")
    assert cfg.feature_version == "trajectory_features.v2"


def test_config_falls_back_when_encoder_mode_empty() -> None:
    cfg = TrajectoryBuildConfig(encoder_mode="")
    assert cfg.encoder_mode == "feature_only"


# ---------------------------------------------------------------------------
# _regime_from_features
# ---------------------------------------------------------------------------


def test_regime_predicts_spike_when_early_velocity_dominates() -> None:
    pred, probs, conf = _regime_from_features(
        early_velocity=2.0, late_lift=-1.0, durability_ratio=0.0, stability=0.0
    )
    assert pred == "spike"
    assert probs["spike"] > probs["balanced"]
    assert probs["spike"] > probs["durable"]
    assert 0.0 <= conf <= 1.0


def test_regime_predicts_durable_when_late_lift_dominates() -> None:
    pred, probs, conf = _regime_from_features(
        early_velocity=0.0, late_lift=2.0, durability_ratio=0.8, stability=1.0
    )
    assert pred == "durable"
    assert probs["durable"] > probs["spike"]
    assert probs["durable"] > probs["balanced"]
    assert 0.0 <= conf <= 1.0


def test_regime_predicts_balanced_for_low_signal_centered_inputs() -> None:
    pred, probs, _ = _regime_from_features(
        early_velocity=0.1, late_lift=0.1, durability_ratio=0.5, stability=0.0
    )
    assert pred == "balanced"
    assert probs["balanced"] > probs["spike"]
    assert probs["balanced"] > probs["durable"]


def test_regime_probabilities_sum_to_one() -> None:
    _, probs, _ = _regime_from_features(
        early_velocity=0.3, late_lift=0.4, durability_ratio=0.6, stability=0.2
    )
    total = probs["spike"] + probs["balanced"] + probs["durable"]
    # rounded to 6 digits internally; allow a small tolerance
    assert total == pytest.approx(1.0, abs=1e-5)


def test_regime_keys_match_module_constants() -> None:
    _, probs, _ = _regime_from_features(
        early_velocity=0.0, late_lift=0.0, durability_ratio=0.0, stability=0.0
    )
    assert set(probs.keys()) == set(TRAJECTORY_REGIMES)


def test_regime_confidence_clamped_to_unit_interval() -> None:
    # Extreme inputs should never push confidence outside [0, 1]
    _, _, conf = _regime_from_features(
        early_velocity=1000.0, late_lift=-1000.0, durability_ratio=0.0, stability=0.0
    )
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _profile_from_row
# ---------------------------------------------------------------------------


def test_profile_extracts_video_id_from_explicit_field() -> None:
    row = _make_row(video_id="explicit-id")
    profile = _profile_from_row(row, windows_hours=(6, 24, 96))
    assert profile["video_id"] == "explicit-id"


def test_profile_falls_back_to_row_id_for_video_id() -> None:
    row = _make_row(video_id="vid-9")
    row["video_id"] = ""  # blank explicit field
    row["row_id"] = "vid-9::2026-04-01T00:00:00Z"
    profile = _profile_from_row(row, windows_hours=(6, 24, 96))
    assert profile["video_id"] == "vid-9"


def test_profile_includes_all_expected_feature_keys() -> None:
    profile = _profile_from_row(_make_row(), windows_hours=(6, 24, 96))
    expected = {
        "early_velocity",
        "core_velocity",
        "late_lift",
        "stability",
        "late_velocity",
        "acceleration_proxy",
        "curvature_proxy",
        "durability_ratio",
        "peak_lag_hours",
        "available_ratio",
        "missing_component_count",
        "regime_pred",
        "regime_probabilities",
        "regime_confidence",
        "objectives",
    }
    assert expected.issubset(profile["features"].keys())


def test_profile_includes_per_objective_breakdown() -> None:
    profile = _profile_from_row(_make_row(), windows_hours=(6, 24, 96))
    objectives = profile["features"]["objectives"]
    assert set(objectives.keys()) == {"reach", "engagement", "conversion"}
    for payload in objectives.values():
        assert "composite_z" in payload
        assert "objective_available" in payload
        assert set(payload["components"].keys()) == {
            "early_velocity",
            "core_velocity",
            "late_lift",
            "stability",
        }


def test_profile_acceleration_proxy_is_core_minus_early() -> None:
    row = _make_row(early=0.2, core=0.7)
    profile = _profile_from_row(row, windows_hours=(6, 24, 96))
    features = profile["features"]
    assert features["acceleration_proxy"] == pytest.approx(
        features["core_velocity"] - features["early_velocity"], abs=1e-6
    )


def test_profile_available_ratio_is_one_when_all_components_available() -> None:
    profile = _profile_from_row(_make_row(available=True), windows_hours=(6, 24, 96))
    assert profile["features"]["available_ratio"] == 1.0
    assert profile["features"]["missing_component_count"] == 0


def test_profile_available_ratio_is_zero_when_no_availability_payload() -> None:
    row = _make_row()
    row["target_availability"] = {}
    profile = _profile_from_row(row, windows_hours=(6, 24, 96))
    assert profile["features"]["available_ratio"] == 0.0
    # 4 components × 3 objectives = 12 missing
    assert profile["features"]["missing_component_count"] == 12


def test_profile_handles_missing_label_components_gracefully() -> None:
    row = _make_row()
    row["labels_trajectory"] = {}
    profile = _profile_from_row(row, windows_hours=(6, 24, 96))
    # All averaged components should be 0.0 when labels are absent
    assert profile["features"]["early_velocity"] == 0.0
    assert profile["features"]["core_velocity"] == 0.0
    assert profile["features"]["late_lift"] == 0.0
    assert profile["features"]["stability"] == 0.0


# ---------------------------------------------------------------------------
# _trajectory_feature_vector
# ---------------------------------------------------------------------------


def test_feature_vector_has_14_dimensions() -> None:
    profile = _profile_from_row(_make_row(), windows_hours=(6, 24, 96))
    vec = _trajectory_feature_vector(profile["features"])
    assert len(vec) == 14


def test_feature_vector_normalizes_peak_lag_by_96() -> None:
    features = {"peak_lag_hours": 96.0}
    vec = _trajectory_feature_vector(features)
    # peak_lag_hours sits at index 8 and should be divided by 96
    assert vec[8] == pytest.approx(1.0, abs=1e-9)


def test_feature_vector_treats_missing_regime_probabilities_as_zero() -> None:
    vec = _trajectory_feature_vector({})
    # Indices 9, 10, 11 are spike/balanced/durable probabilities
    assert vec[9] == 0.0
    assert vec[10] == 0.0
    assert vec[11] == 0.0


# ---------------------------------------------------------------------------
# build_trajectory_bundle
# ---------------------------------------------------------------------------


def test_build_bundle_dedups_to_latest_per_video() -> None:
    rows = [
        _make_row(video_id="vid-A", as_of="2026-04-01T00:00:00Z"),
        _make_row(video_id="vid-A", as_of="2026-04-02T00:00:00Z"),  # newer
        _make_row(video_id="vid-B", as_of="2026-04-01T00:00:00Z"),
    ]
    bundle = build_trajectory_bundle(rows)
    assert len(bundle.profiles) == 2
    vid_a = bundle.profile_by_video["vid-A"]
    assert vid_a["as_of_time"] == "2026-04-02T00:00:00Z"


def test_build_bundle_skips_rows_after_as_of_filter() -> None:
    rows = [
        _make_row(video_id="vid-keep", as_of="2026-04-01T00:00:00Z"),
        _make_row(video_id="vid-drop", as_of="2026-05-01T00:00:00Z"),
    ]
    bundle = build_trajectory_bundle(rows, as_of_time="2026-04-15T00:00:00Z")
    ids = {p["video_id"] for p in bundle.profiles}
    assert ids == {"vid-keep"}


def test_build_bundle_skips_rows_after_run_cutoff() -> None:
    rows = [
        _make_row(
            video_id="vid-keep",
            as_of="2026-04-01T00:00:00Z",
            ingested_at="2026-04-01T00:00:00Z",
        ),
        _make_row(
            video_id="vid-drop",
            as_of="2026-04-01T00:00:00Z",
            ingested_at="2026-05-01T00:00:00Z",
        ),
    ]
    bundle = build_trajectory_bundle(rows, run_cutoff_time="2026-04-15T00:00:00Z")
    ids = {p["video_id"] for p in bundle.profiles}
    assert ids == {"vid-keep"}


def test_build_bundle_uses_default_config_when_none_provided() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    assert bundle.config.windows_hours == (6, 24, 96)
    assert bundle.config.embedding_dim == 16


def test_build_bundle_yields_deterministic_manifest_id() -> None:
    rows = [_make_row(video_id="vid-1"), _make_row(video_id="vid-2")]
    bundle_a = build_trajectory_bundle(rows)
    bundle_b = build_trajectory_bundle(rows)
    assert bundle_a.trajectory_manifest_id == bundle_b.trajectory_manifest_id
    assert bundle_a.trajectory_schema_hash == bundle_b.trajectory_schema_hash


def test_build_bundle_skips_rows_with_no_video_id() -> None:
    bad = _make_row(video_id="")
    bad["row_id"] = ""
    bundle = build_trajectory_bundle([bad, _make_row(video_id="vid-good")])
    assert {p["video_id"] for p in bundle.profiles} == {"vid-good"}


def test_build_bundle_skips_rows_missing_as_of_time() -> None:
    bad = _make_row()
    bad["as_of_time"] = None
    bundle = build_trajectory_bundle([bad, _make_row(video_id="vid-good")])
    assert {p["video_id"] for p in bundle.profiles} == {"vid-good"}


def test_build_bundle_returns_empty_for_empty_input() -> None:
    bundle = build_trajectory_bundle([])
    assert bundle.profiles == []
    assert bundle.embeddings_by_video == {}
    assert bundle.profile_by_video == {}


def test_build_bundle_emits_unit_norm_embeddings() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    [(_, vec)] = bundle.embeddings_by_video.items()
    norm_sq = sum(v * v for v in vec)
    # Normalized vectors should have unit L2 norm (or zero for all-zero input)
    assert norm_sq == pytest.approx(1.0, abs=1e-5)


def test_build_bundle_uses_module_version_constant() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    assert bundle.version == TRAJECTORY_VERSION


# ---------------------------------------------------------------------------
# TrajectoryBundle.save / load round trip
# ---------------------------------------------------------------------------


def test_save_writes_manifest_with_expected_keys(tmp_path: Path) -> None:
    bundle = build_trajectory_bundle([_make_row()])
    manifest_path = bundle.save(tmp_path / "out")
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "version",
        "trajectory_manifest_id",
        "trajectory_schema_hash",
        "created_at",
        "config",
        "tables",
        "profile_count",
        "embedding_count",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["profile_count"] == 1
    assert payload["embedding_count"] == 1


def test_save_then_load_round_trips_bundle(tmp_path: Path) -> None:
    rows = [_make_row(video_id="vid-A"), _make_row(video_id="vid-B")]
    original = build_trajectory_bundle(rows)
    out_dir = tmp_path / "bundle"
    original.save(out_dir)

    # Remove the pickle so we exercise the manifest-driven load path
    (out_dir / "bundle.pkl").unlink()
    loaded = TrajectoryBundle.load(out_dir)

    assert loaded.version == original.version
    assert loaded.trajectory_manifest_id == original.trajectory_manifest_id
    assert loaded.trajectory_schema_hash == original.trajectory_schema_hash
    assert set(loaded.profile_by_video.keys()) == set(original.profile_by_video.keys())
    assert set(loaded.embeddings_by_video.keys()) == set(
        original.embeddings_by_video.keys()
    )


def test_load_prefers_pickle_when_present(tmp_path: Path) -> None:
    bundle = build_trajectory_bundle([_make_row()])
    out_dir = tmp_path / "bundle"
    bundle.save(out_dir)
    # Pickle still present → fast path
    loaded = TrajectoryBundle.load(out_dir)
    assert isinstance(loaded, TrajectoryBundle)
    assert loaded.trajectory_manifest_id == bundle.trajectory_manifest_id


# ---------------------------------------------------------------------------
# TrajectoryBundle.query_embedding
# ---------------------------------------------------------------------------


def test_query_embedding_uses_row_features_when_present() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    query = {
        "video_id": "vid-other",
        "features": {
            "trajectory_features": {
                "early_velocity": 0.5,
                "regime_pred": "spike",
                "regime_confidence": 0.8,
            }
        },
    }
    vec, meta = bundle.query_embedding(query)
    assert vec.shape == (bundle.config.embedding_dim,)
    assert meta["source"] == "row_feature"
    assert meta["regime_pred"] == "spike"
    assert meta["regime_confidence"] == pytest.approx(0.8, abs=1e-6)


def test_query_embedding_falls_back_to_profile_lookup() -> None:
    bundle = build_trajectory_bundle([_make_row(video_id="vid-known")])
    query: Dict[str, Any] = {"video_id": "vid-known"}
    vec, meta = bundle.query_embedding(query)
    assert vec.shape == (bundle.config.embedding_dim,)
    assert meta["video_id"] == "vid-known"
    # The fallback fills features from profile_by_video, marking source as row_feature
    assert meta["source"] == "row_feature"


def test_query_embedding_returns_unavailable_when_no_data() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    vec, meta = bundle.query_embedding({"video_id": "vid-unknown"})
    assert vec.shape == (bundle.config.embedding_dim,)
    assert meta["source"] == "unavailable"
    assert meta["regime_pred"] == "balanced"


def test_query_embedding_extracts_video_id_from_row_id() -> None:
    bundle = build_trajectory_bundle([_make_row()])
    _, meta = bundle.query_embedding({"row_id": "vid-derived::2026-04-01T00:00:00Z"})
    assert meta["video_id"] == "vid-derived"


# ---------------------------------------------------------------------------
# annotate_rows_with_trajectory_features
# ---------------------------------------------------------------------------


def test_annotate_is_no_op_for_none_bundle() -> None:
    rows = [{"video_id": "vid-1", "features": {}}]
    annotate_rows_with_trajectory_features(rows, None)
    assert rows == [{"video_id": "vid-1", "features": {}}]


def test_annotate_attaches_trajectory_features_to_matching_video() -> None:
    bundle = build_trajectory_bundle([_make_row(video_id="vid-1")])
    rows: List[Dict[str, Any]] = [{"video_id": "vid-1", "features": {}}]
    annotate_rows_with_trajectory_features(rows, bundle)
    assert "trajectory_features" in rows[0]["features"]
    assert "_trajectory_profile" in rows[0]
    assert rows[0]["_trajectory_profile"]["video_id"] == "vid-1"
    assert rows[0]["_trajectory_profile"]["trajectory_version"] == TRAJECTORY_VERSION


def test_annotate_skips_rows_with_no_video_id() -> None:
    bundle = build_trajectory_bundle([_make_row(video_id="vid-1")])
    rows: List[Dict[str, Any]] = [{"video_id": "", "row_id": "", "features": {}}]
    annotate_rows_with_trajectory_features(rows, bundle)
    assert "trajectory_features" not in rows[0]["features"]
    assert "_trajectory_profile" not in rows[0]


def test_annotate_skips_rows_missing_features_dict() -> None:
    bundle = build_trajectory_bundle([_make_row(video_id="vid-1")])
    rows: List[Dict[str, Any]] = [{"video_id": "vid-1"}]  # no "features" key
    annotate_rows_with_trajectory_features(rows, bundle)
    assert "_trajectory_profile" not in rows[0]


def test_annotate_skips_rows_for_unknown_video_id() -> None:
    bundle = build_trajectory_bundle([_make_row(video_id="vid-known")])
    rows: List[Dict[str, Any]] = [{"video_id": "vid-unknown", "features": {}}]
    annotate_rows_with_trajectory_features(rows, bundle)
    assert rows[0]["features"] == {}
    assert "_trajectory_profile" not in rows[0]
