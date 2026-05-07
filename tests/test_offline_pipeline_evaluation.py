from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pipeline.py"
    spec = importlib.util.spec_from_file_location("evaluate_pipeline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_candidate_uses_only_point_in_time_metrics() -> None:
    module = _load_module()
    row = {
        "row_id": "row-1",
        "video_id": "video-1",
        "caption": "hello #world",
        "author_id": "author-1",
        "posted_at": "2026-01-01T00:00:00Z",
        "as_of_time": "2026-01-02T00:00:00Z",
        "features": {
            "pre_metrics": {"views": 7, "likes": 2, "comments_count": 1, "shares": 0},
            "comment_intelligence": {"available": False},
            "trajectory_features": {"regime_pred": "balanced"},
        },
        "labels": {
            "future_views": 999999,
            "future_engagement_rate": 0.9,
            "future_shares_per_1k_views": 100.0,
        },
        "targets_z": {"engagement": 5.0},
    }

    candidate = module._row_to_runtime_candidate(row)

    assert candidate["views"] == 7
    assert candidate["likes"] == 2
    assert candidate["comments_count"] == 1
    assert "future_views" not in candidate
    assert "labels" not in candidate
    assert "targets_z" not in candidate
    assert candidate["signal_hints"]["comment_intelligence"] == {"available": False}
    assert candidate["signal_hints"]["trajectory_features"] == {"regime_pred": "balanced"}


def test_resolve_bundle_dir_accepts_stale_absolute_pointer(tmp_path: Path) -> None:
    module = _load_module()
    bundle = module.REPO_ROOT / "artifacts" / "recommender" / "example-bundle"
    pointer = tmp_path / "latest"
    pointer.write_text(f"/stale/machine/path/{bundle.name}\n", encoding="utf-8")

    try:
        bundle.mkdir(parents=True)
        resolved = module._resolve_bundle_dir(pointer)
        assert resolved == bundle.resolve()
    finally:
        bundle.rmdir()


def test_markdown_renders_dynamic_passes() -> None:
    module = _load_module()
    payload = {
        "passes": ["retrieval_only", "production_pipeline"],
        "retrieval_only": {
            "ndcg@5": 0.1,
            "mrr@5": 0.2,
            "recall@5": 0.3,
        },
        "production_pipeline": {
            "ndcg@5": 0.4,
            "mrr@5": 0.5,
            "recall@5": 0.6,
        },
        "queries_evaluated": 2,
        "test_set_size": 3,
        "candidate_pool_size": 4,
        "config": {
            "k_values": [5],
            "bundle_dir": "artifacts/recommender/latest",
            "split_hashes": {"test": "abc123"},
        },
    }

    text = module._markdown_table(payload)

    assert "| retrieval_only | 0.100 | 0.200 | 0.300 |" in text
    assert "| production_pipeline | 0.400 | 0.500 | 0.600 |" in text
    assert "Split hash" in text
