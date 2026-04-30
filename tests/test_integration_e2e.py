"""End-to-end contract test for ``POST /v1/recommendations``.

Asserts the four invariants the report cites without needing a real bundle on
disk: response shape, score monotonicity (descending), top-k respect, and no
raw caption/transcript leakage. Uses the same FakeRuntime + TestClient pattern
as ``test_recommendation_service.py`` so it runs in CI without ML deps.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from src.recommendation import service  # noqa: E402

# A caption + transcript that the test will guard against leaking back in the
# response. Long and distinctive so a substring search is a strong signal.
_FIXTURE_CAPTION = (
    "growth marketing tutorial with engagement secrets you have not seen yet"
)
_FIXTURE_TRANSCRIPT = (
    "this transcript content must never appear in the recommendations response"
)


class _MonotonicFakeRuntime:
    """Returns top_k items with strictly descending scores for monotonicity check."""

    def __init__(self) -> None:
        self.last_kwargs = None

    def recommend(self, **kwargs):
        self.last_kwargs = kwargs
        top_k = int(kwargs.get("top_k", 10))
        items = []
        for rank in range(1, top_k + 1):
            items.append({
                "candidate_id": f"c{rank:03d}",
                "rank": rank,
                "score": round(1.0 - 0.05 * (rank - 1), 4),  # 1.00, 0.95, 0.90, ...
                "similarity": {
                    "sparse": round(0.9 - 0.04 * (rank - 1), 4),
                    "dense": round(0.85 - 0.03 * (rank - 1), 4),
                    "fused": round(0.88 - 0.035 * (rank - 1), 4),
                },
                "trace": {
                    "objective_model": kwargs.get("objective", "engagement"),
                    "ranker_backend": "test_fake",
                },
            })
        return {
            "objective": kwargs.get("objective", "engagement"),
            "objective_effective": "engagement",
            "generated_at": "2026-04-30T00:00:00Z",
            "fallback_mode": False,
            "items": items,
        }


def _build_request_payload(top_k: int) -> dict:
    return {
        "objective": "engagement",
        "as_of_time": "2026-04-30T00:00:00Z",
        "query": {
            "text": _FIXTURE_CAPTION,
            "transcript": _FIXTURE_TRANSCRIPT,
            "language": "en",
            "signal_hints": {"duration_seconds": 30, "visual_motion_score": 0.45},
        },
        "candidates": [
            {
                "candidate_id": f"c{rank:03d}",
                "text": f"candidate {rank} text",
                "as_of_time": "2026-04-29T00:00:00Z",
                "language": "en",
            }
            for rank in range(1, 21)
        ],
        "language": "en",
        "locale": "en-us",
        "content_type": "tutorial",
        "top_k": top_k,
        "retrieve_k": top_k * 2,
    }


def test_recommendations_endpoint_full_contract(monkeypatch):
    """Single E2E test asserting the four contract invariants the report cites."""
    client = testclient.TestClient(service.app)
    monkeypatch.setattr(service, "_bundle_marker", lambda: ("cached", 1))
    runtime = _MonotonicFakeRuntime()
    service._runtime = runtime
    service._runtime_marker = ("cached", 1)

    requested_k = 10
    response = client.post("/v1/recommendations", json=_build_request_payload(requested_k))

    # ---- contract: status + top-level shape ------------------------------
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    for key in ("objective", "objective_effective", "generated_at", "items"):
        assert key in body, f"top-level field missing: {key}"
    assert isinstance(body["items"], list)

    items = body["items"]
    # ---- contract: top-k respect -----------------------------------------
    assert len(items) <= requested_k, (
        f"endpoint returned {len(items)} items but top_k was {requested_k}"
    )
    assert len(items) > 0, "endpoint must return at least one item for a populated request"

    # ---- contract: per-item shape ----------------------------------------
    for item in items:
        assert isinstance(item.get("candidate_id"), str)
        assert isinstance(item.get("rank"), int)
        score = item.get("score")
        assert isinstance(score, (int, float)), f"score must be numeric, got {type(score)}"

    # ---- contract: score monotonicity (descending, ties allowed) ---------
    scores = [item["score"] for item in items]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"scores not monotonic non-increasing at index {i}: "
            f"{scores[i]} < {scores[i + 1]}"
        )

    # ---- contract: no raw caption / transcript leakage -------------------
    serialised = str(items)
    assert _FIXTURE_CAPTION not in serialised, (
        "raw caption text leaked into the response items"
    )
    assert _FIXTURE_TRANSCRIPT not in serialised, (
        "raw transcript content leaked into the response items"
    )


def test_recommendations_endpoint_top_k_smaller_than_default(monkeypatch):
    """Asking for a smaller top_k must shrink the result list accordingly."""
    client = testclient.TestClient(service.app)
    monkeypatch.setattr(service, "_bundle_marker", lambda: ("cached", 1))
    service._runtime = _MonotonicFakeRuntime()
    service._runtime_marker = ("cached", 1)

    response = client.post("/v1/recommendations", json=_build_request_payload(top_k=3))
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert [item["rank"] for item in items] == [1, 2, 3]
