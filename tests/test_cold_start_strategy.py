from datetime import datetime, timezone

from src.recommendation.learning.candidate_support import prepare_candidate
from src.recommendation.learning.inference import _performance_quality_score


def _query_profile():
    return {"tokens": ["ramen"], "lexical_tokens": ["ramen"]}


def test_prepare_candidate_marks_new_content_cold_start():
    candidate = prepare_candidate(
        payload={
            "candidate_id": "new-video",
            "caption": "ramen prep #ramen",
            "hashtags": ["ramen"],
            "keywords": ["noodles"],
            "topic_key": "food",
            "author_id": "chef",
            "posted_at": "2026-05-05T12:00:00Z",
            "language": "en",
            "locale": "en-us",
            "content_type": "tutorial",
        },
        as_of=datetime(2026, 5, 6, tzinfo=timezone.utc),
        query_profile=_query_profile(),
        manifest_comment_lookup=lambda video_id, as_of: None,
    )

    assert candidate is not None
    trace = candidate["cold_start_trace"]
    assert trace["is_cold_start"] is True
    assert trace["strategy"] == "semantic_topic_freshness_exploration"
    assert "cold_start_no_engagement_history" in candidate["support_flags"]
    assert "freshness_exploration" in trace["flags"]
    assert 0.0 < trace["fallback_rank_signal"] <= 1.0
    assert _performance_quality_score(candidate) > 0.0


def test_prepare_candidate_uses_observed_engagement_when_available():
    candidate = prepare_candidate(
        payload={
            "candidate_id": "seen-video",
            "caption": "ramen prep #ramen",
            "hashtags": ["ramen"],
            "topic_key": "food",
            "author_id": "chef",
            "views": 1000,
            "likes": 100,
        },
        as_of=datetime(2026, 5, 6, tzinfo=timezone.utc),
        query_profile=_query_profile(),
        manifest_comment_lookup=lambda video_id, as_of: None,
    )

    assert candidate is not None
    assert candidate["cold_start_trace"]["is_cold_start"] is False
    assert "cold_start_no_engagement_history" not in candidate["support_flags"]
    assert _performance_quality_score(candidate) > 0.0
