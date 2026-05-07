"""Unit tests for src/recommendation/hashtag_recommender.py

Tests cover the pure utility functions — no FAISS, no SBERT, no GPU required.
All heavy ML dependencies are avoided so these run in plain CI with no model
downloads.

Coverage:
- clean_caption: URL/mention/hashtag removal, whitespace normalisation
- extract_hashtags_from_text: pattern extraction, deduplication, casing
- extract_combined_hashtags: merging caption + explicit field, dedup
- diversity reranking: verifies the inline n-gram Jaccard diversity logic
  inside recommend() via a white-box helper test
"""

from __future__ import annotations


from src.recommendation.hashtag_recommender import (
    clean_caption,
    extract_combined_hashtags,
    extract_hashtags_from_text,
    hashtag_jaccard,
    rerank_hashtags_mmr,
)


# ---------------------------------------------------------------------------
# clean_caption
# ---------------------------------------------------------------------------

class TestCleanCaption:
    def test_removes_url_http(self):
        result = clean_caption("Check this out http://example.com cool video")
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_url_https(self):
        result = clean_caption("Follow https://tiktok.com/@user for more")
        assert "https" not in result

    def test_removes_url_www(self):
        result = clean_caption("Visit www.example.com today")
        assert "www.example.com" not in result

    def test_removes_mentions(self):
        result = clean_caption("Thanks @johndoe and @janedoe for the collab")
        assert "@" not in result

    def test_removes_hashtags(self):
        result = clean_caption("Love this #football #goals #skills")
        assert "#" not in result

    def test_preserves_regular_text(self):
        result = clean_caption("This is a normal caption about cooking")
        assert "normal caption about cooking" in result

    def test_normalises_whitespace(self):
        result = clean_caption("  too   many    spaces  ")
        assert result == "too many spaces"

    def test_empty_string(self):
        assert clean_caption("") == ""

    def test_only_hashtags_and_mentions(self):
        result = clean_caption("#football @user #goals")
        assert result == ""

    def test_mixed_content(self):
        result = clean_caption("Great game #football @espn https://link.com watch now")
        assert "Great game" in result
        assert "watch now" in result
        assert "#" not in result
        assert "@" not in result
        assert "http" not in result

    def test_non_string_input_coerced(self):
        result = clean_caption(123)
        assert result == "123"


# ---------------------------------------------------------------------------
# extract_hashtags_from_text
# ---------------------------------------------------------------------------

class TestExtractHashtagsFromText:
    def test_extracts_single_hashtag(self):
        result = extract_hashtags_from_text("Great #football video")
        assert "#football" in result

    def test_extracts_multiple_hashtags(self):
        result = extract_hashtags_from_text("#football #goals #skills")
        assert len(result) == 3

    def test_returns_lowercase(self):
        result = extract_hashtags_from_text("#Football #GOALS #Skills")
        assert "#football" in result
        assert "#goals" in result
        assert "#skills" in result

    def test_deduplicates(self):
        result = extract_hashtags_from_text("#football #football #football")
        assert result.count("#football") == 1

    def test_empty_string(self):
        assert extract_hashtags_from_text("") == []

    def test_whitespace_only(self):
        assert extract_hashtags_from_text("   ") == []

    def test_no_hashtags(self):
        result = extract_hashtags_from_text("just a regular caption")
        assert result == []

    def test_preserves_order(self):
        result = extract_hashtags_from_text("#alpha #beta #gamma")
        assert result == ["#alpha", "#beta", "#gamma"]

    def test_hashtag_with_numbers(self):
        result = extract_hashtags_from_text("#top10 #2024goals")
        assert "#top10" in result
        assert "#2024goals" in result


# ---------------------------------------------------------------------------
# extract_combined_hashtags
# ---------------------------------------------------------------------------

class TestExtractCombinedHashtags:
    def test_combines_caption_and_explicit(self):
        result = extract_combined_hashtags(
            "Great #football video",
            ["goals", "skills"]
        )
        assert "#football" in result
        assert "#goals" in result
        assert "#skills" in result

    def test_deduplicates_across_sources(self):
        result = extract_combined_hashtags(
            "#football is great",
            ["football", "goals"]
        )
        count = sum(1 for t in result if t == "#football")
        assert count == 1

    def test_normalises_explicit_tags_without_hash(self):
        result = extract_combined_hashtags("caption", ["Football", "GOALS"])
        assert "#football" in result
        assert "#goals" in result

    def test_normalises_explicit_tags_with_hash(self):
        result = extract_combined_hashtags("caption", ["#Football"])
        assert "#football" in result

    def test_empty_caption_empty_list(self):
        result = extract_combined_hashtags("", [])
        assert result == []

    def test_skips_blank_explicit_tags(self):
        result = extract_combined_hashtags("caption", ["", "  ", "football"])
        assert "" not in result
        assert "  " not in result
        assert "#football" in result

    def test_caption_tags_appear_first(self):
        result = extract_combined_hashtags("#alpha caption", ["beta"])
        assert result[0] == "#alpha"

    def test_explicit_only_no_caption_tags(self):
        result = extract_combined_hashtags("no hashtags here", ["cooking", "food"])
        assert "#cooking" in result
        assert "#food" in result


# ---------------------------------------------------------------------------
# Diversity logic — white-box tests of the inline Jaccard helper
# ---------------------------------------------------------------------------

class TestInlineDiversityLogic:
    """Tests for the n-gram Jaccard and MMR-style diversity helpers."""

    def test_identical_strings_return_one(self):
        assert hashtag_jaccard("#football", "#football") == 1.0

    def test_completely_different_strings_return_low(self):
        score = hashtag_jaccard("#football", "#xyz")
        assert score < 0.3

    def test_overlapping_strings_return_intermediate(self):
        score = hashtag_jaccard("#football", "#footballskills")
        assert 0.2 < score < 1.0

    def test_symmetry(self):
        a = hashtag_jaccard("#football", "#goals")
        b = hashtag_jaccard("#goals", "#football")
        assert abs(a - b) < 1e-9

    def test_short_string_returns_zero(self):
        assert hashtag_jaccard("ab", "#footballskills") == 0.0

    def test_score_bounded_zero_to_one(self):
        pairs = [
            ("#football", "#footballgoals"),
            ("#dance", "#cooking"),
        ]
        for a, b in pairs:
            score = hashtag_jaccard(a, b)
            assert 0.0 <= score <= 1.0

    def test_diversity_penalty_reduces_similar_tag_score(self):
        """Similar tags should get a larger diversity penalty."""
        recs = [
            {"hashtag": "#football", "frequency": 3, "score": 0.9},
            {"hashtag": "#footballskills", "frequency": 3, "score": 0.85},
        ]
        reranked = rerank_hashtags_mmr(recs, top_n=2, diversity_weight=0.5)
        similar = [item for item in reranked if item["hashtag"] == "#footballskills"][0]
        assert similar["score"] < 0.85

    def test_diverse_tag_not_penalised_much(self):
        """A completely different tag should barely be penalised."""
        recs = [
            {"hashtag": "#football", "frequency": 3, "score": 0.9},
            {"hashtag": "#cooking", "frequency": 3, "score": 0.7},
        ]
        reranked = rerank_hashtags_mmr(recs, top_n=2, diversity_weight=0.5)
        diverse = [item for item in reranked if item["hashtag"] == "#cooking"][0]
        assert diverse["score"] > 0.55
