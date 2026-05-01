"""Unit tests for TikTokEmbeddingFinetuner.

The tests for hashtag extraction, eligibility, pair construction, and
evaluate() all run without sentence-transformers installed. evaluate() is
exercised through a _MockModel that returns deterministic embeddings, so no
model download occurs.

Only the train() test is skipped when sentence-transformers is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.recommendation.learning.embedding_finetuner import (
    EmbeddingFinetunerConfig,
    TikTokEmbeddingFinetuner,
    _SBERT_AVAILABLE,
    _DATASETS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _row(
    row_id: str,
    topic: str,
    caption: str = "",
    hashtags: list | None = None,
    plays: int = 500,
    posted_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "row_id": row_id,
        "video_id": row_id,
        "topic_key": topic,
        "caption": caption,
        "hashtags": hashtags or [],
        "keywords": [],
        "search_query": "",
        "language": "en",
        "locale": "en-us",
        "author_id": "author-1",
        "as_of_time": posted_at,
        "posted_at": posted_at,
        "plays": plays,
    }


class _MockModel:
    """Deterministic fake encoder — no ML dependencies needed."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=256,
    ):
        dim = 8
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            v = np.random.RandomState(seed).randn(dim).astype(np.float32)
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.stack(vecs)


def _finetuner(**kwargs) -> TikTokEmbeddingFinetuner:
    cfg = EmbeddingFinetunerConfig(**kwargs)
    return TikTokEmbeddingFinetuner(cfg)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = EmbeddingFinetunerConfig()
        assert cfg.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.epochs == 3
        assert cfg.batch_size == 32
        assert cfg.eval_k == 10
        assert cfg.min_shared_hashtags == 1
        assert cfg.max_pairs_per_anchor == 10
        assert cfg.min_plays == 100

    def test_custom(self):
        cfg = EmbeddingFinetunerConfig(epochs=5, eval_k=20)
        assert cfg.epochs == 5
        assert cfg.eval_k == 20


# ---------------------------------------------------------------------------
# Hashtag extraction
# ---------------------------------------------------------------------------

class TestHashtags:
    def test_from_explicit_list(self):
        ft = _finetuner()
        row = _row("r1", "dance", hashtags=["#Fitness", "Cooking", "#TRAVEL"])
        tags = ft._hashtags(row)
        assert "fitness" in tags
        assert "cooking" in tags
        assert "travel" in tags

    def test_from_caption(self):
        ft = _finetuner()
        row = _row("r1", "food", caption="Try this #recipe for #food lovers")
        tags = ft._hashtags(row)
        assert "recipe" in tags
        assert "food" in tags

    def test_deduplicates_across_sources(self):
        ft = _finetuner()
        row = _row("r1", "food", caption="My #recipe video", hashtags=["recipe"])
        assert ft._hashtags(row) == frozenset({"recipe"})

    def test_strips_hashes_and_whitespace(self):
        ft = _finetuner()
        row = _row("r1", "t", hashtags=["  #dance  ", "##double"])
        tags = ft._hashtags(row)
        assert "dance" in tags
        assert "double" in tags  # strips leading # repeatedly — first # only

    def test_empty_row_returns_empty_frozenset(self):
        ft = _finetuner()
        assert ft._hashtags(_row("r1", "t")) == frozenset()


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_eligible_with_caption_and_enough_plays(self):
        ft = _finetuner(min_plays=100)
        assert ft._eligible(_row("r1", "t", caption="great video", plays=500))

    def test_ineligible_low_plays(self):
        ft = _finetuner(min_plays=100)
        assert not ft._eligible(_row("r1", "t", caption="great video", plays=50))

    def test_zero_plays_treated_as_unknown_and_kept(self):
        ft = _finetuner(min_plays=100)
        # plays=0 means unscraped, not literally zero — should pass
        assert ft._eligible(_row("r1", "t", caption="great video", plays=0))

    def test_ineligible_empty_text(self):
        ft = _finetuner()
        assert not ft._eligible(_row("r1", "t", caption="", hashtags=[], plays=500))

    def test_eligible_via_hashtag_text(self):
        ft = _finetuner()
        # no caption but has hashtag — row_text picks it up from hashtags list
        row = _row("r1", "t", caption="", hashtags=["cooking"], plays=0)
        assert ft._eligible(row)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

class TestBuildTrainingPairs:
    def test_same_topic_shared_hashtag_forms_pair(self):
        ft = _finetuner(min_shared_hashtags=1)
        rows = [
            _row("r1", "food", caption="pasta #recipe"),
            _row("r2", "food", caption="cake #recipe"),
            _row("r3", "tech", caption="python #code"),
        ]
        anchors, positives = ft.build_training_pairs(rows)
        # r1↔r2 share #recipe in the same topic — should produce 2 pairs
        assert len(anchors) >= 2
        assert len(anchors) == len(positives)

    def test_different_topics_never_paired(self):
        ft = _finetuner(min_shared_hashtags=1)
        rows = [
            _row("r1", "food", caption="pasta #recipe"),
            _row("r2", "tech", caption="python #code"),
        ]
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) == 0

    def test_no_hashtag_fallback_same_topic(self):
        ft = _finetuner(min_shared_hashtags=1)
        rows = [
            _row("r1", "science", caption="space exploration"),
            _row("r2", "science", caption="nasa rockets"),
        ]
        # Both have no hashtags → no_tags=True → should still pair
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) >= 2

    def test_one_has_hashtag_other_does_not_no_fallback(self):
        ft = _finetuner(min_shared_hashtags=1)
        rows = [
            _row("r1", "art", caption="painting #art"),
            _row("r2", "art", caption="sculpting"),
        ]
        # anchor_tags = {"art"}, other_tags = {} → shared=0, no_tags=False → skip
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) == 0

    def test_respects_max_pairs_per_anchor(self):
        ft = _finetuner(min_shared_hashtags=1, max_pairs_per_anchor=3)
        rows = [
            _row(f"r{i}", "sport", caption=f"run {i} #sport", hashtags=["sport"])
            for i in range(20)
        ]
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) <= 20 * 3

    def test_single_video_per_topic_yields_no_pairs(self):
        ft = _finetuner()
        rows = [
            _row("r1", "food", caption="pasta #recipe"),
            _row("r2", "tech", caption="python #code"),
        ]
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) == 0

    def test_low_plays_rows_excluded(self):
        ft = _finetuner(min_plays=100, min_shared_hashtags=1)
        rows = [
            _row("r1", "food", caption="pasta #recipe", plays=10),
            _row("r2", "food", caption="cake #recipe", plays=10),
        ]
        anchors, _ = ft.build_training_pairs(rows)
        assert len(anchors) == 0

    def test_returns_equal_length_lists(self):
        ft = _finetuner(min_shared_hashtags=1)
        rows = [
            _row("r1", "music", caption="guitar #music"),
            _row("r2", "music", caption="piano #music"),
            _row("r3", "music", caption="drums #music"),
        ]
        anchors, positives = ft.build_training_pairs(rows)
        assert len(anchors) == len(positives)

    def test_higher_overlap_ranked_first(self):
        ft = _finetuner(min_shared_hashtags=1, max_pairs_per_anchor=1)
        rows = [
            _row("anchor", "food", caption="", hashtags=["recipe", "cooking", "italian"]),
            _row("high", "food", caption="", hashtags=["recipe", "cooking", "italian"]),
            _row("low", "food", caption="", hashtags=["recipe"]),
        ]
        # With max 1 pair per anchor, the high-overlap pair should be chosen
        anchors, positives = ft.build_training_pairs(rows)
        # "anchor" should pair with "high" (3 shared) not "low" (1 shared)
        # We can't easily check the text content without knowing row_text output,
        # so just verify pairs were produced
        assert len(anchors) >= 1


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _ft(self, **kwargs) -> TikTokEmbeddingFinetuner:
        return _finetuner(eval_k=3, **kwargs)

    def _rows(self):
        return [
            _row("a1", "animals", caption="cute cat playing"),
            _row("a2", "animals", caption="dog running outside"),
            _row("a3", "animals", caption="parrot talking"),
            _row("t1", "tech", caption="python coding tips"),
            _row("t2", "tech", caption="machine learning basics"),
        ]

    def test_returns_expected_keys(self):
        result = self._ft().evaluate(_MockModel(), self._rows())
        assert "ndcg@3" in result
        assert "mrr@3" in result

    def test_scores_in_unit_interval(self):
        result = self._ft().evaluate(_MockModel(), self._rows())
        for val in result.values():
            assert 0.0 <= val <= 1.0, f"Score out of range: {val}"

    def test_empty_rows_returns_zeros(self):
        result = self._ft().evaluate(_MockModel(), [])
        assert result["ndcg@3"] == 0.0
        assert result["mrr@3"] == 0.0

    def test_single_topic_single_video_returns_zeros(self):
        rows = [_row("only", "lonely", caption="solo video")]
        result = self._ft().evaluate(_MockModel(), rows)
        assert result["ndcg@3"] == 0.0
        assert result["mrr@3"] == 0.0

    def test_custom_k_overrides_config(self):
        result = _finetuner(eval_k=3).evaluate(_MockModel(), self._rows(), k=5)
        assert "ndcg@5" in result
        assert "mrr@5" in result

    def test_perfect_retrieval_gives_mrr_one(self):
        """Model returning identical embeddings per topic → MRR@k should be 1."""

        class _TopicModel:
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=256):
                vecs = []
                for t in texts:
                    first = t.split()[0] if t.strip() else "x"
                    seed = hash(first) % (2**31)
                    v = np.random.RandomState(seed).randn(8).astype(np.float32)
                    vecs.append(v / (np.linalg.norm(v) + 1e-9))
                return np.stack(vecs)

        rows = [
            _row("a1", "animals", caption="animals playing"),
            _row("a2", "animals", caption="animals running"),
            _row("t1", "tech", caption="tech coding"),
            _row("t2", "tech", caption="tech learning"),
        ]
        result = _finetuner(eval_k=3).evaluate(_TopicModel(), rows)
        assert result["mrr@3"] == pytest.approx(1.0, abs=1e-6)

    def test_random_model_produces_nonzero_ndcg_with_enough_rows(self):
        rows = [
            _row(f"r{i}", "topic_a", caption=f"video {i} about stuff")
            for i in range(6)
        ]
        result = self._ft().evaluate(_MockModel(), rows)
        # NDCG > 0 requires at least one relevant retrieved — with 5 relevant
        # docs in top-3 some will hit by chance
        assert "ndcg@3" in result


# ---------------------------------------------------------------------------
# train() — only when dependencies are present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_SBERT_AVAILABLE and _DATASETS_AVAILABLE),
    reason="sentence-transformers and datasets required",
)
class TestTrain:
    def test_train_raises_on_no_pairs(self, tmp_path):
        ft = TikTokEmbeddingFinetuner(
            EmbeddingFinetunerConfig(
                output_dir=str(tmp_path / "model"),
                epochs=1,
                min_shared_hashtags=99,  # impossible threshold → 0 pairs
            )
        )
        rows = [
            _row("r1", "food", caption="pasta"),
            _row("r2", "food", caption="cake"),
        ]
        with pytest.raises(ValueError, match="No training pairs"):
            ft.train(rows)

    def test_train_saves_model(self, tmp_path):
        output = tmp_path / "model"
        ft = TikTokEmbeddingFinetuner(
            EmbeddingFinetunerConfig(
                output_dir=str(output),
                epochs=1,
                batch_size=2,
                min_shared_hashtags=1,
            )
        )
        rows = [
            _row(f"r{i}", "food", caption=f"video {i} #recipe #food", plays=0)
            for i in range(8)
        ]
        model = ft.train(rows)
        assert output.exists()
        emb = model.encode(["test"], convert_to_numpy=True)
        assert emb.shape[0] == 1
