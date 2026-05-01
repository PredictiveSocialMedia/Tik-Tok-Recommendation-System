"""Unit tests for LoraHashtagFinetuner.

All tests except TestTrain run without peft, trl, or torch installed.
The evaluate() and generate() tests use monkeypatching on _model_generate
so no real model or tokenizer download occurs.
"""
from __future__ import annotations

import pytest

from src.recommendation.learning.lora_hashtag_finetuner import (
    LoraHashtagFinetunerConfig,
    LoraHashtagFinetuner,
    _LORA_AVAILABLE,
    _DATASETS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _row(
    row_id: str,
    caption: str = "",
    hashtags: list | None = None,
    keywords: list | None = None,
    posted_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "row_id": row_id,
        "video_id": row_id,
        "caption": caption,
        "hashtags": hashtags or [],
        "keywords": keywords or [],
        "search_query": "",
        "topic_key": (hashtags or ["unknown"])[0],
        "author_id": "author-1",
        "language": "en",
        "locale": "en-us",
        "as_of_time": posted_at,
        "posted_at": posted_at,
    }


def _ft(**kwargs) -> LoraHashtagFinetuner:
    return LoraHashtagFinetuner(LoraHashtagFinetunerConfig(**kwargs))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = LoraHashtagFinetunerConfig()
        assert cfg.base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.epochs == 3
        assert cfg.batch_size == 4
        assert cfg.min_hashtags == 2
        assert cfg.eval_k == 10
        assert cfg.load_in_4bit is False

    def test_custom(self):
        cfg = LoraHashtagFinetunerConfig(lora_r=16, epochs=5, min_hashtags=3)
        assert cfg.lora_r == 16
        assert cfg.epochs == 5
        assert cfg.min_hashtags == 3


# ---------------------------------------------------------------------------
# Hashtag extraction
# ---------------------------------------------------------------------------

class TestExtractHashtags:
    def test_from_explicit_list(self):
        ft = _ft()
        row = _row("r1", hashtags=["#Travel", "Italy", "#FYP"])
        tags = ft._extract_hashtags(row)
        assert "travel" in tags
        assert "italy" in tags
        assert "fyp" in tags

    def test_from_caption(self):
        ft = _ft()
        row = _row("r1", caption="Explore #italy with me #travel #fyp")
        tags = ft._extract_hashtags(row)
        assert "italy" in tags
        assert "travel" in tags
        assert "fyp" in tags

    def test_deduplicates(self):
        ft = _ft()
        row = _row("r1", caption="My #recipe video", hashtags=["recipe"])
        tags = ft._extract_hashtags(row)
        assert tags.count("recipe") == 1

    def test_strips_leading_hash(self):
        ft = _ft()
        row = _row("r1", hashtags=["##double", "  #dance  "])
        tags = ft._extract_hashtags(row)
        assert "double" in tags
        assert "dance" in tags

    def test_empty_returns_empty_list(self):
        assert _ft()._extract_hashtags(_row("r1")) == []

    def test_single_char_tags_excluded(self):
        ft = _ft()
        row = _row("r1", hashtags=["a", "ok", "go"])
        tags = ft._extract_hashtags(row)
        assert "a" not in tags
        assert "ok" in tags
        assert "go" in tags


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_eligible_enough_hashtags(self):
        ft = _ft(min_hashtags=2)
        row = _row("r1", hashtags=["travel", "italy"])
        assert ft._eligible(row) is True

    def test_ineligible_too_few_hashtags(self):
        ft = _ft(min_hashtags=2)
        row = _row("r1", hashtags=["travel"])
        assert ft._eligible(row) is False

    def test_eligible_hashtags_in_caption(self):
        ft = _ft(min_hashtags=2)
        row = _row("r1", caption="Visit #italy and #rome today")
        assert ft._eligible(row) is True

    def test_ineligible_no_hashtags(self):
        ft = _ft(min_hashtags=2)
        row = _row("r1", caption="Just a plain caption")
        assert ft._eligible(row) is False

    def test_combined_caption_and_list(self):
        ft = _ft(min_hashtags=3)
        row = _row("r1", caption="See #italy today", hashtags=["travel", "fyp"])
        # italy from caption + travel + fyp from list = 3 tags
        assert ft._eligible(row) is True


# ---------------------------------------------------------------------------
# Prompt and target building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_prompt_contains_caption(self):
        ft = _ft()
        row = _row("r1", caption="Amazing sunset in Paris", hashtags=["travel"])
        prompt = ft._build_prompt(row)
        assert "Amazing sunset in Paris" in prompt

    def test_prompt_contains_topics(self):
        ft = _ft()
        row = _row("r1", caption="Travel vlog", keywords=["paris", "sunset"])
        prompt = ft._build_prompt(row)
        assert "paris" in prompt
        assert "sunset" in prompt

    def test_prompt_contains_none_when_no_topics(self):
        ft = _ft()
        row = _row("r1", caption="A video")
        prompt = ft._build_prompt(row)
        assert "none" in prompt.lower()

    def test_prompt_has_hashtags_header(self):
        ft = _ft()
        row = _row("r1", caption="Test")
        assert "### Hashtags:" in ft._build_prompt(row)

    def test_target_formats_with_hash(self):
        ft = _ft()
        row = _row("r1", hashtags=["travel", "italy", "fyp"])
        target = ft._build_target(row)
        assert target == "#travel #italy #fyp"

    def test_target_respects_max(self):
        ft = _ft(max_hashtags_target=3)
        row = _row("r1", hashtags=["a1", "a2", "a3", "a4", "a5"])
        target = ft._build_target(row)
        assert len(target.split()) == 3

    def test_prompt_plus_target_is_full_text(self):
        ft = _ft()
        row = _row("r1", caption="Cool video", hashtags=["cool", "viral"])
        full = ft._build_prompt(row) + ft._build_target(row)
        assert "#cool" in full
        assert "#viral" in full


# ---------------------------------------------------------------------------
# parse_hashtags
# ---------------------------------------------------------------------------

class TestParseHashtags:
    def test_extracts_hash_tokens(self):
        tags = LoraHashtagFinetuner.parse_hashtags("#travel #italy #fyp")
        assert tags == ["travel", "italy", "fyp"]

    def test_ignores_non_hash_tokens(self):
        tags = LoraHashtagFinetuner.parse_hashtags("here are some #tags today")
        assert tags == ["tags"]

    def test_deduplicates(self):
        tags = LoraHashtagFinetuner.parse_hashtags("#travel #italy #travel")
        assert tags.count("travel") == 1

    def test_normalizes_to_lowercase(self):
        tags = LoraHashtagFinetuner.parse_hashtags("#Travel #ITALY")
        assert "travel" in tags
        assert "italy" in tags

    def test_empty_string(self):
        assert LoraHashtagFinetuner.parse_hashtags("") == []

    def test_excludes_single_char(self):
        tags = LoraHashtagFinetuner.parse_hashtags("#a #ok #go")
        assert "a" not in tags
        assert "ok" in tags
        assert "go" in tags

    def test_stops_at_alphanumeric_boundary(self):
        tags = LoraHashtagFinetuner.parse_hashtags("#travel, #italy.")
        assert "travel" in tags
        assert "italy" in tags


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DATASETS_AVAILABLE, reason="datasets not installed")
class TestBuildDataset:
    def test_returns_dataset_with_text_column(self):
        ft = _ft(min_hashtags=2)
        rows = [
            _row("r1", caption="Travel vlog", hashtags=["travel", "italy"]),
            _row("r2", caption="Food recipe", hashtags=["food", "recipe"]),
        ]
        ds = ft.build_dataset(rows)
        assert "text" in ds.column_names
        assert len(ds) == 2

    def test_filters_ineligible_rows(self):
        ft = _ft(min_hashtags=2)
        rows = [
            _row("r1", caption="Has tags", hashtags=["travel", "italy"]),
            _row("r2", caption="No tags"),  # ineligible
        ]
        ds = ft.build_dataset(rows)
        assert len(ds) == 1

    def test_text_contains_prompt_and_target(self):
        ft = _ft(min_hashtags=1)
        row = _row("r1", caption="My video", hashtags=["fyp", "viral"])
        ds = ft.build_dataset([row])
        text = ds[0]["text"]
        assert "My video" in text
        assert "#fyp" in text

    def test_empty_rows_raises(self):
        ft = _ft(min_hashtags=99)
        rows = [_row("r1", caption="x", hashtags=["one"])]
        ds = ft.build_dataset(rows)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# generate() and evaluate() — monkeypatched _model_generate
# ---------------------------------------------------------------------------

class TestGenerateAndEvaluate:
    def _ft_with_mock(self, response: str, **kwargs) -> LoraHashtagFinetuner:
        ft = _ft(**kwargs)
        ft._model_generate = lambda model, tok, prompt: response
        return ft

    def test_generate_returns_parsed_tags(self):
        ft = self._ft_with_mock("#travel #italy #fyp")
        row = _row("r1", caption="Trip", hashtags=["travel", "italy"])
        result = ft.generate(None, None, row)
        assert "travel" in result
        assert "italy" in result
        assert "fyp" in result

    def test_generate_respects_k(self):
        ft = self._ft_with_mock("#a1 #a2 #a3 #a4 #a5", eval_k=3)
        row = _row("r1", caption="Test")
        result = ft.generate(None, None, row)
        assert len(result) <= 3

    def test_generate_empty_response(self):
        ft = self._ft_with_mock("no hashtags here at all")
        row = _row("r1", caption="Test")
        result = ft.generate(None, None, row)
        assert result == []

    def test_evaluate_returns_expected_keys(self):
        ft = self._ft_with_mock("#travel #italy", eval_k=5)
        rows = [_row("r1", caption="Trip", hashtags=["travel", "italy", "fyp"])]
        result = ft.evaluate(None, None, rows)
        assert "precision@5" in result
        assert "recall@5" in result
        assert "f1@5" in result

    def test_evaluate_scores_in_unit_interval(self):
        ft = self._ft_with_mock("#travel #italy", eval_k=5)
        rows = [_row(f"r{i}", caption="Trip", hashtags=["travel", "italy", "fyp"]) for i in range(5)]
        result = ft.evaluate(None, None, rows)
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_evaluate_perfect_prediction(self):
        ft = self._ft_with_mock("#travel #italy", eval_k=10)
        rows = [_row("r1", caption="Trip", hashtags=["travel", "italy"])]
        result = ft.evaluate(None, None, rows)
        assert result["precision@10"] == pytest.approx(1.0)
        assert result["recall@10"] == pytest.approx(1.0)
        assert result["f1@10"] == pytest.approx(1.0)

    def test_evaluate_no_overlap(self):
        ft = self._ft_with_mock("#dance #music", eval_k=10)
        rows = [_row("r1", caption="Trip", hashtags=["travel", "italy"])]
        result = ft.evaluate(None, None, rows)
        assert result["precision@10"] == pytest.approx(0.0)
        assert result["recall@10"] == pytest.approx(0.0)

    def test_evaluate_empty_rows_returns_zeros(self):
        ft = self._ft_with_mock("#travel", eval_k=5)
        result = ft.evaluate(None, None, [])
        assert result["precision@5"] == 0.0
        assert result["recall@5"] == 0.0
        assert result["f1@5"] == 0.0

    def test_evaluate_skips_ineligible_rows(self):
        ft = self._ft_with_mock("#travel #italy", eval_k=5, min_hashtags=2)
        rows = [
            _row("r1", caption="x", hashtags=["one"]),  # ineligible (1 tag)
            _row("r2", caption="y", hashtags=["travel", "italy"]),  # eligible
        ]
        result = ft.evaluate(None, None, rows)
        # Only r2 counted; perfect prediction → f1=1.0
        assert result["f1@5"] == pytest.approx(1.0)

    def test_evaluate_custom_k_overrides_config(self):
        ft = self._ft_with_mock("#travel #italy", eval_k=5)
        rows = [_row("r1", caption="Trip", hashtags=["travel", "italy"])]
        result = ft.evaluate(None, None, rows, k=3)
        assert "precision@3" in result
        assert "f1@3" in result


# ---------------------------------------------------------------------------
# train() — only when all dependencies are present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_LORA_AVAILABLE and _DATASETS_AVAILABLE),
    reason="peft, trl, transformers, and datasets required",
)
class TestTrain:
    def test_train_raises_on_no_eligible_rows(self, tmp_path):
        ft = LoraHashtagFinetuner(
            LoraHashtagFinetunerConfig(
                output_dir=str(tmp_path / "model"),
                epochs=1,
                min_hashtags=99,
            )
        )
        rows = [_row("r1", caption="x", hashtags=["one"])]
        with pytest.raises(ValueError, match="No eligible"):
            ft.train(rows)
