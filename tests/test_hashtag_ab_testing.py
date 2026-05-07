from __future__ import annotations

from src.recommendation.learning.hashtag_ab_testing import (
    HashtagVariant,
    TfidfHashtagBaseline,
    evaluate_hashtag_variants,
    extract_hashtags,
    format_hashtag_ab_markdown,
)


def test_extract_hashtags_combines_caption_and_field() -> None:
    row = {"caption": "hello #Food #Food", "hashtags": ["Travel", "#NYC"]}

    assert extract_hashtags(row) == ["#food", "#travel", "#nyc"]


def test_tfidf_baseline_recommends_tags_from_similar_training_rows() -> None:
    train = [
        {"caption": "easy pasta recipe #pasta #food", "hashtags": []},
        {"caption": "dance challenge tutorial #dance #trend", "hashtags": []},
    ]
    baseline = TfidfHashtagBaseline(train)

    predicted = baseline.recommend({"caption": "pasta tutorial for dinner"}, k=2)

    assert "#pasta" in predicted
    assert "#food" in predicted


def test_evaluate_hashtag_variants_reports_precision_recall_and_coverage() -> None:
    rows = [
        {"caption": "easy pasta recipe #pasta #food"},
        {"caption": "dance challenge #dance"},
    ]
    report = evaluate_hashtag_variants(
        [
            HashtagVariant(
                name="perfect",
                recommend=lambda row, k: extract_hashtags(row)[:k],
            ),
            HashtagVariant(
                name="miss",
                recommend=lambda row, k: ["#wrong"],
            ),
        ],
        rows,
        k_values=(2,),
    )

    perfect = report["variants"][0]["metrics"]
    miss = report["variants"][1]["metrics"]
    assert perfect["precision@2"] == 1.0
    assert perfect["recall@2"] == 1.0
    assert perfect["catalog_coverage"] == 1.0
    assert miss["recall@2"] == 0.0


def test_format_hashtag_ab_markdown_contains_variant_rows() -> None:
    report = {
        "n_rows": 2,
        "n_evaluable_rows": 1,
        "unique_ground_truth_hashtags": 2,
        "k_values": [5],
        "variants": [
            {
                "name": "tfidf_baseline",
                "metrics": {
                    "precision@5": 0.1,
                    "recall@5": 0.2,
                    "f1@5": 0.133333,
                    "diversity@5": 0.9,
                    "catalog_coverage": 0.4,
                },
            }
        ],
    }

    text = format_hashtag_ab_markdown(report)

    assert "# Hashtag recommender A/B test" in text
    assert "| tfidf_baseline | 0.100 | 0.200 | 0.133 | 0.900 | 0.400 |" in text
