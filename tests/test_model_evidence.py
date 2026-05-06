from src.recommendation.learning.model_evidence import (
    TopicPriorHashtagBaseline,
    build_comparison,
    evaluate_tag_predictor,
    format_metric_report,
    metric_delta,
    summarize_improvement,
)


def test_metric_delta_and_verdict_use_shared_metrics():
    baseline = {"ndcg@10": 0.2, "mrr@10": 0.5}
    candidate = {"ndcg@10": 0.3, "mrr@10": 0.4, "extra": 1.0}

    assert metric_delta(baseline, candidate) == {
        "ndcg@10": 0.09999999999999998,
        "mrr@10": -0.09999999999999998,
    }
    assert summarize_improvement(baseline, candidate, "ndcg@10") == "improved"
    assert summarize_improvement(baseline, candidate, "mrr@10") == "regressed"


def test_topic_prior_baseline_prefers_matching_topic_tags():
    baseline = TopicPriorHashtagBaseline(
        [
            {"topic_key": "food", "hashtags": ["ramen", "noodles"]},
            {"topic_key": "food", "caption": "late dinner #ramen"},
            {"topic_key": "fitness", "hashtags": ["gym"]},
        ]
    )

    assert baseline.recommend({"topic_key": "food"}, 2)[0] == "ramen"


def test_evaluate_tag_predictor_computes_precision_recall_f1():
    rows = [{"hashtags": ["ramen", "noodles"]}, {"caption": "lift #gym"}]

    metrics = evaluate_tag_predictor(
        rows,
        lambda row, k: ["ramen", "gym"][:k],
        k=2,
    )

    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.75
    assert round(metrics["f1@2"], 3) == 0.583


def test_format_metric_report_includes_models_and_candidate_missing():
    report = {
        "title": "Evidence",
        "status": "candidate_missing",
        "data": {"train_rows": 10, "test_rows": 2},
        "models": [
            {"name": "base", "role": "baseline", "metrics": {"ndcg@10": 0.2}},
        ],
        "comparison": build_comparison(
            {"ndcg@10": 0.2},
            None,
            primary_metric="ndcg@10",
        ),
    }

    markdown = format_metric_report(report)

    assert "# Evidence" in markdown
    assert "candidate_missing" in markdown
    assert "| base | baseline | 0.2000 |" in markdown
