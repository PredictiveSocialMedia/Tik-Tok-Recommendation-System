import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hashtag_diversity_report.py"
spec = importlib.util.spec_from_file_location("run_hashtag_diversity_report", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_diversity_metrics_report_similarity_and_coverage():
    recommendations = [
        ["#football", "#footballskills", "#cooking"],
        ["#ramen", "#noodles"],
    ]
    truth = [["#football", "#cooking"], ["#ramen", "#travel"]]

    metrics = module.diversity_metrics(recommendations, truth)

    assert metrics["unique_tags"] == 5.0
    assert metrics["ground_truth_coverage"] == 0.75
    assert metrics["avg_pairwise_similarity"] > 0.0
    assert metrics["redundant_tag_rate"] > 0.0


def test_format_diversity_markdown_includes_delta():
    report = {
        "rows_evaluated": 2,
        "top_n": 5,
        "variants": {
            "before_no_mmr": {
                "unique_tags": 3.0,
                "ground_truth_coverage": 0.2,
                "avg_pairwise_similarity": 0.5,
                "redundant_tag_rate": 0.4,
            },
            "after_mmr": {
                "unique_tags": 5.0,
                "ground_truth_coverage": 0.3,
                "avg_pairwise_similarity": 0.2,
                "redundant_tag_rate": 0.1,
            },
        },
        "delta": {
            "unique_tags": 2.0,
            "ground_truth_coverage": 0.1,
            "avg_pairwise_similarity": -0.3,
            "redundant_tag_rate": -0.3,
        },
    }

    markdown = module.format_diversity_markdown(report)

    assert "# Hashtag diversity report" in markdown
    assert "| after_mmr | 5 | 0.300 | 0.200 | 0.100 |" in markdown
    assert "Redundant tag rate: -0.300" in markdown
