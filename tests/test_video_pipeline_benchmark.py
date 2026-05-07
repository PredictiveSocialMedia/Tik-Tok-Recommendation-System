import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_video_pipeline.py"
spec = importlib.util.spec_from_file_location("benchmark_video_pipeline", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(benchmark)


def test_total_stats_and_speedup():
    stats = benchmark._total_stats(
        [
            {"total_seconds": 10.0},
            {"total_seconds": 20.0},
            {"total_seconds": 30.0},
        ]
    )

    assert stats["mean_s"] == 20.0
    assert stats["p50_s"] == 20.0
    assert benchmark._speedup(30.0, 10.0) == 3.0
    assert benchmark._speedup(0.0, 10.0) == 0.0


def test_markdown_summary_includes_before_after_comparison():
    payload = {
        "videos_benchmarked": 1,
        "optimisations_applied": ["mock optimisation"],
        "total_latency": {"mean_s": 5.0, "p50_s": 5.0, "p95_s": 5.0},
        "per_video": [
            {
                "video_id": "clip",
                "video_duration_s": 3.0,
                "total_seconds": 5.0,
                "branch_count": 2,
            }
        ],
        "per_branch": {
            "visual": {"mean_s": 1.0, "p50_s": 1.0, "p95_s": 1.0, "n": 1}
        },
        "baseline_comparison": {
            "baseline_mean_s": 10.0,
            "baseline_p50_s": 9.0,
            "baseline_p95_s": 15.0,
            "current_mean_s": 5.0,
            "current_p50_s": 5.0,
            "current_p95_s": 5.0,
            "mean_speedup": 2.0,
            "p50_speedup": 1.8,
            "p95_speedup": 3.0,
        },
    }

    markdown = benchmark._markdown_summary(payload)

    assert "Before/after speedup" in markdown
    assert "| mean | 10.00 | 5.00 | 2.0x |" in markdown
