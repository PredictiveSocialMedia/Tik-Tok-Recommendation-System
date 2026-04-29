"""Marimo notebook: end-to-end view of the resubmission deliverables.

Walks the reader through:
  1. The temporal train/val/test split exported from the production datamart.
  2. NDCG@k / MRR@k / Recall@k on the held-out test set, retrieval-only and
     full pipeline.
  3. The video pipeline latency benchmark.
  4. The scoring regression tests.

Run with:

    pip install marimo
    marimo edit notebooks/resubmission_metrics.py

(Or `marimo run` for read-only HTML output.)
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _intro(mo):
    mo.md(
        """
        # Resubmission deliverables

        This notebook accompanies the four artifacts requested by the professor:
        evaluation metrics, a temporal split, a video benchmark, and unit tests.
        Each section loads the on-disk output of one of the scripts and renders
        it interactively, so a reviewer can verify the numbers without running
        anything heavy.
        """
    )
    return


@app.cell
def _imports():
    import json
    from pathlib import Path

    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    return REPO_ROOT, json, pd


@app.cell
def _splits_section(REPO_ROOT, json, mo, pd):
    mo.md("## 1. Temporal split")
    splits_meta_path = REPO_ROOT / "data" / "splits" / "splits_metadata.json"
    if not splits_meta_path.exists():
        mo.md(
            "_Run `python scripts/export_temporal_splits.py --respect-existing-splits` "
            "to generate the split files._"
        )
        return splits_meta_path,

    splits_meta = json.loads(splits_meta_path.read_text(encoding="utf-8"))
    splits_df = pd.DataFrame(splits_meta["splits"])
    mo.md(
        f"Source datamart: `{splits_meta['source_datamart']}` "
        f"(sha256 `{splits_meta['source_datamart_sha256'][:12]}...`)\n\n"
        f"- Leakage check: **{splits_meta['leakage_check']}**\n"
        f"- Temporal order: **{splits_meta['temporal_order_check']}**"
    )
    return splits_df, splits_meta, splits_meta_path


@app.cell
def _splits_table(mo, splits_df):
    mo.md("### Row counts and time ranges")
    return mo.ui.table(splits_df, selection=None)


@app.cell
def _eval_section(REPO_ROOT, json, mo, pd):
    mo.md("## 2. Evaluation metrics on the held-out test set")
    eval_path = REPO_ROOT / "evaluation_results.json"
    if not eval_path.exists():
        mo.md(
            "_Run `python scripts/evaluate_pipeline.py` to generate "
            "`evaluation_results.json`._"
        )
        return eval_path,

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    k_values = eval_payload["config"]["k_values"]
    rows = []
    for pass_name in ("retrieval_only", "full_pipeline"):
        pass_metrics = eval_payload[pass_name]
        for k in k_values:
            rows.append(
                {
                    "pass": pass_name,
                    "k": k,
                    "ndcg": pass_metrics[f"ndcg@{k}"],
                    "mrr": pass_metrics[f"mrr@{k}"],
                    "recall": pass_metrics[f"recall@{k}"],
                }
            )
    eval_df = pd.DataFrame(rows)
    mo.md(
        f"- Test queries used: **{eval_payload['queries_evaluated']}** "
        f"(of {eval_payload['test_set_size']})\n"
        f"- Candidate pool: **{eval_payload['candidate_pool_size']}** "
        f"rows (`train + validation`)\n"
        f"- Retriever: `{eval_payload['config']['retriever']}` "
        f"(reranker blend: cosine "
        f"{eval_payload['config']['ranker_blend']['cosine_weight']} + "
        f"engagement_z "
        f"{eval_payload['config']['ranker_blend']['engagement_weight']})"
    )
    return eval_df, eval_path, eval_payload, k_values


@app.cell
def _eval_table(eval_df, mo):
    mo.md("### Metrics per pass")
    return mo.ui.table(eval_df, selection=None)


@app.cell
def _eval_chart(eval_df, mo):
    """Quick visual: NDCG by K for each pass."""
    import altair as alt  # marimo has altair built in

    chart = (
        alt.Chart(eval_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="K"),
            y=alt.Y("ndcg:Q", title="NDCG@K"),
            color=alt.Color("pass:N", title=""),
        )
        .properties(width=420, height=240, title="NDCG@K, retrieval-only vs full pipeline")
    )
    return mo.ui.altair_chart(chart),


@app.cell
def _bench_section(REPO_ROOT, json, mo, pd):
    mo.md("## 3. Video pipeline latency")
    bench_path = REPO_ROOT / "benchmarks" / "video_pipeline_results.json"
    if not bench_path.exists():
        mo.md(
            "_Requires the ML deps installed locally. "
            "Run `python scripts/benchmark_video_pipeline.py` "
            "after `pip install -r requirements-service.txt` plus the video deps "
            "listed in the Dockerfile._"
        )
        return bench_path,

    bench_payload = json.loads(bench_path.read_text(encoding="utf-8"))
    branches_df = (
        pd.DataFrame(bench_payload["per_branch"])
        .T.reset_index()
        .rename(columns={"index": "branch"})
        .sort_values("mean_s", ascending=False)
    )
    videos_df = pd.DataFrame(bench_payload["per_video"])
    mo.md(
        f"- Videos benchmarked: **{bench_payload['videos_benchmarked']}**\n"
        f"- Optimisations applied: "
        + ", ".join(bench_payload["optimisations_applied"])
    )
    return bench_path, bench_payload, branches_df, videos_df


@app.cell
def _bench_tables(branches_df, mo, videos_df):
    mo.md("### Per-branch latency")
    branch_table = mo.ui.table(branches_df, selection=None)
    mo.md("### Per-video totals")
    video_table = mo.ui.table(videos_df[["video_id", "video_duration_s", "total_seconds", "branch_count"]], selection=None)
    return branch_table, video_table


@app.cell
def _tests_section(REPO_ROOT, mo):
    mo.md("## 4. Scoring regression tests")
    test_file = REPO_ROOT / "tests" / "test_scoring_regressions.py"
    if not test_file.exists():
        mo.md("_Test file not found._")
        return test_file,

    src = test_file.read_text(encoding="utf-8")
    test_names = [
        line.strip().split("(")[0].replace("def ", "")
        for line in src.splitlines()
        if line.strip().startswith("def test_")
    ]
    mo.md(
        "Tests are in `tests/test_scoring_regressions.py`. "
        f"Total: **{len(test_names)}**.\n\n"
        + "\n".join(f"- `{name}`" for name in test_names)
        + "\n\nRun with:\n\n```\npython -m pytest tests/test_scoring_regressions.py -v\n```"
    )
    return src, test_file, test_names


@app.cell
def _closing(mo):
    mo.md(
        """
        ## Reproducing everything

        ```bash
        # 1. Splits (writes data/splits/{train,val,test}.jsonl + metadata)
        python scripts/export_temporal_splits.py --respect-existing-splits

        # 2. Evaluation metrics (writes evaluation_results.{json,md})
        python scripts/evaluate_pipeline.py

        # 3. Video benchmark (writes benchmarks/video_pipeline_*.{json,md})
        python scripts/benchmark_video_pipeline.py

        # 4. Tests
        python -m pytest tests/test_scoring_regressions.py -v
        ```

        Each script is self-documenting (`--help` for options) and reuses the
        production scoring code in `src/recommendation/learning/evaluator.py`,
        so the metrics here are the same ones the training pipeline already
        emits internally.
        """
    )
    return


@app.cell
def _setup(mo):
    return mo


if __name__ == "__main__":
    app.run()
