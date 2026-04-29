"""Marimo notebook: end-to-end view of the resubmission deliverables.

Loads the on-disk outputs of the four resubmission scripts and renders
them interactively (markdown + tables + an Altair chart).

Run with:
    pip install marimo altair pandas
    marimo edit notebooks/resubmission_metrics.py
    # Or for read-only HTML:
    marimo export html notebooks/resubmission_metrics.py -o notebooks/resubmission_metrics.html
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _intro(mo):
    intro = mo.md(
        """
        # Resubmission deliverables

        This notebook accompanies the four artifacts requested by the professor:
        evaluation metrics, a temporal split, a video benchmark, and unit tests.
        Each section loads the on-disk output of one of the scripts and renders
        it so a reviewer can verify the numbers without running anything heavy.
        """
    )
    return (intro,)


@app.cell
def _imports():
    import json
    from pathlib import Path

    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    return REPO_ROOT, json, pd


@app.cell
def _splits_section(REPO_ROOT, json, mo, pd):
    splits_meta_path = REPO_ROOT / "data" / "splits" / "splits_metadata.json"
    if splits_meta_path.exists():
        splits_meta = json.loads(splits_meta_path.read_text(encoding="utf-8"))
        splits_df = pd.DataFrame(splits_meta["splits"])
        splits_md = mo.md(
            f"## 1. Temporal split\n\n"
            f"Source datamart: `{splits_meta['source_datamart']}` "
            f"(sha256 `{splits_meta['source_datamart_sha256'][:12]}...`)\n\n"
            f"- Leakage check: **{splits_meta['leakage_check']}**\n"
            f"- Temporal order: **{splits_meta['temporal_order_check']}**"
        )
    else:
        splits_meta = None
        splits_df = pd.DataFrame()
        splits_md = mo.md(
            "## 1. Temporal split\n\n"
            "_Run `python scripts/export_temporal_splits.py --respect-existing-splits` "
            "to generate the split files._"
        )
    splits_md
    return splits_df, splits_md, splits_meta


@app.cell
def _splits_table(mo, splits_df):
    splits_table = (
        mo.ui.table(splits_df, selection=None)
        if not splits_df.empty
        else mo.md("_(no split data on disk)_")
    )
    splits_table
    return (splits_table,)


@app.cell
def _eval_section(REPO_ROOT, json, mo, pd):
    eval_path = REPO_ROOT / "evaluation_results.json"
    if eval_path.exists():
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
        blend = eval_payload["config"]["ranker_blend"]
        eval_md = mo.md(
            f"## 2. Evaluation metrics on the held-out test set\n\n"
            f"- Test queries used: **{eval_payload['queries_evaluated']}** "
            f"(of {eval_payload['test_set_size']})\n"
            f"- Candidate pool: **{eval_payload['candidate_pool_size']}** "
            f"rows (`train + validation`)\n"
            f"- Retriever: `{eval_payload['config']['retriever']}` "
            f"(reranker blend: cosine {blend['cosine_weight']} + "
            f"engagement_z {blend['engagement_weight']})"
        )
    else:
        eval_payload = None
        eval_df = pd.DataFrame()
        eval_md = mo.md(
            "## 2. Evaluation metrics on the held-out test set\n\n"
            "_Run `python scripts/evaluate_pipeline.py` to generate "
            "`evaluation_results.json`._"
        )
    eval_md
    return eval_df, eval_md, eval_payload


@app.cell
def _eval_table(mo, eval_df):
    eval_table = (
        mo.ui.table(eval_df, selection=None)
        if not eval_df.empty
        else mo.md("_(no evaluation results on disk)_")
    )
    eval_table
    return (eval_table,)


@app.cell
def _eval_chart(mo, eval_df):
    if eval_df.empty:
        chart_view = mo.md("_(no chart -- run evaluate_pipeline.py first)_")
    else:
        import altair as alt
        chart = (
            alt.Chart(eval_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("k:O", title="K"),
                y=alt.Y("ndcg:Q", title="NDCG@K"),
                color=alt.Color("pass:N", title=""),
            )
            .properties(
                width=440,
                height=260,
                title="NDCG@K -- retrieval-only vs full pipeline",
            )
        )
        chart_view = mo.ui.altair_chart(chart)
    chart_view
    return (chart_view,)


@app.cell
def _bench_section(REPO_ROOT, json, mo, pd):
    bench_path = REPO_ROOT / "benchmarks" / "video_pipeline_results.json"
    if bench_path.exists():
        bench_payload = json.loads(bench_path.read_text(encoding="utf-8"))
        branches_df = (
            pd.DataFrame(bench_payload["per_branch"])
            .T.reset_index()
            .rename(columns={"index": "branch"})
            .sort_values("mean_s", ascending=False)
        )
        videos_df = pd.DataFrame(bench_payload["per_video"])
        bench_md = mo.md(
            f"## 3. Video pipeline latency\n\n"
            f"- Videos benchmarked: **{bench_payload['videos_benchmarked']}**\n"
            f"- Optimisations applied: "
            + ", ".join(bench_payload["optimisations_applied"])
        )
    else:
        bench_payload = None
        branches_df = pd.DataFrame()
        videos_df = pd.DataFrame()
        bench_md = mo.md(
            "## 3. Video pipeline latency\n\n"
            "_Requires the ML deps installed locally. "
            "Run `python scripts/benchmark_video_pipeline.py` after "
            "`pip install -r requirements-service.txt` plus the video deps "
            "listed in the Dockerfile._"
        )
    bench_md
    return bench_md, bench_payload, branches_df, videos_df


@app.cell
def _bench_tables(mo, branches_df, videos_df):
    if branches_df.empty:
        bench_view = mo.md("_(no benchmark output on disk)_")
    else:
        bench_view = mo.vstack(
            [
                mo.md("### Per-branch latency"),
                mo.ui.table(branches_df, selection=None),
                mo.md("### Per-video totals"),
                mo.ui.table(
                    videos_df[
                        ["video_id", "video_duration_s", "total_seconds", "branch_count"]
                    ],
                    selection=None,
                ),
            ]
        )
    bench_view
    return (bench_view,)


@app.cell
def _tests_section(REPO_ROOT, mo):
    test_file = REPO_ROOT / "tests" / "test_scoring_regressions.py"
    if test_file.exists():
        src = test_file.read_text(encoding="utf-8")
        test_names = [
            line.strip().split("(")[0].replace("def ", "")
            for line in src.splitlines()
            if line.strip().startswith("def test_")
        ]
        tests_md = mo.md(
            "## 4. Scoring regression tests\n\n"
            f"Tests are in `tests/test_scoring_regressions.py`. "
            f"Total: **{len(test_names)}**.\n\n"
            + "\n".join(f"- `{name}`" for name in test_names)
            + "\n\nRun with:\n\n```\npython -m pytest tests/test_scoring_regressions.py -v\n```"
        )
    else:
        tests_md = mo.md("## 4. Scoring regression tests\n\n_Test file not found._")
    tests_md
    return (tests_md,)


@app.cell
def _closing(mo):
    closing = mo.md(
        """
        ## Reproducing everything

        ```bash
        python scripts/export_temporal_splits.py --respect-existing-splits
        python scripts/evaluate_pipeline.py
        python scripts/benchmark_video_pipeline.py
        python -m pytest tests/test_scoring_regressions.py -v
        ```

        Each script is self-documenting (`--help` for options) and reuses the
        production scoring code in `src/recommendation/learning/evaluator.py`,
        so the metrics here are the same ones the training pipeline already
        emits internally.
        """
    )
    closing
    return (closing,)


@app.cell
def _setup():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
