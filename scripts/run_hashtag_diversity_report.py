#!/usr/bin/env python3
"""Measure hashtag recommender diversity before and after MMR reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.hashtag_recommender import (  # noqa: E402
    HashtagRecommender,
    hashtag_jaccard,
)
from src.recommendation.learning.hashtag_ab_testing import extract_hashtags  # noqa: E402


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _pairwise_similarities(tags: Sequence[str]) -> List[float]:
    sims: List[float] = []
    for i, left in enumerate(tags):
        for right in tags[i + 1:]:
            sims.append(hashtag_jaccard(left, right))
    return sims


def diversity_metrics(recommendations: Sequence[Sequence[str]], ground_truth: Sequence[Sequence[str]]) -> Dict[str, float]:
    unique_predicted = {tag for tags in recommendations for tag in tags}
    unique_truth = {tag for tags in ground_truth for tag in tags}
    sims = [value for tags in recommendations for value in _pairwise_similarities(tags)]
    redundant = [value for value in sims if value >= 0.45]
    return {
        "unique_tags": float(len(unique_predicted)),
        "ground_truth_coverage": (
            len(unique_predicted & unique_truth) / max(1, len(unique_truth))
        ),
        "avg_pairwise_similarity": sum(sims) / max(1, len(sims)),
        "redundant_tag_rate": len(redundant) / max(1, len(sims)),
    }


def format_diversity_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Hashtag diversity report",
        "",
        f"- Rows evaluated: **{report['rows_evaluated']}**",
        f"- Top N: **{report['top_n']}**",
        "",
        "| Variant | Unique Tags | Ground-Truth Coverage | Avg Tag Similarity | Redundant Tag Rate |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in report["variants"].items():
        lines.append(
            f"| {name} | {metrics['unique_tags']:.0f} | "
            f"{metrics['ground_truth_coverage']:.3f} | "
            f"{metrics['avg_pairwise_similarity']:.3f} | "
            f"{metrics['redundant_tag_rate']:.3f} |"
        )
    delta = report.get("delta") or {}
    if delta:
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- Unique tags: {delta['unique_tags']:+.0f}",
                f"- Ground-truth coverage: {delta['ground_truth_coverage']:+.3f}",
                f"- Average tag similarity: {delta['avg_pairwise_similarity']:+.3f}",
                f"- Redundant tag rate: {delta['redundant_tag_rate']:+.3f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=REPO_ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--recommender-dir", type=Path, default=REPO_ROOT / "artifacts" / "hashtag_recommender")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--neighbours", type=int, default=25)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "hashtag_diversity_report.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "hashtag_diversity_report.md")
    parser.add_argument("--allow-model-downloads", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(args.test_path)
    if args.max_test_rows is not None:
        rows = rows[: max(1, int(args.max_test_rows))]
    recommender = HashtagRecommender.load(
        args.recommender_dir,
        local_files_only=not args.allow_model_downloads,
    )

    before: List[List[str]] = []
    after: List[List[str]] = []
    truth: List[List[str]] = []
    for row in rows:
        caption = str(row.get("caption") or "")
        actual = extract_hashtags(row)
        if not caption.strip() or not actual:
            continue
        truth.append(actual)
        before.append(
            [
                str(item.get("hashtag") or "")
                for item in recommender.recommend(
                    caption,
                    k=args.neighbours,
                    top_n=args.top_n,
                    diversity_weight=0.0,
                )
            ]
        )
        after.append(
            [
                str(item.get("hashtag") or "")
                for item in recommender.recommend(
                    caption,
                    k=args.neighbours,
                    top_n=args.top_n,
                    diversity_weight=0.5,
                )
            ]
        )

    before_metrics = diversity_metrics(before, truth)
    after_metrics = diversity_metrics(after, truth)
    report = {
        "rows_evaluated": len(truth),
        "top_n": args.top_n,
        "test_path": str(args.test_path),
        "recommender_dir": str(args.recommender_dir),
        "variants": {
            "before_no_mmr": before_metrics,
            "after_mmr": after_metrics,
        },
        "delta": {
            key: after_metrics[key] - before_metrics[key]
            for key in before_metrics.keys()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(format_diversity_markdown(report), encoding="utf-8")
    print(format_diversity_markdown(report))
    print(f"Wrote {args.output_json} and {args.output_md}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
