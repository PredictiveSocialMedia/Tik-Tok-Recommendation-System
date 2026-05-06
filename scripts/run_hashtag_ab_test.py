#!/usr/bin/env python3
"""Compare the production hashtag recommender against a TF-IDF baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.hashtag_recommender import HashtagRecommender  # noqa: E402
from src.recommendation.learning.hashtag_ab_testing import (  # noqa: E402
    HashtagVariant,
    TfidfHashtagBaseline,
    evaluate_hashtag_variants,
    format_hashtag_ab_markdown,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data" / "splits" / "train.jsonl")
    parser.add_argument("--test-path", type=Path, default=REPO_ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--recommender-dir", type=Path, default=REPO_ROOT / "artifacts" / "hashtag_recommender")
    parser.add_argument("--k-values", nargs="+", default=["5", "10"])
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "hashtag_ab_test_report.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "hashtag_ab_test_report.md")
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Allow SentenceTransformer to contact model hubs instead of requiring local cached files.",
    )
    args = parser.parse_args()

    train_rows = _load_jsonl(args.train_path)
    test_rows = _load_jsonl(args.test_path)
    if args.max_test_rows is not None:
        test_rows = test_rows[: max(1, int(args.max_test_rows))]
    k_values = [int(item) for item in args.k_values]

    tfidf = TfidfHashtagBaseline(train_rows)
    if not args.recommender_dir.exists():
        raise SystemExit(f"Hashtag recommender artifact not found: {args.recommender_dir}")
    recommender = HashtagRecommender.load(
        args.recommender_dir,
        local_files_only=not args.allow_model_downloads,
    )

    variants = [
        HashtagVariant(
            name="tfidf_baseline",
            description="Lexical nearest-neighbour TF-IDF over train split captions.",
            recommend=lambda row, k: tfidf.recommend(row, k=k),
        ),
        HashtagVariant(
            name="hashtag_recommender",
            description="Production SBERT/FAISS semantic hashtag recommender artifact.",
            recommend=lambda row, k: [
                str(item.get("hashtag") or "")
                for item in recommender.recommend(str(row.get("caption") or ""), top_n=k, k=25)
            ],
        ),
    ]
    report = evaluate_hashtag_variants(variants, test_rows, k_values=k_values)
    report["train_path"] = str(args.train_path)
    report["test_path"] = str(args.test_path)
    report["recommender_dir"] = str(args.recommender_dir)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(format_hashtag_ab_markdown(report), encoding="utf-8")
    print(format_hashtag_ab_markdown(report))
    print(f"Wrote {args.output_json} and {args.output_md}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
