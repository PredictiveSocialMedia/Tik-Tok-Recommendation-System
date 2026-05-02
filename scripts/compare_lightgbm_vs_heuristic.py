#!/usr/bin/env python3
"""
Train a LightGBM ranker on the project's temporal splits and compare
its held-out NDCG@k / MRR@k against a transparent hand-coded heuristic
baseline.

Implements suggestion #7 from the prof's email:
  "Implement LightGBM ranking trained on real scraped TikTok data and
  compare against the heuristic baseline."

Trains per-objective LightGBM regressors on
``data/splits/train.jsonl``, evaluates on ``data/splits/test.jsonl``,
and reports a paired bootstrap CI on (LightGBM − heuristic) lift for
each NDCG@k / MRR@k metric. Both rankers consume the same feature set,
so any difference is attributable to the model class, not the input.

Usage:
    python scripts/compare_lightgbm_vs_heuristic.py
    python scripts/compare_lightgbm_vs_heuristic.py --bootstrap-resamples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.ranker_comparison import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_K_VALUES,
    DEFAULT_OBJECTIVES,
    OBJECTIVE_TARGETS,
    LightGBMRankerConfig,
    compare_rankers,
    format_comparison_markdown,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Bad JSONL row in {path}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LightGBM ranker on the temporal train split and compare "
            "held-out NDCG@k / MRR@k against a hand-coded heuristic baseline."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "train.jsonl",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "test.jsonl",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(DEFAULT_OBJECTIVES),
        help=f"Objectives to compare (default: {' '.join(DEFAULT_OBJECTIVES)}).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[str(k) for k in DEFAULT_K_VALUES],
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=(
            f"Number of paired bootstrap resamples for the lift CI "
            f"(default: {DEFAULT_BOOTSTRAP_RESAMPLES})."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "recommender" / "lightgbm_ranker",
        help="Where to save trained per-objective LightGBM models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the comparison metrics as JSON.",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    for objective in args.objectives:
        if objective not in OBJECTIVE_TARGETS:
            raise SystemExit(
                f"Unknown objective {objective!r}; supported: {tuple(OBJECTIVE_TARGETS.keys())}"
            )
    try:
        k_values = [int(k) for k in args.k_values]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--k-values must be ints, got {args.k_values!r}") from exc

    if not args.train_path.exists():
        raise SystemExit(f"Train split not found: {args.train_path}")
    if not args.test_path.exists():
        raise SystemExit(f"Test split not found: {args.test_path}")

    train_rows = _load_jsonl(args.train_path)
    test_rows = _load_jsonl(args.test_path)
    print(f"Loaded {len(train_rows)} train rows from {args.train_path}.")
    print(f"Loaded {len(test_rows)} test rows from {args.test_path}.")
    print()

    config = LightGBMRankerConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        random_state=args.random_state,
    )

    metrics, trained = compare_rankers(
        train_rows=train_rows,
        test_rows=test_rows,
        objectives=tuple(args.objectives),
        k_values=tuple(k_values),
        n_resamples=int(args.bootstrap_resamples),
        config=config,
    )

    # Persist each per-objective model so reviewers / CI can inspect.
    for objective, ranker in trained.items():
        path = args.artifacts_root / objective / "model.pkl"
        ranker.save(path)
        print(f"Saved trained {objective} ranker -> {path}")
    print()

    # Save JSON FIRST (so a console-encoding error on the markdown print
    # doesn't lose the metrics).
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_path": str(args.train_path),
            "test_path": str(args.test_path),
            "n_train_rows": len(train_rows),
            "n_test_rows": len(test_rows),
            "objectives": list(args.objectives),
            "k_values": k_values,
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "config": {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "num_leaves": config.num_leaves,
                "learning_rate": config.learning_rate,
                "min_child_samples": config.min_child_samples,
                "random_state": config.random_state,
            },
            "per_objective": metrics,
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote comparison metrics to {args.output_json}.")
        print()

    print("Held-out comparison (test split):")
    print()
    try:
        print(format_comparison_markdown(metrics, k_values=k_values))
    except UnicodeEncodeError:
        # Windows cp1252 console can choke on Unicode chars in the
        # markdown — fall back to ASCII so the run finishes cleanly.
        # The full table is in the JSON output anyway.
        print(
            format_comparison_markdown(metrics, k_values=k_values).encode(
                "ascii", "replace"
            ).decode("ascii")
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
