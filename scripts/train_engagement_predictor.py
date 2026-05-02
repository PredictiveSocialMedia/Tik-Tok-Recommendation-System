#!/usr/bin/env python3
"""
Train the metadata-based engagement predictor on the project's
temporal splits and report held-out metrics.

Trains one regressor per target (default: log_views, engagement_rate),
saves each to ``artifacts/recommender/engagement_predictor/<target>/model.pkl``,
evaluates on the held-out test split, and writes a metrics JSON +
markdown summary that can be pasted into a PR description.

Usage:
    python scripts/train_engagement_predictor.py
    python scripts/train_engagement_predictor.py --targets log_views engagement_rate
    python scripts/train_engagement_predictor.py --output-json artifacts/control_plane/engagement_predictor_metrics.json
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

from src.recommendation.learning.engagement_predictor import (  # noqa: E402
    DEFAULT_TARGETS,
    SUPPORTED_TARGETS,
    EngagementPredictor,
    EngagementPredictorConfig,
    baseline_metrics,
    format_metrics_markdown,
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
                raise ValueError(f"Bad JSONL row in {path}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train metadata-based engagement predictors and report held-out "
            "MAE / RMSE / R² / Spearman against a constant-mean baseline."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "train.jsonl",
        help="Path to the training split (default: data/splits/train.jsonl).",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "test.jsonl",
        help="Path to the held-out test split (default: data/splits/test.jsonl).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help=f"Targets to train (default: {' '.join(DEFAULT_TARGETS)}).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "recommender" / "engagement_predictor",
        help="Where to save trained models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the metrics dict as JSON.",
    )
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    for target in args.targets:
        if target not in SUPPORTED_TARGETS:
            raise SystemExit(
                f"Unknown target {target!r}; supported: {SUPPORTED_TARGETS}"
            )

    if not args.train_path.exists():
        raise SystemExit(f"Train split not found: {args.train_path}")
    if not args.test_path.exists():
        raise SystemExit(f"Test split not found: {args.test_path}")

    try:
        train_rows = _load_jsonl(args.train_path)
        test_rows = _load_jsonl(args.test_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(train_rows)} train rows from {args.train_path}.")
    print(f"Loaded {len(test_rows)} test rows from {args.test_path}.")
    print()

    config = EngagementPredictorConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    per_target: Dict[str, Dict[str, Dict[str, float]]] = {}
    for target in args.targets:
        print(f"Training target: {target} ...")
        predictor = EngagementPredictor(target=target, config=config)
        predictor.fit(train_rows)

        target_dir = args.artifacts_root / target
        model_path = target_dir / "model.pkl"
        predictor.save(model_path)

        held_out = predictor.evaluate(test_rows)
        baseline = baseline_metrics(test_rows, target=target, train_rows=train_rows)
        per_target[target] = {"trained": held_out, "baseline": baseline}
        print(
            f"  saved -> {model_path} | "
            f"trained MAE={held_out['mae']:.4f} R²={held_out['r2']:.4f} | "
            f"baseline MAE={baseline['mae']:.4f}"
        )

    print()
    print("Held-out metrics (test split):")
    print()
    print(format_metrics_markdown(per_target))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_path": (args.train_path.relative_to(REPO_ROOT) if args.train_path.is_relative_to(REPO_ROOT) else args.train_path).as_posix(),
            "test_path": (args.test_path.relative_to(REPO_ROOT) if args.test_path.is_relative_to(REPO_ROOT) else args.test_path).as_posix(),
            "n_train_rows": len(train_rows),
            "n_test_rows": len(test_rows),
            "targets": list(args.targets),
            "config": {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "min_samples_leaf": config.min_samples_leaf,
                "random_state": config.random_state,
            },
            "per_target": per_target,
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Wrote metrics to {args.output_json}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
