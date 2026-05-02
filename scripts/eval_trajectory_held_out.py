#!/usr/bin/env python3
"""
Held-out evaluation runner for the trajectory-aware ranking module.

Implements suggestion #1 from the prof's email:
  "Add proper evaluation metrics (NDCG@k, MRR@k) to the trajectory-aware
  ranking and report numbers on a held-out set."

Loads the held-out test split (default: ``data/splits/test.jsonl``),
calls ``evaluate_trajectory_held_out`` for the configured objectives,
and prints a markdown summary table. Optionally saves the metrics dict
to JSON so the numbers can be checked into ``artifacts/``.

Usage:
    python scripts/eval_trajectory_held_out.py
    python scripts/eval_trajectory_held_out.py --splits-path data/splits/test.jsonl
    python scripts/eval_trajectory_held_out.py --output-json artifacts/control_plane/trajectory_eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.trajectory_eval import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_OBJECTIVES,
    evaluate_trajectory_held_out,
    format_metrics_markdown,
    summarize_metrics,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSONL row in {path}: {exc}") from exc
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _parse_int_list(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"--k-values entries must be integers, got {value!r}") from exc
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trajectory-aware ranking on a held-out split and print "
            "NDCG@k / MRR@k per objective."
        )
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "test.jsonl",
        help="Path to the held-out JSONL split (default: data/splits/test.jsonl).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional path to write the metrics dict as JSON. Parents are "
            "created automatically."
        ),
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(DEFAULT_OBJECTIVES),
        help="Objectives to evaluate (default: reach engagement conversion).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[str(k) for k in DEFAULT_K_VALUES],
        help="Cut-offs for NDCG@k / MRR@k (default: 10 20).",
    )
    args = parser.parse_args(argv)

    if not args.splits_path.exists():
        raise SystemExit(f"Split file not found: {args.splits_path}")

    rows = _load_jsonl(args.splits_path)
    k_values = _parse_int_list(args.k_values)
    metrics = evaluate_trajectory_held_out(
        rows, objectives=tuple(args.objectives), k_values=tuple(k_values)
    )

    print(f"Loaded {len(rows)} rows from {args.splits_path}.")
    print()
    print("Per-objective metrics:")
    print()
    print(format_metrics_markdown(metrics, k_values=k_values))
    print()
    summary = summarize_metrics(metrics)
    if summary:
        print("Macro-average across objectives:")
        for key in sorted(summary.keys()):
            print(f"  {key}: {summary[key]:.4f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "splits_path": (args.splits_path.relative_to(REPO_ROOT) if args.splits_path.is_relative_to(REPO_ROOT) else args.splits_path).as_posix(),
            "n_rows_loaded": len(rows),
            "objectives": list(args.objectives),
            "k_values": list(k_values),
            "per_objective": metrics,
            "macro_average": summary,
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
