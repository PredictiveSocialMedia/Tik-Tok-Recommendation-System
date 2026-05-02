#!/usr/bin/env python3
"""
Run a systematic A/B test of three ranker variants on the held-out
test split, for every supported objective.

Implements suggestion #8 from the prof's email:
  "Add A/B testing infrastructure to compare ranking approaches
  systematically."

Variants compared (per objective):
  1. ``random``        — uniform random scores; sanity-check baseline
  2. ``heuristic_v1``  — transparent hand-coded rules over metadata
  3. ``gbr``           — sklearn GradientBoostingRegressor trained
                         inline on the train split

For each objective (reach, engagement, conversion) we report:
  - per-variant absolute NDCG@k / MRR@k on the held-out test rows
  - all-pairs paired-bootstrap 95% CI on the lift between every pair

Outputs a JSON report and prints a markdown summary.

Usage:
    python scripts/run_ab_test.py
    python scripts/run_ab_test.py --bootstrap-resamples 2000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.ab_testing import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_K_VALUES,
    RankerVariant,
    assign_relevance_grades,
    format_report_markdown,
    run_ab_test,
)

CONTENT_TYPES = ("general", "tutorial", "review", "story", "other")
OBJECTIVES = ("reach", "engagement", "conversion")


# ---------------------------------------------------------------------------
# Row + feature helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _parse_posted_at(value: Any) -> Tuple[float, float]:
    if not isinstance(value, str):
        return (-1.0, -1.0)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return (-1.0, -1.0)
    return (float(dt.hour), float(dt.weekday()))


def _features(row: Dict[str, Any]) -> Dict[str, float]:
    payload = row.get("features")
    metadata = payload if isinstance(payload, dict) else {}
    caption = str(row.get("caption") or "")
    content_type = str(row.get("content_type") or "").lower()
    if content_type not in CONTENT_TYPES:
        content_type = "other"
    posted_hour, posted_dow = _parse_posted_at(row.get("posted_at"))
    out: Dict[str, float] = {
        "caption_word_count": _as_float(metadata.get("caption_word_count")),
        "caption_length_chars": float(len(caption)),
        "caption_has_question": 1.0 if "?" in caption else 0.0,
        "hashtag_count": _as_float(metadata.get("hashtag_count")),
        "keyword_count": _as_float(metadata.get("keyword_count")),
        "duration_seconds": _as_float(metadata.get("duration_seconds")),
        "posted_hour": posted_hour,
        "posted_day_of_week": posted_dow,
    }
    for ct in CONTENT_TYPES:
        out[f"content_type_{ct}"] = 1.0 if content_type == ct else 0.0
    return out


def _extract_target_z(row: Dict[str, Any], objective: str) -> Optional[float]:
    targets_z = row.get("targets_z")
    if not isinstance(targets_z, dict):
        return None
    raw = targets_z.get(objective)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _candidate_id(row: Dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("video_id") or "")


def _filter_rows(
    rows: Sequence[Dict[str, Any]],
    objective: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[float]]:
    """Drop rows missing the objective target or a candidate id."""
    keep_rows: List[Dict[str, Any]] = []
    keep_ids: List[str] = []
    keep_z: List[float] = []
    for row in rows:
        cid = _candidate_id(row)
        z = _extract_target_z(row, objective)
        if not cid or z is None:
            continue
        keep_rows.append(row)
        keep_ids.append(cid)
        keep_z.append(z)
    return keep_rows, keep_ids, keep_z


# ---------------------------------------------------------------------------
# Variant scorers
# ---------------------------------------------------------------------------


def _random_scorer(seed: int) -> Any:
    rng = np.random.default_rng(seed)

    def _score(rows: Sequence[Dict[str, Any]]) -> List[float]:
        return list(rng.uniform(0.0, 1.0, size=len(rows)))

    return _score


def _heuristic_score(features: Dict[str, float], objective: str) -> float:
    """
    Transparent hand-coded rules per objective. Same heuristic family as
    the one shipped in PR #75 (suggestion #7), kept self-contained here
    so this script doesn't depend on that PR being merged.
    """
    f = features
    if objective == "reach":
        return (
            0.50 * f.get("hashtag_count", 0.0)
            + 0.10 * f.get("caption_word_count", 0.0)
            + 0.30 * f.get("content_type_general", 0.0)
            - 0.02 * abs(f.get("duration_seconds", 0.0) - 25.0)
        )
    if objective == "engagement":
        wc = f.get("caption_word_count", 0.0)
        length_bonus = -0.05 * (wc - 10.0) ** 2
        return (
            0.40 * f.get("caption_has_question", 0.0)
            + 0.50 * f.get("content_type_tutorial", 0.0)
            + 0.20 * f.get("content_type_review", 0.0)
            + 0.10 * f.get("hashtag_count", 0.0)
            + length_bonus
        )
    if objective == "conversion":
        hour = f.get("posted_hour", -1.0)
        hour_bonus = max(0.0, 1.0 - abs(hour - 19.0) / 6.0) if hour >= 0.0 else 0.0
        return (
            0.60 * f.get("content_type_review", 0.0)
            + 0.45 * f.get("content_type_tutorial", 0.0)
            + 0.20 * f.get("caption_has_question", 0.0)
            + 0.15 * hour_bonus
            - 0.02 * f.get("caption_word_count", 0.0)
        )
    return 0.0


def _heuristic_scorer(objective: str) -> Any:
    def _score(rows: Sequence[Dict[str, Any]]) -> List[float]:
        return [_heuristic_score(_features(row), objective) for row in rows]

    return _score


def _gbr_scorer(
    train_rows: Sequence[Dict[str, Any]],
    objective: str,
    random_state: int,
) -> Any:
    """Train a small GradientBoostingRegressor on the train split for ``objective``."""
    from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

    feature_names: Optional[List[str]] = None
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in train_rows:
        z = _extract_target_z(row, objective)
        if z is None:
            continue
        feats = _features(row)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        X_rows.append([feats[name] for name in feature_names])
        y_rows.append(z)
    if feature_names is None or not X_rows:
        raise SystemExit(
            f"Cannot train GBR for objective {objective!r}: no usable train rows."
        )
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=15,
        random_state=random_state,
    )
    model.fit(np.asarray(X_rows), np.asarray(y_rows))

    def _score(rows: Sequence[Dict[str, Any]]) -> List[float]:
        x_list: List[List[float]] = []
        for row in rows:
            feats = _features(row)
            x_list.append([feats.get(name, 0.0) for name in feature_names])
        if not x_list:
            return []
        return list(model.predict(np.asarray(x_list)))

    return _score


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a systematic 3-variant A/B test (random / heuristic / GBR) "
            "for every objective on the held-out test split."
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
        default=list(OBJECTIVES),
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
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    if not args.train_path.exists():
        raise SystemExit(f"Train split not found: {args.train_path}")
    if not args.test_path.exists():
        raise SystemExit(f"Test split not found: {args.test_path}")
    try:
        k_values = [int(k) for k in args.k_values]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--k-values must be ints: {args.k_values!r}") from exc

    train_rows = _load_jsonl(args.train_path)
    test_rows = _load_jsonl(args.test_path)
    print(f"Loaded {len(train_rows)} train rows, {len(test_rows)} test rows.")
    print()

    per_objective: Dict[str, Dict[str, Any]] = {}
    for objective in args.objectives:
        if objective not in OBJECTIVES:
            print(f"Skipping unsupported objective: {objective}")
            continue
        eval_rows, ids, z_values = _filter_rows(test_rows, objective)
        if not eval_rows:
            print(f"Skipping {objective}: no eval rows after filtering.")
            continue
        grades = assign_relevance_grades(z_values)

        variants = [
            RankerVariant(
                name="random",
                score=_random_scorer(seed=args.random_state),
                description="Uniform random scores — sanity-check baseline.",
            ),
            RankerVariant(
                name="heuristic_v1",
                score=_heuristic_scorer(objective),
                description="Transparent hand-coded per-objective rules over metadata.",
            ),
            RankerVariant(
                name="gbr",
                score=_gbr_scorer(
                    train_rows, objective, random_state=args.random_state
                ),
                description=(
                    "sklearn GradientBoostingRegressor trained on the train "
                    "split predicting targets_z[objective]."
                ),
            ),
        ]

        report = run_ab_test(
            variants,
            eval_rows,
            ids,
            grades,
            k_values=tuple(k_values),
            n_resamples=int(args.bootstrap_resamples),
            random_state=args.random_state,
        )
        per_objective[objective] = report

    payload = {
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "objectives": list(args.objectives),
        "k_values": k_values,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "per_objective": per_objective,
    }

    # Save JSON first so a console-encoding hiccup on the markdown print
    # never loses the run output.
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote A/B test report to {args.output_json}.")
        print()

    for objective, report in per_objective.items():
        print(f"=== Objective: {objective} ===")
        print()
        try:
            print(format_report_markdown(report))
        except UnicodeEncodeError:
            print(
                format_report_markdown(report).encode("ascii", "replace").decode("ascii")
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
