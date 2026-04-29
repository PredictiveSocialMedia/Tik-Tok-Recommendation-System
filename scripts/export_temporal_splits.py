#!/usr/bin/env python3
"""Export the production training datamart into on-disk train/val/test JSONL splits.

The split itself is delegated to ``datamart._split_rows_by_time`` so this stays in
lockstep with what ``train_full_pipeline.py`` actually trains on. The script adds
three things on top:

* Writes one JSONL file per split under ``data/splits/`` (one row per line).
* Validates that the splits are temporally ordered AND that no ``video_id`` appears
  in more than one split (true leakage check, not just row-id check).
* Emits ``data/splits/splits_metadata.json`` with row counts, posted_at /
  as_of_time ranges, the source datamart sha256 and the generation timestamp so
  the splits can be cited reproducibly in the report.

Usage::

    python scripts/export_temporal_splits.py
    python scripts/export_temporal_splits.py \\
        --datamart data/real/training_datamart.json \\
        --output-dir data/splits \\
        --train-ratio 0.70 --validation-ratio 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.datamart import _split_rows_by_time  # noqa: E402


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarise(split_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"split": split_name, "n_rows": 0, "n_unique_videos": 0}
    posted = sorted(r.get("posted_at", "") for r in rows if r.get("posted_at"))
    as_of = sorted(r.get("as_of_time", "") for r in rows if r.get("as_of_time"))
    video_ids = {str(r.get("video_id")) for r in rows if r.get("video_id")}
    return {
        "split": split_name,
        "n_rows": len(rows),
        "n_unique_videos": len(video_ids),
        "posted_at_min": posted[0] if posted else None,
        "posted_at_max": posted[-1] if posted else None,
        "as_of_time_min": as_of[0] if as_of else None,
        "as_of_time_max": as_of[-1] if as_of else None,
    }


def _assert_no_video_leakage(buckets: Dict[str, List[Dict[str, Any]]]) -> None:
    """No ``video_id`` may appear in more than one split."""
    sets: Dict[str, set] = {
        name: {str(r.get("video_id")) for r in rows if r.get("video_id")}
        for name, rows in buckets.items()
    }
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for left, right in pairs:
        overlap = sets[left] & sets[right]
        if overlap:
            sample = sorted(overlap)[:5]
            raise RuntimeError(
                f"Leakage detected: {len(overlap)} video_id(s) appear in both "
                f"'{left}' and '{right}' splits (sample: {sample})."
            )


def _assert_temporal_order(buckets: Dict[str, List[Dict[str, Any]]]) -> None:
    """train.max(as_of) <= val.min(as_of) <= test.min(as_of)."""

    def _bounds(name: str) -> Tuple[str, str]:
        rows = buckets.get(name, [])
        times = sorted(r.get("as_of_time", "") for r in rows if r.get("as_of_time"))
        if not times:
            return ("", "")
        return (times[0], times[-1])

    tr_min, tr_max = _bounds("train")
    va_min, va_max = _bounds("validation")
    te_min, te_max = _bounds("test")

    if tr_max and va_min and tr_max > va_min:
        raise RuntimeError(
            f"Temporal order violated: train.max(as_of_time)={tr_max} "
            f"> validation.min(as_of_time)={va_min}."
        )
    if va_max and te_min and va_max > te_min:
        raise RuntimeError(
            f"Temporal order violated: validation.max(as_of_time)={va_max} "
            f"> test.min(as_of_time)={te_min}."
        )


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datamart",
        type=Path,
        default=REPO_ROOT / "data" / "real" / "training_datamart.json",
        help="Path to the training datamart JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits",
        help="Directory to write JSONL splits + metadata into.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--respect-existing-splits",
        action="store_true",
        help=(
            "If the datamart already carries a 'split' field on each row "
            "(true for production datamarts), use those instead of recomputing."
        ),
    )
    args = parser.parse_args()

    if not args.datamart.exists():
        print(f"Datamart not found: {args.datamart}", file=sys.stderr)
        return 1

    payload = json.loads(args.datamart.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = payload.get("rows", [])
    if not rows:
        print("Datamart has no rows.", file=sys.stderr)
        return 1

    has_existing = all(isinstance(r.get("split"), str) for r in rows)
    if not args.respect_existing_splits or not has_existing:
        _split_rows_by_time(
            rows,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )

    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = row.get("split", "test")
        buckets.setdefault(split, []).append(row)

    _assert_temporal_order(buckets)
    _assert_no_video_leakage(buckets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        name: _write_jsonl(buckets[name], args.output_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_datamart": str(args.datamart.relative_to(REPO_ROOT)),
        "source_datamart_sha256": _sha256_of_file(args.datamart),
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": round(1.0 - args.train_ratio - args.validation_ratio, 4),
        },
        "totals": {
            "rows_in_datamart": len(rows),
            "rows_written": written,
        },
        "splits": [
            _summarise("train", buckets["train"]),
            _summarise("validation", buckets["validation"]),
            _summarise("test", buckets["test"]),
        ],
        "leakage_check": "passed: no video_id appears in more than one split",
        "temporal_order_check": "passed: as_of_time is monotonically non-decreasing across train -> val -> test",
    }
    (args.output_dir / "splits_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for split in ("train", "validation", "test"):
        s = next(s for s in metadata["splits"] if s["split"] == split)
        print(
            f"  {split:<11s} n={s['n_rows']:>5d}  unique_videos={s['n_unique_videos']:>5d}  "
            f"as_of_time {s.get('as_of_time_min')} -> {s.get('as_of_time_max')}"
        )
    print(f"\nWrote splits + metadata to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
