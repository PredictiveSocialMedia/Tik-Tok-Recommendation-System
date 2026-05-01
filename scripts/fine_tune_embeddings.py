#!/usr/bin/env python3
"""Fine-tune a sentence transformer on TikTok caption and hashtag data.

Usage (internal row format):
    python scripts/fine_tune_embeddings.py --data-path data/rows.json

Usage (raw scraper output):
    python scripts/fine_tune_embeddings.py --data-path scraper/output.json

The script auto-detects the data format. If rows carry a "split" field
(train/validation/test) it honours it; otherwise it performs a temporal
split ordered by as_of_time (80 / 10 / 10).

After training the fine-tuned model is saved to --output-dir. To use it
in the retriever, set dense_model_name to that path in
HybridRetrieverTrainerConfig.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        data = [data]
    rows = []
    for item in data:
        if "row_id" in item or ("video_id" in item and "topic_key" in item):
            rows.append(item)
        else:
            rows.append(_scraped_to_row(item))
    return rows


def _scraped_to_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw scraped TikTok JSON to the internal row schema."""
    meta = item.get("metadata") or {}
    stats = meta.get("stats") or {}
    author = meta.get("author") or {}
    hashtags = [str(h).lstrip("#") for h in (meta.get("hashtags") or [])]
    caption = str(item.get("caption") or item.get("video_caption") or "")

    # Pull hashtags out of caption if not present in metadata
    if not hashtags:
        hashtags = [
            w.lstrip("#") for w in caption.split() if w.startswith("#")
        ]

    video_id = str(meta.get("id") or item.get("asset_id") or "")
    topic_key = hashtags[0] if hashtags else "unknown"

    create_time = meta.get("createTime")
    try:
        posted_at = (
            datetime.fromtimestamp(int(create_time), tz=timezone.utc).isoformat()
            if create_time
            else "2026-01-01T00:00:00Z"
        )
    except (ValueError, TypeError):
        posted_at = "2026-01-01T00:00:00Z"

    return {
        "row_id": video_id or f"row_{hash(caption) & 0xFFFFFF}",
        "video_id": video_id,
        "caption": caption,
        "hashtags": hashtags,
        "keywords": list(item.get("analysis", {}).get("keyTopics") or []),
        "search_query": "",
        "topic_key": topic_key,
        "author_id": str(author.get("uniqueId") or "unknown"),
        "language": str(item.get("detected_language") or "en") or "en",
        "locale": "en-us",
        "as_of_time": posted_at,
        "posted_at": posted_at,
        "plays": int(stats.get("playCount") or 0),
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_by_field(
    rows: List[Dict[str, Any]],
) -> Tuple[List, List, List]:
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "validation"]
    test = [r for r in rows if r.get("split") == "test"]
    return train, val, test


def _temporal_split(
    rows: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List, List, List]:
    def _ts(row: Dict[str, Any]) -> float:
        raw = row.get("as_of_time") or row.get("posted_at") or ""
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    ordered = sorted(rows, key=_ts)
    n = len(ordered)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = ordered[-n_test:]
    val = ordered[-(n_test + n_val) : -n_test]
    train = ordered[: -(n_test + n_val)]
    return train, val, test


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence transformer on TikTok data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", type=Path, required=True, help="JSON file of rows to train on"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tiktok-sbert"),
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID or local path to start from",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument(
        "--min-plays",
        type=int,
        default=100,
        help="Skip videos with fewer plays (0 = keep all)",
    )
    parser.add_argument(
        "--min-shared-hashtags",
        type=int,
        default=1,
        help="Minimum shared hashtags to form a positive pair",
    )
    parser.add_argument("--max-pairs-per-anchor", type=int, default=10)
    args = parser.parse_args()

    try:
        from src.recommendation.learning.embedding_finetuner import (
            EmbeddingFinetunerConfig,
            TikTokEmbeddingFinetuner,
        )
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    if not args.data_path.exists():
        logger.error("Data file not found: %s", args.data_path)
        return 1

    rows = _load_rows(args.data_path)
    if not rows:
        logger.error("No rows loaded from %s", args.data_path)
        return 1
    logger.info("Loaded %d rows from %s", len(rows), args.data_path)

    if all("split" in r for r in rows):
        train, val, test = _split_by_field(rows)
        logger.info("Using pre-assigned splits")
    else:
        train, val, test = _temporal_split(rows)
        logger.info("Performed temporal split (no leakage)")

    logger.info("train=%d  val=%d  test=%d", len(train), len(val), len(test))

    cfg = EmbeddingFinetunerConfig(
        base_model=args.base_model,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        eval_k=args.eval_k,
        min_plays=args.min_plays,
        min_shared_hashtags=args.min_shared_hashtags,
        max_pairs_per_anchor=args.max_pairs_per_anchor,
    )

    finetuner = TikTokEmbeddingFinetuner(cfg)

    try:
        model = finetuner.train(train, val_rows=val if val else None)
    except (ImportError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1

    logger.info("Evaluating on held-out test set (%d rows) …", len(test))
    metrics = finetuner.evaluate(model, test)

    print("\n── Evaluation results ──────────────────────────")
    for key, value in sorted(metrics.items()):
        print(f"  {key:<12} {value:.4f}")
    print("────────────────────────────────────────────────\n")

    results_path = args.output_dir / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Results written to %s", results_path)
    logger.info(
        "To use the fine-tuned model set dense_model_name=%r in HybridRetrieverTrainerConfig",
        str(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
