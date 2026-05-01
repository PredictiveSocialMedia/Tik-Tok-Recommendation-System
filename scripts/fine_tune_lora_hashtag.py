#!/usr/bin/env python3
"""Fine-tune a small LLM with LoRA for TikTok hashtag generation.

Trains TinyLlama (or any causal LM) to generate contextually relevant
hashtags from a video caption + visual topics, replacing the rule-based
suggest_hashtags() in caption_suggest.py.

Usage (scraped or internal row format):
    python scripts/fine_tune_lora_hashtag.py --data-path data/rows.json

Usage (4-bit quantization for limited GPU memory):
    python scripts/fine_tune_lora_hashtag.py \\
        --data-path data/rows.json \\
        --load-in-4bit \\
        --batch-size 2

Requirements:
    pip install 'peft>=0.10' 'trl>=0.8' 'transformers>=4.40' torch datasets
    # Optional: bitsandbytes>=0.43 for 4-bit quantization

After training, load the adapter:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = PeftModel.from_pretrained(base, "models/tiktok-hashtag-lora")
    tokenizer = AutoTokenizer.from_pretrained("models/tiktok-hashtag-lora")
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
        if "hashtags" in item or "caption" in item:
            rows.append(item if "row_id" in item else _scraped_to_row(item))
        else:
            rows.append(_scraped_to_row(item))
    return rows


def _scraped_to_row(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata") or {}
    stats = meta.get("stats") or {}
    author = meta.get("author") or {}
    hashtags = [str(h).lstrip("#") for h in (meta.get("hashtags") or [])]
    caption = str(item.get("caption") or item.get("video_caption") or "")
    if not hashtags:
        hashtags = [w.lstrip("#") for w in caption.split() if w.startswith("#")]
    video_id = str(meta.get("id") or item.get("asset_id") or "")
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
        "keywords": list((item.get("analysis") or {}).get("keyTopics") or []),
        "topic_key": hashtags[0] if hashtags else "unknown",
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
    return (
        ordered[: -(n_test + n_val)],
        ordered[-(n_test + n_val) : -n_test],
        ordered[-n_test:],
    )


def _split_by_field(rows: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    return (
        [r for r in rows if r.get("split") == "train"],
        [r for r in rows if r.get("split") == "validation"],
        [r for r in rows if r.get("split") == "test"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a small LLM for TikTok hashtag generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tiktok-hashtag-lora"))
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--min-hashtags", type=int, default=2)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization via bitsandbytes (requires GPU)",
    )
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    try:
        from src.recommendation.learning.lora_hashtag_finetuner import (
            LoraHashtagFinetunerConfig,
            LoraHashtagFinetuner,
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

    cfg = LoraHashtagFinetunerConfig(
        base_model=args.base_model,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len,
        eval_k=args.eval_k,
        min_hashtags=args.min_hashtags,
        load_in_4bit=args.load_in_4bit,
        fp16=args.fp16,
    )

    finetuner = LoraHashtagFinetuner(cfg)

    try:
        model, tokenizer = finetuner.train(train, val_rows=val if val else None)
    except (ImportError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1

    logger.info("Evaluating on held-out test set (%d rows) …", len(test))
    metrics = finetuner.evaluate(model, tokenizer, test)

    print("\n── Evaluation results ──────────────────────────")
    for key, value in sorted(metrics.items()):
        print(f"  {key:<14} {value:.4f}")
    print("────────────────────────────────────────────────\n")

    results_path = args.output_dir / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Results written to %s", results_path)
    logger.info(
        "Adapter saved — load with: PeftModel.from_pretrained(base, %r)",
        str(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
