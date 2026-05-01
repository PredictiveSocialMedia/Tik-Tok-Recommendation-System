#!/usr/bin/env python3
"""Train the RL content optimization policy on historical TikTok data.

The REINFORCE agent learns which content strategy (REACH / ENGAGEMENT /
CONVERSION / RETENTION) to recommend for a given video, based on the
video's observed engagement outcomes.

Usage:
    python scripts/train_rl_content_policy.py --data-path data/rows.json

Usage (custom hyperparameters):
    python scripts/train_rl_content_policy.py \\
        --data-path data/rows.json \\
        --epochs 20 \\
        --learning-rate 5e-4 \\
        --hidden-dim 128

After training, load and use the policy:
    from src.recommendation.learning.rl_content_policy import REINFORCEAgent
    agent = REINFORCEAgent.load("models/rl-content-policy/policy.json")
    rec = agent.recommend(row)
    print(rec["action"], rec["advice"])

No ML dependencies (torch, peft, etc.) required — pure numpy.
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
        if "row_id" in item or "video_id" in item:
            rows.append(item)
        else:
            rows.append(_scraped_to_row(item))
    return rows


def _scraped_to_row(item: Dict[str, Any]) -> Dict[str, Any]:
    meta   = item.get("metadata") or {}
    stats  = meta.get("stats")  or {}
    author = meta.get("author") or {}
    hashtags = [str(h).lstrip("#") for h in (meta.get("hashtags") or [])]
    caption  = str(item.get("caption") or item.get("video_caption") or "")
    if not hashtags:
        hashtags = [w.lstrip("#") for w in caption.split() if w.startswith("#")]

    plays    = int(stats.get("playCount")    or 0)
    likes    = int(stats.get("likeCount")    or 0)
    comments = int(stats.get("commentCount") or 0)
    shares   = int(stats.get("shareCount")   or 0)
    er = (likes + comments + shares) / max(1, plays) if plays else 0.0

    video_id   = str(meta.get("id") or item.get("asset_id") or "")
    create_time = meta.get("createTime")
    try:
        posted_at = (
            datetime.fromtimestamp(int(create_time), tz=timezone.utc).isoformat()
            if create_time else "2026-01-01T00:00:00Z"
        )
    except (ValueError, TypeError):
        posted_at = "2026-01-01T00:00:00Z"

    return {
        "row_id":   video_id or f"row_{hash(caption) & 0xFFFFFF}",
        "video_id": video_id,
        "caption":  caption,
        "hashtags": hashtags,
        "keywords": list((item.get("analysis") or {}).get("keyTopics") or []),
        "topic_key": hashtags[0] if hashtags else "unknown",
        "author_id": str(author.get("uniqueId") or "unknown"),
        "language": "en",
        "locale":   "en-us",
        "as_of_time": posted_at,
        "posted_at":  posted_at,
        "plays":      plays,
        "likes":      likes,
        "comments_count": comments,
        "shares":     shares,
        "engagement_metrics": {
            "views":           plays,
            "engagement_rate": er,
        },
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _temporal_split(
    rows: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List, List, List]:
    def _ts(r: Dict[str, Any]) -> float:
        raw = r.get("as_of_time") or r.get("posted_at") or ""
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    ordered = sorted(rows, key=_ts)
    n      = len(ordered)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
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
        description="Train REINFORCE content optimization policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path",    type=Path, required=True)
    parser.add_argument("--output-dir",   type=Path, default=Path("models/rl-content-policy"))
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--hidden-dim",   type=int,   default=64)
    parser.add_argument("--learning-rate",type=float, default=1e-3)
    parser.add_argument("--baseline-decay",type=float,default=0.95)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--demo-row",
        action="store_true",
        help="Print a sample recommendation from the test set after training.",
    )
    args = parser.parse_args()

    try:
        from src.recommendation.learning.rl_content_policy import (
            REINFORCEAgent,
            RLContentPolicyConfig,
            ContentAction,
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
    logger.info("Loaded %d rows", len(rows))

    if all("split" in r for r in rows):
        train, val, test = _split_by_field(rows)
        logger.info("Using pre-assigned splits")
    else:
        train, val, test = _temporal_split(rows)
        logger.info("Performed temporal split (no leakage)")
    logger.info("train=%d  val=%d  test=%d", len(train), len(val), len(test))

    cfg = RLContentPolicyConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        baseline_decay=args.baseline_decay,
        seed=args.seed,
    )
    agent = REINFORCEAgent(cfg)

    try:
        agent.train(train, val_rows=val if val else None)
    except ValueError as exc:
        logger.error("Training failed: %s", exc)
        return 1

    logger.info("Evaluating on held-out test set (%d rows) …", len(test))
    metrics = agent.evaluate(test)

    print("\n── Evaluation results ─────────────────────────────")
    for key, value in sorted(metrics.items()):
        print(f"  {key:<25} {value:.4f}")
    print("────────────────────────────────────────────────────\n")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "policy.json"
    agent.save(str(policy_path))

    results_path = output / "eval_results.json"
    with open(results_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Results written to %s", results_path)

    if args.demo_row and test:
        sample = test[0]
        rec = agent.recommend(sample)
        print("── Sample recommendation ───────────────────────────")
        print(f"  Action     : {rec['action']} (confidence {rec['confidence']:.2%})")
        print(f"  Probs      : {', '.join(f'{k}={v:.2%}' for k, v in rec['probs'].items())}")
        print(f"  Advice     : {rec['advice']}")
        print("────────────────────────────────────────────────────\n")

    logger.info(
        "Load the trained policy with: REINFORCEAgent.load(%r)", str(policy_path)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
