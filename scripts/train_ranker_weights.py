"""
Train LambdaRank ranking weights from an existing training datamart and artifact bundle.

Usage:
    python scripts/train_ranker_weights.py \
        --datamart  data/mock/training_datamart.json \
        --bundle    artifacts/recommender/latest \
        [--objectives reach engagement conversion] \
        [--max-queries 256] \
        [--max-iter 300] \
        [--min-pairs 30]

The script:
  1. Loads the datamart to get rows, pair_rows, and relevance labels.
  2. Loads the existing artifact bundle to get the trained retriever.
  3. For each objective, collects ranking training examples by running the
     retriever + baseline scorer on validation queries.
  4. Runs LambdaRank optimisation (scipy L-BFGS-B) over the score_components.
  5. Saves lambdarank_weights/manifest.json under each
     <bundle>/rankers/<objective>/ directory so inference picks it up automatically.

Re-running this script updates the weights in-place without retraining the retriever.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ranker_weights")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LambdaRank ranking weights")
    parser.add_argument("--datamart", required=True, help="Path to training_datamart.json")
    parser.add_argument("--bundle", required=True, help="Path to artifact bundle directory")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["reach", "engagement", "conversion"],
        help="Objectives to train (default: all three)",
    )
    parser.add_argument("--max-queries", type=int, default=256, help="Max validation queries per objective")
    parser.add_argument("--max-iter", type=int, default=300, help="L-BFGS-B max iterations")
    parser.add_argument("--min-pairs", type=int, default=30, help="Min pairwise examples required to optimise")
    args = parser.parse_args()

    datamart_path = Path(args.datamart)
    bundle_path = Path(args.bundle)

    if not datamart_path.exists():
        logger.error("Datamart not found: %s", datamart_path)
        sys.exit(1)
    if not bundle_path.exists():
        logger.error("Bundle not found: %s", bundle_path)
        sys.exit(1)

    logger.info("Loading datamart from %s", datamart_path)
    datamart = json.loads(datamart_path.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    pair_rows = list(datamart.get("pair_rows") or [])
    logger.info("Datamart: %d rows, %d pair_rows", len(rows), len(pair_rows))

    from src.recommendation.learning.pipeline import (
        _collect_ranking_training_examples,
        _group_relevance_by_query,
        _row_candidate_payload,
        _row_query_payload,
    )
    from src.recommendation.learning.ranker_weight_optimizer import RankerWeightOptimizer
    from src.recommendation.learning.temporal import split_rows

    rows_split = split_rows(rows)

    for objective in args.objectives:
        logger.info("=== Objective: %s ===", objective)

        ranker_dir = bundle_path / "rankers" / objective
        if not ranker_dir.exists():
            logger.warning("Ranker dir not found for objective=%s, skipping: %s", objective, ranker_dir)
            continue

        relevance_by_query = _group_relevance_by_query(pair_rows, objective)
        if not relevance_by_query:
            logger.warning("No pair_rows found for objective=%s, skipping", objective)
            continue
        logger.info("Queries with relevance labels: %d", len(relevance_by_query))

        # Get retrieve_k from the bundle manifest if available
        retrieve_k = 100
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.exists():
            bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            retrieve_k = int(bundle_manifest.get("retrieve_k") or 100)
            max_age_days = int(bundle_manifest.get("max_age_days") or 180)
        else:
            max_age_days = 180

        groups = _collect_ranking_training_examples(
            objective=objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=retrieve_k,
            max_age_days=max_age_days,
            max_queries=args.max_queries,
        )

        n_pairs = sum(g.n_pairs() for g in groups)
        logger.info(
            "Collected %d query groups, %d pairwise comparisons for objective=%s",
            len(groups), n_pairs, objective,
        )

        optimizer = RankerWeightOptimizer()
        results = optimizer.train(
            groups,
            objectives=[objective],
            max_iter=args.max_iter,
            min_pairs=args.min_pairs,
        )

        summary = optimizer.train_summary.get(objective, {})
        status = summary.get("status", "unknown")
        logger.info(
            "Training complete: status=%s ndcg_before=%.4f ndcg_after=%.4f",
            status,
            summary.get("ndcg_before") or 0.0,
            summary.get("ndcg_after") or 0.0,
        )
        logger.info("Learned weights: %s", json.dumps(results.get(objective, {}), indent=2))

        out_dir = ranker_dir / "lambdarank_weights"
        optimizer.save(out_dir)
        logger.info("Saved to %s", out_dir / "manifest.json")

    logger.info("Done.")


if __name__ == "__main__":
    main()
