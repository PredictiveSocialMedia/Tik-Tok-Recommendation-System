"""
Run Bayesian optimisation for LambdaRank ranking hyperparameters.

Usage:
    python scripts/run_bayesian_ranker_search.py \
        --datamart data/mock/training_datamart.json \
        --bundle artifacts/recommender/latest \
        --objectives reach engagement conversion \
        --iterations 25 \
        --initial-random 5

For each objective, this script:
  1. Collects ranking groups with the same pipeline helpers used by
     scripts/train_ranker_weights.py.
  2. Splits those groups into an inner train/validation set.
  3. Searches over LambdaRank sigma and regularisation.
  4. Saves bayesian_ranker_search.json under <bundle>/rankers/<objective>/.
  5. Retrains LambdaRank on all collected groups with the best parameters and
     writes <bundle>/rankers/<objective>/lambdarank_weights/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Sequence, TypeVar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_bayesian_ranker_search")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T = TypeVar("T")


def _split_groups(groups: Sequence[T], *, val_fraction: float, seed: int) -> tuple[list[T], list[T]]:
    """Deterministically split ranking groups for inner hyperparameter validation."""
    groups_list = list(groups)
    if len(groups_list) <= 1:
        return groups_list, groups_list

    rng = random.Random(seed)
    shuffled = list(groups_list)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    val_groups = shuffled[:n_val]
    train_groups = shuffled[n_val:]
    return train_groups, val_groups


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian search for LambdaRank hyperparameters")
    parser.add_argument("--datamart", required=True, help="Path to training_datamart.json")
    parser.add_argument("--bundle", required=True, help="Path to artifact bundle directory")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["reach", "engagement", "conversion"],
        help="Objectives to optimise (default: all three)",
    )
    parser.add_argument("--max-queries", type=int, default=256, help="Max query groups to collect per objective")
    parser.add_argument("--retrieve-k", type=int, default=None, help="Override retrieve_k from bundle manifest")
    parser.add_argument("--max-age-days", type=int, default=None, help="Override max_age_days from bundle manifest")
    parser.add_argument("--iterations", type=int, default=25, help="Total Bayesian search trials")
    parser.add_argument("--initial-random", type=int, default=5, help="Random trials before GP-guided search")
    parser.add_argument("--candidates", type=int, default=2000, help="Random acquisition candidates per BO trial")
    parser.add_argument("--sigma-min", type=_positive_float, default=0.1)
    parser.add_argument("--sigma-max", type=_positive_float, default=5.0)
    parser.add_argument("--log-reg-min", type=float, default=-2.0)
    parser.add_argument("--log-reg-max", type=float, default=2.0)
    parser.add_argument("--xi", type=float, default=0.01, help="Expected Improvement exploration bonus")
    parser.add_argument("--max-inner-iter", type=int, default=300, help="Inner LambdaRank max iterations")
    parser.add_argument("--min-pairs", type=int, default=30, help="Minimum pairs required to optimise LambdaRank")
    parser.add_argument("--ndcg-k", type=int, default=10, help="Validation NDCG cutoff")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Fraction of groups used for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-save-weights",
        action="store_true",
        help="Only save search results; do not overwrite lambdarank_weights",
    )
    args = parser.parse_args()

    datamart_path = Path(args.datamart)
    bundle_path = Path(args.bundle)

    if not datamart_path.exists():
        logger.error("Datamart not found: %s", datamart_path)
        sys.exit(1)
    if not bundle_path.exists():
        logger.error("Bundle not found: %s", bundle_path)
        sys.exit(1)
    if not 0.0 < args.val_fraction < 1.0:
        logger.error("--val-fraction must be between 0 and 1.")
        sys.exit(1)

    from src.recommendation.learning.bayesian_ranker_search import (
        BayesianRankerSearch,
        BayesianRankerSearchConfig,
    )
    from src.recommendation.learning.pipeline import (
        _collect_ranking_training_examples,
        _group_relevance_by_query,
    )
    from src.recommendation.learning.ranker_weight_optimizer import RankerWeightOptimizer
    from src.recommendation.learning.temporal import split_rows

    logger.info("Loading datamart from %s", datamart_path)
    datamart = json.loads(datamart_path.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    pair_rows = list(datamart.get("pair_rows") or [])
    rows_split = split_rows(rows)
    logger.info("Datamart: %d rows, %d pair_rows", len(rows), len(pair_rows))

    manifest_path = bundle_path / "manifest.json"
    bundle_manifest = {}
    if manifest_path.exists():
        bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    retrieve_k = int(args.retrieve_k or bundle_manifest.get("retrieve_k") or 100)
    max_age_days = int(args.max_age_days or bundle_manifest.get("max_age_days") or 180)
    searched_any = False

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

        groups = _collect_ranking_training_examples(
            objective=objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=retrieve_k,
            max_age_days=max_age_days,
            max_queries=args.max_queries,
        )
        n_pairs = sum(g.n_pairs() for g in groups)
        logger.info("Collected %d groups, %d pairwise comparisons", len(groups), n_pairs)
        if not groups:
            logger.warning("No usable ranking groups for objective=%s, skipping", objective)
            continue

        train_groups, val_groups = _split_groups(
            groups,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        logger.info("Inner split: %d train groups, %d validation groups", len(train_groups), len(val_groups))

        cfg = BayesianRankerSearchConfig(
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            log_reg_min=args.log_reg_min,
            log_reg_max=args.log_reg_max,
            n_iterations=args.iterations,
            n_initial_random=args.initial_random,
            n_candidates=args.candidates,
            xi=args.xi,
            max_inner_iter=args.max_inner_iter,
            min_pairs=args.min_pairs,
            ndcg_k=args.ndcg_k,
            seed=args.seed,
        )
        result = BayesianRankerSearch(cfg).run(train_groups, val_groups)
        result_path = ranker_dir / "bayesian_ranker_search.json"
        result.save(str(result_path))
        logger.info("Saved search result to %s", result_path)

        if not args.no_save_weights:
            optimizer = RankerWeightOptimizer()
            weights = optimizer.train(
                groups,
                objectives=[objective],
                max_iter=args.max_inner_iter,
                sigma=result.best_sigma,
                reg=result.best_reg,
                min_pairs=args.min_pairs,
            )
            out_dir = ranker_dir / "lambdarank_weights"
            optimizer.save(out_dir)
            logger.info("Saved tuned weights to %s", out_dir / "manifest.json")
            logger.info("Learned weights: %s", json.dumps(weights.get(objective, {}), indent=2))

        searched_any = True

    if not searched_any:
        logger.error("No objectives were searched.")
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
