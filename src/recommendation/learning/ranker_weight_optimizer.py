from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.special import softmax as _scipy_softmax
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

from .baseline_common import (
    DEFAULT_RANKING_WEIGHTS,
    OBJECTIVE_RANKING_WEIGHTS,
    clamp,
    round_score,
)

OPTIMIZER_VERSION = "ranker_weights.lambdarank.v1"
OPTIMIZER_ID = "lambdarank_ranker_weights"

# Must match score_components_for_candidate output keys
COMPONENT_NAMES: List[str] = [
    "semantic_relevance",
    "intent_alignment",
    "performance_quality",
    "reference_usefulness",
    "support_confidence",
    "trajectory_alignment",
]

# Minimum pairwise comparisons required to attempt optimisation per objective
MIN_PAIRS_PER_OBJECTIVE = 30


# ---------------------------------------------------------------------------
# Training example containers
# ---------------------------------------------------------------------------

@dataclass
class RankingCandidate:
    candidate_id: str
    components: Dict[str, float]  # score_components from score_components_for_candidate
    relevance_label: float        # 0.0–3.0 from datamart pair_rows


@dataclass
class RankingGroupExample:
    """All scored candidates for a single query, ready for LambdaRank training."""
    query_id: str
    objective: str
    candidates: List[RankingCandidate] = field(default_factory=list)

    def n_pairs(self) -> int:
        """Count of (winner, loser) pairs with distinct labels."""
        return sum(
            1
            for i, a in enumerate(self.candidates)
            for b in self.candidates[i + 1:]
            if a.relevance_label != b.relevance_label
        )


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    if _SCIPY_AVAILABLE:
        return _scipy_softmax(x)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    n = min(k, len(scores))
    if n == 0:
        return 0.0
    order = np.argsort(-scores)[:n]
    dcg = sum((2.0 ** labels[order[i]] - 1.0) / math.log2(i + 2) for i in range(n))
    ideal_order = np.argsort(-labels)[:n]
    idcg = sum((2.0 ** labels[ideal_order[i]] - 1.0) / math.log2(i + 2) for i in range(n))
    return dcg / idcg if idcg > 0 else 0.0


def _mean_ndcg(groups: Sequence[RankingGroupExample], w: np.ndarray, k: int = 10) -> float:
    scores_list = []
    for group in groups:
        if len(group.candidates) < 2:
            continue
        C = np.array([[float(c.components.get(name, 0.0)) for name in COMPONENT_NAMES] for c in group.candidates])
        labels = np.array([float(c.relevance_label) for c in group.candidates])
        if labels.max() == labels.min():
            continue
        scores_list.append(_ndcg_at_k(C @ w, labels, k=k))
    return float(np.mean(scores_list)) if scores_list else 0.0


def _lambda_loss_and_grad(
    theta: np.ndarray,
    groups: Sequence[RankingGroupExample],
    sigma: float = 1.0,
    theta0: Optional[np.ndarray] = None,
    reg: float = 5.0,
) -> Tuple[float, np.ndarray]:
    """
    LambdaRank pairwise loss + L2 regularisation + analytical gradient w.r.t. theta.

    For each query group, for each pair (i, j) where label_i > label_j:
      Loss += |ΔNDCG_ij| × log(1 + exp(−σ(s_i − s_j)))
      where s = C @ softmax(theta)

    Regularisation (prevents weight collapse):
      Loss += reg/2 × ||theta − theta0||²
      pulls learned weights back toward the warm-start (hardcoded) values.

    The gradient w.r.t. theta uses the softmax Jacobian:
      ∂L/∂θ_k = w_k × (∂L/∂w_k − Σ_m w_m × ∂L/∂w_m)
    """
    w = _softmax(theta)
    total_loss = 0.0
    grad_w = np.zeros(len(COMPONENT_NAMES))

    for group in groups:
        cands = group.candidates
        n = len(cands)
        if n < 2:
            continue

        C = np.array([[float(c.components.get(name, 0.0)) for name in COMPONENT_NAMES] for c in cands])
        labels = np.array([float(c.relevance_label) for c in cands])
        scores = C @ w

        # Ideal DCG for ΔNDCG normalisation
        ideal_order = np.argsort(-labels)
        idcg = sum((2.0 ** labels[ideal_order[i]] - 1.0) / math.log2(i + 2) for i in range(n))
        if idcg == 0.0:
            continue

        # Current ranks (0-indexed, lower = better)
        rank = np.empty(n, dtype=int)
        rank[np.argsort(-scores)] = np.arange(n)

        for i in range(n):
            for j in range(n):
                if labels[i] <= labels[j]:
                    continue

                ri, rj = int(rank[i]), int(rank[j])
                delta_ndcg = abs(
                    (2.0 ** labels[i] - 1.0) * (1.0 / math.log2(rj + 2) - 1.0 / math.log2(ri + 2))
                    + (2.0 ** labels[j] - 1.0) * (1.0 / math.log2(ri + 2) - 1.0 / math.log2(rj + 2))
                ) / idcg

                if delta_ndcg < 1e-9:
                    continue

                s_diff = sigma * (scores[i] - scores[j])
                # Numerically stable: log(1 + exp(-s)) = log1p(exp(-s))
                total_loss += delta_ndcg * math.log1p(math.exp(max(-500.0, -s_diff)))
                # P(j beats i) = sigmoid(-s_diff); gradient of loss w.r.t. w
                p_ji = 1.0 / (1.0 + math.exp(min(500.0, s_diff)))
                grad_w -= delta_ndcg * sigma * p_ji * (C[i] - C[j])

    # Chain rule through softmax: ∂L/∂θ_k = w_k × (g_k − g·w)
    g_dot_w = float(grad_w @ w)
    grad_theta = w * (grad_w - g_dot_w)

    # L2 regularisation toward warm-start theta0 — prevents weight collapse
    if reg > 0.0 and theta0 is not None:
        diff = theta - theta0
        total_loss += 0.5 * reg * float(diff @ diff)
        grad_theta += reg * diff

    return total_loss, grad_theta


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class RankerWeightOptimizer:
    """
    Learns per-objective ranking component weights via LambdaRank.

    Training flow:
        groups = collect_ranking_training_examples(...)
        opt = RankerWeightOptimizer()
        opt.train(groups)
        opt.save(artifact_dir / "lambdarank_weights")

    Inference flow:
        opt = RankerWeightOptimizer.load(artifact_dir / "lambdarank_weights")
        weights = opt.get_weights("engagement")   # Dict[str, float]
    """

    def __init__(self) -> None:
        self._learned_weights: Dict[str, Dict[str, float]] = {}
        self._train_summary: Dict[str, Any] = {}
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        groups: Sequence[RankingGroupExample],
        objectives: Optional[List[str]] = None,
        max_iter: int = 300,
        sigma: float = 1.0,
        reg: float = 5.0,
        min_pairs: int = MIN_PAIRS_PER_OBJECTIVE,
    ) -> Dict[str, Dict[str, float]]:
        """Fit per-objective weight vectors. Returns learned weight dicts."""
        if objectives is None:
            objectives = list({g.objective for g in groups})

        results: Dict[str, Dict[str, float]] = {}
        summary: Dict[str, Any] = {}

        for obj in objectives:
            obj_groups = [g for g in groups if g.objective == obj]
            n_pairs = sum(g.n_pairs() for g in obj_groups)

            if not _SCIPY_AVAILABLE or n_pairs < min_pairs:
                fallback = dict(OBJECTIVE_RANKING_WEIGHTS.get(obj, DEFAULT_RANKING_WEIGHTS))
                results[obj] = fallback
                reason = "no_scipy" if not _SCIPY_AVAILABLE else "insufficient_pairs"
                summary[obj] = {
                    "status": f"fallback_{reason}",
                    "n_groups": len(obj_groups),
                    "n_pairs": n_pairs,
                    "min_pairs_required": min_pairs,
                    "ndcg_before": None,
                    "ndcg_after": None,
                    "weights": fallback,
                }
                continue

            # Warm-start theta from current hardcoded weights (faster convergence)
            current_w = OBJECTIVE_RANKING_WEIGHTS.get(obj, DEFAULT_RANKING_WEIGHTS)
            w0 = np.array([max(1e-6, float(current_w.get(n, 1.0 / len(COMPONENT_NAMES)))) for n in COMPONENT_NAMES])
            # Inverse softmax: theta = log(w) (shift doesn't matter for softmax)
            theta0 = np.log(w0)
            theta0 -= theta0.mean()

            ndcg_before = _mean_ndcg(obj_groups, _softmax(theta0))

            result = _scipy_minimize(
                fun=_lambda_loss_and_grad,
                x0=theta0,
                args=(obj_groups, sigma, theta0, reg),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8},
            )

            w_opt = _softmax(result.x)
            ndcg_after = _mean_ndcg(obj_groups, w_opt)

            # Safety guard: revert if optimisation degraded quality
            if ndcg_after < ndcg_before - 0.005:
                w_opt = _softmax(theta0)
                status = "reverted_ndcg_regression"
            elif result.success:
                status = "converged"
            else:
                status = f"stopped_iter_{result.nit}"

            # Renormalise to exactly 1.0 and enforce non-negativity
            w_clipped = np.maximum(w_opt, 1e-6)
            w_clipped /= w_clipped.sum()

            learned: Dict[str, float] = {
                name: round_score(float(w_clipped[k]), 6)
                for k, name in enumerate(COMPONENT_NAMES)
            }
            # Final renorm after rounding
            total = sum(learned.values())
            learned = {k: round_score(v / total, 6) for k, v in learned.items()}

            results[obj] = learned
            summary[obj] = {
                "status": status,
                "n_groups": len(obj_groups),
                "n_pairs": n_pairs,
                "n_iter": result.nit,
                "ndcg_before": round(ndcg_before, 4),
                "ndcg_after": round(_mean_ndcg(obj_groups, w_clipped), 4),
                "weights": learned,
            }

        self._learned_weights = results
        self._train_summary = summary
        self._trained = True
        return results

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_weights(self, objective: str) -> Optional[Dict[str, float]]:
        """Return learned weights for objective, or None if not trained / insufficient data."""
        if not self._trained:
            return None
        weights = self._learned_weights.get(objective)
        if weights is None:
            return None
        # If status was fallback, return None so inference uses hardcoded fallback
        summary = self._train_summary.get(objective, {})
        if str(summary.get("status", "")).startswith("fallback_"):
            return None
        return dict(weights)

    @property
    def train_summary(self) -> Dict[str, Any]:
        return dict(self._train_summary)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": OPTIMIZER_VERSION,
            "optimizer_id": OPTIMIZER_ID,
            "component_names": COMPONENT_NAMES,
            "learned_weights": self._learned_weights,
            "train_summary": self._train_summary,
            "trained": self._trained,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, artifact_dir: Path) -> "RankerWeightOptimizer":
        artifact_dir = Path(artifact_dir)
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No lambdarank manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("version") != OPTIMIZER_VERSION:
            raise ValueError(
                f"Version mismatch: saved={manifest.get('version')} current={OPTIMIZER_VERSION}"
            )
        saved_components = manifest.get("component_names", [])
        if set(saved_components) != set(COMPONENT_NAMES):
            raise ValueError(
                f"Component name mismatch — retrain. saved={saved_components} current={COMPONENT_NAMES}"
            )

        opt = cls()
        opt._learned_weights = manifest.get("learned_weights") or {}
        opt._train_summary = manifest.get("train_summary") or {}
        opt._trained = bool(manifest.get("trained", False))
        return opt
