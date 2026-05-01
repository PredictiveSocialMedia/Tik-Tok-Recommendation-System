"""Bayesian optimisation for LambdaRank hyperparameters.

The existing RankerWeightOptimizer.train() fixes sigma=1.0 and reg=5.0.
This module wraps it in a Gaussian Process + Expected Improvement loop
to find the (sigma, reg) pair that maximises NDCG@k on a validation set.

Architecture
------------
Outer loop  – Bayesian optimisation (GP surrogate + EI acquisition)
Inner loop  – L-BFGS-B LambdaRank (the existing RankerWeightOptimizer)

Search space (2-D, normalised to [0,1]² internally)
  sigma   ∈ [sigma_min,  sigma_max]   linear scale
  reg     ∈ [10^log_reg_min, 10^log_reg_max]  log10 scale

Surrogate model
  sklearn GaussianProcessRegressor with Matérn-2.5 kernel.
  Kernel hyperparameters are re-optimised each iteration.

Acquisition function
  Expected Improvement (EI) with exploration bonus xi.
  Maximised by exhaustive evaluation over n_candidates random points.

Dependencies
  scikit-learn ≥ 1.3  (in requirements-base.txt)
  scipy              (already used by RankerWeightOptimizer)
  Both are available without any additional installs.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ranker_weight_optimizer import (
    COMPONENT_NAMES,
    RankerWeightOptimizer,
    RankingGroupExample,
    _mean_ndcg,
    _softmax,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    from scipy.stats import norm as _scipy_norm

    _SCIPY_STATS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_STATS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config + result container
# ---------------------------------------------------------------------------

@dataclass
class BayesianRankerSearchConfig:
    # Search space
    sigma_min: float = 0.1
    sigma_max: float = 5.0
    log_reg_min: float = -2.0   # reg = 10^(-2) = 0.01
    log_reg_max: float = 2.0    # reg = 10^(2)  = 100.0
    # Bayesian optimisation
    n_iterations: int = 25
    n_initial_random: int = 5   # purely random before GP starts
    n_candidates: int = 2_000   # random points evaluated for EI maximisation
    xi: float = 0.01            # EI exploration bonus
    # Inner LambdaRank solver
    max_inner_iter: int = 300
    min_pairs: int = 30
    # Evaluation
    ndcg_k: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        if self.log_reg_max <= self.log_reg_min:
            raise ValueError("Require log_reg_min < log_reg_max.")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        if self.n_initial_random < 1:
            raise ValueError("n_initial_random must be >= 1.")
        if self.n_initial_random > self.n_iterations:
            raise ValueError("n_initial_random must be <= n_iterations.")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1.")
        if self.max_inner_iter < 1:
            raise ValueError("max_inner_iter must be >= 1.")
        if self.min_pairs < 1:
            raise ValueError("min_pairs must be >= 1.")
        if self.ndcg_k < 1:
            raise ValueError("ndcg_k must be >= 1.")


@dataclass
class SearchResult:
    best_sigma: float
    best_reg: float
    best_ndcg: float
    baseline_ndcg: float        # NDCG with default sigma=1.0, reg=5.0
    improvement: float          # best_ndcg - baseline_ndcg
    n_iterations: int
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_sigma":     self.best_sigma,
            "best_reg":       self.best_reg,
            "best_ndcg":      self.best_ndcg,
            "baseline_ndcg":  self.baseline_ndcg,
            "improvement":    self.improvement,
            "n_iterations":   self.n_iterations,
            "history":        self.history,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SearchResult":
        with open(path) as fh:
            d = json.load(fh)
        return cls(**d)


# ---------------------------------------------------------------------------
# Core search class
# ---------------------------------------------------------------------------

class BayesianRankerSearch:
    """Bayesian optimisation over (sigma, reg) for LambdaRank training.

    Usage::

        search = BayesianRankerSearch()
        result = search.run(train_groups, val_groups)
        print(result.best_sigma, result.best_reg, result.improvement)

        # Apply the best hyperparameters
        opt = RankerWeightOptimizer()
        opt.train(all_groups, sigma=result.best_sigma, reg=result.best_reg)
    """

    def __init__(self, cfg: Optional[BayesianRankerSearchConfig] = None) -> None:
        self.cfg = cfg or BayesianRankerSearchConfig()
        self._rng = np.random.RandomState(self.cfg.seed)

    # ------------------------------------------------------------------
    # Search space encoding (sigma, log10(reg)) → normalised [0,1]²
    # ------------------------------------------------------------------

    def _encode(self, sigma: float, log_reg: float) -> np.ndarray:
        x1 = (sigma - self.cfg.sigma_min) / (self.cfg.sigma_max - self.cfg.sigma_min)
        x2 = (log_reg - self.cfg.log_reg_min) / (self.cfg.log_reg_max - self.cfg.log_reg_min)
        return np.array([np.clip(x1, 0.0, 1.0), np.clip(x2, 0.0, 1.0)], dtype=np.float64)

    def _decode(self, x: np.ndarray) -> Tuple[float, float]:
        """Return (sigma, reg) from a normalised [0,1]² point."""
        sigma   = self.cfg.sigma_min + x[0] * (self.cfg.sigma_max - self.cfg.sigma_min)
        log_reg = self.cfg.log_reg_min + x[1] * (self.cfg.log_reg_max - self.cfg.log_reg_min)
        reg = 10.0 ** log_reg
        sigma = float(np.clip(sigma, self.cfg.sigma_min, self.cfg.sigma_max))
        reg   = float(np.clip(reg, 10.0 ** self.cfg.log_reg_min, 10.0 ** self.cfg.log_reg_max))
        return sigma, reg

    def _sample_random(self, n: int) -> np.ndarray:
        """Sample n points uniformly from the normalised search space."""
        return self._rng.uniform(0.0, 1.0, size=(n, 2))

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective(
        self,
        sigma: float,
        reg: float,
        train_groups: Sequence[RankingGroupExample],
        val_groups: Sequence[RankingGroupExample],
    ) -> float:
        """Run LambdaRank with (sigma, reg), return mean NDCG@k on val_groups.

        Returns 0.0 on any failure so the GP can still model the space.
        Falls back to training NDCG when val_groups is empty.
        """
        try:
            opt = RankerWeightOptimizer()
            opt.train(
                list(train_groups),
                sigma=sigma,
                reg=reg,
                max_iter=self.cfg.max_inner_iter,
                min_pairs=self.cfg.min_pairs,
            )
            eval_groups = list(val_groups) if val_groups else list(train_groups)
            ndcgs: List[float] = []
            for obj, learned_w in opt._learned_weights.items():
                w_arr = np.array(
                    [float(learned_w.get(name, 0.0)) for name in COMPONENT_NAMES]
                )
                obj_groups = [g for g in eval_groups if g.objective == obj]
                if obj_groups:
                    ndcgs.append(_mean_ndcg(obj_groups, w_arr, k=self.cfg.ndcg_k))
            return float(np.mean(ndcgs)) if ndcgs else 0.0
        except Exception as exc:
            logger.warning("Objective failed (sigma=%.3f, reg=%.3f): %s", sigma, reg, exc)
            return 0.0

    # ------------------------------------------------------------------
    # Acquisition function
    # ------------------------------------------------------------------

    def _expected_improvement(
        self,
        X: np.ndarray,
        gp: "GaussianProcessRegressor",
        y_best: float,
    ) -> np.ndarray:
        """Expected Improvement at candidate points X.

        EI(x) = (μ(x) - y* - ξ) Φ(Z) + σ(x) φ(Z)
        where Z = (μ(x) - y* - ξ) / σ(x)
        """
        mu, sigma = gp.predict(X, return_std=True)
        improvement = mu - y_best - self.cfg.xi
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(sigma > 1e-9, improvement / sigma, 0.0)

        if _SCIPY_STATS_AVAILABLE:
            ei = improvement * _scipy_norm.cdf(Z) + sigma * _scipy_norm.pdf(Z)
        else:
            # Fallback: pure numpy approximation of the normal CDF
            ei = improvement * _normal_cdf(Z) + sigma * _normal_pdf(Z)

        ei = np.where(sigma > 1e-9, ei, np.maximum(improvement, 0.0))
        return ei

    def _next_candidate(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
    ) -> np.ndarray:
        """Fit a GP on current observations and return the EI-maximising candidate."""
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
            nu=2.5,
            length_scale=[1.0, 1.0],
            length_scale_bounds=(1e-2, 10.0),
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,         # numerical stability
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.cfg.seed,
        )
        gp.fit(X_obs, y_obs)

        X_cand = self._sample_random(self.cfg.n_candidates)
        ei = self._expected_improvement(X_cand, gp, float(y_obs.max()))
        return X_cand[int(np.argmax(ei))]

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def run(
        self,
        train_groups: Sequence[RankingGroupExample],
        val_groups: Sequence[RankingGroupExample],
        objective_fn: Optional[Callable[[float, float], float]] = None,
    ) -> SearchResult:
        """Run Bayesian optimisation and return the best (sigma, reg) found.

        Args:
            train_groups: Groups used to fit LambdaRank weights each iteration.
            val_groups:   Groups used to evaluate NDCG (held-out, no leakage).
            objective_fn: Optional override for the objective. Signature:
                          ``fn(sigma: float, reg: float) -> float``.
                          Useful for testing or custom evaluation logic.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn>=1.3 is required for Bayesian optimisation. "
                "Install with: pip install scikit-learn"
            )

        def _eval(sigma: float, reg: float) -> float:
            if objective_fn is not None:
                return float(objective_fn(sigma, reg))
            return self._objective(sigma, reg, train_groups, val_groups)

        # --- Baseline: default hyperparameters ---
        baseline_ndcg = _eval(1.0, 5.0)
        logger.info("Baseline NDCG@%d (sigma=1.0, reg=5.0): %.4f",
                    self.cfg.ndcg_k, baseline_ndcg)

        history: List[Dict[str, Any]] = []
        X_obs_list: List[np.ndarray] = []
        y_obs_list: List[float]       = []

        # --- Phase 1: random exploration ---
        random_points = self._sample_random(self.cfg.n_initial_random)
        for i, x in enumerate(random_points):
            sigma, reg = self._decode(x)
            ndcg = _eval(sigma, reg)
            X_obs_list.append(x)
            y_obs_list.append(ndcg)
            history.append({
                "iteration": i,
                "type":      "random",
                "sigma":     round(sigma, 4),
                "reg":       round(reg, 4),
                "ndcg":      round(ndcg, 4),
            })
            logger.info("[%2d/%d] random  sigma=%.3f reg=%.3f  NDCG=%.4f",
                        i + 1, self.cfg.n_iterations, sigma, reg, ndcg)

        # --- Phase 2: GP-guided acquisition ---
        n_bo = self.cfg.n_iterations - self.cfg.n_initial_random
        for i in range(n_bo):
            it = self.cfg.n_initial_random + i
            X_obs = np.array(X_obs_list)
            y_obs = np.array(y_obs_list)

            x_next = self._next_candidate(X_obs, y_obs)
            sigma, reg = self._decode(x_next)
            ndcg = _eval(sigma, reg)

            X_obs_list.append(x_next)
            y_obs_list.append(ndcg)
            history.append({
                "iteration": it,
                "type":      "bo",
                "sigma":     round(sigma, 4),
                "reg":       round(reg, 4),
                "ndcg":      round(ndcg, 4),
            })
            logger.info("[%2d/%d] bo      sigma=%.3f reg=%.3f  NDCG=%.4f",
                        it + 1, self.cfg.n_iterations, sigma, reg, ndcg)

        # --- Pick best observed ---
        best_idx  = int(np.argmax(y_obs_list))
        best_x    = X_obs_list[best_idx]
        best_sigma, best_reg = self._decode(best_x)
        best_ndcg = float(y_obs_list[best_idx])

        logger.info(
            "Best: sigma=%.4f  reg=%.4f  NDCG=%.4f  improvement=%.4f",
            best_sigma, best_reg, best_ndcg, best_ndcg - baseline_ndcg,
        )

        return SearchResult(
            best_sigma=round(best_sigma, 4),
            best_reg=round(best_reg, 4),
            best_ndcg=round(best_ndcg, 4),
            baseline_ndcg=round(baseline_ndcg, 4),
            improvement=round(best_ndcg - baseline_ndcg, 4),
            n_iterations=len(history),
            history=history,
        )


# ---------------------------------------------------------------------------
# Pure-numpy fallbacks for scipy.stats (used when scipy is not installed)
# ---------------------------------------------------------------------------

def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun approximation, max error 1.5e-7
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1.0 - poly * np.exp(-(x ** 2)))
