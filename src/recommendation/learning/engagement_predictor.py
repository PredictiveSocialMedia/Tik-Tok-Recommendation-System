"""
Engagement prediction from metadata signals.

Implements suggestion #9 from the prof's email:
  "Improve the video analysis pipeline by adding engagement prediction
  from metadata signals (views, likes, comments)."

Trains per-target gradient-boosted regressors on metadata features
(caption stats, hashtag/keyword counts, duration, content type, posted
hour / day-of-week) and predicts:

  - log_views          (future_reach_log_delta = log(future_views))
  - engagement_rate    (future_engagement_rate ≈ likes+comments / views)
  - shares_per_1k      (future_shares_per_1k_views)

All feature extraction is pure-function so the unit tests don't need
sklearn fixtures. Metrics: MAE, RMSE, R², Spearman rank correlation,
plus a constant-mean baseline for context.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ENGAGEMENT_PREDICTOR_VERSION = "engagement_predictor.v1"

# Categorical buckets for content_type. Anything not in this list
# is mapped to "other" before one-hot encoding.
CONTENT_TYPES: Tuple[str, ...] = ("general", "tutorial", "review", "story", "other")

# Targets the predictor knows how to extract from a row's labels.
SUPPORTED_TARGETS: Tuple[str, ...] = ("log_views", "engagement_rate", "shares_per_1k")
DEFAULT_TARGETS: Tuple[str, ...] = ("log_views", "engagement_rate")


# ---------------------------------------------------------------------------
# Pure feature extraction
# ---------------------------------------------------------------------------


def _parse_posted_at(value: Any) -> Tuple[float, float]:
    """Returns (hour_of_day, day_of_week). (-1.0, -1.0) when unparseable."""
    if not isinstance(value, str):
        return (-1.0, -1.0)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return (-1.0, -1.0)
    return (float(dt.hour), float(dt.weekday()))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def extract_features(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Build a fixed-shape feature dict from a single row.

    The dict's keys form the feature schema — every row produces the same
    keys regardless of which fields are missing, so callers can stack them
    into a 2-D matrix without alignment logic.
    """
    features_payload = row.get("features")
    metadata = features_payload if isinstance(features_payload, dict) else {}

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


def extract_target(row: Dict[str, Any], target: str) -> Optional[float]:
    """Pull a single training target out of a row's ``labels`` payload."""
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; supported: {SUPPORTED_TARGETS}"
        )
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return None
    if target == "log_views":
        raw = labels.get("future_reach_log_delta")
    elif target == "engagement_rate":
        raw = labels.get("future_engagement_rate")
    else:  # shares_per_1k
        raw = labels.get("future_shares_per_1k_views")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def build_dataset(
    rows: Sequence[Dict[str, Any]],
    target: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Stack rows into (X, y, feature_names). Rows missing the target are
    silently dropped — the caller can compare ``len(X)`` to ``len(rows)``
    to see how many were usable.
    """
    feature_names: Optional[List[str]] = None
    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    for row in rows:
        y_value = extract_target(row, target)
        if y_value is None:
            continue
        feats = extract_features(row)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        x_rows.append([feats[name] for name in feature_names])
        y_rows.append(y_value)
    if feature_names is None:
        feature_names = []
    if not x_rows:
        return np.zeros((0, len(feature_names)), dtype=np.float64), np.zeros((0,), dtype=np.float64), feature_names
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64), feature_names


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy — no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank version of ``scipy.stats.rankdata`` for ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    # Handle ties: average ranks of equal values
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < len(values):
        rank_sum = np.zeros(len(unique), dtype=np.float64)
        for idx, rank in zip(inverse, ranks):
            rank_sum[idx] += rank
        avg_rank = rank_sum / counts
        ranks = avg_rank[inverse]
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    r_true = _rankdata(y_true)
    r_pred = _rankdata(y_pred)
    var_true = float(np.var(r_true))
    var_pred = float(np.var(r_pred))
    if var_true <= 0.0 or var_pred <= 0.0:
        return 0.0
    cov = float(np.mean((r_true - r_true.mean()) * (r_pred - r_pred.mean())))
    rho = cov / math.sqrt(var_true * var_pred)
    if not math.isfinite(rho):
        return 0.0
    return float(max(-1.0, min(1.0, rho)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R², Spearman. Returns zeroed metrics for empty input."""
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return {"n_rows": 0.0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "spearman": 0.0}
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff * diff)))
    var = float(np.var(y_true))
    if var <= 0.0:
        r2 = 0.0
    else:
        r2 = 1.0 - float(np.mean(diff * diff)) / var
    spearman = _spearman(y_true, y_pred)
    return {
        "n_rows": float(n),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "spearman": round(spearman, 6),
    }


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


@dataclass
class EngagementPredictorConfig:
    # Defaults tuned for the project's small (~500 row) splits — deeper or
    # broader configs overfit quickly. See the held-out comparison study
    # in the PR description for the sweep that motivated these.
    n_estimators: int = 50
    max_depth: int = 2
    learning_rate: float = 0.05
    min_samples_leaf: int = 15
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")


@dataclass
class EngagementPredictor:
    """Per-target regression model with deferred sklearn import."""

    target: str
    config: EngagementPredictorConfig = field(default_factory=EngagementPredictorConfig)
    model: Optional[Any] = None
    feature_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_TARGETS:
            raise ValueError(
                f"Unknown target {self.target!r}; supported: {SUPPORTED_TARGETS}"
            )

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "EngagementPredictor":
        # Deferred import keeps the module importable in environments
        # where sklearn isn't installed (e.g. lightweight test runs).
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

        X, y, names = build_dataset(rows, self.target)
        if len(X) == 0:
            raise ValueError(
                f"No rows produced a usable {self.target!r} target — cannot fit."
            )
        self.feature_names = names
        self.model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.model is None or not self.feature_names:
            raise RuntimeError("Predictor must be fit before calling predict().")
        x_list: List[List[float]] = []
        for row in rows:
            feats = extract_features(row)
            x_list.append([feats.get(name, 0.0) for name in self.feature_names])
        if not x_list:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(self.model.predict(np.asarray(x_list)), dtype=np.float64)

    def evaluate(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        X, y_true, _ = build_dataset(rows, self.target)
        if len(X) == 0:
            return regression_metrics(np.zeros((0,)), np.zeros((0,)))
        if self.model is None:
            raise RuntimeError("Predictor must be fit before calling evaluate().")
        y_pred = np.asarray(self.model.predict(X), dtype=np.float64)
        return regression_metrics(y_true, y_pred)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ENGAGEMENT_PREDICTOR_VERSION,
            "config": asdict(self.config),
            "target": self.target,
            "feature_names": list(self.feature_names),
            "model": self.model,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> "EngagementPredictor":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != ENGAGEMENT_PREDICTOR_VERSION:
            raise ValueError(
                f"Version mismatch: saved={payload.get('version')!r} "
                f"current={ENGAGEMENT_PREDICTOR_VERSION!r}"
            )
        target = payload.get("target")
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"Saved target {target!r} not in {SUPPORTED_TARGETS}.")
        cfg_payload = payload.get("config") or {}
        cfg = EngagementPredictorConfig(**cfg_payload)
        instance = cls(target=target, config=cfg)
        instance.feature_names = list(payload.get("feature_names") or [])
        instance.model = payload.get("model")
        return instance


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def baseline_metrics(
    eval_rows: Sequence[Dict[str, Any]],
    target: str,
    train_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Constant-mean predictor metrics — the fair baseline a trained model
    has to beat. By default the constant is the mean of ``train_rows``
    (the *honest* zero-feature predictor: it doesn't peek at the test
    targets). When ``train_rows`` is omitted, the in-sample mean of
    ``eval_rows`` is used instead.
    """
    _, y_eval, _ = build_dataset(eval_rows, target)
    if len(y_eval) == 0:
        return regression_metrics(np.zeros((0,)), np.zeros((0,)))
    if train_rows is None:
        constant = float(np.mean(y_eval))
    else:
        _, y_train, _ = build_dataset(train_rows, target)
        if len(y_train) == 0:
            constant = float(np.mean(y_eval))
        else:
            constant = float(np.mean(y_train))
    y_pred = np.full_like(y_eval, constant)
    return regression_metrics(y_eval, y_pred)


def format_metrics_markdown(
    per_target: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """
    Render a {target: {"trained": metrics, "baseline": metrics}} dict
    as a markdown comparison table.
    """
    if not per_target:
        return "_No metrics — held-out evaluation produced no rows._"
    lines: List[str] = []
    lines.append("| Target | Variant | N | MAE | RMSE | R² | Spearman |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for target, payload in per_target.items():
        for variant in ("trained", "baseline"):
            metrics = payload.get(variant, {})
            lines.append(
                "| {target} | {variant} | {n} | {mae:.4f} | {rmse:.4f} | {r2:.4f} | {sp:.4f} |".format(
                    target=target,
                    variant=variant,
                    n=int(metrics.get("n_rows", 0)),
                    mae=metrics.get("mae", 0.0),
                    rmse=metrics.get("rmse", 0.0),
                    r2=metrics.get("r2", 0.0),
                    sp=metrics.get("spearman", 0.0),
                )
            )
    return "\n".join(lines)


__all__ = [
    "CONTENT_TYPES",
    "DEFAULT_TARGETS",
    "ENGAGEMENT_PREDICTOR_VERSION",
    "SUPPORTED_TARGETS",
    "EngagementPredictor",
    "EngagementPredictorConfig",
    "baseline_metrics",
    "build_dataset",
    "extract_features",
    "extract_target",
    "format_metrics_markdown",
    "regression_metrics",
]
