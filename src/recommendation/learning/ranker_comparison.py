"""
LightGBM vs heuristic ranker comparison harness.

Implements suggestion #7 from the prof's email:
  "Implement LightGBM ranking trained on real scraped TikTok data and
  compare against the heuristic baseline."

Both rankers consume the same metadata features (caption stats,
hashtag/keyword counts, duration, content type, posted hour/day-of-week)
so the comparison is apples-to-apples — any difference in NDCG/MRR is
attributable to the model class, not the input.

For each objective (reach, engagement, conversion):
  1. Train a LightGBM regressor on the train split predicting
     ``targets_z[objective]``.
  2. Define a transparent hand-coded heuristic scorer using the same
     features.
  3. Rank held-out test rows by each scorer.
  4. Report NDCG@k / MRR@k for both, plus a paired-bootstrap 95% CI
     on the lift (LightGBM − heuristic).

The hand-coded heuristic is intentionally simple and reproducible so
the prof can audit it line-by-line — magic numbers are documented
inline.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluator import mrr_at_k, ndcg_at_k

RANKER_COMPARISON_VERSION = "ranker_comparison.v1"

# Maps an objective to the label field used as ground truth for ranking.
OBJECTIVE_TARGETS: Dict[str, str] = {
    "reach": "reach",
    "engagement": "engagement",
    "conversion": "conversion",
}

# Categorical buckets for content_type one-hot encoding.
CONTENT_TYPES: Tuple[str, ...] = ("general", "tutorial", "review", "story", "other")

DEFAULT_OBJECTIVES: Tuple[str, ...] = ("reach", "engagement", "conversion")
DEFAULT_K_VALUES: Tuple[int, ...] = (10, 20)
DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000

# Relevance grades 0-3 (matches the LearnedPairwiseReranker's relevance scale).
RELEVANCE_GRADES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Pure feature extraction (shared by both scorers)
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
    """(hour_of_day, day_of_week); (-1, -1) when unparseable."""
    if not isinstance(value, str):
        return (-1.0, -1.0)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return (-1.0, -1.0)
    return (float(dt.hour), float(dt.weekday()))


def extract_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Build a fixed-shape feature dict from a single row."""
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


def extract_target_z(row: Dict[str, Any], objective: str) -> Optional[float]:
    """Extract ``targets_z[objective]`` (the z-scored ground-truth label)."""
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


def build_dataset(
    rows: Sequence[Dict[str, Any]],
    objective: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Stack rows into ``(X, y, feature_names, candidate_ids)``. Rows with
    no ``targets_z[objective]`` or no candidate id are silently dropped.
    """
    feature_names: Optional[List[str]] = None
    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    ids: List[str] = []
    for row in rows:
        cid = _candidate_id(row)
        if not cid:
            continue
        y_value = extract_target_z(row, objective)
        if y_value is None:
            continue
        feats = extract_features(row)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        x_rows.append([feats[name] for name in feature_names])
        y_rows.append(y_value)
        ids.append(cid)
    if feature_names is None:
        feature_names = []
    if not x_rows:
        return (
            np.zeros((0, len(feature_names)), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            feature_names,
            ids,
        )
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.float64),
        feature_names,
        ids,
    )


# ---------------------------------------------------------------------------
# Hand-coded heuristic scorer (transparent, reproducible)
# ---------------------------------------------------------------------------


def heuristic_score(features: Dict[str, float], objective: str) -> float:
    """
    Per-objective hand-coded scorer using the same metadata features as
    LightGBM. Coefficients are deliberately small and well-rounded so a
    reviewer can sanity-check the rules; they are NOT tuned or learned.

    Reach (viral spread):
        Hashtag-discoverable, longer captions tend to surface broadly.
        General-content gets a small bonus over niche types.

    Engagement (interaction rate):
        Questions invite replies; tutorials pull comment threads; very
        short or very long captions hurt.

    Conversion (action-taking, share-rate proxy):
        Reviews and tutorials drive saves/shares; questions help; the
        evening posting window is generally stronger.
    """
    f = features
    if objective == "reach":
        return (
            0.50 * f.get("hashtag_count", 0.0)
            + 0.10 * f.get("caption_word_count", 0.0)
            + 0.30 * f.get("content_type_general", 0.0)
            + 0.10 * f.get("content_type_review", 0.0)
            - 0.02 * abs(f.get("duration_seconds", 0.0) - 25.0)
        )
    if objective == "engagement":
        word_count = f.get("caption_word_count", 0.0)
        # Sweet-spot caption length: ~10 words, with a mild quadratic penalty.
        length_bonus = -0.05 * (word_count - 10.0) ** 2
        return (
            0.40 * f.get("caption_has_question", 0.0)
            + 0.50 * f.get("content_type_tutorial", 0.0)
            + 0.20 * f.get("content_type_review", 0.0)
            + 0.10 * f.get("hashtag_count", 0.0)
            + length_bonus
        )
    if objective == "conversion":
        # Hour-of-day bonus peaks around 19:00, falls off ±6h.
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


# ---------------------------------------------------------------------------
# LightGBM scorer
# ---------------------------------------------------------------------------


@dataclass
class LightGBMRankerConfig:
    n_estimators: int = 100
    max_depth: int = 4
    num_leaves: int = 15
    learning_rate: float = 0.05
    min_child_samples: int = 10
    n_jobs: int = -1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be >= 2.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be >= 1.")


@dataclass
class LightGBMRanker:
    objective: str
    config: LightGBMRankerConfig = field(default_factory=LightGBMRankerConfig)
    model: Optional[Any] = None
    feature_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVE_TARGETS:
            raise ValueError(
                f"Unknown objective {self.objective!r}; supported: {tuple(OBJECTIVE_TARGETS.keys())}"
            )

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "LightGBMRanker":
        # Deferred import keeps the module importable when lightgbm
        # isn't available (the unit tests mostly don't need it).
        from lightgbm import LGBMRegressor  # noqa: PLC0415

        X, y, names, _ = build_dataset(rows, self.objective)
        if len(X) == 0:
            raise ValueError(
                f"No rows produced a usable target for objective {self.objective!r}."
            )
        self.feature_names = names
        self.model = LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            min_child_samples=self.config.min_child_samples,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def score(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.model is None or not self.feature_names:
            raise RuntimeError("Ranker must be fit before scoring.")
        x_list: List[List[float]] = []
        for row in rows:
            feats = extract_features(row)
            x_list.append([feats.get(name, 0.0) for name in self.feature_names])
        if not x_list:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(self.model.predict(np.asarray(x_list)), dtype=np.float64)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": RANKER_COMPARISON_VERSION,
            "objective": self.objective,
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "model": self.model,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> "LightGBMRanker":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != RANKER_COMPARISON_VERSION:
            raise ValueError(
                f"Version mismatch: saved={payload.get('version')!r} "
                f"current={RANKER_COMPARISON_VERSION!r}"
            )
        objective = payload.get("objective")
        if objective not in OBJECTIVE_TARGETS:
            raise ValueError(f"Saved objective {objective!r} not supported.")
        instance = cls(
            objective=objective,
            config=LightGBMRankerConfig(**(payload.get("config") or {})),
        )
        instance.feature_names = list(payload.get("feature_names") or [])
        instance.model = payload.get("model")
        return instance


# ---------------------------------------------------------------------------
# Relevance grading + ranking metrics
# ---------------------------------------------------------------------------


def assign_relevance_grades(
    z_scores: Sequence[float],
    grades: Sequence[float] = RELEVANCE_GRADES,
) -> List[float]:
    """Quantile-bucket continuous z-scores into discrete relevance grades."""
    if not z_scores:
        return []
    n_grades = len(grades)
    if n_grades == 0:
        raise ValueError("grades must contain at least one value.")
    sorted_scores = sorted(z_scores)
    thresholds: List[float] = []
    for k in range(1, n_grades):
        idx = int(round(k * len(sorted_scores) / n_grades))
        idx = max(0, min(len(sorted_scores) - 1, idx))
        thresholds.append(sorted_scores[idx])
    out: List[float] = []
    for value in z_scores:
        grade_idx = 0
        for threshold in thresholds:
            if value >= threshold:
                grade_idx += 1
        out.append(float(grades[grade_idx]))
    return out


def _ranking_metrics_for_scored(
    scored: Sequence[Tuple[str, float, float]],
    k_values: Sequence[int],
) -> Dict[str, float]:
    """Given (id, predicted_score, relevance_grade) triples, return NDCG/MRR @ each k."""
    if not scored:
        return {}
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    recommended_ids = [item[0] for item in ordered]
    relevance_dict = {item[0]: item[2] for item in ordered}
    relevant_set = {item[0] for item in ordered if item[2] > 0}
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg_at_k(recommended_ids, relevance_dict, k)
        out[f"mrr@{k}"] = mrr_at_k(recommended_ids, relevant_set, k)
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI for the paired lift
# ---------------------------------------------------------------------------


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def paired_bootstrap_lift(
    learned_scored: Sequence[Tuple[str, float, float]],
    heuristic_scored: Sequence[Tuple[str, float, float]],
    k_values: Sequence[int],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Resample the held-out rows with replacement ``n_resamples`` times
    and report the 95% bootstrap CI on (learned − heuristic) NDCG@k /
    MRR@k. The two inputs MUST have the same row IDs in the same order
    so the resample uses paired indices.
    """
    n = len(learned_scored)
    if n == 0 or n != len(heuristic_scored):
        return {}
    rng = np.random.default_rng(random_state)
    metric_keys = []
    for k in k_values:
        metric_keys.append(f"ndcg@{k}")
        metric_keys.append(f"mrr@{k}")

    diffs: Dict[str, List[float]] = {key: [] for key in metric_keys}
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        learned_sample = [learned_scored[i] for i in idx]
        heuristic_sample = [heuristic_scored[i] for i in idx]
        learned_metrics = _ranking_metrics_for_scored(learned_sample, k_values)
        heuristic_metrics = _ranking_metrics_for_scored(heuristic_sample, k_values)
        for key in metric_keys:
            diffs[key].append(
                float(learned_metrics.get(key, 0.0) - heuristic_metrics.get(key, 0.0))
            )

    out: Dict[str, Dict[str, float]] = {}
    for key, values in diffs.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "lift_mean": float(arr.mean()) if arr.size else 0.0,
            "lift_ci_low": _percentile(arr, 2.5),
            "lift_ci_high": _percentile(arr, 97.5),
            "lift_positive_share": float(np.mean(arr > 0.0)) if arr.size else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# End-to-end comparison runner
# ---------------------------------------------------------------------------


def _scored_for(
    rows: Sequence[Dict[str, Any]],
    objective: str,
    scorer: str,
    learned_ranker: Optional[LightGBMRanker] = None,
) -> List[Tuple[str, float, float]]:
    """Build ``(id, predicted_score, relevance_grade)`` triples for one scorer."""
    _, y_eval, _, ids = build_dataset(rows, objective)
    if len(ids) == 0:
        return []

    # Filter rows down to the same IDs we used for y to keep alignment exact.
    keep_set = set(ids)
    aligned_rows = [row for row in rows if _candidate_id(row) in keep_set]

    if scorer == "learned":
        if learned_ranker is None:
            raise ValueError("learned_ranker is required when scorer='learned'.")
        predictions = learned_ranker.score(aligned_rows)
    elif scorer == "heuristic":
        predictions = np.asarray(
            [heuristic_score(extract_features(r), objective) for r in aligned_rows],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown scorer {scorer!r}.")

    grades = assign_relevance_grades(list(y_eval))
    return [
        (cid, float(predictions[i]), grades[i])
        for i, cid in enumerate(ids)
    ]


def compare_rankers(
    train_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    objectives: Sequence[str] = DEFAULT_OBJECTIVES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    config: Optional[LightGBMRankerConfig] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, LightGBMRanker]]:
    """
    Train a per-objective LightGBM ranker on ``train_rows``, score
    ``test_rows`` with both LightGBM and the heuristic, and return a
    metrics dict keyed by objective alongside the trained rankers.
    """
    cfg = config or LightGBMRankerConfig()
    metrics: Dict[str, Dict[str, Any]] = {}
    trained: Dict[str, LightGBMRanker] = {}

    for objective in objectives:
        if objective not in OBJECTIVE_TARGETS:
            continue
        ranker = LightGBMRanker(objective=objective, config=cfg)
        ranker.fit(train_rows)
        trained[objective] = ranker

        learned_scored = _scored_for(test_rows, objective, "learned", ranker)
        heuristic_scored = _scored_for(test_rows, objective, "heuristic")
        if not learned_scored or not heuristic_scored:
            continue

        learned_metrics = _ranking_metrics_for_scored(learned_scored, k_values)
        heuristic_metrics = _ranking_metrics_for_scored(heuristic_scored, k_values)
        bootstrap = paired_bootstrap_lift(
            learned_scored,
            heuristic_scored,
            k_values=k_values,
            n_resamples=n_resamples,
        )

        metrics[objective] = {
            "n_rows": len(learned_scored),
            "learned": learned_metrics,
            "heuristic": heuristic_metrics,
            "lift": bootstrap,
        }

    return metrics, trained


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_comparison_markdown(
    metrics: Dict[str, Dict[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> str:
    """Render the comparison dict as a markdown table."""
    if not metrics:
        return "_No metrics — comparison produced no usable rows._"
    rows: List[str] = []
    headers = ["Objective", "N", "Variant"]
    for k in k_values:
        headers.append(f"NDCG@{k}")
    for k in k_values:
        headers.append(f"MRR@{k}")
    rows.append("| " + " | ".join(headers) + " |")
    rows.append(
        "|" + "|".join(["---"] + [" ---: "] * (len(headers) - 1)) + "|"
    )
    for objective, payload in metrics.items():
        n = int(payload.get("n_rows", 0))
        for variant in ("learned", "heuristic"):
            v_metrics = payload.get(variant, {})
            cells = [objective, str(n), variant]
            for k in k_values:
                cells.append(f"{v_metrics.get(f'ndcg@{k}', 0.0):.4f}")
            for k in k_values:
                cells.append(f"{v_metrics.get(f'mrr@{k}', 0.0):.4f}")
            rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("**Lift (LightGBM minus heuristic) with 95% paired bootstrap CI:**")
    rows.append("")
    rows.append("| Objective | Metric | Lift | 95% CI | P(lift > 0) |")
    rows.append("|---|---|---:|:---:|---:|")
    for objective, payload in metrics.items():
        lift = payload.get("lift", {})
        for metric_name, lift_payload in lift.items():
            rows.append(
                "| {obj} | {metric} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.2f} |".format(
                    obj=objective,
                    metric=metric_name,
                    mean=lift_payload.get("lift_mean", 0.0),
                    lo=lift_payload.get("lift_ci_low", 0.0),
                    hi=lift_payload.get("lift_ci_high", 0.0),
                    p=lift_payload.get("lift_positive_share", 0.0),
                )
            )
    return "\n".join(rows)


__all__ = [
    "CONTENT_TYPES",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_K_VALUES",
    "DEFAULT_OBJECTIVES",
    "LightGBMRanker",
    "LightGBMRankerConfig",
    "OBJECTIVE_TARGETS",
    "RANKER_COMPARISON_VERSION",
    "RELEVANCE_GRADES",
    "assign_relevance_grades",
    "build_dataset",
    "compare_rankers",
    "extract_features",
    "extract_target_z",
    "format_comparison_markdown",
    "heuristic_score",
    "paired_bootstrap_lift",
]
