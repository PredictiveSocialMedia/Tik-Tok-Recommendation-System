from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .baseline_common import (
    DEFAULT_RANKING_WEIGHTS,
    OBJECTIVE_RANKING_WEIGHTS,
    as_float,
    clamp,
    round_score,
    sanitize_probability,
)
from .candidate_support import prepare_candidate
from .query_contract import build_query_profile
from .retrieval_baseline import (
    audience_compatibility,
    content_type_compatibility,
    cta_alignment,
    locale_compatibility,
    objective_compatibility,
    retrieve_shortlist,
)
from .temporal import parse_dt, row_text

BASELINE_LOGISTIC_RANKER_ID = "baseline_logreg"
BASELINE_LOGISTIC_RANKER_VERSION = "recommender.ranker.logreg.v1"
BASELINE_LOGISTIC_FEATURE_NAMES = [
    "semantic_relevance",
    "intent_alignment",
    "performance_quality",
    "reference_usefulness",
    "support_confidence",
]


@dataclass
class ConstantProbabilityClassifier:
    positive_probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        row_count = int(x.shape[0]) if hasattr(x, "shape") else len(x)
        positive = sanitize_probability(self.positive_probability, 0.0)
        negative = round_score(1.0 - positive, 6)
        return np.tile(np.array([[negative, positive]], dtype=np.float32), (row_count, 1))


def support_confidence_score(level: str, score: float) -> float:
    tier_floor = 0.82 if level == "full" else 0.52 if level == "partial" else 0.0
    return round_score(clamp((tier_floor * 0.55) + (score * 0.45), 0.0, 1.0), 6)


FRESHNESS_HALF_LIFE_DAYS = 18.0


def freshness_score(posted_at: Optional[datetime], reference_date: datetime) -> float:
    if posted_at is None:
        return 0.55
    age_days = max(0.0, (reference_date - posted_at).total_seconds() / 86400.0)
    return round_score(
        clamp(math.exp((-math.log(2.0) * age_days) / FRESHNESS_HALF_LIFE_DAYS), 0.0, 1.0),
        6,
    )


def performance_quality_score(candidate: Dict[str, Any]) -> float:
    metrics = candidate.get("engagement_metrics") or {}
    views = as_float(metrics.get("views"), 0.0)
    engagement_rate = as_float(metrics.get("engagement_rate"), 0.0)
    view_signal = math.log1p(views) / math.log1p(10_000_000) if views > 0 else 0.0
    er_signal = min(engagement_rate / 0.10, 1.0)
    return round_score(clamp(view_signal * 0.55 + er_signal * 0.45, 0.0, 1.0), 6)


def reference_usefulness(candidate: Dict[str, Any], reference_date: datetime) -> float:
    comment_trace = candidate["comment_trace"]
    metadata_quality = sanitize_probability(candidate["support_score"], 0.0)
    freshness = freshness_score(candidate.get("posted_at"), reference_date)
    comment_richness = sanitize_probability(comment_trace.get("value_prop_coverage"), 0.0)
    share_signal = sanitize_probability(comment_trace.get("on_topic_ratio"), 0.0)
    fabric = candidate.get("fabric_signals") or {}
    content_quality = clamp(
        (as_float(fabric.get("clarity_score"), 0.5) * 0.5)
        + (as_float(fabric.get("pacing_score"), 0.5) * 0.3)
        + (min(as_float(fabric.get("cta_keyword_count"), 0), 3) / 3.0 * 0.2),
        0.0,
        1.0,
    )
    return round_score(
        clamp(
            (metadata_quality * 0.25)
            + (freshness * 0.20)
            + (content_quality * 0.20)
            + (comment_richness * 0.15)
            + (share_signal * 0.10)
            + (candidate["support_score"] * 0.10),
            0.0,
            1.0,
        ),
        6,
    )


def score_components_for_candidate(
    *,
    query_profile: Dict[str, Any],
    candidate: Dict[str, Any],
    reference_date: datetime,
) -> Dict[str, float]:
    semantic_relevance = round_score(
        clamp(
            (candidate["retrieval_branch_scores"]["semantic"] * 0.70)
            + (candidate["retrieval_branch_scores"]["hashtag_topic"] * 0.30),
            0.0,
            1.0,
        ),
        6,
    )
    intent_alignment = round_score(
        clamp(
            (
                objective_compatibility(
                    query_profile["objective"], candidate["objective_guess"]
                )
                * 0.25
            )
            + (
                content_type_compatibility(
                    query_profile["content_type"], candidate["content_type"]
                )
                * 0.30
            )
            + (cta_alignment(query_profile["primary_cta"], candidate["text"]) * 0.20)
            + (
                audience_compatibility(query_profile["audience"], candidate["audience_tokens"])
                * 0.15
            )
            + (
                locale_compatibility(
                    query_profile["locale"],
                    query_profile["language"],
                    candidate["locale"],
                    candidate["language"],
                )
                * 0.10
            ),
            0.0,
            1.0,
        ),
        6,
    )
    return {
        "semantic_relevance": semantic_relevance,
        "intent_alignment": intent_alignment,
        "performance_quality": performance_quality_score(candidate),
        "reference_usefulness": reference_usefulness(candidate, reference_date),
        "support_confidence": support_confidence_score(
            candidate["support_level"], candidate["support_score"]
        ),
    }


def ranking_reasons(candidate: Dict[str, Any], score_components: Dict[str, float]) -> List[str]:
    ordered = sorted(score_components.items(), key=lambda item: item[1], reverse=True)
    reasons = [f"strong_{name}" for name, _ in ordered[:2]]
    if len(candidate["retrieval_branches"]) >= 2:
        reasons.append("multi_branch_retrieval_match")
    if candidate["support_level"] == "full":
        reasons.append("fully_supported_reference")
    return reasons


def _feature_vector_from_components(score_components: Dict[str, float]) -> np.ndarray:
    return np.array(
        [float(score_components.get(name, 0.0)) for name in BASELINE_LOGISTIC_FEATURE_NAMES],
        dtype=np.float32,
    )


def _component_weights_from_model(model: LogisticRegression) -> Dict[str, float]:
    if not hasattr(model, "coef_"):
        return dict(DEFAULT_RANKING_WEIGHTS)
    raw = [abs(float(value)) for value in list(model.coef_[0])]
    denom = sum(raw)
    if denom <= 0.0:
        return dict(DEFAULT_RANKING_WEIGHTS)
    return {
        name: round_score(raw[idx] / denom, 6)
        for idx, name in enumerate(BASELINE_LOGISTIC_FEATURE_NAMES)
    }


def _query_payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    topic_key = str(row.get("topic_key") or "").strip()
    return {
        "query_id": str(row.get("row_id") or "query"),
        "text": row_text(row),
        "description": str(row.get("caption") or row_text(row)),
        "hashtags": list(row.get("hashtags") or ([] if not topic_key else [f"#{topic_key}"])),
        "mentions": [],
        "content_type": row.get("content_type"),
        "primary_cta": "none",
        "language": row.get("language"),
        "locale": row.get("locale"),
        "topic_key": topic_key or None,
        "keywords": list(row.get("keywords") or []),
    }


def _candidate_payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    topic_key = str(row.get("topic_key") or "").strip()
    features = row.get("features")
    if not isinstance(features, dict):
        features = {}
    return {
        "candidate_id": str(row.get("row_id") or ""),
        "row_id": str(row.get("row_id") or ""),
        "video_id": str(row.get("video_id") or ""),
        "author_id": str(row.get("author_id") or ""),
        "text": row_text(row),
        "caption": str(row.get("caption") or row_text(row)),
        "hashtags": list(row.get("hashtags") or ([] if not topic_key else [f"#{topic_key}"])),
        "keywords": list(row.get("keywords") or ([] if not topic_key else [topic_key])),
        "search_query": row.get("search_query"),
        "posted_at": row.get("posted_at"),
        "as_of_time": row.get("as_of_time"),
        "language": row.get("language") or features.get("language"),
        "locale": row.get("locale"),
        "content_type": row.get("content_type"),
        "topic_key": topic_key or None,
        "signal_hints": {
            key: value
            for key, value in {
                "comment_intelligence": features.get("comment_intelligence"),
                "trajectory_features": features.get("trajectory_features"),
            }.items()
            if isinstance(value, dict)
        },
    }


def _baseline_training_frame(
    *,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    target_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_rows_by_query: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        if str(pair.get("target_source") or "scalar_v1") != target_source:
            continue
        query_id = str(pair.get("query_row_id") or "").strip()
        if not query_id:
            continue
        pair_rows_by_query.setdefault(query_id, []).append(pair)

    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    sample_weights: List[float] = []
    for query_id, query_pairs in pair_rows_by_query.items():
        query_row = rows_by_id.get(query_id)
        if query_row is None or str(query_row.get("split")) != "train":
            continue
        query_as_of = parse_dt(query_row.get("as_of_time"))
        if query_as_of is None:
            continue
        query_payload = _query_payload_from_row(query_row)
        query_profile = build_query_profile(
            objective=objective,
            query=query_payload,
            fallback_language=query_payload.get("language"),
            fallback_locale=query_payload.get("locale"),
            fallback_content_type=query_payload.get("content_type"),
        )

        prepared_candidates: List[Dict[str, Any]] = []
        labels_by_candidate: Dict[str, int] = {}
        for pair in query_pairs:
            candidate_id = str(pair.get("candidate_row_id") or "").strip()
            candidate_row = rows_by_id.get(candidate_id)
            if candidate_row is None:
                continue
            prepared = prepare_candidate(
                payload=_candidate_payload_from_row(candidate_row),
                as_of=query_as_of,
                query_profile=query_profile,
                manifest_comment_lookup=lambda _row_id, _point_in_time: None,
            )
            if prepared is None:
                continue
            prepared_candidates.append(prepared)
            labels_by_candidate[str(prepared.get("candidate_id") or candidate_id)] = int(
                pair.get("relevance_label") or 0
            )

        usable_candidates = [
            item for item in prepared_candidates if str(item.get("support_level") or "") != "low"
        ]
        if len(usable_candidates) < 2:
            continue

        shortlist, _ = retrieve_shortlist(
            usable_candidates=usable_candidates,
            query_profile=query_profile,
            retrieve_k=len(usable_candidates),
        )
        reference_date = max(
            [
                item["posted_at"]
                for item in shortlist
                if isinstance(item.get("posted_at"), datetime)
            ]
            or [query_as_of]
        )
        for item in shortlist:
            candidate_id = str(item.get("candidate_id") or "").strip()
            label = labels_by_candidate.get(candidate_id)
            if label is None:
                continue
            components = score_components_for_candidate(
                query_profile=query_profile,
                candidate=item,
                reference_date=reference_date,
            )
            x_rows.append(_feature_vector_from_components(components))
            y_rows.append(1 if int(label) >= 2 else 0)
            sample_weights.append(1.0 + (0.5 if int(label) >= 3 else 0.0))

    if not x_rows:
        return (
            np.zeros((0, len(BASELINE_LOGISTIC_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.stack(x_rows, axis=0),
        np.asarray(y_rows, dtype=np.int32),
        np.asarray(sample_weights, dtype=np.float32),
    )


@dataclass
class BaselineLogisticRanker:
    objective: str
    model: Any
    model_type: str
    train_summary: Dict[str, Any]

    @classmethod
    def train(
        cls,
        *,
        objective: str,
        rows_by_id: Dict[str, Dict[str, Any]],
        pair_rows: Sequence[Dict[str, Any]],
        target_source: str,
    ) -> Optional["BaselineLogisticRanker"]:
        x_train, y_train, sample_weight = _baseline_training_frame(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=objective,
            target_source=target_source,
        )
        if x_train.shape[0] < 4:
            return None
        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            model = ConstantProbabilityClassifier(float(np.mean(y_train)))
            model_type = "constant_probability_classifier"
        else:
            model = LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=13,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            model_type = "logistic_regression_classifier"
        train_scores = model.predict_proba(x_train)[:, 1]
        weighted_correct = float(
            np.sum(((train_scores >= 0.5) == (y_train >= 1)).astype(np.float32) * sample_weight)
        )
        weighted_total = float(np.sum(sample_weight))
        train_summary = {
            "row_count": int(x_train.shape[0]),
            "positive_row_count": int(np.sum(y_train == 1)),
            "negative_row_count": int(np.sum(y_train == 0)),
            "weighted_train_accuracy": round_score(
                0.0 if weighted_total <= 0.0 else weighted_correct / weighted_total, 6
            ),
            "feature_names": list(BASELINE_LOGISTIC_FEATURE_NAMES),
            "coefficient_weights": _component_weights_from_model(model),
            "degenerate_single_class": bool(len(unique_labels) < 2),
        }
        return cls(
            objective=objective,
            model=model,
            model_type=model_type,
            train_summary=train_summary,
        )

    def predict_score(self, score_components: Dict[str, float]) -> float:
        vector = _feature_vector_from_components(score_components).reshape(1, -1)
        return sanitize_probability(float(self.model.predict_proba(vector)[0][1]), 0.0)

    def score_weights(self) -> Dict[str, float]:
        return _component_weights_from_model(self.model)

    def save(self, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "ranker_id": BASELINE_LOGISTIC_RANKER_ID,
            "version": BASELINE_LOGISTIC_RANKER_VERSION,
            "objective": self.objective,
            "feature_names": list(BASELINE_LOGISTIC_FEATURE_NAMES),
            "model_type": self.model_type,
            "train_summary": dict(self.train_summary),
            "fallback_weights": dict(
                OBJECTIVE_RANKING_WEIGHTS.get(self.objective, DEFAULT_RANKING_WEIGHTS)
            ),
        }
        (output_dir / "baseline_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "baseline_model.pkl").open("wb") as handle:
            pickle.dump(self.model, handle)
        return manifest

    @classmethod
    def load(cls, output_dir: Path) -> "BaselineLogisticRanker":
        manifest = json.loads((output_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
        with (output_dir / "baseline_model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        return cls(
            objective=str(manifest["objective"]),
            model=model,
            model_type=str(
                manifest.get("model_type") or "logistic_regression_classifier"
            ),
            train_summary=dict(manifest.get("train_summary") or {}),
        )


def rank_shortlist(
    *,
    shortlist: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
    effective_objective: str,
    portfolio: Optional[Dict[str, Any]],
    rankers_available: Sequence[str],
    baseline_ranker: Optional[BaselineLogisticRanker] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reference_date = max(
        [item["posted_at"] for item in shortlist if isinstance(item.get("posted_at"), datetime)]
        or [datetime.now(timezone.utc)]
    )
    ranking_weights = (
        baseline_ranker.score_weights()
        if baseline_ranker is not None
        else OBJECTIVE_RANKING_WEIGHTS.get(effective_objective, DEFAULT_RANKING_WEIGHTS)
    )
    ranker_id = (
        BASELINE_LOGISTIC_RANKER_ID if baseline_ranker is not None else "baseline_weighted"
    )
    ranked: List[Dict[str, Any]] = []
    for item in shortlist:
        components = score_components_for_candidate(
            query_profile=query_profile,
            candidate=item,
            reference_date=reference_date,
        )
        normalized_components = {
            key: sanitize_probability(value, 0.0) for key, value in components.items()
        }
        raw_score = (
            baseline_ranker.predict_score(normalized_components)
            if baseline_ranker is not None
            else round_score(
                clamp(
                    sum(components[k] * ranking_weights.get(k, 0.0) for k in components),
                    0.0,
                    1.0,
                ),
                6,
            )
        )
        ranked.append(
            {
                **item,
                "score_components": normalized_components,
                "score_raw": raw_score,
                "score_calibrated": raw_score,
                "score": raw_score,
                "score_mean": raw_score,
                "score_std": 0.0,
                "confidence": normalized_components["support_confidence"],
                "global_score_mean": raw_score,
                "segment_blend_weight": 0.0,
                "selected_ranker_id": ranker_id,
                "policy_penalty": 0.0,
                "policy_bonus": 0.0,
                "policy_adjusted_score": raw_score,
                "calibration_trace": {
                    "score_raw": raw_score,
                    "score_calibrated": raw_score,
                    "calibrator_segment_id": "baseline_identity",
                    "requested_segment_id": ranker_id,
                    "calibrator_method": "identity",
                    "calibrator_support_count": 0,
                    "calibration_fallback_used": False,
                },
                "policy_trace": {
                    "policy_version": "policy.baseline.v1",
                    "applied_rules": [],
                },
                "ranking_reasons": ranking_reasons(item, normalized_components),
            }
        )

    portfolio_payload = portfolio if isinstance(portfolio, dict) else {}
    portfolio_requested = bool(portfolio_payload.get("enabled", False))
    available_rankers = set(str(item) for item in rankers_available)
    portfolio_supported = (
        portfolio_requested and "reach" in available_rankers and "conversion" in available_rankers
    )
    portfolio_fallback_reason = None
    if portfolio_requested and not portfolio_supported:
        portfolio_fallback_reason = next(
            (
                f"missing_ranker_{objective_id}"
                for objective_id in ("reach", "conversion")
                if objective_id not in available_rankers
            ),
            "portfolio_unavailable",
        )

    portfolio_weights_raw = (
        portfolio_payload.get("weights")
        if isinstance(portfolio_payload.get("weights"), dict)
        else {}
    )
    reach_weight = max(0.0, as_float(portfolio_weights_raw.get("reach"), 0.45))
    conversion_weight = max(0.0, as_float(portfolio_weights_raw.get("conversion"), 0.35))
    durability_weight = max(0.0, as_float(portfolio_weights_raw.get("durability"), 0.20))
    total_weight = max(1e-9, reach_weight + conversion_weight + durability_weight)
    portfolio_weights = {
        "reach": reach_weight / total_weight,
        "conversion": conversion_weight / total_weight,
        "durability": durability_weight / total_weight,
    }
    risk_aversion = max(0.0, as_float(portfolio_payload.get("risk_aversion"), 0.10))

    for item in ranked:
        utility_before = float(item["score"])
        if portfolio_supported:
            reach_score = item["score_components"]["semantic_relevance"]
            conversion_score = item["score_components"]["intent_alignment"]
            durability_score = item["score_components"]["reference_usefulness"]
            utility_before = round_score(
                (portfolio_weights["reach"] * reach_score)
                + (portfolio_weights["conversion"] * conversion_score)
                + (portfolio_weights["durability"] * durability_score)
                - (risk_aversion * max(0.0, 1.0 - item["confidence"])),
                6,
            )
            item["portfolio_trace"] = {
                "portfolio_version": "policy.baseline.v1",
                "weights": {key: round_score(value, 6) for key, value in portfolio_weights.items()},
                "risk_aversion": round_score(risk_aversion, 6),
                "utility_before_policy": utility_before,
                "utility_after_policy": utility_before,
                "components": {
                    "reach_score": reach_score,
                    "conversion_score": conversion_score,
                    "durability_score": durability_score,
                },
            }
        item["portfolio_utility"] = utility_before

    ranked.sort(
        key=lambda item: (
            -(float(item["portfolio_utility"]) if portfolio_supported else float(item["score"])),
            -float(item["score"]),
            -float(item["retrieval_branch_scores"]["fused_retrieval"]),
        )
    )

    return ranked, {
        "weights": ranking_weights,
        "ranker_id": ranker_id,
        "portfolio_requested": portfolio_requested,
        "portfolio_supported": portfolio_supported,
        "portfolio_fallback_reason": portfolio_fallback_reason,
        "portfolio_weights": portfolio_weights,
        "risk_aversion": risk_aversion,
        "candidate_pool_cap": int(portfolio_payload.get("candidate_pool_cap", 120) or 120),
    }
