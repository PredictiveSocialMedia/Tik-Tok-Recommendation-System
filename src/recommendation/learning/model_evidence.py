"""Evidence reporting helpers for model comparison runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .hashtag_ab_testing import normalize_hashtag


def metric_delta(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
) -> Dict[str, float]:
    """Return candidate-minus-baseline deltas for shared numeric metrics."""
    deltas: Dict[str, float] = {}
    for key, base_value in baseline.items():
        cand_value = candidate.get(key)
        if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
            deltas[key] = float(cand_value) - float(base_value)
    return deltas


def summarize_improvement(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    primary_metric: str,
) -> str:
    """Classify whether candidate improves, regresses, or ties on a metric."""
    if primary_metric not in baseline or primary_metric not in candidate:
        return "unknown"
    delta = float(candidate[primary_metric]) - float(baseline[primary_metric])
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "regressed"
    return "tied"


def format_metric_report(report: Mapping[str, Any]) -> str:
    """Render a compact markdown report for evidence artifacts."""
    title = str(report.get("title") or "Model evidence report")
    lines = [f"# {title}", ""]
    status = report.get("status")
    if status:
        lines.append(f"- Status: **{status}**")
    data = report.get("data") or {}
    if data:
        for key in ("train_rows", "validation_rows", "test_rows", "dataset_rows"):
            if key in data:
                lines.append(f"- {key.replace('_', ' ').title()}: **{data[key]}**")
    split = report.get("split") or {}
    if split:
        split_parts = [f"{key}={value}" for key, value in sorted(split.items())]
        lines.append(f"- Split: `{', '.join(split_parts)}`")
    lines.append("")

    rows = report.get("models") or []
    metric_keys: List[str] = []
    for row in rows:
        for key, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and key not in metric_keys:
                metric_keys.append(key)
    if rows and metric_keys:
        header = ["Model", "Role", *metric_keys]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            values = [
                str(row.get("name") or ""),
                str(row.get("role") or ""),
            ]
            metrics = row.get("metrics") or {}
            values.extend(
                f"{float(metrics[key]):.4f}" if key in metrics else ""
                for key in metric_keys
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    comparison = report.get("comparison") or {}
    if comparison:
        verdict = comparison.get("verdict")
        primary = comparison.get("primary_metric")
        if verdict and primary:
            lines.append(f"- Primary comparison: **{verdict}** on `{primary}`")
        deltas = comparison.get("deltas") or {}
        if deltas:
            lines.append("- Deltas:")
            for key, value in sorted(deltas.items()):
                lines.append(f"  - `{key}`: {float(value):+.4f}")

    notes = report.get("notes") or []
    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class TopicPriorHashtagBaseline:
    """Hashtag baseline using train-split topic and global priors."""

    rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        self.topic_counts: Dict[str, Dict[str, int]] = {}
        self.global_counts: Dict[str, int] = {}
        for row in self.rows:
            topic = str(row.get("topic_key") or "unknown").strip().lower() or "unknown"
            bucket = self.topic_counts.setdefault(topic, {})
            for tag in extract_tags(row):
                bucket[tag] = bucket.get(tag, 0) + 1
                self.global_counts[tag] = self.global_counts.get(tag, 0) + 1

    def recommend(self, row: Mapping[str, Any], k: int) -> List[str]:
        topic = str(row.get("topic_key") or "unknown").strip().lower() or "unknown"
        counts = dict(self.global_counts)
        for tag, count in self.topic_counts.get(topic, {}).items():
            counts[tag] = counts.get(tag, 0) + count * 2
        return [
            tag
            for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]
        ]


def extract_tags(row: Mapping[str, Any]) -> List[str]:
    """Extract normalized no-hash tags from row fields and caption text."""
    tags: List[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        tag = normalize_hashtag(str(value)).lstrip("#")
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    raw_tags = row.get("hashtags") or []
    if isinstance(raw_tags, str):
        raw_tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
    for tag in raw_tags:
        add(tag)
    for word in str(row.get("caption") or "").split():
        if word.startswith("#"):
            add(word)
    return tags


def evaluate_tag_predictor(
    rows: Iterable[Mapping[str, Any]],
    predictor: Callable[[Mapping[str, Any], int], Sequence[str]],
    *,
    k: int,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for a hashtag predictor."""
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for row in rows:
        actual = set(extract_tags(row))
        if not actual:
            continue
        predicted = [normalize_hashtag(tag).lstrip("#") for tag in predictor(row, k)]
        predicted_set = {tag for tag in predicted[:k] if tag}
        tp = len(predicted_set & actual)
        precision = tp / max(1, len(predicted_set))
        recall = tp / max(1, len(actual))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if not precisions:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"f1@{k}": 0.0}
    n = float(len(precisions))
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        f"f1@{k}": sum(f1s) / n,
    }


def build_comparison(
    baseline: Mapping[str, float],
    candidate: Optional[Mapping[str, float]],
    *,
    primary_metric: str,
) -> Dict[str, Any]:
    """Build a serializable comparison payload."""
    if candidate is None:
        return {
            "primary_metric": primary_metric,
            "verdict": "candidate_missing",
            "deltas": {},
        }
    return {
        "primary_metric": primary_metric,
        "verdict": summarize_improvement(baseline, candidate, primary_metric),
        "deltas": metric_delta(baseline, candidate),
    }
