"""A/B evaluation for hashtag recommender variants.

The harness compares a production hashtag recommender against a lexical TF-IDF
baseline on held-out rows with known caption hashtags. It is intentionally
dependency-light so the metric contract is testable without FAISS/SBERT.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

HASHTAG_AB_TEST_VERSION = "hashtag_ab_test.v1"


def normalize_hashtag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text if text.startswith("#") else f"#{text}"
    return re.sub(r"[^\w#]+", "", text)


def extract_hashtags(row: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    caption = str(row.get("caption") or "")
    tags.extend(re.findall(r"#\w+", caption.lower()))
    raw = row.get("hashtags") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",")]
    tags.extend(normalize_hashtag(item) for item in raw)
    return list(dict.fromkeys(tag for tag in tags if tag and tag != "#"))


def row_text(row: Dict[str, Any]) -> str:
    parts = [str(row.get("caption") or "")]
    parts.extend(str(item) for item in row.get("keywords") or [])
    if row.get("topic_key"):
        parts.append(str(row["topic_key"]))
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.sub(r"[^0-9a-zA-Z#_]+", " ", text.lower()).split() if tok]


class TfidfHashtagBaseline:
    """Nearest-neighbour hashtag recommender using TF-IDF cosine similarity."""

    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        docs: List[List[str]] = []
        self.tags: List[List[str]] = []
        for row in rows:
            tags = extract_hashtags(row)
            if not tags:
                continue
            docs.append(_tokenize(row_text(row)))
            self.tags.append(tags)
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        self.vocab = {term: idx for idx, term in enumerate(sorted(df))}
        self.idf = {
            term: math.log((1 + len(docs)) / (1 + count)) + 1.0
            for term, count in df.items()
        }
        self.matrix = np.vstack([self._vector_from_tokens(tokens) for tokens in docs]) if docs else np.zeros((0, 0))

    def _vector_from_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros((len(self.vocab),), dtype=np.float64)
        if not tokens:
            return vec
        counts = Counter(tokens)
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (count / len(tokens)) * self.idf.get(term, 1.0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def recommend(self, row: Dict[str, Any], *, k: int = 10, neighbours: int = 25) -> List[str]:
        if self.matrix.size == 0:
            return []
        q = self._vector_from_tokens(_tokenize(row_text(row)))
        scores = self.matrix @ q
        order = np.argsort(-scores)[: max(1, neighbours)]
        tag_scores: Dict[str, float] = defaultdict(float)
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            for tag in self.tags[idx]:
                tag_scores[tag] += score / rank
        ranked = sorted(tag_scores.items(), key=lambda item: (-item[1], item[0]))
        return [tag for tag, _ in ranked[:k]]


@dataclass
class HashtagVariant:
    name: str
    recommend: Callable[[Dict[str, Any], int], Sequence[str]]
    description: str = ""


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall <= 0 else (2 * precision * recall) / (precision + recall)


def _avg_pairwise_distance(tags: Sequence[str]) -> float:
    clean = [tag.lstrip("#") for tag in tags if tag]
    if len(clean) < 2:
        return 0.0
    distances: List[float] = []
    for i, left in enumerate(clean):
        left_grams = {left[j:j + 3] for j in range(max(1, len(left) - 2))}
        for right in clean[i + 1:]:
            right_grams = {right[j:j + 3] for j in range(max(1, len(right) - 2))}
            union = left_grams | right_grams
            sim = (len(left_grams & right_grams) / len(union)) if union else 0.0
            distances.append(1.0 - sim)
    return float(np.mean(distances)) if distances else 0.0


def evaluate_hashtag_variants(
    variants: Sequence[HashtagVariant],
    rows: Sequence[Dict[str, Any]],
    *,
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, Any]:
    evaluable = [row for row in rows if extract_hashtags(row)]
    actual_by_row = [set(extract_hashtags(row)) for row in evaluable]
    all_actual = set().union(*actual_by_row) if actual_by_row else set()
    reports: List[Dict[str, Any]] = []
    for variant in variants:
        metric_lists: Dict[str, List[float]] = defaultdict(list)
        unique_predicted: set[str] = set()
        for row, actual in zip(evaluable, actual_by_row):
            max_k = max(k_values)
            predicted_all = [normalize_hashtag(tag) for tag in variant.recommend(row, max_k)]
            predicted_all = [tag for tag in dict.fromkeys(predicted_all) if tag]
            unique_predicted.update(predicted_all)
            for k in k_values:
                predicted = predicted_all[:k]
                hits = len(set(predicted) & actual)
                precision = hits / max(1, len(predicted))
                recall = hits / max(1, len(actual))
                metric_lists[f"precision@{k}"].append(precision)
                metric_lists[f"recall@{k}"].append(recall)
                metric_lists[f"f1@{k}"].append(_f1(precision, recall))
                metric_lists[f"diversity@{k}"].append(_avg_pairwise_distance(predicted))
        metrics = {
            key: round(float(np.mean(values)), 6) if values else 0.0
            for key, values in sorted(metric_lists.items())
        }
        metrics["catalog_coverage"] = round(len(unique_predicted) / max(1, len(all_actual)), 6)
        reports.append({
            "name": variant.name,
            "description": variant.description,
            "metrics": metrics,
        })
    return {
        "version": HASHTAG_AB_TEST_VERSION,
        "n_rows": len(rows),
        "n_evaluable_rows": len(evaluable),
        "k_values": list(k_values),
        "unique_ground_truth_hashtags": len(all_actual),
        "variants": reports,
    }


def format_hashtag_ab_markdown(report: Dict[str, Any]) -> str:
    if not report.get("variants"):
        return "_No hashtag A/B report - variants list was empty._\n"
    k_values = list(report.get("k_values") or [])
    headers = ["Variant"]
    for k in k_values:
        headers.extend([f"Precision@{k}", f"Recall@{k}", f"F1@{k}", f"Diversity@{k}"])
    headers.append("Catalog Coverage")
    lines = ["# Hashtag recommender A/B test\n"]
    lines.append(f"- Evaluable rows: **{report.get('n_evaluable_rows', 0)}** of {report.get('n_rows', 0)}")
    lines.append(f"- Unique ground-truth hashtags: **{report.get('unique_ground_truth_hashtags', 0)}**")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for variant in report["variants"]:
        metrics = variant.get("metrics") or {}
        row = [str(variant.get("name") or "")]
        for k in k_values:
            row.extend([
                f"{metrics.get(f'precision@{k}', 0.0):.3f}",
                f"{metrics.get(f'recall@{k}', 0.0):.3f}",
                f"{metrics.get(f'f1@{k}', 0.0):.3f}",
                f"{metrics.get(f'diversity@{k}', 0.0):.3f}",
            ])
        row.append(f"{metrics.get('catalog_coverage', 0.0):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"

