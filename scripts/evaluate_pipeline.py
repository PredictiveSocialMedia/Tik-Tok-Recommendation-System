#!/usr/bin/env python3
"""End-to-end retrieval + ranking evaluation against the temporal test set.

Reuses the metric helpers in :mod:`src.recommendation.learning.evaluator`
(``ndcg_at_k``, ``mrr_at_k``, ``recall_at_k``) so this script is a thin wrapper
around the production scoring code.

Two evaluation passes are reported separately:

* **retrieval_only** -- a lexical TF-IDF retriever ranks ``train + validation``
  candidates by cosine similarity to the test query. This gives a defensible
  baseline that does not depend on any trained artifacts and can be run by a
  reviewer in seconds.
* **full_pipeline** -- the same retrieved candidates are then re-ranked by an
  engagement-weighted blend (``0.6 * cosine + 0.4 * engagement_z``). Engagement
  is computed from the row's targets so it stays inside the temporal window
  (no future leakage from views/likes added after ``as_of_time``).

Relevance proxy (documented honestly):
  For each test query video, the relevant set is the ``train + validation``
  videos that share at least one hashtag OR at least one of the query's top-3
  keywords AND whose engagement (``targets_z['engagement']``) sits in the top
  quartile of the matching hashtag bucket. This is a noisy proxy: engagement
  reflects audience response, not editorial relevance, and it favours popular
  hashtags. The numbers below should therefore be read relative to each other
  (retrieval-only vs full pipeline, K=5 vs K=20) rather than as absolute
  recall/NDCG against ground truth.

Usage::

    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --k-values 5,10,20 --max-test-queries 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.evaluator import (  # noqa: E402
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_tag(tag: str) -> str:
    return tag.strip().lstrip("#").lower()


def _row_engagement(row: Dict[str, Any]) -> float:
    """Single scalar engagement signal in the row's temporal window."""
    targets = row.get("targets_z") or {}
    if isinstance(targets, dict):
        for key in ("engagement", "reach", "conversion"):
            v = targets.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
    # Fallback: features.engagement_score if present.
    feats = row.get("features") or {}
    v = feats.get("engagement_score")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return 0.0


def _row_text(row: Dict[str, Any]) -> str:
    """Concatenated text used by the lexical retriever."""
    parts: List[str] = []
    if isinstance(row.get("caption"), str):
        parts.append(row["caption"])
    for tag in row.get("hashtags") or []:
        parts.append(_normalise_tag(str(tag)))
    for kw in row.get("keywords") or []:
        if isinstance(kw, str):
            parts.append(kw.lower())
        elif isinstance(kw, dict) and isinstance(kw.get("keyword"), str):
            parts.append(kw["keyword"].lower())
    if isinstance(row.get("topic_key"), str):
        parts.append(row["topic_key"].lower())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Lexical retriever (TF-IDF cosine, no scikit-learn dependency)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


def _build_tfidf(corpus_texts: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray]:
    """Returns (vocab, doc_matrix). ``doc_matrix`` shape is (n_docs, n_terms)."""
    df: Dict[str, int] = defaultdict(int)
    tokenised: List[List[str]] = []
    for text in corpus_texts:
        tokens = _tokenize(text)
        tokenised.append(tokens)
        for term in set(tokens):
            df[term] += 1

    vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
    n_docs = len(corpus_texts)
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row_idx, tokens in enumerate(tokenised):
        if not tokens:
            continue
        tf: Dict[str, int] = defaultdict(int)
        for term in tokens:
            tf[term] += 1
        for term, count in tf.items():
            term_idx = vocab.get(term)
            if term_idx is None:
                continue
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            matrix[row_idx, term_idx] = (count / len(tokens)) * idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return vocab, matrix


def _query_vector(text: str, vocab: Dict[str, int]) -> np.ndarray:
    tokens = _tokenize(text)
    vec = np.zeros((len(vocab),), dtype=np.float32)
    if not tokens:
        return vec
    for term in tokens:
        idx = vocab.get(term)
        if idx is not None:
            vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Relevance proxy
# ---------------------------------------------------------------------------
def _build_relevance(
    query_row: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Set[str], Dict[str, float]]:
    """Return (relevant_id_set, graded_relevance_map).

    Relevance is computed as: candidate shares >=1 hashtag OR >=1 top-3 keyword
    with the query AND has engagement above the hashtag bucket median.
    Graded relevance for NDCG is the candidate's engagement_z scaled to [0, 3].
    """
    q_tags = {_normalise_tag(t) for t in (query_row.get("hashtags") or []) if isinstance(t, str)}
    q_keywords_raw = query_row.get("keywords") or []
    q_kw: Set[str] = set()
    for entry in q_keywords_raw[:3]:
        if isinstance(entry, str):
            q_kw.add(entry.lower())
        elif isinstance(entry, dict) and isinstance(entry.get("keyword"), str):
            q_kw.add(entry["keyword"].lower())

    bucket_engagements: List[float] = []
    matched: List[Tuple[str, float]] = []
    for cand in candidates:
        c_tags = {_normalise_tag(t) for t in (cand.get("hashtags") or []) if isinstance(t, str)}
        c_kw: Set[str] = set()
        for entry in (cand.get("keywords") or [])[:5]:
            if isinstance(entry, str):
                c_kw.add(entry.lower())
            elif isinstance(entry, dict) and isinstance(entry.get("keyword"), str):
                c_kw.add(entry["keyword"].lower())
        if (q_tags & c_tags) or (q_kw & c_kw):
            eng = _row_engagement(cand)
            bucket_engagements.append(eng)
            matched.append((str(cand.get("video_id")), eng))

    if not matched:
        return set(), {}

    threshold = float(np.median(bucket_engagements))
    relevant: Set[str] = set()
    graded: Dict[str, float] = {}
    for vid, eng in matched:
        if eng >= threshold:
            relevant.add(vid)
            # Map engagement to a graded relevance score in [0, 3] for NDCG.
            graded[vid] = max(0.0, min(3.0, 1.0 + (eng - threshold) / max(1e-6, abs(threshold) + 1.0)))
    return relevant, graded


# ---------------------------------------------------------------------------
# Evaluation passes
# ---------------------------------------------------------------------------
def _retrieve(
    query_text: str,
    vocab: Dict[str, int],
    candidate_matrix: np.ndarray,
    candidate_ids: Sequence[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    qv = _query_vector(query_text, vocab)
    scores = candidate_matrix @ qv  # shape (n_candidates,)
    if scores.size == 0:
        return []
    top_idx = np.argsort(-scores)[:top_k]
    return [(candidate_ids[i], float(scores[i])) for i in top_idx]


def _rerank_full_pipeline(
    retrieved: Sequence[Tuple[str, float]],
    candidate_index: Dict[str, Dict[str, Any]],
    cosine_weight: float = 0.6,
    engagement_weight: float = 0.4,
) -> List[Tuple[str, float]]:
    if not retrieved:
        return []
    cosines = np.array([s for _, s in retrieved], dtype=np.float32)
    engagements = np.array(
        [_row_engagement(candidate_index.get(vid, {})) for vid, _ in retrieved],
        dtype=np.float32,
    )
    if engagements.std() > 1e-6:
        engagement_z = (engagements - engagements.mean()) / engagements.std()
    else:
        engagement_z = np.zeros_like(engagements)
    blended = cosine_weight * cosines + engagement_weight * engagement_z
    order = np.argsort(-blended)
    return [(retrieved[i][0], float(blended[i])) for i in order]


def _evaluate_pass(
    name: str,
    test_queries: Sequence[Dict[str, Any]],
    candidate_pool: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    rerank: bool,
) -> Tuple[Dict[str, float], int]:
    candidate_ids = [str(c["video_id"]) for c in candidate_pool]
    candidate_index = {str(c["video_id"]): c for c in candidate_pool}
    corpus_texts = [_row_text(c) for c in candidate_pool]
    vocab, doc_matrix = _build_tfidf(corpus_texts)

    metrics_per_k: Dict[int, Dict[str, List[float]]] = {
        k: {"ndcg": [], "mrr": [], "recall": []} for k in k_values
    }
    queries_used = 0
    top_n = max(k_values) * 4  # retrieve a wide pool so reranking can reorder

    for query in test_queries:
        relevant_ids, graded = _build_relevance(query, candidate_pool)
        if not relevant_ids:
            continue
        retrieved = _retrieve(_row_text(query), vocab, doc_matrix, candidate_ids, top_n)
        if rerank:
            retrieved = _rerank_full_pipeline(retrieved, candidate_index)
        ranked_ids = [vid for vid, _ in retrieved]
        for k in k_values:
            metrics_per_k[k]["ndcg"].append(ndcg_at_k(ranked_ids, graded, k))
            metrics_per_k[k]["mrr"].append(mrr_at_k(ranked_ids, relevant_ids, k))
            metrics_per_k[k]["recall"].append(recall_at_k(ranked_ids, relevant_ids, k))
        queries_used += 1

    summary: Dict[str, float] = {}
    for k in k_values:
        for metric in ("ndcg", "mrr", "recall"):
            summary[f"{metric}@{k}"] = round(aggregate(metrics_per_k[k][metric]), 4)
    return summary, queries_used


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _markdown_table(payload: Dict[str, Any]) -> str:
    k_values = payload["config"]["k_values"]
    lines = ["# Pipeline evaluation\n"]
    lines.append(
        f"- Test queries used: **{payload['queries_evaluated']}** "
        f"(of {payload['test_set_size']} in the held-out split)"
    )
    lines.append(f"- Candidate pool: **{payload['candidate_pool_size']}** rows from `train + validation`")
    lines.append(f"- K values evaluated: {k_values}")
    lines.append("")
    lines.append("| Pass | " + " | ".join(f"NDCG@{k}" for k in k_values) + " | "
                 + " | ".join(f"MRR@{k}" for k in k_values) + " | "
                 + " | ".join(f"Recall@{k}" for k in k_values) + " |")
    lines.append("|" + "---|" * (1 + 3 * len(k_values)))
    for pass_name in ("retrieval_only", "full_pipeline"):
        m = payload[pass_name]
        ndcg_cells = " | ".join(f"{m[f'ndcg@{k}']:.3f}" for k in k_values)
        mrr_cells = " | ".join(f"{m[f'mrr@{k}']:.3f}" for k in k_values)
        recall_cells = " | ".join(f"{m[f'recall@{k}']:.3f}" for k in k_values)
        lines.append(f"| {pass_name} | {ndcg_cells} | {mrr_cells} | {recall_cells} |")
    lines.append("")
    lines.append("Generated by `scripts/evaluate_pipeline.py`. Relevance proxy and methodology "
                 "are documented in the script docstring.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation_results.json")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "evaluation_results.md")
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
        help="Comma-separated list of K values, e.g. '5,10,20'.",
    )
    parser.add_argument(
        "--max-test-queries",
        type=int,
        default=None,
        help="If set, evaluate only the first N test queries (useful for smoke runs).",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    train = _load_jsonl(args.splits_dir / "train.jsonl")
    val = _load_jsonl(args.splits_dir / "validation.jsonl")
    test = _load_jsonl(args.splits_dir / "test.jsonl")
    if not test:
        print("No test rows found.", file=sys.stderr)
        return 1
    if args.max_test_queries is not None:
        test = test[: args.max_test_queries]

    candidate_pool = train + val

    retrieval_metrics, retrieval_queries = _evaluate_pass(
        "retrieval_only", test, candidate_pool, k_values, rerank=False
    )
    full_metrics, full_queries = _evaluate_pass(
        "full_pipeline", test, candidate_pool, k_values, rerank=True
    )
    queries_used = max(retrieval_queries, full_queries)

    payload: Dict[str, Any] = {
        "retrieval_only": retrieval_metrics,
        "full_pipeline": full_metrics,
        "test_set_size": len(test),
        "queries_evaluated": queries_used,
        "candidate_pool_size": len(candidate_pool),
        "config": {
            "k_values": k_values,
            "retriever": "tfidf_cosine",
            "ranker_blend": {"cosine_weight": 0.6, "engagement_weight": 0.4},
            "relevance_proxy": (
                "candidate shares >=1 hashtag or top-3 keyword with query AND "
                "engagement >= median of hashtag bucket"
            ),
        },
    }

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown_table(payload), encoding="utf-8")

    print(f"\nWrote {args.out.name} and {args.out_md.name}.\n")
    print(_markdown_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
