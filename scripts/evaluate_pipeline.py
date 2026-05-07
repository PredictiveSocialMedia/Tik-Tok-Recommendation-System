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
* **production_pipeline** -- the actual :class:`RecommenderRuntime` loaded from
  a bundle directory. It exercises the same retrieval, ranking, learned
  reranker, policy, and calibration path used by the serving API.

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
    python scripts/evaluate_pipeline.py --bundle-dir artifacts/recommender/latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in minimal installs
    mlflow = None  # type: ignore
    _MLFLOW_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.evaluator import (  # noqa: E402
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.recommendation.learning.inference import RecommenderRuntime  # noqa: E402


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


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _resolve_bundle_dir(path: Path) -> Path:
    """Resolve symlink/text-pointer bundle paths, including stale absolute pointers."""
    raw = Path(path)
    if raw.is_file() and not raw.name.endswith(".json"):
        target_text = raw.read_text(encoding="utf-8").strip()
        target = Path(target_text)
        if target.exists():
            return target.resolve()
        local_by_name = REPO_ROOT / "artifacts" / "recommender" / target.name
        if local_by_name.exists():
            return local_by_name.resolve()
        return target
    return raw.resolve()


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


def _row_to_runtime_query(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": str(row.get("video_id") or row.get("row_id") or "query"),
        "description": str(row.get("caption") or ""),
        "text": _row_text(row),
        "hashtags": list(row.get("hashtags") or []),
        "keywords": list(row.get("keywords") or []),
        "topic_key": row.get("topic_key") or row.get("search_query"),
        "author_id": row.get("author_id"),
        "language": row.get("language"),
        "locale": row.get("locale"),
        "content_type": row.get("content_type"),
        "as_of_time": row.get("as_of_time"),
    }


def _row_to_runtime_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    pre = features.get("pre_metrics") if isinstance(features.get("pre_metrics"), dict) else {}
    signal_hints: Dict[str, Any] = {}
    comment_intelligence = features.get("comment_intelligence")
    if isinstance(comment_intelligence, dict):
        signal_hints["comment_intelligence"] = comment_intelligence
    trajectory_features = features.get("trajectory_features")
    if isinstance(trajectory_features, dict):
        signal_hints["trajectory_features"] = trajectory_features
    return {
        "candidate_id": str(row.get("video_id") or row.get("row_id") or ""),
        "row_id": str(row.get("row_id") or row.get("video_id") or ""),
        "video_id": str(row.get("video_id") or row.get("row_id") or ""),
        "caption": str(row.get("caption") or ""),
        "text": _row_text(row),
        "hashtags": list(row.get("hashtags") or []),
        "keywords": list(row.get("keywords") or []),
        "topic_key": row.get("topic_key") or row.get("search_query"),
        "author_id": row.get("author_id"),
        "as_of_time": row.get("as_of_time"),
        "posted_at": row.get("posted_at"),
        "duration_seconds": features.get("duration_seconds"),
        "language": row.get("language"),
        "locale": row.get("locale"),
        "content_type": row.get("content_type"),
        # Only point-in-time/pre-publication metrics are exposed to the runtime.
        "views": pre.get("views"),
        "likes": pre.get("likes"),
        "comments_count": pre.get("comments_count"),
        "shares": pre.get("shares"),
        "signal_hints": signal_hints,
    }


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


def _evaluate_production_pipeline(
    *,
    runtime: RecommenderRuntime,
    test_queries: Sequence[Dict[str, Any]],
    candidate_pool: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    objective: str,
    retrieve_k: int,
) -> Tuple[Dict[str, float], int, Dict[str, Any]]:
    candidates = [_row_to_runtime_candidate(row) for row in candidate_pool]
    metrics_per_k: Dict[int, Dict[str, List[float]]] = {
        k: {"ndcg": [], "mrr": [], "recall": []} for k in k_values
    }
    queries_used = 0
    failures: List[Dict[str, str]] = []
    top_k = max(k_values)

    for query in test_queries:
        relevant_ids, graded = _build_relevance(query, candidate_pool)
        if not relevant_ids:
            continue
        try:
            response = runtime.recommend(
                objective=objective,
                as_of_time=query.get("as_of_time"),
                query=_row_to_runtime_query(query),
                candidates=candidates,
                top_k=top_k,
                retrieve_k=max(retrieve_k, top_k),
                language=query.get("language"),
                locale=query.get("locale"),
                content_type=query.get("content_type"),
                debug=False,
            )
        except Exception as exc:
            failures.append({
                "query_id": str(query.get("video_id") or query.get("row_id") or ""),
                "error": str(exc),
            })
            continue

        ranked_ids = [str(item.get("candidate_id") or item.get("candidate_row_id")) for item in response.get("items", [])]
        for k in k_values:
            metrics_per_k[k]["ndcg"].append(ndcg_at_k(ranked_ids, graded, k))
            metrics_per_k[k]["mrr"].append(mrr_at_k(ranked_ids, relevant_ids, k))
            metrics_per_k[k]["recall"].append(recall_at_k(ranked_ids, relevant_ids, k))
        queries_used += 1

    summary: Dict[str, float] = {}
    for k in k_values:
        for metric in ("ndcg", "mrr", "recall"):
            summary[f"{metric}@{k}"] = round(aggregate(metrics_per_k[k][metric]), 4)
    diagnostics = {
        "failed_queries": len(failures),
        "failure_examples": failures[:5],
        "bundle_id": runtime.bundle_id,
        "retriever_loaded": runtime.retriever is not None,
        "retriever_load_warning": runtime.retriever_load_warning,
    }
    return summary, queries_used, diagnostics


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
    lines.append(f"- Bundle: `{payload['config'].get('bundle_dir', 'not used')}`")
    lines.append(f"- Split hash: `{payload['config']['split_hashes'].get('test', 'unknown')[:12]}` (`test.jsonl`)")
    lines.append("")
    lines.append("| Pass | " + " | ".join(f"NDCG@{k}" for k in k_values) + " | "
                 + " | ".join(f"MRR@{k}" for k in k_values) + " | "
                 + " | ".join(f"Recall@{k}" for k in k_values) + " |")
    lines.append("|" + "---|" * (1 + 3 * len(k_values)))
    for pass_name in payload["passes"]:
        m = payload[pass_name]
        ndcg_cells = " | ".join(f"{m[f'ndcg@{k}']:.3f}" for k in k_values)
        mrr_cells = " | ".join(f"{m[f'mrr@{k}']:.3f}" for k in k_values)
        recall_cells = " | ".join(f"{m[f'recall@{k}']:.3f}" for k in k_values)
        lines.append(f"| {pass_name} | {ndcg_cells} | {mrr_cells} | {recall_cells} |")
    lines.append("")
    lines.append("Generated by `scripts/evaluate_pipeline.py`. Relevance proxy and methodology "
                 "are documented in the script docstring.")
    return "\n".join(lines) + "\n"


def _metric_name(name: str) -> str:
    return name.replace("@", "_at_").replace(" ", "_")


def _log_mlflow_run(
    *,
    payload: Dict[str, Any],
    experiment_name: str,
    run_name: str,
    artifact_paths: Sequence[Path],
) -> None:
    if not _MLFLOW_AVAILABLE:
        return
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        mlflow.log_param("git_sha", payload.get("git_sha", "unknown"))
        mlflow.log_param("candidate_pool_size", payload.get("candidate_pool_size", 0))
        mlflow.log_param("test_set_size", payload.get("test_set_size", 0))
        mlflow.log_param("queries_evaluated", payload.get("queries_evaluated", 0))
        mlflow.log_param("passes", ",".join(payload.get("passes") or []))
        for key in ("bundle_dir", "production_objective", "retrieve_k", "baseline_retriever"):
            if key in config:
                mlflow.log_param(key, config[key])
        split_hashes = config.get("split_hashes") if isinstance(config.get("split_hashes"), dict) else {}
        for split_name, digest in split_hashes.items():
            mlflow.log_param(f"split_hash.{split_name}", digest)
        diagnostics = payload.get("production_diagnostics")
        if isinstance(diagnostics, dict):
            mlflow.log_param("bundle_id", diagnostics.get("bundle_id", ""))
            mlflow.log_param("retriever_loaded", diagnostics.get("retriever_loaded", False))
            mlflow.log_metric("production_failed_queries", float(diagnostics.get("failed_queries") or 0))
        for pass_name in payload.get("passes") or []:
            metrics = payload.get(pass_name)
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(_metric_name(f"{pass_name}.{key}"), float(value))
        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="offline_evaluation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / "splits")
    parser.add_argument("--bundle-dir", type=Path, default=REPO_ROOT / "artifacts" / "recommender" / "latest")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation_results.json")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "evaluation_results.md")
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "artifacts" / "evaluation" / "offline")
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
    parser.add_argument("--objective", default="engagement", help="Production objective to evaluate.")
    parser.add_argument("--retrieve-k", type=int, default=200)
    parser.add_argument("--skip-production", action="store_true", help="Only run the TF-IDF retrieval baseline.")
    parser.add_argument("--no-versioned-artifact", action="store_true", help="Do not write timestamped artifact copies.")
    parser.add_argument("--experiment-name", default="recommender-offline-evaluation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
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
    passes = ["retrieval_only"]
    production_metrics: Optional[Dict[str, float]] = None
    production_queries = 0
    production_diagnostics: Dict[str, Any] = {}
    resolved_bundle_dir: Optional[Path] = None
    if not args.skip_production:
        resolved_bundle_dir = _resolve_bundle_dir(args.bundle_dir)
        if not resolved_bundle_dir.exists():
            raise SystemExit(f"Bundle directory not found: {resolved_bundle_dir}")
        runtime = RecommenderRuntime(bundle_dir=resolved_bundle_dir)
        production_metrics, production_queries, production_diagnostics = _evaluate_production_pipeline(
            runtime=runtime,
            test_queries=test,
            candidate_pool=candidate_pool,
            k_values=k_values,
            objective=args.objective,
            retrieve_k=args.retrieve_k,
        )
        passes.append("production_pipeline")
    queries_used = max(retrieval_queries, production_queries)
    split_hashes = {
        name: _sha256_of_file(args.splits_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "passes": passes,
        "retrieval_only": retrieval_metrics,
        "test_set_size": len(test),
        "queries_evaluated": queries_used,
        "candidate_pool_size": len(candidate_pool),
        "config": {
            "k_values": k_values,
            "baseline_retriever": "tfidf_cosine",
            "production_objective": args.objective,
            "retrieve_k": int(args.retrieve_k),
            "bundle_dir": str(resolved_bundle_dir) if resolved_bundle_dir is not None else None,
            "split_hashes": split_hashes,
            "relevance_proxy": (
                "candidate shares >=1 hashtag or top-3 keyword with query AND "
                "engagement >= median of hashtag bucket"
            ),
        },
    }
    if production_metrics is not None:
        payload["production_pipeline"] = production_metrics
        payload["production_diagnostics"] = production_diagnostics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown_table(payload), encoding="utf-8")
    mlflow_artifacts = [args.out, args.out_md]
    if not args.no_versioned_artifact:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_dir = args.artifact_dir / stamp
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "evaluation_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "evaluation_results.md").write_text(
            _markdown_table(payload),
            encoding="utf-8",
        )
        mlflow_artifacts.extend(
            [artifact_dir / "evaluation_results.json", artifact_dir / "evaluation_results.md"]
        )

    if not args.disable_mlflow:
        _log_mlflow_run(
            payload=payload,
            experiment_name=args.experiment_name,
            run_name=args.run_name or f"offline-eval-{payload['generated_at_utc']}",
            artifact_paths=mlflow_artifacts,
        )

    print(f"\nWrote {args.out.name} and {args.out_md.name}.\n")
    print(_markdown_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
