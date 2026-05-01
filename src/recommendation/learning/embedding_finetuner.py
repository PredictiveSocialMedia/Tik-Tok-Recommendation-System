"""Fine-tune a sentence transformer on TikTok-specific caption/hashtag data.

Training signal: positive pairs are videos sharing the same topic_key and
>= min_shared_hashtags hashtags. MultipleNegativesRankingLoss treats all
other items in each batch as implicit negatives — no explicit negative
labelling required.

Requires sentence-transformers>=3.0 and datasets for training.
The evaluate() method only needs numpy and a model with .encode(), so a mock
works in unit tests without any ML dependencies installed.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .evaluator import aggregate, mrr_at_k, ndcg_at_k
from .temporal import row_text

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    _SBERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SBERT_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset  # type: ignore

    _DATASETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DATASETS_AVAILABLE = False


@dataclass
class EmbeddingFinetunerConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: str = "models/tiktok-sbert"
    epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    min_shared_hashtags: int = 1
    max_pairs_per_anchor: int = 10
    min_plays: int = 100
    eval_k: int = 10
    seed: int = 42


class TikTokEmbeddingFinetuner:
    """Fine-tunes a sentence transformer on TikTok caption/hashtag data."""

    def __init__(self, cfg: Optional[EmbeddingFinetunerConfig] = None) -> None:
        self.cfg = cfg or EmbeddingFinetunerConfig()
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ------------------------------------------------------------------
    # Pure-Python helpers (no SBERT dependency)
    # ------------------------------------------------------------------

    def _hashtags(self, row: Dict[str, Any]) -> frozenset:
        tags: set = set()
        for tag in row.get("hashtags") or []:
            clean = str(tag).strip().lower().lstrip("#").strip()
            if clean:
                tags.add(clean)
        for word in str(row.get("caption") or "").split():
            if word.startswith("#"):
                tags.add(word.lower().lstrip("#").strip())
        return frozenset(t for t in tags if t)

    def _eligible(self, row: Dict[str, Any]) -> bool:
        plays = int((row.get("features") or {}).get("plays") or row.get("plays") or 0)
        # plays == 0 means unknown (not scraped), not literally zero views — keep it
        if plays and plays < self.cfg.min_plays:
            return False
        # Require real content, not just topic_key (which row_text falls back to)
        has_caption = bool(str(row.get("caption") or "").strip())
        has_tags = bool(row.get("hashtags") or [])
        has_keywords = bool(row.get("keywords") or [])
        has_query = bool(str(row.get("search_query") or "").strip())
        return has_caption or has_tags or has_keywords or has_query

    # ------------------------------------------------------------------
    # Pair construction (pure Python, testable without SBERT)
    # ------------------------------------------------------------------

    def build_training_pairs(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Return (anchors, positives) text lists for MNRL training.

        Positive pair rule:
          - Same topic_key AND >= min_shared_hashtags shared hashtags, OR
          - Same topic_key AND both videos have no hashtags (topic-only fallback).
        Up to max_pairs_per_anchor positives per anchor, ranked by overlap.
        """
        eligible = [r for r in rows if self._eligible(r)]

        by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            topic = str(row.get("topic_key") or "unknown").strip()
            by_topic[topic].append(row)

        anchors: List[str] = []
        positives: List[str] = []

        for group in by_topic.values():
            if len(group) < 2:
                continue
            for i, anchor in enumerate(group):
                anchor_tags = self._hashtags(anchor)
                anchor_text = row_text(anchor)
                scored: List[Tuple[int, int, Dict[str, Any]]] = []
                for j, other in enumerate(group):
                    if i == j:
                        continue
                    other_tags = self._hashtags(other)
                    shared = len(anchor_tags & other_tags)
                    no_tags = not anchor_tags and not other_tags
                    if shared >= self.cfg.min_shared_hashtags or no_tags:
                        scored.append((shared, j, other))
                scored.sort(key=lambda x: -x[0])
                for _, _, other in scored[: self.cfg.max_pairs_per_anchor]:
                    anchors.append(anchor_text)
                    positives.append(row_text(other))

        if anchors:
            combined = list(zip(anchors, positives))
            random.shuffle(combined)
            anchors, positives = zip(*combined)
            anchors, positives = list(anchors), list(positives)

        logger.info(
            "Built %d training pairs from %d eligible rows", len(anchors), len(eligible)
        )
        return anchors, positives

    # ------------------------------------------------------------------
    # Training (requires sentence-transformers>=3.0 + datasets)
    # ------------------------------------------------------------------

    def train(
        self,
        train_rows: List[Dict[str, Any]],
        val_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> "SentenceTransformer":
        """Fine-tune the base model on train_rows and save to output_dir."""
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers>=3.0 is required. "
                "Install with: pip install 'sentence-transformers>=3.0'"
            )
        if not _DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install with: pip install datasets"
            )

        anchors, positives = self.build_training_pairs(train_rows)
        if not anchors:
            raise ValueError(
                "No training pairs constructed — check topic_key and hashtag coverage."
            )

        train_dataset = HFDataset.from_dict({"anchor": anchors, "positive": positives})
        model = SentenceTransformer(self.cfg.base_model)
        loss = losses.MultipleNegativesRankingLoss(model)
        evaluator = self._ir_evaluator(val_rows, name="val") if val_rows else None

        output = Path(self.cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output),
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            warmup_ratio=self.cfg.warmup_ratio,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            seed=self.cfg.seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()
        model.save_pretrained(str(output))
        logger.info("Fine-tuned model saved to %s", output)
        return model

    # ------------------------------------------------------------------
    # IR evaluator (used during training validation callback)
    # ------------------------------------------------------------------

    def _ir_evaluator(
        self, rows: List[Dict[str, Any]], name: str = "eval"
    ) -> "InformationRetrievalEvaluator":
        by_topic: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for i, row in enumerate(rows):
            topic = str(row.get("topic_key") or "unknown").strip()
            by_topic[topic].append((i, row))

        queries: Dict[str, str] = {}
        corpus: Dict[str, str] = {}
        relevant_docs: Dict[str, set] = {}

        for indexed_group in by_topic.values():
            if len(indexed_group) < 2:
                continue
            for idx, row in indexed_group:
                rid = str(row.get("row_id") or row.get("video_id") or f"row_{idx}")
                corpus[rid] = row_text(row)

            idx0, anchor = indexed_group[0]
            qid = str(anchor.get("row_id") or anchor.get("video_id") or f"row_{idx0}")
            queries[qid] = corpus[qid]
            relevant_docs[qid] = {
                str(r.get("row_id") or r.get("video_id") or f"row_{i}")
                for i, r in indexed_group[1:]
            }

        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=name,
        )

    # ------------------------------------------------------------------
    # Evaluation (only needs a model with .encode(); mock-friendly)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        test_rows: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate retrieval quality on test_rows.

        Groups rows by topic_key. Each video is used as a query; same-topic
        videos are the relevant set; the full test set is the corpus.
        Returns NDCG@k and MRR@k averaged over all queries.
        """
        k = k or self.cfg.eval_k
        zero = {f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}
        if not test_rows:
            return zero

        all_ids = [
            str(r.get("row_id") or r.get("video_id") or f"row_{i}")
            for i, r in enumerate(test_rows)
        ]
        all_texts = [row_text(r) for r in test_rows]
        all_embs: np.ndarray = model.encode(
            all_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=256,
        )
        idx_of = {rid: i for i, rid in enumerate(all_ids)}

        by_topic: Dict[str, List[str]] = defaultdict(list)
        for rid, row in zip(all_ids, test_rows):
            topic = str(row.get("topic_key") or "unknown").strip()
            by_topic[topic].append(rid)

        ndcg_scores: List[float] = []
        mrr_scores: List[float] = []

        for group_ids in by_topic.values():
            if len(group_ids) < 2:
                continue
            relevant_set = set(group_ids)
            for qid in group_ids:
                q_emb = all_embs[idx_of[qid]]
                cand_ids = [rid for rid in all_ids if rid != qid]
                cand_embs = all_embs[[idx_of[c] for c in cand_ids]]
                scores = (cand_embs @ q_emb).tolist()
                ranked = [
                    cid for _, cid in sorted(zip(scores, cand_ids), reverse=True)
                ]
                relevant = relevant_set - {qid}
                relevance = {rid: 1.0 for rid in relevant}
                ndcg_scores.append(ndcg_at_k(ranked, relevance, k))
                mrr_scores.append(mrr_at_k(ranked, relevant, k))

        if not ndcg_scores:
            return zero

        return {
            f"ndcg@{k}": aggregate(ndcg_scores),
            f"mrr@{k}": aggregate(mrr_scores),
        }
