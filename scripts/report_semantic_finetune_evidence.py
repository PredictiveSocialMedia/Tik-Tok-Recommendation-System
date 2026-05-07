#!/usr/bin/env python3
"""Compare baseline SBERT against a fine-tuned sentence transformer artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.embedding_finetuner import (  # noqa: E402
    EmbeddingFinetunerConfig,
    TikTokEmbeddingFinetuner,
)
from src.recommendation.learning.model_evidence import (  # noqa: E402
    build_comparison,
    format_metric_report,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _load_model(model_name: str, *, local_files_only: bool) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data" / "splits" / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data" / "splits" / "validation.jsonl")
    parser.add_argument("--test-path", type=Path, default=REPO_ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--baseline-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--finetuned-model-dir", type=Path, default=REPO_ROOT / "models" / "tiktok-sbert")
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "semantic_finetune_evidence.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "semantic_finetune_evidence.md")
    parser.add_argument("--allow-model-downloads", action="store_true")
    args = parser.parse_args()

    train = _load_jsonl(args.train_path)
    val = _load_jsonl(args.val_path)
    test = _load_jsonl(args.test_path)
    cfg = EmbeddingFinetunerConfig(eval_k=args.eval_k)
    evaluator = TikTokEmbeddingFinetuner(cfg)
    local_only = not args.allow_model_downloads

    baseline_model = _load_model(args.baseline_model, local_files_only=local_only)
    baseline_metrics = evaluator.evaluate(baseline_model, test, k=args.eval_k)
    models = [
        {
            "name": args.baseline_model,
            "role": "baseline_sbert",
            "metrics": baseline_metrics,
        }
    ]

    finetuned_metrics = None
    status = "candidate_missing"
    notes = []
    if args.finetuned_model_dir.exists():
        finetuned_model = _load_model(str(args.finetuned_model_dir), local_files_only=local_only)
        finetuned_metrics = evaluator.evaluate(finetuned_model, test, k=args.eval_k)
        models.append(
            {
                "name": str(args.finetuned_model_dir),
                "role": "fine_tuned_sentence_transformer",
                "metrics": finetuned_metrics,
            }
        )
        status = "evaluated"
    else:
        notes.append(
            f"Fine-tuned model directory not found: {args.finetuned_model_dir}. "
            "Run scripts/fine_tune_embeddings.py first, then rerun this evidence report."
        )

    primary_metric = f"ndcg@{args.eval_k}"
    report = {
        "title": "Semantic fine-tuning evidence",
        "status": status,
        "data": {
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
        },
        "split": {
            "train": str(args.train_path),
            "validation": str(args.val_path),
            "test": str(args.test_path),
        },
        "models": models,
        "comparison": build_comparison(
            baseline_metrics,
            finetuned_metrics,
            primary_metric=primary_metric,
        ),
        "notes": notes,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(format_metric_report(report), encoding="utf-8")
    print(format_metric_report(report))
    print(f"Wrote {args.output_json} and {args.output_md}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
