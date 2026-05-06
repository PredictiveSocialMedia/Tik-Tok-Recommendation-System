#!/usr/bin/env python3
"""Evaluate LoRA hashtag generation against a topic-prior baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.lora_hashtag_finetuner import (  # noqa: E402
    LoraHashtagFinetuner,
    LoraHashtagFinetunerConfig,
)
from src.recommendation.learning.model_evidence import (  # noqa: E402
    TopicPriorHashtagBaseline,
    build_comparison,
    evaluate_tag_predictor,
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


def _load_lora(base_model: str, adapter_dir: Path, *, local_files_only: bool) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        local_files_only=local_files_only,
    )
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data" / "splits" / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data" / "splits" / "validation.jsonl")
    parser.add_argument("--test-path", type=Path, default=REPO_ROOT / "data" / "splits" / "test.jsonl")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", type=Path, default=REPO_ROOT / "models" / "tiktok-hashtag-lora")
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "lora_evidence.json")
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / "artifacts" / "control_plane" / "lora_evidence.md")
    parser.add_argument("--allow-model-downloads", action="store_true")
    args = parser.parse_args()

    train = _load_jsonl(args.train_path)
    val = _load_jsonl(args.val_path)
    test = _load_jsonl(args.test_path)
    baseline = TopicPriorHashtagBaseline(train)
    baseline_metrics = evaluate_tag_predictor(
        test,
        lambda row, k: baseline.recommend(row, k),
        k=args.eval_k,
    )
    models = [
        {
            "name": "topic_prior_hashtag_baseline",
            "role": "baseline",
            "metrics": baseline_metrics,
        }
    ]

    cfg = LoraHashtagFinetunerConfig(base_model=args.base_model, eval_k=args.eval_k)
    finetuner = LoraHashtagFinetuner(cfg)
    lora_metrics = None
    status = "candidate_missing"
    notes = []
    if args.adapter_dir.exists():
        model, tokenizer = _load_lora(
            args.base_model,
            args.adapter_dir,
            local_files_only=not args.allow_model_downloads,
        )
        lora_metrics = finetuner.evaluate(model, tokenizer, test, k=args.eval_k)
        models.append(
            {
                "name": str(args.adapter_dir),
                "role": "lora_adapter",
                "metrics": lora_metrics,
            }
        )
        status = "evaluated"
    else:
        notes.append(
            f"LoRA adapter directory not found: {args.adapter_dir}. "
            "Run scripts/fine_tune_lora_hashtag.py first, then rerun this evidence report."
        )

    primary_metric = f"f1@{args.eval_k}"
    report = {
        "title": "LoRA hashtag fine-tuning evidence",
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
            lora_metrics,
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
