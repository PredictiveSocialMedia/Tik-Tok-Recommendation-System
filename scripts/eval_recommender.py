#!/usr/bin/env python3
"""Evaluate trained recommender artifacts against datamart.v1 JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning.artifacts import ArtifactRegistry


def sanitize_metric_name(name: str) -> str:
    return (
        name.replace("@", "_at_")
        .replace(" ", "_")
    )
def main() -> int:
    parser = argparse.ArgumentParser(description="Print recommender evaluation summary.")
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Path to trained recommender bundle.",
    )
    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Print bundle manifest as well.",
    )
    parser.add_argument(
        "--show-ablation",
        action="store_true",
        help="Include retriever ablation diagnostics when available.",
    )
    parser.add_argument(
        "--experiment-name",
        default="recommender-evaluation",
        help="MLflow experiment name.",
    )
    args = parser.parse_args()

    start_time = time.time()

    bundle_dir = args.bundle_dir
    if bundle_dir.is_file():
        resolved = bundle_dir.read_text(encoding="utf-8").strip()
        bundle_dir = Path(resolved)

    registry = ArtifactRegistry(bundle_dir.parent)
    manifest = registry.load_manifest(bundle_dir)
    metrics_path = bundle_dir / "metrics" / "objective_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=bundle_dir.name):
        mlflow.log_param("bundle_dir", str(bundle_dir))
        mlflow.log_param("metrics_path", str(metrics_path))
        mlflow.log_param("show_manifest", args.show_manifest)
        mlflow.log_param("show_ablation", args.show_ablation)

        if isinstance(manifest, dict):
            for key in [
                "dataset_version",
                "retrieval_approach",
                "embedding_model",
                "top_k",
                "graph_version",
                "trajectory_version",
            ]:
                value = manifest.get(key)
                if value is not None:
                    mlflow.log_param(key, value)

        if isinstance(metrics, dict):
            for objective, payload in metrics.items():
                if not isinstance(payload, dict):
                    continue

                spec = payload.get("spec", {})
                if isinstance(spec, dict):
                    for key in [
                        "objective_id",
                        "label_key",
                        "primary_metric",
                        "training_loss",
                        "calibration",
                    ]:
                        value = spec.get(key)
                        if value is not None:
                            mlflow.log_param(f"{objective}.{key}", value)

                retriever_metrics = payload.get("retriever", {})
                if isinstance(retriever_metrics, dict):
                    for metric_name, metric_value in retriever_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            safe_name = sanitize_metric_name(
                                f"{objective}.retriever.{metric_name}"
                            )
                            mlflow.log_metric(safe_name, metric_value)

                ranker_metrics = payload.get("ranker", {})
                if isinstance(ranker_metrics, dict):
                    for metric_name, metric_value in ranker_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            safe_name = sanitize_metric_name(
                                f"{objective}.ranker.{metric_name}"
                            )
                            mlflow.log_metric(safe_name, metric_value)

        runtime_seconds = time.time() - start_time
        mlflow.log_metric("runtime_seconds", runtime_seconds)

    output = {"bundle_dir": str(bundle_dir), "metrics": metrics}
    if args.show_manifest:
        output["manifest"] = manifest
    if args.show_ablation:
        ablation: dict[str, dict[str, object]] = {}
        reports = manifest.get("objective_ablation_reports")
        if isinstance(reports, dict):
            for objective, meta in reports.items():
                if not isinstance(meta, dict):
                    continue
                rel_path = str(meta.get("path") or "").strip()
                if not rel_path:
                    continue
                report_path = bundle_dir / rel_path
                if not report_path.exists():
                    continue
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    ablation[str(objective)] = payload
        if not ablation and isinstance(metrics, dict):
            for objective, payload in metrics.items():
                if not isinstance(payload, dict):
                    continue
                summary = payload.get("retriever_ablation")
                if isinstance(summary, dict):
                    ablation[str(objective)] = {
                        "objective": str(objective),
                        "retriever_ablation": summary,
                    }
        output["ablation"] = ablation
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())