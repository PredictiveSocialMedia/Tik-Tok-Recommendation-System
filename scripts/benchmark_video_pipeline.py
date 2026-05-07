#!/usr/bin/env python3
"""End-to-end latency benchmark for the video analysis pipeline.

The analyzer in :mod:`src.recommendation.video.analyzer` already emits per-branch
timings via ``logger.info("Branch [%s] completed in %.1fs", ...)``. This script
attaches a structured log capture handler, runs ``VideoAnalyzer.analyze`` over a
small set of representative videos, and writes:

* ``benchmarks/video_pipeline_results.json`` -- per-video totals + per-branch
  latencies, plus aggregate p50/p95.
* ``benchmarks/video_pipeline_summary.md`` -- two markdown tables (per-video
  and per-branch).

Optimisations already applied (documented for the report):

* Timeline frames reduced from 20 -> 10
* OCR sampling reduced from 5 frames -> 3
* Per-frame face detection dropped (was the dominant CPU cost)
* Whisper model switched from ``small`` to ``base`` (int8 quantised)

Usage::

    python scripts/benchmark_video_pipeline.py
    python scripts/benchmark_video_pipeline.py --videos-dir frontend/artifacts/uploaded_assets --max-videos 10
    python scripts/benchmark_video_pipeline.py --baseline-json benchmarks/video_pipeline_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BRANCH_PATTERN = re.compile(r"Branch \[(?P<name>[^\]]+)\] completed in (?P<seconds>[0-9.]+)s")
TOTAL_PATTERN = re.compile(
    r"Video analysis total: (?P<total>[0-9.]+)s \(video duration: (?P<duration>[0-9.]+)s\)"
)


class _BranchTimingHandler(logging.Handler):
    """Captures the analyzer's per-branch timing log lines."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.branches: Dict[str, float] = {}
        self.total_seconds: Optional[float] = None
        self.video_duration_seconds: Optional[float] = None

    def reset(self) -> None:
        self.branches = {}
        self.total_seconds = None
        self.video_duration_seconds = None

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        message = record.getMessage()
        m = BRANCH_PATTERN.search(message)
        if m:
            self.branches[m.group("name")] = float(m.group("seconds"))
            return
        m = TOTAL_PATTERN.search(message)
        if m:
            self.total_seconds = float(m.group("total"))
            self.video_duration_seconds = float(m.group("duration"))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def _total_stats(per_video: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [float(entry.get("total_seconds") or 0.0) for entry in per_video]
    values = [value for value in values if value > 0]
    if not values:
        return {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0}
    return {
        "mean_s": round(mean(values), 3),
        "p50_s": round(_percentile(values, 50), 3),
        "p95_s": round(_percentile(values, 95), 3),
    }


def _speedup(before: float, after: float) -> float:
    if before <= 0 or after <= 0:
        return 0.0
    return round(before / after, 3)


def _select_videos(videos_dir: Path, max_videos: int) -> List[Path]:
    candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        candidates = sorted(videos_dir.rglob("*.mp4"))
    return candidates[:max_videos]


def _markdown_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["# Video pipeline latency benchmark\n"]
    lines.append(f"- Videos benchmarked: **{payload['videos_benchmarked']}**")
    lines.append(f"- Optimisations applied: {', '.join(payload['optimisations_applied'])}")
    if payload.get("total_latency"):
        stats = payload["total_latency"]
        lines.append(
            f"- Current total latency: mean **{stats['mean_s']:.2f}s**, "
            f"p50 **{stats['p50_s']:.2f}s**, p95 **{stats['p95_s']:.2f}s**"
        )
    comparison = payload.get("baseline_comparison") or {}
    if comparison:
        lines.append(
            f"- Before/after speedup: mean **{comparison['mean_speedup']}x**, "
            f"p50 **{comparison['p50_speedup']}x**, p95 **{comparison['p95_speedup']}x**"
        )
    lines.append("")
    lines.append("## Per-video totals\n")
    lines.append("| video_id | duration (s) | total wall-clock (s) | branches succeeded |")
    lines.append("|---|---|---|---|")
    for entry in payload["per_video"]:
        lines.append(
            f"| `{entry['video_id']}` | {entry['video_duration_s']:.1f} | "
            f"{entry['total_seconds']:.2f} | {entry['branch_count']} |"
        )
    lines.append("")
    lines.append("## Per-branch latency (across all videos)\n")
    lines.append("| branch | mean (s) | p50 (s) | p95 (s) | n |")
    lines.append("|---|---|---|---|---|")
    for branch, stats in payload["per_branch"].items():
        lines.append(
            f"| {branch} | {stats['mean_s']:.2f} | {stats['p50_s']:.2f} | "
            f"{stats['p95_s']:.2f} | {stats['n']} |"
        )
    if comparison:
        lines.append("")
        lines.append("## Before/after totals\n")
        lines.append("| metric | baseline (s) | current (s) | speedup |")
        lines.append("|---|---|---|---|")
        for metric in ("mean", "p50", "p95"):
            lines.append(
                f"| {metric} | {comparison[f'baseline_{metric}_s']:.2f} | "
                f"{comparison[f'current_{metric}_s']:.2f} | "
                f"{comparison[f'{metric}_speedup']}x |"
            )
    lines.append("")
    lines.append("Generated by `scripts/benchmark_video_pipeline.py`. The analyzer's existing "
                 "`logger.info(\"Branch [...]\")` lines are the source of truth for per-branch "
                 "timings; this script just captures and aggregates them.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=REPO_ROOT / "frontend" / "artifacts" / "uploaded_assets",
    )
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "video_pipeline_results.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "video_pipeline_summary.md",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional previous benchmark JSON used to compute before/after speedup.",
    )
    args = parser.parse_args()

    videos = _select_videos(args.videos_dir, args.max_videos)
    if not videos:
        print(f"No .mp4 files under {args.videos_dir}.", file=sys.stderr)
        print("Skipping benchmark; pass --videos-dir <path> to a folder with sample videos.", file=sys.stderr)
        return 1

    try:
        from src.recommendation.video.analyzer import VideoAnalyzer  # noqa: WPS433
    except Exception as exc:
        print(
            "Could not import VideoAnalyzer (likely missing ML deps locally): "
            f"{exc}\n"
            "Install requirements-service.txt + the video dependencies listed in "
            "the Dockerfile, then re-run.",
            file=sys.stderr,
        )
        return 2

    analyzer = VideoAnalyzer()
    capture = _BranchTimingHandler()
    analyzer_logger = logging.getLogger("src.recommendation.video.analyzer")
    analyzer_logger.addHandler(capture)
    analyzer_logger.setLevel(logging.INFO)

    per_video: List[Dict[str, Any]] = []
    per_branch_samples: Dict[str, List[float]] = defaultdict(list)

    for video_path in videos:
        capture.reset()
        wall_start = time.perf_counter()
        try:
            response = analyzer.analyze(str(video_path))
        except Exception as exc:
            print(f"  ! {video_path.name}: failed -- {exc}", file=sys.stderr)
            continue
        wall_seconds = time.perf_counter() - wall_start
        total_seconds = capture.total_seconds if capture.total_seconds is not None else wall_seconds
        duration_seconds = (
            capture.video_duration_seconds
            if capture.video_duration_seconds is not None
            else float(getattr(response, "duration_seconds", 0.0) or 0.0)
        )
        for name, secs in capture.branches.items():
            per_branch_samples[name].append(secs)
        per_video.append({
            "video_id": video_path.stem,
            "video_path": str(video_path.relative_to(REPO_ROOT)) if REPO_ROOT in video_path.parents else str(video_path),
            "video_duration_s": round(duration_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "wall_clock_seconds": round(wall_seconds, 2),
            "branch_count": len(capture.branches),
            "branch_latencies_s": dict(capture.branches),
        })
        print(
            f"  {video_path.name}: total={total_seconds:.1f}s  "
            f"branches={len(capture.branches)}"
        )

    per_branch_summary: Dict[str, Dict[str, Any]] = {}
    for name, samples in sorted(per_branch_samples.items()):
        per_branch_summary[name] = {
            "n": len(samples),
            "mean_s": round(mean(samples), 3),
            "p50_s": round(_percentile(samples, 50), 3),
            "p95_s": round(_percentile(samples, 95), 3),
        }

    payload = {
        "videos_benchmarked": len(per_video),
        "optimisations_applied": [
            "timeline frames 20 -> 10",
            "OCR sampling 5 frames -> 3",
            "drop per-frame face detection",
            "Whisper small -> base (int8)",
        ],
        "total_latency": _total_stats(per_video),
        "per_video": per_video,
        "per_branch": per_branch_summary,
    }

    if args.baseline_json and args.baseline_json.exists():
        baseline_payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_stats = baseline_payload.get("total_latency") or _total_stats(
            baseline_payload.get("per_video") or []
        )
        current_stats = payload["total_latency"]
        payload["baseline_comparison"] = {
            "baseline_json": str(args.baseline_json),
            "baseline_mean_s": float(baseline_stats.get("mean_s") or 0.0),
            "baseline_p50_s": float(baseline_stats.get("p50_s") or 0.0),
            "baseline_p95_s": float(baseline_stats.get("p95_s") or 0.0),
            "current_mean_s": float(current_stats.get("mean_s") or 0.0),
            "current_p50_s": float(current_stats.get("p50_s") or 0.0),
            "current_p95_s": float(current_stats.get("p95_s") or 0.0),
            "mean_speedup": _speedup(
                float(baseline_stats.get("mean_s") or 0.0),
                float(current_stats.get("mean_s") or 0.0),
            ),
            "p50_speedup": _speedup(
                float(baseline_stats.get("p50_s") or 0.0),
                float(current_stats.get("p50_s") or 0.0),
            ),
            "p95_speedup": _speedup(
                float(baseline_stats.get("p95_s") or 0.0),
                float(current_stats.get("p95_s") or 0.0),
            ),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown_summary(payload), encoding="utf-8")
    print(f"\nWrote {args.out_json.name} and {args.out_md.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
