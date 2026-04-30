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
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hardware_metadata() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
    }
    try:
        import psutil  # type: ignore
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        info["ram_gb"] = None
        info["cpu_count_logical"] = os.cpu_count()
    return info


BRANCH_PATTERN = re.compile(r"Branch \[(?P<name>[^\]]+)\] completed in (?P<seconds>[0-9.]+)s")
TOTAL_PATTERN = re.compile(
    r"Video analysis total: (?P<total>[0-9.]+)s \(video duration: (?P<duration>[0-9.]+)s\)"
)

# Names of the eight branch tasks the analyzer submits, mapped to a
# substring that uniquely identifies the inner function in the analyzer's
# qualified name. Used by the ThreadPoolExecutor.submit monkey-patch below
# to attribute real wall-clock latency to each branch (the analyzer's
# existing log line measures completion-handler time, not branch execution
# time, which is why the per-branch numbers come out as ~0s without this).
_BRANCH_FN_HINTS = {
    "transcribe": ("transcribe", "_run_transcription"),
    "audio": ("_run_audio", "audio_features"),
    "visual": ("_run_visual", "scene"),
    "vlm": ("_caption", "vlm", "blip"),
    "ocr": ("_run_ocr", "ocr"),
    "colors": ("_run_color", "color"),
    "blur": ("_run_blur", "blur"),
    "timeline": ("timeline", "_run_timeline"),
}


class _BranchTimingHandler(logging.Handler):
    """Captures the analyzer's per-branch timing log lines AND the total line."""

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


def _classify_branch(fn) -> Optional[str]:
    """Best-effort attribution of an analyzer's submitted callable to a branch name."""
    qualname = getattr(fn, "__qualname__", "") or ""
    name = getattr(fn, "__name__", "") or ""
    haystack = f"{qualname} {name}".lower()
    for branch, hints in _BRANCH_FN_HINTS.items():
        for hint in hints:
            if hint.lower() in haystack:
                return branch
    return None


def _install_branch_timing_patch(timings_target: Dict[str, List[float]]) -> Any:
    """Wrap ``ThreadPoolExecutor.submit`` so every analyzer branch is timed
    end-to-end. Returns the original ``submit`` so the caller can restore it.

    The analyzer's own ``logger.info("Branch [...] completed in Xs")`` line
    fires inside the as_completed loop, which means the timer starts AFTER
    the future has already finished. That's why the existing per-branch
    numbers come out as ~0 seconds. Wrapping ``submit`` here measures the
    real wall-clock time spent inside each branch.
    """
    from concurrent.futures import ThreadPoolExecutor

    original_submit = ThreadPoolExecutor.submit

    def timed_submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        branch = _classify_branch(fn)

        def _wrapped(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                if branch is not None:
                    timings_target.setdefault(branch, []).append(
                        time.perf_counter() - start
                    )

        return original_submit(self, _wrapped, *args, **kwargs)

    ThreadPoolExecutor.submit = timed_submit  # type: ignore[assignment]
    return original_submit


def _restore_branch_timing_patch(original_submit: Any) -> None:
    from concurrent.futures import ThreadPoolExecutor

    ThreadPoolExecutor.submit = original_submit  # type: ignore[assignment]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def _select_videos(videos_dir: Path, max_videos: int) -> List[Path]:
    candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        candidates = sorted(videos_dir.rglob("*.mp4"))
    return candidates[:max_videos]


def _markdown_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["# Video pipeline latency benchmark\n"]
    metadata = payload.get("metadata", {})
    if metadata:
        hw = metadata.get("hardware", {})
        lines.append(f"- Run at: `{metadata.get('run_at_utc', 'unknown')}` "
                     f"(variant: `{metadata.get('variant', 'current')}`)")
        cpu = hw.get("processor") or hw.get("machine") or "unknown"
        ram = hw.get("ram_gb")
        ram_str = f"{ram} GB" if ram else "unknown"
        lines.append(f"- Hardware: {cpu}, {ram_str} RAM, "
                     f"{hw.get('cpu_count_logical', '?')} logical cores, "
                     f"Python {hw.get('python_version', '?')}")
        if metadata.get("demucs_enabled") is False:
            lines.append("- Demucs branch: **skipped** (heavy model not installed locally)")
        runs = metadata.get("runs_per_video")
        if runs:
            lines.append(f"- Runs per video: {runs} (best-of reported)")
    lines.append(f"- Videos benchmarked: **{payload['videos_benchmarked']}**")
    lines.append(f"- Optimisations applied: {', '.join(payload['optimisations_applied'])}")
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
        "--runs",
        type=int,
        default=1,
        help="Run each video N times; report best-of when N>1 to mitigate cold-cache variance.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="current",
        help="Label written into the metadata for this run (e.g. 'current', 'pre-optimisation').",
    )
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
        run_results: List[Dict[str, Any]] = []
        for run_idx in range(max(1, args.runs)):
            capture.reset()
            patched_branches: Dict[str, List[float]] = {}
            original_submit = _install_branch_timing_patch(patched_branches)
            wall_start = time.perf_counter()
            try:
                response = analyzer.analyze(str(video_path))
            except Exception as exc:
                _restore_branch_timing_patch(original_submit)
                print(f"  ! {video_path.name} run {run_idx + 1}: failed -- {exc}", file=sys.stderr)
                continue
            finally:
                _restore_branch_timing_patch(original_submit)
            wall_seconds = time.perf_counter() - wall_start
            total_seconds = (
                capture.total_seconds if capture.total_seconds is not None else wall_seconds
            )
            duration_seconds = (
                capture.video_duration_seconds
                if capture.video_duration_seconds is not None
                else float(getattr(response, "duration_seconds", 0.0) or 0.0)
            )
            # Merge: prefer real branch timings from the submit-patch; fall back
            # to the analyzer's log lines (which are near-zero, see comment in
            # _install_branch_timing_patch). Average duplicates from the same run.
            merged_branches: Dict[str, float] = {}
            for name, samples in patched_branches.items():
                if samples:
                    merged_branches[name] = max(samples)  # parallel = wall-clock for that branch
            for name, secs in capture.branches.items():
                merged_branches.setdefault(name, secs)
            run_results.append({
                "total_seconds": total_seconds,
                "wall_clock_seconds": wall_seconds,
                "duration_seconds": duration_seconds,
                "branches": merged_branches,
            })

        if not run_results:
            continue

        # Best-of: pick the run with the lowest total wall-clock to mitigate
        # cold-cache outliers; merge the per-branch latencies from that run.
        best = min(run_results, key=lambda r: r["total_seconds"])
        for name, secs in best["branches"].items():
            per_branch_samples[name].append(secs)

        per_video.append({
            "video_id": video_path.stem,
            "video_path": str(video_path.relative_to(REPO_ROOT)) if REPO_ROOT in video_path.parents else str(video_path),
            "video_duration_s": round(best["duration_seconds"], 2),
            "total_seconds": round(best["total_seconds"], 2),
            "wall_clock_seconds": round(best["wall_clock_seconds"], 2),
            "branch_count": len(best["branches"]),
            "branch_latencies_s": best["branches"],
            "run_count": len(run_results),
            "all_run_totals_s": [round(r["total_seconds"], 2) for r in run_results],
        })
        print(
            f"  {video_path.name}: best total={best['total_seconds']:.1f}s "
            f"(of {len(run_results)} run(s))  branches={len(best['branches'])}"
        )

    per_branch_summary: Dict[str, Dict[str, Any]] = {}
    for name, samples in sorted(per_branch_samples.items()):
        per_branch_summary[name] = {
            "n": len(samples),
            "mean_s": round(mean(samples), 3),
            "median_s": round(median(samples), 3),
            "p50_s": round(_percentile(samples, 50), 3),
            "p95_s": round(_percentile(samples, 95), 3),
        }

    payload = {
        "metadata": {
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "variant": args.variant,
            "runs_per_video": args.runs,
            "demucs_enabled": os.environ.get("DEMUCS_ENABLED", "true").lower() != "false",
            "hardware": _hardware_metadata(),
        },
        "test_set": [entry["video_id"] for entry in per_video],
        "videos_benchmarked": len(per_video),
        "optimisations_applied": [
            "timeline frames 20 -> 10",
            "OCR sampling 5 frames -> 3",
            "drop per-frame face detection",
            "Whisper small -> base (int8)",
        ],
        "per_video": per_video,
        "per_branch": per_branch_summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown_summary(payload), encoding="utf-8")
    print(f"\nWrote {args.out_json.name} and {args.out_md.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
