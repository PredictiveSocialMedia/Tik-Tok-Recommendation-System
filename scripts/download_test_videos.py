#!/usr/bin/env python3
"""Download a small fixed set of TikTok videos for the video-pipeline benchmark.

Reads ``data/real/tiktok_posts_real.jsonl`` (the project's scraped dataset)
and uses yt-dlp to pull a deterministic sample of N videos into
``data/test_videos/``. The sampled ``video_id`` list is pinned to
``benchmarks/video_pipeline_test_set.json`` after the first successful run so
re-runs reuse the same set.

Failures are logged to ``data/test_videos/_download_log.json`` and skipped
rather than aborting the run, since TikTok rate-limits anonymous requests.

Usage::

    python scripts/download_test_videos.py --n 8
    python scripts/download_test_videos.py --pinned   # use the pinned list
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _sample_diverse(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pick N rows that cover short / medium / long buckets when duration is known.

    If durations are unknown (the scraped dataset does not always have them),
    fall back to a deterministic stride sample so the function still picks the
    same N rows on every run.
    """
    with_url = [r for r in rows if isinstance(r.get("video_url"), str)]
    with_duration = [r for r in with_url if isinstance(r.get("duration_seconds"), (int, float))]

    if len(with_duration) >= n:
        with_duration.sort(key=lambda r: r["duration_seconds"])
        # Stride evenly across the duration distribution so we cover short,
        # medium and long videos without manually bucketing.
        step = max(1, len(with_duration) // n)
        return with_duration[::step][:n]

    if not with_url:
        return []
    step = max(1, len(with_url) // n)
    return with_url[::step][:n]


def _yt_dlp_download(url: str, out_path: Path) -> Optional[str]:
    """Download a single TikTok URL into ``out_path``. Returns error string on failure."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return "yt-dlp is not installed (pip install yt-dlp)"

    if out_path.exists():
        return None

    out_template = str(out_path.with_suffix("")) + ".%(ext)s"
    options = {
        "outtmpl": out_template,
        "format": "mp4/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 2,
        "fragment_retries": 2,
        "merge_output_format": "mp4",
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:  # yt-dlp raises a wide variety of error types
        return f"{type(exc).__name__}: {exc}"

    # yt-dlp may write .mp4, .webm, etc. depending on availability.
    candidates = list(out_path.parent.glob(f"{out_path.stem}.*"))
    mp4 = next((p for p in candidates if p.suffix.lower() == ".mp4"), None)
    if mp4 is None and candidates:
        # Take the first non-mp4 and rename it to .mp4 if ffmpeg already merged.
        first = candidates[0]
        first.rename(first.with_suffix(".mp4"))
        mp4 = first.with_suffix(".mp4")
    if mp4 is None:
        return "no output file produced"
    if mp4.resolve() != out_path.resolve():
        # yt-dlp can name files differently; normalise.
        shutil.move(str(mp4), str(out_path))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "real" / "tiktok_posts_real.jsonl",
        help="Path to the scraped TikTok dataset (JSONL with video_url + video_id).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "test_videos",
        help="Directory to save downloaded mp4 files into.",
    )
    parser.add_argument(
        "--pinned-list",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "video_pipeline_test_set.json",
        help="Pinned list of video_ids for reproducible benchmarks. Created after first run.",
    )
    parser.add_argument("--n", type=int, default=8, help="Number of videos to sample.")
    parser.add_argument(
        "--pinned",
        action="store_true",
        help="Use the pinned video_id list instead of resampling.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pinned_list.parent.mkdir(parents=True, exist_ok=True)

    if args.pinned and args.pinned_list.exists():
        pinned_payload = json.loads(args.pinned_list.read_text(encoding="utf-8"))
        wanted_ids = pinned_payload.get("video_ids", [])
        all_rows = _read_dataset(args.dataset)
        by_id = {str(r.get("video_id")): r for r in all_rows}
        sample = [by_id[vid] for vid in wanted_ids if vid in by_id]
        print(f"Using pinned set: {len(sample)} video_ids from {args.pinned_list.name}")
    else:
        rows = _read_dataset(args.dataset)
        sample = _sample_diverse(rows, args.n)
        print(f"Sampled {len(sample)} videos from {len(rows)} rows in {args.dataset.name}")

    log: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downloads": [],
    }

    successes: List[str] = []
    for row in sample:
        vid = str(row.get("video_id", "")).strip()
        url = row.get("video_url")
        if not vid or not isinstance(url, str):
            continue
        out_path = args.out_dir / f"{vid}.mp4"
        if out_path.exists():
            print(f"  [skip] {vid} (already downloaded)")
            successes.append(vid)
            log["downloads"].append({"video_id": vid, "status": "cached", "path": str(out_path.relative_to(REPO_ROOT))})
            continue
        print(f"  [pull] {vid}: {url[:60]}...")
        err = _yt_dlp_download(url, out_path)
        if err is None:
            successes.append(vid)
            log["downloads"].append({"video_id": vid, "status": "ok", "path": str(out_path.relative_to(REPO_ROOT))})
        else:
            log["downloads"].append({"video_id": vid, "status": "failed", "error": err, "url": url})
            print(f"    failed: {err}", file=sys.stderr)

    log["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["successes"] = len(successes)
    log["failures"] = len(sample) - len(successes)

    log_path = args.out_dir / "_download_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if successes and not args.pinned_list.exists():
        pinned_payload = {
            "generated_at_utc": log["started_at"],
            "source_dataset": str(args.dataset.relative_to(REPO_ROOT)),
            "video_ids": successes,
        }
        args.pinned_list.write_text(json.dumps(pinned_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Pinned {len(successes)} video_ids to {args.pinned_list.relative_to(REPO_ROOT)}")

    print(f"\nDownloaded {len(successes)} of {len(sample)} videos. Log: {log_path.relative_to(REPO_ROOT)}")
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
