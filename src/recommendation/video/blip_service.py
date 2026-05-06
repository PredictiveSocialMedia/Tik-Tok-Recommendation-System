"""Standalone BLIP caption worker service for Cloud Run."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

from .analyzer import _extract_frames, _generate_video_caption_cpu, _load_blip


app = FastAPI(title="TikTok BLIP Worker", version="v1")


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("PRELOAD_BLIP", "true").lower() in {"1", "true", "yes"}:
        _load_blip()


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "blip",
        "model_id": os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base"),
    }


@app.post("/v1/blip/caption")
async def caption_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    started = time.perf_counter()
    try:
        tmp.write(await file.read())
        tmp.close()
        frames, _, fps, duration, width, height = _extract_frames(tmp.name, n_vlm=6)
        caption = _generate_video_caption_cpu(frames) or ""
        return {
            "video_caption": caption,
            "frames_used": len(frames),
            "fps": round(float(fps), 2),
            "duration_seconds": round(float(duration), 2),
            "resolution": f"{width}x{height}" if width and height else "",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "blip_caption_failed", "reason": str(error)},
        ) from error
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
