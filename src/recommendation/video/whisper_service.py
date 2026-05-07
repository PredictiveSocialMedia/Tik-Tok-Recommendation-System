"""Standalone Whisper worker service for Cloud Run."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

from .analyzer import _extract_audio_track, _load_whisper_model, _transcribe


app = FastAPI(title="TikTok Whisper Worker", version="v1")


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("PRELOAD_WHISPER", "true").lower() in {"1", "true", "yes"}:
        _load_whisper_model()


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "whisper",
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "base"),
    }


@app.post("/v1/whisper/transcribe")
async def transcribe_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    wav_path = None
    started = time.perf_counter()
    try:
        tmp.write(await file.read())
        tmp.close()
        wav_path = _extract_audio_track(tmp.name)
        transcript, speech_seconds, language = _transcribe(wav_path)
        return {
            "transcript": transcript,
            "speech_seconds": round(float(speech_seconds), 2),
            "detected_language": language,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "whisper_transcription_failed", "reason": str(error)},
        ) from error
    finally:
        for path in (tmp.name, wav_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
