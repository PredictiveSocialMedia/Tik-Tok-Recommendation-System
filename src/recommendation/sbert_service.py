"""Standalone SBERT hashtag recommender service for Cloud Run."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common.config import settings
from src.recommendation.hashtag_recommender import HashtagRecommender


class HashtagSuggestRequest(BaseModel):
    caption: str
    k: int = Field(default=25, ge=1, le=200)
    top_n: int = Field(default=10, ge=1, le=50)
    diversity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    exclude_tags: List[str] = Field(default_factory=list)
    include_neighbours: bool = False


app = FastAPI(title="TikTok SBERT Hashtag Worker", version="v1")
_recommender: Optional[HashtagRecommender] = None


def _artifact_dir() -> Path:
    return Path(os.getenv("HASHTAG_RECOMMENDER_DIR") or settings.hashtag_recommender_dir)


def _load_recommender() -> HashtagRecommender:
    global _recommender
    if _recommender is None:
        allow_downloads = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        _recommender = HashtagRecommender.load(
            _artifact_dir(),
            local_files_only=not allow_downloads,
        )
    return _recommender


@app.on_event("startup")
def _startup() -> None:
    if os.getenv("PRELOAD_SBERT", "true").lower() in {"1", "true", "yes"}:
        _load_recommender()


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    loaded = _recommender is not None
    return {
        "ok": True,
        "service": "sbert-hashtag",
        "artifact_dir": str(_artifact_dir()),
        "loaded": loaded,
        "corpus_size": len(_recommender.corpus_captions) if loaded else 0,
    }


@app.post("/v1/hashtags/suggest")
def suggest_hashtags(request: HashtagSuggestRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        recommender = _load_recommender()
        if request.include_neighbours:
            result = recommender.recommend_with_neighbours(
                caption=request.caption,
                k=request.k,
                top_n=request.top_n,
            )
        else:
            result = {
                "hashtags": recommender.recommend(
                    request.caption,
                    k=request.k,
                    top_n=request.top_n,
                    exclude_tags=request.exclude_tags,
                    diversity_weight=request.diversity_weight,
                )
            }
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        result["corpus_size"] = len(recommender.corpus_captions)
        return result
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={"error": "sbert_hashtag_recommender_failed", "reason": str(error)},
        ) from error
