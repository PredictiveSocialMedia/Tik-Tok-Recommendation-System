"""YAKE keyword extraction — fast, statistical, no model needed.

Designed for short social media text. Outperforms TF-IDF on captions.

Changes (alp/keyword-extractor-multilingual):
- Fixed broken KeywordResult import — model does not exist in video/models.py.
  Return type is now a plain dataclass defined in this module.
- Added automatic language detection via langdetect so non-English captions
  get the correct language passed to YAKE instead of always defaulting to "en".
- Added YAKE_LANGUAGE_MAP covering all languages YAKE supports.
- detect_language() falls back gracefully if langdetect is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KeywordResult:
    """A single extracted keyword with its YAKE relevance score."""
    keyword: str
    score: float


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

YAKE_LANGUAGE_MAP: dict[str, str] = {
    "en": "en", "es": "es", "pt": "pt", "fr": "fr", "de": "de",
    "it": "it", "nl": "nl", "pl": "pl", "ro": "ro", "tr": "tr",
    "ar": "ar", "zh-cn": "zh", "zh-tw": "zh", "zh": "zh",
    "ja": "ja", "ko": "ko", "ru": "ru", "uk": "uk", "sv": "sv",
    "no": "no", "da": "da", "fi": "fi", "hu": "hu", "cs": "cs",
    "sk": "sk", "hr": "hr", "bg": "bg", "el": "el", "he": "he",
    "id": "id", "vi": "vi", "th": "th",
}


def detect_language(text: str, fallback: str = "en") -> str:
    """Detect language of text, return YAKE-compatible code or fallback."""
    if not text or len(text.strip()) < 10:
        return fallback
    try:
        from langdetect import detect, LangDetectException  # type: ignore
        detected = detect(text)
        return YAKE_LANGUAGE_MAP.get(detected, fallback)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords(
    text: str,
    max_keywords: int = 20,
    language: str = "en",
    n_gram_size: int = 2,
    dedup_threshold: float = 0.7,
    auto_detect_language: bool = True,
) -> List[KeywordResult]:
    """Extract keywords from text using YAKE."""
    if not text or len(text.strip()) < 5:
        return []

    effective_language = (
        detect_language(text, fallback=language)
        if auto_detect_language
        else language
    )

    import yake

    extractor = yake.KeywordExtractor(
        lan=effective_language,
        n=n_gram_size,
        dedupLim=dedup_threshold,
        top=max_keywords,
        features=None,
    )
    raw = extractor.extract_keywords(text)
    return [
        KeywordResult(keyword=kw.lower().strip(), score=round(score, 6))
        for kw, score in raw
        if kw.strip()
    ]
