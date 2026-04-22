"""Unit tests for src/recommendation/video/extractors/keywords.py

Tests cover language detection and keyword extraction behaviour.
Uses unittest.mock to patch langdetect and yake so zero ML dependencies
are needed — runs in plain CI with no model downloads.

Coverage:
- YAKE_LANGUAGE_MAP: spot-checks key language mappings
- detect_language: fallback on short text, import error, detection error,
  unsupported language code
- extract_keywords: empty/short input guard, auto_detect_language flag,
  language routing, result formatting
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.recommendation.video.extractors.keywords import (
    YAKE_LANGUAGE_MAP,
    detect_language,
    extract_keywords,
)


# ---------------------------------------------------------------------------
# YAKE_LANGUAGE_MAP
# ---------------------------------------------------------------------------

class TestYakeLanguageMap:
    def test_english_maps_to_en(self):
        assert YAKE_LANGUAGE_MAP["en"] == "en"

    def test_spanish_maps_to_es(self):
        assert YAKE_LANGUAGE_MAP["es"] == "es"

    def test_portuguese_maps_to_pt(self):
        assert YAKE_LANGUAGE_MAP["pt"] == "pt"

    def test_arabic_maps_to_ar(self):
        assert YAKE_LANGUAGE_MAP["ar"] == "ar"

    def test_chinese_simplified_maps_to_zh(self):
        assert YAKE_LANGUAGE_MAP["zh-cn"] == "zh"

    def test_chinese_traditional_maps_to_zh(self):
        assert YAKE_LANGUAGE_MAP["zh-tw"] == "zh"

    def test_all_values_are_non_empty_strings(self):
        for key, value in YAKE_LANGUAGE_MAP.items():
            assert isinstance(value, str) and value.strip(), \
                f"Bad value for {key!r}: {value!r}"


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_returns_fallback_for_empty_string(self):
        assert detect_language("") == "en"

    def test_returns_fallback_for_whitespace_only(self):
        assert detect_language("   ") == "en"

    def test_returns_fallback_for_very_short_text(self):
        assert detect_language("hi there") == "en"

    def test_custom_fallback_on_short_text(self):
        assert detect_language("short", fallback="es") == "es"

    def test_returns_fallback_when_langdetect_not_installed(self):
        with patch.dict("sys.modules", {"langdetect": None}):
            result = detect_language(
                "Este es un texto en español suficientemente largo para detectar",
                fallback="en",
            )
            assert result == "en"

    def test_returns_fallback_on_detection_exception(self):
        mock_langdetect = MagicMock()
        mock_langdetect.detect.side_effect = Exception("detection failed")
        mock_langdetect.LangDetectException = Exception
        with patch.dict("sys.modules", {"langdetect": mock_langdetect}):
            result = detect_language(
                "Some sufficiently long text that would normally be detected",
                fallback="fr",
            )
            assert result == "fr"

    def test_maps_detected_language_via_yake_map(self):
        mock_langdetect = MagicMock()
        mock_langdetect.detect.return_value = "es"
        mock_langdetect.LangDetectException = Exception
        with patch.dict("sys.modules", {"langdetect": mock_langdetect}):
            result = detect_language(
                "Este es un texto suficientemente largo en español para detección"
            )
            assert result == "es"

    def test_unsupported_language_falls_back(self):
        mock_langdetect = MagicMock()
        mock_langdetect.detect.return_value = "xx"
        mock_langdetect.LangDetectException = Exception
        with patch.dict("sys.modules", {"langdetect": mock_langdetect}):
            result = detect_language(
                "Some sufficiently long text for detection purposes here",
                fallback="en",
            )
            assert result == "en"

    def test_english_detection_returns_en(self):
        mock_langdetect = MagicMock()
        mock_langdetect.detect.return_value = "en"
        mock_langdetect.LangDetectException = Exception
        with patch.dict("sys.modules", {"langdetect": mock_langdetect}):
            result = detect_language(
                "This is a sufficiently long English text for language detection"
            )
            assert result == "en"


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_empty_for_empty_string(self):
        result = extract_keywords("")
        assert result == []

    def test_returns_empty_for_whitespace_only(self):
        result = extract_keywords("   ")
        assert result == []

    def test_returns_empty_for_very_short_text(self):
        result = extract_keywords("hi")
        assert result == []

    def test_auto_detect_false_uses_supplied_language(self):
        mock_yake = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = [("football skills", 0.05)]
        mock_yake.KeywordExtractor.return_value = mock_extractor
        with patch.dict("sys.modules", {"yake": mock_yake}):
            extract_keywords(
                "great football skills and dribbling techniques",
                language="es",
                auto_detect_language=False,
            )
            call_kwargs = mock_yake.KeywordExtractor.call_args
            assert call_kwargs.kwargs["lan"] == "es"

    def test_auto_detect_true_calls_detect_language(self):
        mock_yake = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = [("football", 0.1)]
        mock_yake.KeywordExtractor.return_value = mock_extractor
        with patch.dict("sys.modules", {"yake": mock_yake}):
            with patch(
                "src.recommendation.video.extractors.keywords.detect_language",
                return_value="es",
            ) as mock_detect:
                extract_keywords(
                    "texto suficientemente largo en español sobre fútbol y goles",
                    auto_detect_language=True,
                )
                mock_detect.assert_called_once()
                call_kwargs = mock_yake.KeywordExtractor.call_args
                assert call_kwargs.kwargs["lan"] == "es"

    def test_blank_keywords_filtered_out(self):
        mock_yake = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = [
            ("football", 0.05),
            ("   ", 0.06),
            ("", 0.07),
        ]
        mock_yake.KeywordExtractor.return_value = mock_extractor
        with patch.dict("sys.modules", {"yake": mock_yake}):
            results = extract_keywords(
                "football is a great sport to watch",
                auto_detect_language=False,
            )
            keywords = [r.keyword for r in results]
            assert "" not in keywords
            assert "   " not in keywords

    def test_max_keywords_passed_to_yake(self):
        mock_yake = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = []
        mock_yake.KeywordExtractor.return_value = mock_extractor
        with patch.dict("sys.modules", {"yake": mock_yake}):
            extract_keywords(
                "some sufficiently long text to extract keywords from",
                max_keywords=5,
                auto_detect_language=False,
            )
            call_kwargs = mock_yake.KeywordExtractor.call_args
            assert call_kwargs.kwargs["top"] == 5

    def test_language_fallback_when_detect_fails(self):
        mock_yake = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract_keywords.return_value = []
        mock_yake.KeywordExtractor.return_value = mock_extractor
        with patch.dict("sys.modules", {"yake": mock_yake}):
            with patch(
                "src.recommendation.video.extractors.keywords.detect_language",
                return_value="en",
            ):
                extract_keywords(
                    "some text that is long enough to pass the length check",
                    language="en",
                    auto_detect_language=True,
                )
                call_kwargs = mock_yake.KeywordExtractor.call_args
                assert call_kwargs.kwargs["lan"] == "en"