import numpy as np

from src.recommendation.video import analyzer


def _frame(value: int, shape=(24, 16, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_visual_helpers_detect_motion_and_scene_cuts():
    frames = [_frame(10), _frame(12), _frame(240), _frame(242)]

    assert analyzer._detect_scene_cuts(frames) == 1
    assert analyzer._compute_motion_score(frames) > 0.25
    assert analyzer._compute_hook_motion(frames, fps=1.0) > 0.25


def test_analyze_colors_handles_empty_and_rgb_frames():
    empty = analyzer._analyze_colors([])
    assert empty == ([], 0.0, 0.0, 0.0)

    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    colors, brightness, saturation, contrast = analyzer._analyze_colors([red])

    assert colors
    assert brightness > 0
    assert saturation > 0
    assert contrast >= 0


def test_timeline_generation_returns_stable_entries():
    frames = [_frame(0), _frame(80), _frame(180)]

    timeline = analyzer._generate_timeline(frames, fps=1.0, duration=3.0)

    assert len(timeline) == 3
    assert timeline[0].timestamp_sec == 0.0
    assert timeline[-1].timestamp_sec == 3.0
    assert all(entry.thumbnail_b64 for entry in timeline)


def test_video_analyzer_integration_with_mocked_extractors(monkeypatch, tmp_path):
    video_path = tmp_path / "tiny.mp4"
    video_path.write_bytes(b"not a real video; extractors are mocked")
    frames = [_frame(10), _frame(40), _frame(120)]

    monkeypatch.setattr(
        analyzer,
        "_extract_frames",
        lambda path, n_vlm=6: (frames[:2], frames, 30.0, 3.0, 1080, 1920),
    )
    monkeypatch.setattr(analyzer, "_extract_audio_track", lambda path: None)
    monkeypatch.setattr(analyzer, "_transcribe", lambda wav: ("hello ramen", 1.25, "en"))
    monkeypatch.setattr(
        analyzer,
        "_analyze_audio",
        lambda wav, duration: analyzer.AudioAnalysisResult(
            tempo_bpm=120.0,
            audio_energy=0.2,
            speech_seconds=0.0,
            music_seconds=0.0,
            has_source_separation=False,
        ),
    )
    monkeypatch.setattr(analyzer, "_generate_video_caption", lambda frames: "chef plating noodles")
    monkeypatch.setattr(analyzer, "_extract_ocr_text", lambda frames: "DINNER")
    monkeypatch.setattr(analyzer, "_compute_blur_score", lambda frames: 42.0)
    monkeypatch.setattr(analyzer, "_extract_keywords", lambda text, language="": ["ramen", "noodles"])

    response = analyzer.VideoAnalyzer(max_workers=3).analyze(str(video_path))

    assert response.transcript == "hello ramen"
    assert response.detected_language == "en"
    assert response.keywords == ["ramen", "noodles"]
    assert response.signal_hints.speech_seconds == 1.25
    assert response.signal_hints.music_seconds == 1.75
    assert response.visual_features.aspect_ratio == "9:16"
    assert response.visual_features.resolution == "1080x1920"
    assert response.visual_features.blur_score == 42.0
    assert response.timeline
