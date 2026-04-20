/**
 * signalProfile.ts
 *
 * Changes (alp/signal-profile-content-type-awareness):
 * - Replaced the single `contentType === "tutorial"` check with a full
 *   content-type profile table covering: tutorial, dance, comedy, gaming,
 *   cooking, fitness, travel, music, news, lifestyle, and the original
 *   fallback for unknown types.
 * - Each profile defines expected speech_ratio, visual_motion_score bias,
 *   audio_style, hook timing, and quality_flags so the fallback signal
 *   profile is meaningfully different per niche rather than returning the
 *   same numbers with a different label.
 * - All existing fields and the public buildSignalProfileFallback() API are
 *   fully preserved — callers need no changes.
 */

export interface CandidateSignalProfile {
  pipeline_version: "extractors.v1";
  generated_at: string;
  duration_seconds: number;
  visual: {
    confidence: number;
    shot_change_rate: number;
    visual_motion_score: number;
    visual_style_tags: string[];
    semantic_embedding_proxy: number[];
  };
  audio: {
    confidence: number;
    speech_ratio: number;
    tempo_bpm_estimate: number;
    audio_energy: number;
    music_presence_score: number;
    audio_style_tags: string[];
  };
  transcript_ocr: {
    confidence: number;
    transcript_text: string;
    ocr_text: string;
    combined_text: string;
    token_count: number;
    unique_token_count: number;
    clarity_score: number;
    cta_keywords_detected: string[];
  };
  structure: {
    confidence: number;
    hook_timing_seconds: number;
    payoff_timing_seconds: number;
    step_density: number;
    pacing_score: number;
  };
  overall_confidence: number;
  quality_flags: string[];
}

interface SignalHints {
  duration_seconds?: number;
  transcript_text?: string;
  ocr_text?: string;
  estimated_scene_cuts?: number;
  visual_motion_score?: number;
  speech_seconds?: number;
  music_seconds?: number;
  tempo_bpm?: number;
  audio_energy?: number;
}

// ─── Content-type profile table ────────────────────────────────────────────

/**
 * Per-niche signal defaults.
 *
 * speech_ratio_base   – baseline proportion of audio that is speech [0,1]
 * motion_bias         – added to the computed visual_motion_score [0,1]
 * tempo_bpm_base      – baseline BPM estimate
 * audio_energy_base   – baseline audio energy [0,1]
 * hook_ratio          – hook timing as fraction of duration
 * payoff_ratio        – payoff timing as fraction of duration
 * quality_flags       – content-type-specific flags appended to the profile
 * audio_style_tag     – primary audio style label
 * visual_style_tag    – primary visual style label
 */
interface ContentTypeProfile {
  speech_ratio_base: number;
  motion_bias: number;
  tempo_bpm_base: number;
  audio_energy_base: number;
  hook_ratio: number;
  payoff_ratio: number;
  quality_flags: string[];
  audio_style_tag: string;
  visual_style_tag: string;
}

const CONTENT_TYPE_PROFILES: Record<string, ContentTypeProfile> = {
  tutorial: {
    speech_ratio_base: 0.65,
    motion_bias: -0.05,
    tempo_bpm_base: 105,
    audio_energy_base: 0.52,
    hook_ratio: 0.10,
    payoff_ratio: 0.80,
    quality_flags: ["content_type_tutorial"],
    audio_style_tag: "voice_forward",
    visual_style_tag: "instructional_pacing",
  },
  dance: {
    speech_ratio_base: 0.12,
    motion_bias: 0.25,
    tempo_bpm_base: 128,
    audio_energy_base: 0.82,
    hook_ratio: 0.05,
    payoff_ratio: 0.60,
    quality_flags: ["content_type_dance", "high_motion_expected"],
    audio_style_tag: "music_forward",
    visual_style_tag: "high_motion",
  },
  comedy: {
    speech_ratio_base: 0.55,
    motion_bias: 0.05,
    tempo_bpm_base: 115,
    audio_energy_base: 0.62,
    hook_ratio: 0.08,
    payoff_ratio: 0.70,
    quality_flags: ["content_type_comedy"],
    audio_style_tag: "voice_forward",
    visual_style_tag: "reactive_pacing",
  },
  gaming: {
    speech_ratio_base: 0.45,
    motion_bias: 0.15,
    tempo_bpm_base: 120,
    audio_energy_base: 0.70,
    hook_ratio: 0.06,
    payoff_ratio: 0.75,
    quality_flags: ["content_type_gaming"],
    audio_style_tag: "mixed_audio",
    visual_style_tag: "screen_capture_motion",
  },
  cooking: {
    speech_ratio_base: 0.50,
    motion_bias: -0.08,
    tempo_bpm_base: 100,
    audio_energy_base: 0.45,
    hook_ratio: 0.12,
    payoff_ratio: 0.85,
    quality_flags: ["content_type_cooking"],
    audio_style_tag: "voice_forward",
    visual_style_tag: "slow_reveal_pacing",
  },
  fitness: {
    speech_ratio_base: 0.35,
    motion_bias: 0.18,
    tempo_bpm_base: 130,
    audio_energy_base: 0.78,
    hook_ratio: 0.07,
    payoff_ratio: 0.65,
    quality_flags: ["content_type_fitness", "high_motion_expected"],
    audio_style_tag: "music_forward",
    visual_style_tag: "high_motion",
  },
  travel: {
    speech_ratio_base: 0.30,
    motion_bias: 0.08,
    tempo_bpm_base: 110,
    audio_energy_base: 0.60,
    hook_ratio: 0.10,
    payoff_ratio: 0.72,
    quality_flags: ["content_type_travel"],
    audio_style_tag: "music_forward",
    visual_style_tag: "cinematic_pacing",
  },
  music: {
    speech_ratio_base: 0.10,
    motion_bias: 0.10,
    tempo_bpm_base: 120,
    audio_energy_base: 0.88,
    hook_ratio: 0.05,
    payoff_ratio: 0.55,
    quality_flags: ["content_type_music"],
    audio_style_tag: "music_forward",
    visual_style_tag: "performance_pacing",
  },
  news: {
    speech_ratio_base: 0.80,
    motion_bias: -0.15,
    tempo_bpm_base: 95,
    audio_energy_base: 0.40,
    hook_ratio: 0.08,
    payoff_ratio: 0.90,
    quality_flags: ["content_type_news"],
    audio_style_tag: "voice_forward",
    visual_style_tag: "static_framing",
  },
  lifestyle: {
    speech_ratio_base: 0.48,
    motion_bias: 0.0,
    tempo_bpm_base: 108,
    audio_energy_base: 0.55,
    hook_ratio: 0.12,
    payoff_ratio: 0.75,
    quality_flags: ["content_type_lifestyle"],
    audio_style_tag: "mixed_audio",
    visual_style_tag: "balanced_motion",
  },
};

/** Fallback profile for unknown or missing content types. */
const DEFAULT_PROFILE: ContentTypeProfile = {
  speech_ratio_base: 0.50,
  motion_bias: 0.0,
  tempo_bpm_base: 118,
  audio_energy_base: 0.55,
  hook_ratio: 0.12,
  payoff_ratio: 0.72,
  quality_flags: [],
  audio_style_tag: "mixed_audio",
  visual_style_tag: "balanced_motion",
};

function resolveProfile(contentType?: string): ContentTypeProfile {
  if (!contentType) return DEFAULT_PROFILE;
  return CONTENT_TYPE_PROFILES[contentType.toLowerCase()] ?? DEFAULT_PROFILE;
}

// ─── Utilities ─────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 4): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// ─── Main builder ──────────────────────────────────────────────────────────

export function buildSignalProfileFallback(params: {
  description: string;
  contentType?: string;
  signalHints?: Record<string, unknown>;
}): CandidateSignalProfile {
  const hints = (params.signalHints ?? {}) as SignalHints;
  const profile = resolveProfile(params.contentType);

  const description = params.description.trim();
  const transcriptText =
    typeof hints.transcript_text === "string" ? hints.transcript_text : "";
  const ocrText = typeof hints.ocr_text === "string" ? hints.ocr_text : "";
  const combinedText = [description, transcriptText, ocrText]
    .filter(Boolean)
    .join(" ")
    .trim();
  const tokens = normalizeText(combinedText)
    .split(" ")
    .filter((token) => token.length >= 2);
  const uniqueTokenCount = new Set(tokens).size;

  const durationSeconds = clamp(
    Math.round(hints.duration_seconds ?? 35),
    8,
    600
  );

  const sceneCuts = clamp(
    Math.round(
      hints.estimated_scene_cuts ?? Math.max(8, tokens.length * 0.5)
    ),
    4,
    80
  );
  const shotChangeRate = round(
    sceneCuts / Math.max(1, durationSeconds),
    4
  );

  // Visual motion: hint → computed + content-type bias
  const baseMotion = hints.visual_motion_score ?? shotChangeRate / 1.8;
  const visualMotionScore = round(
    clamp(baseMotion + profile.motion_bias, 0, 1),
    4
  );

  // Speech ratio: hint → content-type base
  const speechRatioHint =
    hints.speech_seconds !== undefined
      ? hints.speech_seconds / Math.max(1, durationSeconds)
      : undefined;
  const speechRatio = round(
    clamp(speechRatioHint ?? profile.speech_ratio_base, 0, 1),
    4
  );

  const speechSeconds = clamp(durationSeconds * speechRatio, 0, durationSeconds);
  const musicSeconds = clamp(
    hints.music_seconds ?? Math.max(0, durationSeconds - speechSeconds),
    0,
    durationSeconds
  );

  const clarityScore = round(
    clamp(
      uniqueTokenCount / Math.max(1, tokens.length || 1) + 0.35,
      0,
      1
    ),
    4
  );

  const pacingScore = round(
    clamp(shotChangeRate * 0.8 + speechRatio * 0.2, 0, 1),
    4
  );

  const hookTimingSeconds = round(
    clamp(durationSeconds * profile.hook_ratio, 0.5, 8),
    2
  );
  const payoffTimingSeconds = round(
    clamp(durationSeconds * profile.payoff_ratio, 3, durationSeconds),
    2
  );

  const ctaKeywordsDetected = [
    "follow",
    "comment",
    "save",
    "share",
    "link",
    "bio",
  ].filter((term) => normalizeText(combinedText).includes(term));

  // Visual style: high motion overrides content-type tag
  const visualStyleTag =
    visualMotionScore > 0.55 ? "high_motion" : profile.visual_style_tag;

  // Audio style: speech ratio overrides content-type tag
  const audioStyleTag =
    speechRatio >= 0.55 ? "voice_forward" : profile.audio_style_tag;

  const qualityFlags: string[] = [
    "ts_signal_fallback",
    ...profile.quality_flags,
  ];

  return {
    pipeline_version: "extractors.v1",
    generated_at: new Date().toISOString(),
    duration_seconds: durationSeconds,
    visual: {
      confidence: 0.42,
      shot_change_rate: shotChangeRate,
      visual_motion_score: visualMotionScore,
      visual_style_tags: [visualStyleTag],
      semantic_embedding_proxy: [
        round(clamp(tokens.length / 80, 0, 1), 6),
        round(clamp(uniqueTokenCount / 60, 0, 1), 6),
        pacingScore,
        visualMotionScore,
      ],
    },
    audio: {
      confidence: 0.4,
      speech_ratio: speechRatio,
      tempo_bpm_estimate: round(
        clamp(hints.tempo_bpm ?? profile.tempo_bpm_base, 60, 220),
        2
      ),
      audio_energy: round(
        clamp(hints.audio_energy ?? profile.audio_energy_base, 0, 1),
        4
      ),
      music_presence_score: round(clamp(1 - speechRatio, 0, 1), 4),
      audio_style_tags: [audioStyleTag],
    },
    transcript_ocr: {
      confidence: 0.38,
      transcript_text: transcriptText,
      ocr_text: ocrText,
      combined_text: combinedText,
      token_count: tokens.length,
      unique_token_count: uniqueTokenCount,
      clarity_score: clarityScore,
      cta_keywords_detected: ctaKeywordsDetected,
    },
    structure: {
      confidence: 0.4,
      hook_timing_seconds: hookTimingSeconds,
      payoff_timing_seconds: payoffTimingSeconds,
      step_density: round(
        clamp(tokens.length / Math.max(1, durationSeconds), 0, 1.25),
        4
      ),
      pacing_score: pacingScore,
    },
    overall_confidence: 0.4,
    quality_flags: qualityFlags,
  };
}