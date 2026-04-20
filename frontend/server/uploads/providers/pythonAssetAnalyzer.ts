/**
 * Asset analysis provider that calls the Python `/v1/video/analyze` endpoint
 * to get full video analysis (transcript, timeline, visual features, etc.).
 * Falls back to the baseline provider if the Python service is unreachable.
 */

import fs from "node:fs/promises";
import type {
  AssetAnalysisProvider,
  AssetAnalysisResult,
  UploadedVideoAsset
} from "../contracts";

interface PythonAssetAnalysisProviderConfig {
  pythonBaseUrl: string;
  timeoutMs?: number;
  fallback: AssetAnalysisProvider;
}

export class PythonAssetAnalysisProvider implements AssetAnalysisProvider {
  public readonly providerId = "python-service";

  private readonly pythonBaseUrl: string;
  private readonly timeoutMs: number;
  private readonly fallback: AssetAnalysisProvider;

  public constructor(config: PythonAssetAnalysisProviderConfig) {
    this.pythonBaseUrl = config.pythonBaseUrl.replace(/\/+$/, "");
    this.timeoutMs = config.timeoutMs ?? 600_000;
    this.fallback = config.fallback;
  }

  public async analyzeAsset(
    asset: UploadedVideoAsset
  ): Promise<AssetAnalysisResult> {
    // Always run baseline first (fast, gives us ffprobe metadata)
    const baselineResult = await this.fallback.analyzeAsset(asset);

    // Then try Python service for rich analysis (timeline, transcript, etc.)
    try {
      const pythonResult = await this.callPythonAnalyzer(asset);

      // Merge signal_hints: baseline provides audio/visual numbers,
      // Python overrides with real measurements and adds transcript_text / ocr_text
      const mergedSignalHints = {
        ...baselineResult.signal_hints,
        ...(pythonResult.signal_hints ?? {}),
        ...(pythonResult.transcript?.trim()
          ? { transcript_text: pythonResult.transcript.trim() }
          : {}),
        ...(pythonResult.ocr_text?.trim()
          ? { ocr_text: pythonResult.ocr_text.trim() }
          : {}),
      };

      // Override analysis.keyTopics with Python keywords when available
      const pythonKeywords =
        Array.isArray(pythonResult.keywords) && pythonResult.keywords.length > 0
          ? pythonResult.keywords.slice(0, 8)
          : null;
      const mergedAnalysis = {
        ...baselineResult.analysis,
        ...(pythonKeywords ? { keyTopics: pythonKeywords } : {}),
        ...(pythonResult.video_caption?.trim()
          ? { summary: pythonResult.video_caption.trim() }
          : {}),
      };

      return {
        ...baselineResult,
        signal_hints: mergedSignalHints as typeof baselineResult.signal_hints,
        analysis: mergedAnalysis,
        transcript: pythonResult.transcript ?? baselineResult.transcript,
        ocr_text: pythonResult.ocr_text ?? baselineResult.ocr_text,
        video_caption:
          pythonResult.video_caption ?? baselineResult.video_caption,
        detected_language:
          pythonResult.detected_language ?? baselineResult.detected_language,
        visual_features:
          pythonResult.visual_features ?? baselineResult.visual_features,
        timeline: pythonResult.timeline,
        duration_seconds:
          pythonResult.duration_seconds ?? baselineResult.duration_seconds
      };
    } catch (err) {
      console.warn(
        "Python video analyzer unavailable, using baseline only:",
        err instanceof Error ? err.message : String(err)
      );
      return baselineResult;
    }
  }

  private async callPythonAnalyzer(
    asset: UploadedVideoAsset
  ): Promise<PythonAnalysisResponse> {
    const fileBuffer = await fs.readFile(asset.storage_path);

    const url = `${this.pythonBaseUrl}/v1/video/analyze`;
    console.log(`[python-analyzer] POST ${url} (${(fileBuffer.byteLength / 1e6).toFixed(1)}MB, timeout=${this.timeoutMs}ms)`);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      // Use FormData API (Node 18+) for reliable multipart encoding
      const blob = new Blob([fileBuffer], { type: asset.mime_type || "video/mp4" });
      const formData = new FormData();
      formData.append("file", blob, asset.file_name || "upload.mp4");

      const response = await fetch(url, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(
          `Python analyzer returned ${response.status}: ${text.slice(0, 200)}`
        );
      }

      const result = (await response.json()) as PythonAnalysisResponse;
      console.log(`[python-analyzer] OK — timeline=${result.timeline?.length ?? 0} frames, caption=${(result.video_caption ?? "").slice(0, 60)}`);
      return result;
    } catch (err) {
      console.error(`[python-analyzer] FAILED:`, err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
}

interface PythonAnalysisResponse {
  transcript?: string;
  ocr_text?: string;
  video_caption?: string;
  detected_language?: string;
  keywords?: string[];
  visual_features?: AssetAnalysisResult["visual_features"];
  timeline?: AssetAnalysisResult["timeline"];
  duration_seconds?: number;
  signal_hints?: Record<string, unknown>;
}
