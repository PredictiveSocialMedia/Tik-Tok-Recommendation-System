import { FFPROBE_BIN, RECOMMENDER_BASE_URL, UPLOAD_ANALYZER_PROVIDER } from "../config";
import type { AssetAnalysisProvider } from "./contracts";
import { createBaselineAssetAnalysisProvider } from "./providers/baselineAssetAnalyzer";
import { PythonAssetAnalysisProvider } from "./providers/pythonAssetAnalyzer";

interface CreateUploadAnalysisProviderOptions {
  provider?: string;
  ffprobeBin?: string;
}

export function createUploadAnalysisProvider(
  options: CreateUploadAnalysisProviderOptions = {}
): AssetAnalysisProvider {
  const provider = (options.provider ?? UPLOAD_ANALYZER_PROVIDER).trim().toLowerCase();

  const baseline = createBaselineAssetAnalysisProvider({
    ffprobeBin: options.ffprobeBin ?? FFPROBE_BIN
  });

  if (provider === "baseline") {
    return baseline;
  }

  if (provider === "python") {
    return new PythonAssetAnalysisProvider({
      pythonBaseUrl: RECOMMENDER_BASE_URL,
      timeoutMs: 120_000,
      fallback: baseline
    });
  }

  throw new Error(`Unsupported upload analysis provider: ${provider}`);
}
