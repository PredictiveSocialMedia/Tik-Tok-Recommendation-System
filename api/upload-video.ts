import type { VercelRequest, VercelResponse } from "@vercel/node";
import { randomUUID } from "crypto";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL =
  process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";

interface UploadJsonBody {
  file_name?: string;
  file_type?: string;
  file_size?: number;
  description?: string;
  hashtags?: string[];
  content_type?: string;
}

interface ContentAnalysis {
  caption: string;
  transcript: string;
  keywords: string[];
  topics: string[];
  suggested_hashtags: string[];
}

function collectBody(req: VercelRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    req.on("data", (chunk: Uint8Array) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

async function analyzeContent(
  fileName: string,
  description: string,
  hashtags: string[],
  contentType: string
): Promise<ContentAnalysis> {
  const empty: ContentAnalysis = {
    caption: "",
    transcript: description,
    keywords: [],
    topics: [],
    suggested_hashtags: hashtags,
  };

  if (!DEEPSEEK_API_KEY) return empty;

  const userInputs = [
    `File name: ${fileName}`,
    description ? `User description: ${description}` : null,
    hashtags.length ? `User hashtags: ${hashtags.join(", ")}` : null,
    contentType ? `Content type: ${contentType}` : null,
  ]
    .filter(Boolean)
    .join("\n");

  try {
    const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        temperature: 0.3,
        max_tokens: 400,
        messages: [
          {
            role: "system",
            content:
              "You are a TikTok content analyst. Given a video's filename and any user-provided metadata, " +
              "extract structured content signals. Return valid JSON only — no markdown, no explanation.\n" +
              "Schema: { \"caption\": string, \"transcript\": string, \"keywords\": string[], \"topics\": string[], \"suggested_hashtags\": string[] }\n" +
              "- caption: 1-2 sentence description of what the video is about\n" +
              "- transcript: a natural language summary of the likely spoken/visual content (2-3 sentences)\n" +
              "- keywords: 5-10 specific content keywords (no hashtag symbol)\n" +
              "- topics: 2-4 broad topic categories (e.g. fitness, cooking, travel)\n" +
              "- suggested_hashtags: 5-8 relevant hashtags including # symbol",
          },
          { role: "user", content: userInputs },
        ],
      }),
      signal: AbortSignal.timeout(20000),
    });

    if (!resp.ok) return empty;

    const data = (await resp.json()) as {
      choices: { message: { content: string } }[];
    };
    const raw = data.choices?.[0]?.message?.content?.trim() ?? "";
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return empty;

    const parsed = JSON.parse(jsonMatch[0]) as Partial<ContentAnalysis>;
    return {
      caption: typeof parsed.caption === "string" ? parsed.caption : empty.caption,
      transcript: typeof parsed.transcript === "string" ? parsed.transcript : description,
      keywords: Array.isArray(parsed.keywords) ? parsed.keywords.map(String) : [],
      topics: Array.isArray(parsed.topics) ? parsed.topics.map(String) : [],
      suggested_hashtags: Array.isArray(parsed.suggested_hashtags)
        ? parsed.suggested_hashtags.map(String)
        : hashtags,
    };
  } catch {
    return empty;
  }
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, x-file-name, x-file-type"
  );

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  // Support both JSON body (new) and raw octet-stream with headers (legacy)
  let fileName = (req.headers["x-file-name"] as string) ?? "uploaded_video.mp4";
  let fileType = (req.headers["x-file-type"] as string) ?? "video/mp4";

  const contentType = (req.headers["content-type"] ?? "").toString().toLowerCase();

  let description = "";
  let hashtags: string[] = [];
  let videoContentType = "";

  if (contentType.includes("application/json")) {
    try {
      const raw = await collectBody(req);
      const jsonBody = JSON.parse(raw.toString("utf-8")) as UploadJsonBody;
      if (jsonBody.file_name) fileName = jsonBody.file_name;
      if (jsonBody.file_type) fileType = jsonBody.file_type;
      if (typeof jsonBody.description === "string") description = jsonBody.description;
      if (Array.isArray(jsonBody.hashtags)) hashtags = jsonBody.hashtags.map(String);
      if (typeof jsonBody.content_type === "string") videoContentType = jsonBody.content_type;
    } catch {
      return res.status(400).json({ error: "Invalid JSON body." });
    }
  } else {
    // Legacy path: consume raw binary body (not stored on Vercel)
    try {
      await collectBody(req);
    } catch {
      return res.status(400).json({ error: "Failed to read upload." });
    }
  }

  const assetId = randomUUID();
  const analysis = await analyzeContent(fileName, description, hashtags, videoContentType);

  return res.status(201).json({
    asset_id: assetId,
    file_name: fileName,
    file_type: fileType,
    duration_seconds: null,
    video_caption: analysis.caption || null,
    transcript: analysis.transcript || null,
    ocr_text: null,
    visual_features: null,
    timeline: null,
    signal_hints: {
      transcript_text: analysis.transcript || description || undefined,
      video_caption: analysis.caption || undefined,
      keywords: analysis.keywords,
      topics: analysis.topics,
      suggested_hashtags: analysis.suggested_hashtags,
    },
    asset: {
      asset_id: assetId,
      original_filename: fileName,
      content_type: fileType,
      size_bytes: 0,
      duration_seconds: null,
    },
  });
}
