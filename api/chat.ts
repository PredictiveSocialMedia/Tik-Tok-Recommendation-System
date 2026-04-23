import type { VercelRequest, VercelResponse } from "@vercel/node";
import { safeParseChatJson } from "./_chat_parse";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";

const CHAT_SYSTEM_PROMPT = [
  "You are a TikTok content strategist helping a creator improve a specific video.",
  "You have access to a recommendation report (top 5 comparable videos with scores and hashtags) and a video analysis (caption, transcript, keywords).",
  "",
  'Respond ONLY with a JSON object matching this exact shape: { "summary": string, "chunks": string[], "follow_ups": string[] }',
  "",
  "Rules:",
  "- summary: one plain-text sentence, <=140 chars, no emojis, no markdown. Captures the single most important diagnosis.",
  "- chunks: 3 to 5 items. Each 80-260 chars. One idea per chunk.",
  '- Each chunk starts with a short lead phrase followed by a colon (e.g., "Hook timing:", "Caption rewrite:", "Pacing fix:").',
  "- Plain text only. Never use **, ##, or any markdown symbol — the client renders plain paragraphs.",
  "- Max 2 emojis total across the ENTIRE response. Only when they add genuine clarity (for example 🎯 hook, 📈 metric, ⚡ pacing). Never decorative.",
  "- Ground every claim in the report/video data (cite candidate numbers, scores, specific keywords, timestamps).",
  "- Avoid filler phrases like 'Based on the data' or 'As a strategist'. Get straight to the point.",
  "",
  "- follow_ups: exactly 3 natural questions the user might ask next, phrased in first person as if the user is speaking.",
  '  Examples: "How do I rewrite my hook?", "Which hashtag should I drop?", "What would a stronger CTA look like?"',
  "  Each <=90 chars, specific to THIS conversation (not generic).",
  "",
  "Output only the JSON object. No preamble, no closing remarks, no code fences.",
].join("\n");

interface ChatBody {
  question?: string;
  report?: Record<string, unknown> | null;
  videoAnalysis?: Record<string, unknown> | null;
}

function buildContext(body: ChatBody): string {
  const parts: string[] = [];

  if (body.report) {
    const r = body.report as Record<string, unknown>;
    if (r.comparables && Array.isArray(r.comparables)) {
      const top = (r.comparables as Record<string, unknown>[]).slice(0, 5);
      parts.push(
        "## Recommendation Report (top 5 comparables)\n" +
          top
            .map((c, i) => {
              const caption = c.caption ?? c.candidate_id ?? `#${i + 1}`;
              const score = typeof c.score === "number" ? c.score.toFixed(3) : "n/a";
              const hashtags = Array.isArray(c.hashtags) ? (c.hashtags as string[]).join(", ") : "";
              return `${i + 1}. [${score}] ${caption}${hashtags ? ` | tags: ${hashtags}` : ""}`;
            })
            .join("\n")
      );
    }
    if (typeof r.summary === "string") {
      parts.push(`## Report Summary\n${r.summary}`);
    }
  }

  if (body.videoAnalysis) {
    const va = body.videoAnalysis as Record<string, unknown>;
    const snippets: string[] = [];
    if (va.video_caption) snippets.push(`Caption: ${va.video_caption}`);
    if (va.transcript) snippets.push(`Transcript: ${va.transcript}`);
    if (va.duration_seconds) snippets.push(`Duration: ${va.duration_seconds}s`);
    if (va.hashtags && Array.isArray(va.hashtags))
      snippets.push(`Hashtags: ${(va.hashtags as string[]).join(", ")}`);
    if (snippets.length > 0) {
      parts.push("## Video Analysis\n" + snippets.join("\n"));
    }
  }

  return parts.join("\n\n");
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const body = req.body as ChatBody;
  const question = typeof body.question === "string" ? body.question.trim() : "";

  if (!question) {
    return res.status(400).json({ error: "A question is required." });
  }

  if (!DEEPSEEK_API_KEY) {
    return res.json({
      answer: "Upload a video and generate a report to start chatting.",
      summary: "",
      chunks: ["Upload a video and generate a report to start chatting."],
      follow_ups: [],
      sources: [],
    });
  }

  let context = buildContext(body);
  // Truncate context to avoid exceeding DeepSeek token limits
  const MAX_CONTEXT_CHARS = 3000;
  if (context.length > MAX_CONTEXT_CHARS) {
    context = context.slice(0, MAX_CONTEXT_CHARS) + "\n\n[...context truncated]";
  }
  const userContent = context
    ? `${context}\n\n---\n\nUser question: ${question}`
    : question;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25_000);

    const isReasoner = DEEPSEEK_MODEL.includes("reasoner");
    const response = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      signal: controller.signal,
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        // deepseek-reasoner does not accept temperature
        ...(isReasoner ? {} : { temperature: 0.4 }),
        max_tokens: 2048,
        // deepseek-chat supports JSON mode; reasoner does not — we parse defensively either way.
        ...(isReasoner ? {} : { response_format: { type: "json_object" as const } }),
        messages: [
          // deepseek-reasoner ignores system messages — prepend instructions to user content instead.
          ...(isReasoner
            ? []
            : [{ role: "system" as const, content: CHAT_SYSTEM_PROMPT }]),
          {
            role: "user",
            content: isReasoner
              ? CHAT_SYSTEM_PROMPT + "\n\n---\n\n" + userContent
              : userContent,
          },
        ],
      }),
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const text = await response.text();
      console.error("DeepSeek error:", response.status, text);
      console.error("DeepSeek request model:", DEEPSEEK_MODEL, "| user content length:", userContent.length);
      const message = "The AI assistant is currently unavailable. Please try again.";
      return res.json({
        answer: message,
        summary: "",
        chunks: [message],
        follow_ups: [],
        sources: [],
      });
    }

    const data = (await response.json()) as {
      choices: { message: { content: string } }[];
    };
    const raw = data.choices?.[0]?.message?.content?.trim() ?? "";
    const parsed = safeParseChatJson(raw);

    // `answer` is kept as a flattened plain-text fallback so any legacy client
    // reading only { answer } still renders something useful.
    const answer = parsed.chunks.length > 0 ? parsed.chunks.join("\n\n") : raw || "No response.";

    return res.json({
      answer,
      summary: parsed.summary,
      chunks: parsed.chunks,
      follow_ups: parsed.follow_ups,
      sources: ["deepseek"],
    });
  } catch (err) {
    console.error("Chat error:", err);
    const message = "The AI assistant is currently unavailable. Please try again.";
    return res.json({
      answer: message,
      summary: "",
      chunks: [message],
      follow_ups: [],
      sources: [],
    });
  }
}
