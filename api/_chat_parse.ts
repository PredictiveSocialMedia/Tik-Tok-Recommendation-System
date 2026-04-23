/**
 * Parses the DeepSeek chat response into the structured shape
 * expected by the frontend (summary + chunks + follow_ups).
 *
 * The model is instructed to emit strict JSON, but we still
 * defend against:
 *   - fenced ```json code blocks the model may emit
 *   - trailing commas
 *   - partial/invalid JSON — we fall back to a single-chunk shape
 *     so the UI always renders something useful.
 */

export interface StructuredChatReply {
  summary: string;
  chunks: string[];
  follow_ups: string[];
}

const MAX_CHUNKS = 6;
const MAX_FOLLOW_UPS = 4;
const CHUNK_MIN_CHARS = 20;
const CHUNK_MAX_CHARS = 600;

function stripFences(raw: string): string {
  const trimmed = raw.trim();
  // ```json ... ```   or   ``` ... ```
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenced && fenced[1]) {
    return fenced[1].trim();
  }
  return trimmed;
}

function coerceStringArray(value: unknown, maxItems: number): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .slice(0, maxItems);
}

function normaliseChunks(chunks: string[]): string[] {
  const bounded = chunks
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.length >= CHUNK_MIN_CHARS)
    .map((chunk) =>
      chunk.length > CHUNK_MAX_CHARS ? chunk.slice(0, CHUNK_MAX_CHARS - 1) + "…" : chunk
    );
  return bounded.slice(0, MAX_CHUNKS);
}

function fallbackChunksFromText(raw: string): string[] {
  // Split on double newlines first, then fall back to single newline.
  const paragraphs = raw.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  if (paragraphs.length > 1) return normaliseChunks(paragraphs);
  return raw.trim().length > 0 ? [raw.trim().slice(0, CHUNK_MAX_CHARS)] : [];
}

export function safeParseChatJson(raw: string): StructuredChatReply {
  const fallback: StructuredChatReply = {
    summary: "",
    chunks: fallbackChunksFromText(raw),
    follow_ups: [],
  };

  if (!raw || typeof raw !== "string") {
    return fallback;
  }

  const cleaned = stripFences(raw);

  // Fast path: JSON.parse
  try {
    const parsed = JSON.parse(cleaned) as unknown;
    if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      const summary = typeof obj.summary === "string" ? obj.summary.trim() : "";
      const chunks = normaliseChunks(coerceStringArray(obj.chunks, MAX_CHUNKS));
      const followUps = coerceStringArray(obj.follow_ups, MAX_FOLLOW_UPS);

      if (chunks.length > 0) {
        return { summary, chunks, follow_ups: followUps };
      }
    }
  } catch {
    // fall through
  }

  // Second attempt: strip trailing commas before `}` / `]`, retry parse.
  const repaired = cleaned.replace(/,\s*([}\]])/g, "$1");
  if (repaired !== cleaned) {
    try {
      const parsed = JSON.parse(repaired) as unknown;
      if (parsed && typeof parsed === "object") {
        const obj = parsed as Record<string, unknown>;
        const chunks = normaliseChunks(coerceStringArray(obj.chunks, MAX_CHUNKS));
        if (chunks.length > 0) {
          return {
            summary: typeof obj.summary === "string" ? obj.summary.trim() : "",
            chunks,
            follow_ups: coerceStringArray(obj.follow_ups, MAX_FOLLOW_UPS),
          };
        }
      }
    } catch {
      // fall through
    }
  }

  return fallback;
}
