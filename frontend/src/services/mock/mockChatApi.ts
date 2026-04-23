import type { IChatService } from "../contracts/IChatService";
import type { ChatMessage, ChatRequest } from "../contracts/models";
import { CHAT_KEYWORD_RESPONSES } from "./fixtures";

function randomDelay(minMs: number, maxMs: number): number {
  return Math.floor(Math.random() * (maxMs - minMs + 1)) + minMs;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function createMessageId(): string {
  return `assistant-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function pickAnswer(message: string): string {
  const normalized = normalizeText(message);

  if (normalized.includes("hashtag")) {
    return CHAT_KEYWORD_RESPONSES.hashtags;
  }

  if (normalized.includes("retention")) {
    return CHAT_KEYWORD_RESPONSES.retention;
  }

  if (normalized.includes("summary")) {
    return CHAT_KEYWORD_RESPONSES.summary;
  }

  return CHAT_KEYWORD_RESPONSES.fallback;
}

/**
 * Splits a longer mock answer into reasonably-sized chunks so the UI
 * still exercises the multi-bubble + follow-up render path in dev mode.
 */
function splitIntoChunks(text: string): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  if (paragraphs.length >= 2) return paragraphs.slice(0, 5);

  // Fallback: break on sentence boundaries for single-paragraph answers.
  const sentences = text.match(/[^.!?\n]+[.!?]+(\s|$)/g) ?? [text];
  const chunks: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    if ((current + sentence).length > 260 && current.length > 0) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current += sentence;
    }
  }
  if (current.trim().length > 0) chunks.push(current.trim());
  return chunks.length > 0 ? chunks.slice(0, 5) : [text];
}

const FOLLOW_UPS_BY_TOPIC: Record<string, string[]> = {
  hashtag: [
    "Which hashtag should I drop first?",
    "How many hashtags should I use?",
    "What niche hashtags would fit?"
  ],
  retention: [
    "How do I fix my hook in 3 seconds?",
    "Where do viewers drop off most?",
    "What pacing works for my niche?"
  ],
  summary: [
    "What is my strongest comparable?",
    "Which objective fits my video best?",
    "How do I rewrite my caption?"
  ],
  fallback: [
    "How do I improve my hook?",
    "Which hashtag should I drop?",
    "What would a stronger CTA look like?"
  ]
};

function pickFollowUps(message: string): string[] {
  const normalized = normalizeText(message);
  if (normalized.includes("hashtag")) return FOLLOW_UPS_BY_TOPIC.hashtag;
  if (normalized.includes("retention")) return FOLLOW_UPS_BY_TOPIC.retention;
  if (normalized.includes("summary")) return FOLLOW_UPS_BY_TOPIC.summary;
  return FOLLOW_UPS_BY_TOPIC.fallback;
}

export class MockChatService implements IChatService {
  public async sendMessage(request: ChatRequest): Promise<ChatMessage> {
    await wait(randomDelay(650, 1050));

    const answer = pickAnswer(request.question);
    const chunks = splitIntoChunks(answer);

    return {
      id: createMessageId(),
      role: "assistant",
      content: answer,
      summary: chunks[0]?.slice(0, 140),
      chunks,
      followUps: pickFollowUps(request.question),
      timestamp: new Date().toISOString()
    };
  }
}
