import { MockChatService } from "../mock/mockChatApi";
import type { IChatService } from "../contracts/IChatService";
import type { ChatMessage, ChatRequest } from "../contracts/models";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const CHAT_API_URL = buildApiUrl("/chat");

interface ChatApiResponse {
  answer: string;
  summary?: string;
  chunks?: string[];
  follow_ups?: string[];
  sources?: string[];
  evidence_refs?: string[];
}

function createAssistantMessage(parsed: ChatApiResponse): ChatMessage {
  const normalisedChunks = Array.isArray(parsed.chunks)
    ? parsed.chunks.filter((chunk): chunk is string => typeof chunk === "string" && chunk.trim().length > 0)
    : [];
  const normalisedFollowUps = Array.isArray(parsed.follow_ups)
    ? parsed.follow_ups.filter((q): q is string => typeof q === "string" && q.trim().length > 0)
    : [];

  return {
    id: `assistant-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    role: "assistant",
    content: parsed.answer,
    summary: typeof parsed.summary === "string" && parsed.summary.trim().length > 0
      ? parsed.summary
      : undefined,
    // If the backend produced structured chunks we keep them; otherwise fall
    // back to a single-chunk wrapping of the flat answer so downstream code
    // can always treat chunks as the source of truth.
    chunks: normalisedChunks.length > 0 ? normalisedChunks : [parsed.answer],
    followUps: normalisedFollowUps.length > 0 ? normalisedFollowUps : undefined,
    timestamp: new Date().toISOString()
  };
}

export class ApiChatService implements IChatService {
  private readonly mockService = new MockChatService();

  public async sendMessage(request: ChatRequest): Promise<ChatMessage> {
    if (MOCK_ONLY_MODE) {
      return this.mockService.sendMessage(request);
    }

    let response: Response;

    try {
      response = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          report: request.report,
          question: request.question,
          videoAnalysis: request.videoAnalysis ?? null,
          history: request.history,
          objective_effective: request.report.meta.objective_effective
        })
      });
    } catch {
      return this.mockService.sendMessage(request);
    }

    if (!response.ok) {
      return this.mockService.sendMessage(request);
    }

    const parsed = (await response.json()) as ChatApiResponse;
    return createAssistantMessage(parsed);
  }
}
