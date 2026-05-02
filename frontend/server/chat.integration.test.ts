import assert from "node:assert/strict";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import http from "node:http";
import type { AddressInfo } from "node:net";
import test, { after, before } from "node:test";

interface MockState {
  ragStatus: number;
  ragRequests: Array<Record<string, unknown>>;
  hashtagRequests: Array<Record<string, unknown>>;
}

interface RunningServer {
  baseUrl: string;
  close: () => Promise<void>;
}

let mockState: MockState;
let mockRecommender: RunningServer;
let appServer: {
  baseUrl: string;
  child: ChildProcessWithoutNullStreams;
  output: () => string;
};

function readJsonBody(request: http.IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw) as Record<string, unknown>);
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function writeJson(
  response: http.ServerResponse,
  status: number,
  payload: Record<string, unknown>
): void {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(payload));
}

async function startMockRecommender(state: MockState): Promise<RunningServer> {
  const server = http.createServer(async (request, response) => {
    try {
      if (request.method === "POST" && request.url === "/v1/chat/rag") {
        const body = await readJsonBody(request);
        state.ragRequests.push(body);
        if (state.ragStatus !== 200) {
          writeJson(response, state.ragStatus, { error: "mock_rag_failed" });
          return;
        }
        writeJson(response, 200, {
          retrieved_videos: [
            {
              video_id: "rag-1",
              caption: "Mock RAG app launch example with crisp onboarding",
              hashtags: ["#appgrowth", "#launch"],
              keywords: ["onboarding", "growth"],
              author_id: "creator-1",
              content_type: "tutorial",
              language: "en",
              fused_score: 0.87,
              branch_scores: { dense_text: 0.8, lexical: 0.6 }
            }
          ],
          retrieval_meta: {
            objective: "engagement",
            weights: { dense_text: 0.6, lexical: 0.4 },
            branch_coverage: { dense_text: 1, lexical: 1 }
          }
        });
        return;
      }

      if (request.method === "POST" && request.url === "/v1/hashtags/suggest") {
        const body = await readJsonBody(request);
        state.hashtagRequests.push(body);
        writeJson(response, 200, {
          hashtags: ["#appgrowth", "#saaslaunch", "#productdemo"],
          latency_ms: 3.2,
          corpus_size: 12
        });
        return;
      }

      writeJson(response, 404, { error: "not_found" });
    } catch (error) {
      writeJson(response, 500, {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      })
  };
}

async function getOpenPort(): Promise<number> {
  const server = http.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address() as AddressInfo;
  const port = address.port;
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
  return port;
}

async function waitForReady(baseUrl: string, getOutput: () => string): Promise<void> {
  const startedAt = Date.now();
  let lastError = "";
  while (Date.now() - startedAt < 12_000) {
    try {
      const response = await fetch(`${baseUrl}/recommender-gateway-metrics`);
      if (response.ok) {
        return;
      }
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Server did not become ready: ${lastError}\n${getOutput()}`);
}

async function startAppServer(recommenderBaseUrl: string): Promise<typeof appServer> {
  const port = await getOpenPort();
  const output: string[] = [];
  const child = spawn(process.execPath, ["--import", "tsx", "server/index.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      PORT: String(port),
      DEEPSEEK_API_KEY: "",
      RECOMMENDER_BASE_URL: recommenderBaseUrl,
      RECOMMENDER_ENABLED: "false",
      RECOMMENDER_FEEDBACK_ENABLED: "false"
    },
    stdio: ["ignore", "pipe", "pipe"]
  });
  child.stdout.on("data", (chunk: Buffer) => output.push(chunk.toString("utf8")));
  child.stderr.on("data", (chunk: Buffer) => output.push(chunk.toString("utf8")));
  child.on("exit", (code, signal) => {
    if (code !== null && code !== 0 && signal === null) {
      output.push(`\n[server exited code=${code}]\n`);
    }
  });

  const server = {
    baseUrl: `http://127.0.0.1:${port}`,
    child,
    output: () => output.join("")
  };
  await waitForReady(server.baseUrl, server.output);
  return server;
}

async function stopAppServer(): Promise<void> {
  if (!appServer || appServer.child.killed) {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeout = setTimeout(() => {
      appServer.child.kill("SIGKILL");
      resolve();
    }, 2_000);
    appServer.child.once("exit", () => {
      clearTimeout(timeout);
      resolve();
    });
    appServer.child.kill("SIGTERM");
  });
}

function buildReport(): Record<string, unknown> {
  return {
    meta: {
      request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
      objective: "engagement",
      objective_effective: "engagement",
      generated_at: "2026-04-04T12:00:00.000Z",
      recommender_source: "python-service",
      fallback_mode: false,
      fallback_reason: null,
      evidence_label: "Strong evidence",
      confidence_label: "High confidence"
    },
    header: {
      title: "Draft report",
      subtitle: "Structured report",
      badges: { candidates_k: 8, model: "baseline", mode: "Guided demo" },
      disclaimer: "Deterministic report."
    },
    executive_summary: {
      metrics: [{ id: "hook_strength", label: "Hook strength", value: "72/100" }],
      extracted_keywords: ["app marketing", "onboarding"],
      meaning_points: ["Top comparables are tutorial-led."],
      summary_text: "Tutorial-style comparables lead this neighborhood."
    },
    comparables: [
      {
        id: "comp-1",
        candidate_id: "cand-1",
        caption: "App marketing tutorial",
        author: "@creator",
        video_url: "https://www.tiktok.com/@creator/video/1",
        thumbnail_url: "https://example.com/thumb.jpg",
        hashtags: ["#appmarketing", "#growthtips"],
        similarity: 0.82,
        support_level: "full",
        confidence_label: "High confidence",
        metrics: {
          views: 1000,
          likes: 100,
          comments_count: 10,
          shares: 5,
          engagement_rate: "11.50%"
        },
        matched_keywords: ["marketing", "onboarding"],
        observations: ["Clear CTA"],
        why_this_was_chosen: "Strong intent alignment.",
        ranking_reasons: ["strong_intent_alignment"],
        score_components: {
          semantic_relevance: 0.7,
          intent_alignment: 0.9,
          performance_quality: 0.6,
          reference_usefulness: 0.8,
          support_confidence: 0.85
        },
        retrieval_branches: ["semantic", "structured_compatibility"]
      }
    ],
    direct_comparison: {
      rows: [
        {
          id: "engagement-rate",
          label: "Engagement rate",
          your_value_label: "7.00%",
          comparable_value_label: "11.50%",
          your_value_pct: 58,
          comparable_value_pct: 96
        }
      ],
      note: "Estimated from dataset."
    },
    relevant_comments: {
      items: [
        {
          id: "comment-1",
          text: "This is useful.",
          topic: "cta",
          polarity: "Positive",
          relevance_note: "Positive response to clear call to action."
        }
      ],
      disclaimer: "Comments are illustrative."
    },
    recommendations: {
      items: [
        {
          id: "rec-1",
          title: "Clarify the CTA",
          priority: "High",
          effort: "Low",
          evidence: "Top comparables use more explicit CTA language.",
          rationale: "The strongest items make the next action obvious.",
          confidence_label: "High confidence",
          effect_area: "cta",
          caveats: [],
          evidence_refs: ["exp-1"]
        }
      ]
    },
    reasoning: {
      evidence_pack: {},
      explanation_units: [],
      recommendation_units: [],
      reasoning_metadata: {}
    }
  };
}

async function postChat(payload: Record<string, unknown>): Promise<Response> {
  return fetch(`${appServer.baseUrl}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

before(async () => {
  mockState = { ragStatus: 200, ragRequests: [], hashtagRequests: [] };
  mockRecommender = await startMockRecommender(mockState);
  appServer = await startAppServer(mockRecommender.baseUrl);
});

after(async () => {
  await stopAppServer();
  await mockRecommender?.close();
});

test("POST /chat grounds local fallback answers with RAG videos and hashtag suggestions", async () => {
  mockState.ragStatus = 200;
  mockState.ragRequests = [];
  mockState.hashtagRequests = [];

  const response = await postChat({
    question: "Can I get similar examples and hashtags for this app launch?",
    report: buildReport(),
    history: [
      {
        role: "assistant",
        content: "Report loaded. Ask me about comparables, recommendations, metrics, or editing strategy."
      }
    ],
    videoAnalysis: {
      transcript: "The clip shows app onboarding and a quick product walkthrough.",
      duration_seconds: 34
    },
    objective_effective: "engagement"
  });
  const body = (await response.json()) as { answer?: string };

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-chat-source"), "baseline-local-with-tools");
  assert.match(body.answer ?? "", /Suggested hashtags: #appgrowth, #saaslaunch, #productdemo/);
  assert.match(body.answer ?? "", /Relevant examples from similar videos/);
  assert.match(body.answer ?? "", /Mock RAG app launch example/);

  assert.equal(mockState.ragRequests.length, 1);
  assert.equal(mockState.hashtagRequests.length, 1);
  const ragRequest = mockState.ragRequests[0];
  assert.deepEqual(ragRequest.report_hashtags, ["#appmarketing", "#growthtips"]);
  assert.deepEqual(ragRequest.report_keywords, ["app marketing", "onboarding", "marketing"]);
  assert.equal(ragRequest.objective, "engagement");
  assert.match(String(ragRequest.transcript_hint ?? ""), /app onboarding/);
});

test("POST /chat still returns local guidance when RAG retrieval fails", async () => {
  mockState.ragStatus = 503;
  mockState.ragRequests = [];
  mockState.hashtagRequests = [];

  const response = await postChat({
    question: "Show me similar examples I can learn from",
    report: buildReport(),
    history: [],
    objective_effective: "engagement"
  });
  const body = (await response.json()) as { answer?: string };

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-chat-source"), "baseline-local-with-tools");
  assert.match(body.answer ?? "", /Quick diagnosis:/);
  assert.doesNotMatch(body.answer ?? "", /Relevant examples from similar videos/);
  assert.equal(mockState.ragRequests.length, 1);
  assert.equal(mockState.hashtagRequests.length, 0);
});

test("POST /chat rejects malformed requests before tool calls", async () => {
  mockState.ragStatus = 200;
  mockState.ragRequests = [];
  mockState.hashtagRequests = [];

  const response = await postChat({ history: "not-an-array" });
  const body = (await response.json()) as { error?: string };

  assert.equal(response.status, 400);
  assert.equal(body.error, "A question is required.");
  assert.equal(mockState.ragRequests.length, 0);
  assert.equal(mockState.hashtagRequests.length, 0);
});
