import type { VercelRequest, VercelResponse } from "@vercel/node";

// ─── Types ────────────────────────────────────────────────────────────────────

interface SignalHints {
  duration_seconds?: number;
  transcript_text?: string;
  ocr_text?: string;
  estimated_scene_cuts?: number;
  audio_tempo_bpm?: number;
  audio_energy?: number;
}

interface ReportRequest {
  description?: string;
  hashtags?: string[];
  mentions?: string[];
  objective?: string;
  audience?: string;
  content_type?: string;
  primary_cta?: string;
  locale?: string;
  signal_hints?: SignalHints;
  candidate_ids?: string[];
}

interface Candidate {
  id: string;
  description?: string;
  hashtags?: string[];
  views?: number;
  likes?: number;
  comments?: number;
  shares?: number;
  author?: string;
  content_type?: string;
  locale?: string;
  [key: string]: unknown;
}

interface ScoredCandidate extends Candidate {
  _score: number;
  _hashtag_overlap: number;
  _semantic_sim: number;
  _popularity_score: number;
  _topical_match: number;
}

// ─── Hashtag normalisation ─────────────────────────────────────────────────

/**
 * Normalise a list of raw hashtag strings:
 *  - strip leading "#"
 *  - lowercase
 *  - trim whitespace
 *  - deduplicate
 */
function normalizeHashtags(raw: string[] | undefined): string[] {
  if (!raw || raw.length === 0) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const tag of raw) {
    const cleaned = tag.replace(/^#+/, "").trim().toLowerCase();
    if (cleaned && !seen.has(cleaned)) {
      seen.add(cleaned);
      out.push(cleaned);
    }
  }
  return out;
}

/**
 * Jaccard-style overlap ratio between two normalised hashtag arrays.
 * Returns a value in [0, 1].
 */
function hashtagOverlapRatio(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const tag of setA) {
    if (setB.has(tag)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// ─── Lightweight text similarity ──────────────────────────────────────────

/**
 * Bag-of-words cosine similarity between two strings.
 * Used as a proxy for semantic similarity when no embedding is available.
 */
function bowCosineSimilarity(a: string, b: string): number {
  const tokenize = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

  const tokensA = tokenize(a);
  const tokensB = tokenize(b);
  if (tokensA.length === 0 || tokensB.length === 0) return 0;

  const freq = (tokens: string[]) => {
    const map: Record<string, number> = {};
    for (const t of tokens) map[t] = (map[t] ?? 0) + 1;
    return map;
  };

  const fa = freq(tokensA);
  const fb = freq(tokensB);
  const vocab = new Set([...Object.keys(fa), ...Object.keys(fb)]);

  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (const word of vocab) {
    const va = fa[word] ?? 0;
    const vb = fb[word] ?? 0;
    dot += va * vb;
    magA += va * va;
    magB += vb * vb;
  }
  return magA === 0 || magB === 0 ? 0 : dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// ─── Popularity score ─────────────────────────────────────────────────────

/**
 * Returns a [0, 1] popularity signal for a candidate.
 * Uses log-scale so viral outliers don't dominate.
 */
function popularityScore(candidate: Candidate): number {
  const views = candidate.views ?? 0;
  const likes = candidate.likes ?? 0;
  const comments = candidate.comments ?? 0;
  const shares = candidate.shares ?? 0;

  // Weighted engagement sum
  const engagement = likes * 1.0 + comments * 1.5 + shares * 2.0;
  const raw = Math.log1p(views) * 0.5 + Math.log1p(engagement) * 0.5;

  // Normalise against a reasonable ceiling (10M views, high engagement)
  const ceiling = Math.log1p(10_000_000) * 0.5 + Math.log1p(500_000) * 0.5;
  return Math.min(raw / ceiling, 1.0);
}

// ─── Core scoring ─────────────────────────────────────────────────────────

/**
 * Score a single candidate against the request.
 *
 * Weight scheme:
 *
 *   When topical match is strong (hashtagOverlap ≥ 0.3 OR semanticSim ≥ 0.4):
 *     hashtag  40%  semantic  35%  popularity  25%
 *
 *   When topical match is weak (both signals below threshold):
 *     hashtag  55%  semantic  35%  popularity  10%
 *     (popularity is penalised to avoid irrelevant viral content floating up)
 */
function scoreCandidate(
  candidate: Candidate,
  queryHashtags: string[],
  queryText: string
): ScoredCandidate {
  const candHashtags = normalizeHashtags(candidate.hashtags);
  const hashtagOverlap = hashtagOverlapRatio(queryHashtags, candHashtags);

  // Combine description + hashtag text for semantic comparison
  const candText = [
    candidate.description ?? "",
    (candidate.hashtags ?? []).join(" "),
  ].join(" ");
  const semanticSim = bowCosineSimilarity(queryText, candText);

  const popScore = popularityScore(candidate);

  // Determine topical match strength
  const strongTopicalMatch = hashtagOverlap >= 0.3 || semanticSim >= 0.4;

  let finalScore: number;
  if (strongTopicalMatch) {
    finalScore = hashtagOverlap * 0.40 + semanticSim * 0.35 + popScore * 0.25;
  } else {
    // Weak topical match: boost hashtag weight, suppress popularity
    finalScore = hashtagOverlap * 0.55 + semanticSim * 0.35 + popScore * 0.10;
  }

  return {
    ...candidate,
    _score: finalScore,
    _hashtag_overlap: hashtagOverlap,
    _semantic_sim: semanticSim,
    _popularity_score: popScore,
    _topical_match: strongTopicalMatch ? 1 : 0,
  };
}

// ─── Minimum relevance gate ────────────────────────────────────────────────

/**
 * Only keep candidates that clear a minimum bar.
 * This prevents "return everything" behaviour when no good match exists.
 *
 * Gate: at least one of
 *  - hashtag overlap > 0
 *  - semantic similarity ≥ 0.05
 */
function isRelevant(scored: ScoredCandidate): boolean {
  return scored._hashtag_overlap > 0 || scored._semantic_sim >= 0.05;
}

// ─── Confidence from match quality ────────────────────────────────────────

function deriveConfidence(
  scored: ScoredCandidate[],
  totalCandidates: number
): { level: "high" | "medium" | "low"; reason: string } {
  if (scored.length === 0) {
    return { level: "low", reason: "No topically relevant candidates found" };
  }
  const topScore = scored[0]._score;
  const relevantRatio = scored.length / Math.max(totalCandidates, 1);

  if (topScore >= 0.5 && scored.length >= 3) {
    return { level: "high", reason: "Strong topical match with multiple comparables" };
  }
  if (topScore >= 0.25 || scored.length >= 2) {
    return { level: "medium", reason: "Moderate topical match" };
  }
  return {
    level: "low",
    reason:
      scored.length < 3
        ? "Limited topical matches found in corpus"
        : `Low match quality (top score: ${topScore.toFixed(2)}, relevant ratio: ${(relevantRatio * 100).toFixed(0)}%)`,
  };
}

// ─── Demo / fallback corpus ────────────────────────────────────────────────

// In production this comes from the recommender or artifact bundle.
// In fallback/static mode it is loaded from demodata.jsonl.
async function loadFallbackCorpus(): Promise<Candidate[]> {
  try {
    // Dynamic import so the bundle doesn't break in environments without FS
    const { readFileSync } = await import("fs");
    const { resolve } = await import("path");

    const bundlePath = resolve(
      process.cwd(),
      "../artifacts/contracts/latest_supabase_bundle.json"
    );
    const raw = readFileSync(bundlePath, "utf-8");
    const bundle = JSON.parse(raw);
    return (bundle.candidates ?? bundle.items ?? []) as Candidate[];
  } catch {
    // Last resort: return empty — caller will set fallback_mode
    return [];
  }
}

// ─── Handler ──────────────────────────────────────────────────────────────

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const body: ReportRequest = req.body ?? {};

  // ── Build query signals ──────────────────────────────────────────────────
  const queryHashtags = normalizeHashtags(body.hashtags);
  const queryText = [
    body.description ?? "",
    queryHashtags.join(" "),
    body.objective ?? "",
    body.content_type ?? "",
  ]
    .join(" ")
    .trim();

  // ── Load corpus ──────────────────────────────────────────────────────────
  let corpus: Candidate[] = [];
  let recommenderSource: "recommender" | "bundle" | "demo" | "empty" = "empty";
  let fallbackMode = false;
  let fallbackReason: string | null = null;

  try {
    corpus = await loadFallbackCorpus();
    if (corpus.length > 0) {
      recommenderSource = "bundle";
    }
  } catch {
    fallbackMode = true;
    fallbackReason = "Corpus bundle unavailable";
  }

  if (corpus.length === 0) {
    fallbackMode = true;
    fallbackReason = fallbackReason ?? "Empty corpus";
    recommenderSource = "empty";
  }

  // Scope to candidate_ids if provided
  if (body.candidate_ids && body.candidate_ids.length > 0) {
    const allowed = new Set(body.candidate_ids);
    corpus = corpus.filter((c) => allowed.has(c.id));
  }

  // ── Score & filter ───────────────────────────────────────────────────────
  const scored: ScoredCandidate[] = corpus
    .map((c) => scoreCandidate(c, queryHashtags, queryText))
    .filter(isRelevant) // ← key: don't return irrelevant candidates
    .sort((a, b) => b._score - a._score);

  // Cap at top 10 comparables
  const comparables = scored.slice(0, 10);

  // ── Confidence ───────────────────────────────────────────────────────────
  const confidence = deriveConfidence(comparables, corpus.length);

  if (comparables.length < 3) {
    fallbackMode = true;
    fallbackReason =
      fallbackReason ?? `Only ${comparables.length} relevant candidate(s) found`;
  }

  // ── Build response ───────────────────────────────────────────────────────
  const report = {
    meta: {
      request_id: `rpt_${Date.now()}`,
      recommender_source: recommenderSource,
      fallback_mode: fallbackMode,
      fallback_reason: fallbackReason,
      confidence: confidence.level,
      confidence_reason: confidence.reason,
      total_corpus_size: corpus.length,
      relevant_candidates_found: comparables.length,
      query_hashtags: queryHashtags,
    },
    comparables: comparables.map(({ _score, _hashtag_overlap, _semantic_sim, _popularity_score, _topical_match, ...rest }) => ({
      ...rest,
      _debug: {
        score: _score,
        hashtag_overlap: _hashtag_overlap,
        semantic_sim: _semantic_sim,
        popularity_score: _popularity_score,
        strong_topical_match: _topical_match === 1,
      },
    })),
    reasoning: {
      explanation_units: [
        queryHashtags.length > 0
          ? `Matched on hashtags: ${queryHashtags.slice(0, 5).join(", ")}`
          : "No hashtags provided — falling back to text similarity only",
        confidence.level !== "high"
          ? `Confidence is ${confidence.level}: ${confidence.reason}`
          : null,
      ].filter(Boolean),
      recommendation_units: comparables.slice(0, 3).map((c) => ({
        id: c.id,
        rationale:
          c._hashtag_overlap > 0
            ? `${(c._hashtag_overlap * 100).toFixed(0)}% hashtag overlap`
            : "Matched by content similarity",
      })),
    },
  };

  return res.status(200).json(report);
}