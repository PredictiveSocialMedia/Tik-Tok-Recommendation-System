import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigation } from "../../app/NavigationContext";
import { ComparableThumbnailImage } from "../report/components/ComparableThumbnailImage";
import {
  createLabelingSession,
  listLabelingSessions,
  listLabelingSources,
  loadLabelingSession,
  updateLabelingCandidateReview
} from "../../services/api/labelingApi";
import type {
  LabelingReviewLabel,
  LabelingSession,
  LabelingSessionCandidate,
  LabelingSessionCase,
  LabelingSessionListItem,
  LabelingSourceSummary
} from "./types";

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => asString(item)).filter(Boolean);
}

function formatDateTime(value: string): string {
  if (!value) {
    return "Unknown";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(parsed);
}

function formatPct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function caseReviewedCount(itemCase: LabelingSessionCase): number {
  return itemCase.candidates.filter((candidate) => candidate.review.label !== null).length;
}

function labelButtonText(label: LabelingReviewLabel): string {
  if (label === "saved") {
    return "Saved";
  }
  if (label === "relevant") {
    return "Relevant";
  }
  return "Not relevant";
}

function candidateVideoUrl(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.video_url) ||
    asString(candidate.candidate_payload.video_url)
  );
}

function candidateCaption(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.caption) ||
    asString(candidate.candidate_payload.caption) ||
    asString(candidate.candidate_payload.text)
  );
}

function candidateAuthor(candidate: LabelingSessionCandidate): string {
  return (
    asString(candidate.display.author_display_name) ||
    asString(candidate.display.author_username) ||
    asString(candidate.candidate_payload.author_id) ||
    "Unknown author"
  );
}

function candidateHashtags(candidate: LabelingSessionCandidate): string[] {
  const fromDisplay = asStringArray(candidate.display.hashtags);
  if (fromDisplay.length > 0) {
    return fromDisplay;
  }
  return asStringArray(candidate.candidate_payload.hashtags);
}

function queryCaption(itemCase: LabelingSessionCase): string {
  return (
    asString(itemCase.query.display.caption) ||
    asString(itemCase.query.query_payload.description) ||
    asString(itemCase.query.query_payload.text)
  );
}

function queryAuthor(itemCase: LabelingSessionCase): string {
  return (
    asString(itemCase.query.display.author_display_name) ||
    asString(itemCase.query.display.author_username) ||
    "Unknown author"
  );
}

function queryHashtags(itemCase: LabelingSessionCase): string[] {
  const displayTags = asStringArray(itemCase.query.display.hashtags);
  if (displayTags.length > 0) {
    return displayTags;
  }
  return asStringArray(itemCase.query.query_payload.hashtags);
}

function queryCommentsPreview(itemCase: LabelingSessionCase): string[] {
  return asStringArray(itemCase.query.display.comments_preview);
}

function preferredSourceId(sources: LabelingSourceSummary[]): string {
  return (
    sources.find((item) => item.file_name.includes("training"))?.source_id ||
    sources.find((item) => item.file_name.includes("seed"))?.source_id ||
    sources[0]?.source_id ||
    ""
  );
}

function preferredSessionId(
  sessions: LabelingSessionListItem[],
  sourceId: string
): string {
  if (!sourceId) {
    return sessions[0]?.session_id ?? "";
  }
  return (
    sessions.find((item) => item.source.source_id === sourceId)?.session_id ||
    sessions[0]?.session_id ||
    ""
  );
}

// ─── Local labeling helpers ───────────────────────────────────────────────────

const LOCAL_SESSION_KEY = "ttrec_local_session";

function extractHashtags(text: string): string[] {
  return (text.match(/#\w+/g) ?? []).map((t) => t.toLowerCase());
}

function computeSessionSummary(cases: LabelingSessionCase[]) {
  const candidates = cases.flatMap((c) => c.candidates);
  const reviewed = candidates.filter((c) => c.review.label !== null);
  return {
    case_count: cases.length,
    candidate_count: candidates.length,
    reviewed_count: reviewed.length,
    saved_count: reviewed.filter((c) => c.review.label === "saved").length,
    relevant_count: reviewed.filter((c) => c.review.label === "relevant").length,
    not_relevant_count: reviewed.filter((c) => c.review.label === "not_relevant").length,
    completion_ratio: candidates.length > 0 ? reviewed.length / candidates.length : 0
  };
}

function parseJsonlToSession(fileName: string, content: string): LabelingSession {
  const lines = content.trim().split("\n").filter(Boolean);
  const videos = lines
    .map((line) => {
      try {
        return JSON.parse(line) as Record<string, unknown>;
      } catch {
        return null;
      }
    })
    .filter((v): v is Record<string, unknown> => v !== null);

  const niche = fileName.replace(/\.jsonl$/i, "").replace(/_/g, " ");
  const sessionId = `local_${Date.now()}`;
  const caseId = `case_${sessionId}`;
  const now = new Date().toISOString();

  const candidates: LabelingSessionCandidate[] = videos.map((video, index) => {
    const meta = (video.metadata as Record<string, unknown>) ?? {};
    const author = (meta.author as Record<string, unknown>) ?? {};
    const caption = asString(video.caption);
    return {
      candidate_id: asString(meta.id) || `cand_${index}`,
      display: {
        video_url: video.url,
        caption,
        author_display_name: author.nickname,
        author_username: author.uniqueId,
        hashtags: extractHashtags(caption)
      },
      candidate_payload: video,
      baseline_rank: index + 1,
      baseline_score: null,
      support_level: null,
      ranking_reasons: [],
      review: { label: null, note: "", updated_at: null }
    };
  });

  const source: LabelingSourceSummary = {
    source_id: `local_${fileName}`,
    file_name: fileName,
    source_path: `local:${fileName}`,
    generated_at: now,
    case_count: 1,
    objectives: [niche]
  };

  const cases: LabelingSessionCase[] = [
    {
      case_id: caseId,
      objective: niche,
      query: {
        query_id: `query_${sessionId}`,
        display: { caption: niche, author_display_name: "Local file" },
        query_payload: { description: niche, file: fileName }
      },
      retrieve_k: candidates.length,
      label_pool_size: candidates.length,
      source_candidate_pool_size: candidates.length,
      notes: "",
      candidates
    }
  ];

  return {
    version: "1.0",
    session_id: sessionId,
    session_name: niche,
    created_at: now,
    updated_at: now,
    storage_path: `local:${fileName}`,
    source,
    rubric: {
      version: "1.0",
      labels: ["saved", "relevant", "not_relevant"],
      instructions: [
        "Saved: strong reference example, keep it.",
        "Relevant: on-topic but not a top pick.",
        "Not relevant: off-topic or low quality."
      ]
    },
    cases,
    summary: computeSessionSummary(cases)
  };
}

function applyLocalLabel(
  session: LabelingSession,
  caseId: string,
  candidateId: string,
  label: LabelingReviewLabel | null
): LabelingSession {
  const now = new Date().toISOString();
  const updatedCases = session.cases.map((c) => {
    if (c.case_id !== caseId) return c;
    return {
      ...c,
      candidates: c.candidates.map((cand) =>
        cand.candidate_id !== candidateId
          ? cand
          : { ...cand, review: { ...cand.review, label, updated_at: now } }
      )
    };
  });
  const updated: LabelingSession = {
    ...session,
    cases: updatedCases,
    updated_at: now,
    summary: computeSessionSummary(updatedCases)
  };
  try {
    localStorage.setItem(LOCAL_SESSION_KEY, JSON.stringify(updated));
  } catch {
    // ignore quota errors
  }
  return updated;
}

function loadLocalSession(): LabelingSession | null {
  try {
    const raw = localStorage.getItem(LOCAL_SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as LabelingSession;
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────

export function LabelingPage(): JSX.Element {
  const { navigate } = useNavigation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [sources, setSources] = useState<LabelingSourceSummary[]>([]);
  const [sessions, setSessions] = useState<LabelingSessionListItem[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");
  const [session, setSession] = useState<LabelingSession | null>(null);
  const [activeCaseId, setActiveCaseId] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isCreatingSession, setIsCreatingSession] = useState<boolean>(false);
  const [savingCandidateId, setSavingCandidateId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [backendAvailable, setBackendAvailable] = useState<boolean>(false);

  const isLocalSession = Boolean(session?.storage_path.startsWith("local:"));

  useEffect(() => {
    let isMounted = true;

    async function initialize(): Promise<void> {
      setIsLoading(true);
      setErrorMessage("");
      try {
        const [nextSources, nextSessions] = await Promise.all([
          listLabelingSources(),
          listLabelingSessions()
        ]);
        if (!isMounted) {
          return;
        }

        const hasBackend = nextSources.length > 0 || nextSessions.length > 0;
        setBackendAvailable(hasBackend);
        setSources(nextSources);
        setSessions(nextSessions);

        if (hasBackend) {
          const preferredSource = preferredSourceId(nextSources);
          setSelectedSourceId((previous) => previous || preferredSource);

          let nextSession: LabelingSession | null = null;
          const preferredSession = preferredSessionId(nextSessions, preferredSource);
          if (preferredSession) {
            nextSession = await loadLabelingSession(preferredSession);
          } else if (preferredSource) {
            nextSession = await createLabelingSession({ source_id: preferredSource });
            const refreshedSessions = await listLabelingSessions();
            if (!isMounted) return;
            setSessions(refreshedSessions);
          }
          if (!isMounted) return;
          setSession(nextSession);
          setActiveCaseId(nextSession?.cases[0]?.case_id ?? "");
        } else {
          // No backend — try to restore a previously saved local session
          const local = loadLocalSession();
          if (local) {
            setSession(local);
            setActiveCaseId(local.cases[0]?.case_id ?? "");
          }
        }
      } catch (error) {
        console.error("labeling_page_init_failed", error);
        if (isMounted) {
          const local = loadLocalSession();
          if (local) {
            setSession(local);
            setActiveCaseId(local.cases[0]?.case_id ?? "");
          }
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    void initialize();

    return () => {
      isMounted = false;
    };
  }, []);

  const activeCase = useMemo(() => {
    if (!session) {
      return null;
    }
    return (
      session.cases.find((item) => item.case_id === activeCaseId) ??
      session.cases[0] ??
      null
    );
  }, [activeCaseId, session]);

  useEffect(() => {
    if (!session) {
      return;
    }
    if (activeCaseId && session.cases.some((item) => item.case_id === activeCaseId)) {
      return;
    }
    setActiveCaseId(session.cases[0]?.case_id ?? "");
  }, [activeCaseId, session]);

  async function handleCreateSession(): Promise<void> {
    if (!selectedSourceId) {
      return;
    }
    setIsCreatingSession(true);
    setErrorMessage("");
    try {
      const nextSession = await createLabelingSession({ source_id: selectedSourceId });
      const refreshedSessions = await listLabelingSessions();
      setSessions(refreshedSessions);
      setSession(nextSession);
      setActiveCaseId(nextSession.cases[0]?.case_id ?? "");
    } catch (error) {
      console.error("labeling_session_create_ui_failed", error);
      setErrorMessage("A new labeling session could not be created.");
    } finally {
      setIsCreatingSession(false);
    }
  }

  async function handleLocalFileLoad(event: React.ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const content = await file.text();
      const parsed = parseJsonlToSession(file.name, content);
      try {
        localStorage.setItem(LOCAL_SESSION_KEY, JSON.stringify(parsed));
      } catch {
        // ignore quota
      }
      setSession(parsed);
      setActiveCaseId(parsed.cases[0]?.case_id ?? "");
      setErrorMessage("");
    } catch (error) {
      console.error("local_file_parse_failed", error);
      setErrorMessage("Could not parse the file. Make sure it is a valid JSONL file.");
    }
    // reset input so the same file can be re-selected
    event.target.value = "";
  }

  function handleClearLocalSession(): void {
    if (!window.confirm("Clear the local session? All labels will be lost.")) return;
    localStorage.removeItem(LOCAL_SESSION_KEY);
    setSession(null);
    setActiveCaseId("");
    setErrorMessage("");
  }

  async function handleSetCandidateLabel(
    itemCase: LabelingSessionCase,
    candidate: LabelingSessionCandidate,
    label: LabelingReviewLabel | null
  ): Promise<void> {
    if (!session) {
      return;
    }

    if (isLocalSession) {
      const updated = applyLocalLabel(session, itemCase.case_id, candidate.candidate_id, label);
      setSession(updated);
      return;
    }

    setSavingCandidateId(candidate.candidate_id);
    setErrorMessage("");
    try {
      const nextSession = await updateLabelingCandidateReview({
        session_id: session.session_id,
        case_id: itemCase.case_id,
        candidate_id: candidate.candidate_id,
        label,
        note: candidate.review.note
      });
      setSession(nextSession);
      setSessions((previous) =>
        previous.map((item) =>
          item.session_id === nextSession.session_id
            ? {
                session_id: nextSession.session_id,
                session_name: nextSession.session_name,
                created_at: nextSession.created_at,
                updated_at: nextSession.updated_at,
                storage_path: nextSession.storage_path,
                source: nextSession.source,
                summary: nextSession.summary
              }
            : item
        )
      );
    } catch (error) {
      console.error("labeling_candidate_update_ui_failed", error);
      setErrorMessage("The label could not be saved.");
    } finally {
      setSavingCandidateId(null);
    }
  }

  if (isLoading) {
    return (
      <section className="glass-card labeling-shell">
        <div className="labeling-empty-state">
          <div className="spinner" aria-hidden="true" />
          <p>Preparing the labeling workspace…</p>
        </div>
      </section>
    );
  }

  if (!session || !activeCase) {
    return (
      <section className="glass-card labeling-shell">
        <div className="labeling-empty-state">
          {backendAvailable ? (
            <>
              <p>{errorMessage || "No labeling session is available yet."}</p>
              <div className="labeling-toolbar">
                <select
                  className="labeling-select"
                  value={selectedSourceId}
                  onChange={(event) => setSelectedSourceId(event.target.value)}
                >
                  {sources.map((source) => (
                    <option key={source.source_id} value={source.source_id}>
                      {source.file_name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="labeling-primary-button"
                  onClick={() => void handleCreateSession()}
                  disabled={!selectedSourceId || isCreatingSession}
                >
                  {isCreatingSession ? "Creating…" : "Create session"}
                </button>
              </div>
            </>
          ) : (
            <>
              <p style={{ fontWeight: 600, marginBottom: 6 }}>No API server detected</p>
              <p style={{ opacity: 0.7, marginBottom: 20, fontSize: "0.88rem" }}>
                Load a local <strong>.jsonl</strong> scrape file to start labeling without a backend.
              </p>
              {errorMessage ? (
                <p className="labeling-error" style={{ marginBottom: 12 }}>{errorMessage}</p>
              ) : null}
              <button
                type="button"
                className="labeling-primary-button"
                onClick={() => fileInputRef.current?.click()}
              >
                Load JSONL file
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jsonl"
                style={{ display: "none" }}
                onChange={(e) => void handleLocalFileLoad(e)}
              />
            </>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className="glass-card labeling-shell">
      <div className="labeling-header">
        <div>
          <p className="labeling-kicker">
            {isLocalSession ? "Local file · browser only" : "Local training labels"}
          </p>
          <h1>Comparable labeling workspace</h1>
          <p className="labeling-subtitle">
            {isLocalSession
              ? "Labels are saved in this browser. They will persist across refreshes but are not synced anywhere."
              : "Labels are stored locally in a disposable session file and stay separate from the live feedback tables."}
          </p>
        </div>
        <div className="labeling-header-actions">
          {isLocalSession ? (
            <>
              <button
                type="button"
                className="report-ghost-action"
                onClick={() => fileInputRef.current?.click()}
              >
                Load different file
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jsonl"
                style={{ display: "none" }}
                onChange={(e) => void handleLocalFileLoad(e)}
              />
              <button
                type="button"
                className="report-ghost-action"
                onClick={handleClearLocalSession}
              >
                Clear session
              </button>
            </>
          ) : (
            <button
              type="button"
              className="report-ghost-action"
              onClick={() => navigate("app")}
            >
              Back to app
            </button>
          )}
        </div>
      </div>

      <div className="labeling-summary-grid">
        <div className="labeling-summary-card">
          <span>Reviewed</span>
          <strong>
            {session.summary.reviewed_count}/{session.summary.candidate_count}
          </strong>
          <small>{formatPct(session.summary.completion_ratio)} complete</small>
        </div>
        <div className="labeling-summary-card">
          <span>Saved</span>
          <strong>{session.summary.saved_count}</strong>
        </div>
        <div className="labeling-summary-card">
          <span>Relevant</span>
          <strong>{session.summary.relevant_count}</strong>
        </div>
        <div className="labeling-summary-card">
          <span>Not relevant</span>
          <strong>{session.summary.not_relevant_count}</strong>
        </div>
      </div>

      {!isLocalSession && (
        <div className="labeling-toolbar">
          <label className="labeling-field">
            <span>Benchmark source</span>
            <select
              className="labeling-select"
              value={selectedSourceId}
              onChange={(event) => setSelectedSourceId(event.target.value)}
            >
              {sources.map((source) => (
                <option key={source.source_id} value={source.source_id}>
                  {source.file_name} ({source.case_count} cases)
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="labeling-primary-button"
            onClick={() => void handleCreateSession()}
            disabled={!selectedSourceId || isCreatingSession}
          >
            {isCreatingSession ? "Creating new session…" : "Create fresh session"}
          </button>
          <div className="labeling-session-meta">
            <span>Session: {session.session_name}</span>
            <span>Updated: {formatDateTime(session.updated_at)}</span>
            <span>Stored sessions: {sessions.length}</span>
          </div>
        </div>
      )}

      {errorMessage ? <p className="labeling-error">{errorMessage}</p> : null}

      <div className="labeling-layout">
        <aside className="labeling-case-list">
          <div className="report-section-head">
            <h3>Cases</h3>
            <p>{session.cases.length} queued</p>
          </div>
          <div className="labeling-case-items">
            {session.cases.map((itemCase) => {
              const reviewedCount = caseReviewedCount(itemCase);
              const isActive = itemCase.case_id === activeCase.case_id;
              return (
                <button
                  type="button"
                  key={itemCase.case_id}
                  className={`labeling-case-item ${isActive ? "labeling-case-item-active" : ""}`}
                  onClick={() => setActiveCaseId(itemCase.case_id)}
                >
                  <span className="labeling-case-item-top">
                    <strong>{itemCase.objective}</strong>
                    <span>
                      {reviewedCount}/{itemCase.candidates.length}
                    </span>
                  </span>
                  <span className="labeling-case-item-caption">
                    {queryCaption(itemCase) || itemCase.case_id}
                  </span>
                </button>
              );
            })}
          </div>
        </aside>

        <div className="labeling-main">
          <section className="report-section labeling-query-card">
            <div className="report-section-head">
              <h3>Query case</h3>
              <p>
                {activeCase.objective} · {activeCase.candidates.length} candidates
              </p>
            </div>
            <p className="labeling-query-caption">{queryCaption(activeCase)}</p>
            <p className="labeling-query-author">{queryAuthor(activeCase)}</p>
            <div className="labeling-tag-row">
              {queryHashtags(activeCase).map((tag) => (
                <span className="comparable-tag" key={`${activeCase.case_id}-${tag}`}>
                  {tag}
                </span>
              ))}
            </div>
            {queryCommentsPreview(activeCase).length > 0 ? (
              <div className="labeling-query-comments">
                <strong>Comment preview</strong>
                {queryCommentsPreview(activeCase).map((comment) => (
                  <p key={`${activeCase.case_id}-${comment}`}>{comment}</p>
                ))}
              </div>
            ) : null}
          </section>

          <section className="report-section">
            <div className="report-section-head">
              <h3>Candidates</h3>
              <p>
                Click one label per candidate. Use saved for the strongest examples you would keep as references.
              </p>
            </div>
            <div className="labeling-candidate-list" role="list">
              {activeCase.candidates.map((candidate) => {
                const videoUrl = candidateVideoUrl(candidate);
                const currentLabel = candidate.review.label;
                const isSaving = savingCandidateId === candidate.candidate_id;
                return (
                  <article
                    className="comparable-item labeling-candidate-item"
                    key={candidate.candidate_id}
                    role="listitem"
                  >
                    {videoUrl ? (
                      <a
                        href={videoUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="comparable-thumb-link"
                      >
                        <ComparableThumbnailImage
                          className="comparable-thumb-image"
                          thumbnailUrl=""
                          videoUrl={videoUrl}
                          alt="Comparable thumbnail"
                          fallbackClassName="comparable-thumb comparable-thumb-fallback"
                        />
                        <span className="comparable-thumb-overlay">Open</span>
                      </a>
                    ) : (
                      <span className="comparable-thumb comparable-thumb-fallback" aria-hidden="true" />
                    )}

                    <div className="comparable-main-shell">
                      <div className="labeling-candidate-meta">
                        <span>Rank: {candidate.baseline_rank ?? "n/a"}</span>
                        {candidate.baseline_score !== null && (
                          <span>Score: {candidate.baseline_score.toFixed(3)}</span>
                        )}
                        {candidate.support_level && (
                          <span>Support: {candidate.support_level}</span>
                        )}
                      </div>
                      <p className="comparable-caption">{candidateCaption(candidate)}</p>
                      <p className="comparable-author">{candidateAuthor(candidate)}</p>
                      <div className="comparable-hashtags">
                        {candidateHashtags(candidate).map((tag) => (
                          <span className="comparable-tag" key={`${candidate.candidate_id}-${tag}`}>
                            {tag}
                          </span>
                        ))}
                      </div>
                      {candidate.ranking_reasons.length > 0 && (
                        <p className="labeling-candidate-reasons">
                          Reasons: {candidate.ranking_reasons.join(", ")}
                        </p>
                      )}
                      <div className="labeling-action-row">
                        {(["saved", "relevant", "not_relevant"] as LabelingReviewLabel[]).map(
                          (label) => (
                            <button
                              key={label}
                              type="button"
                              className={`labeling-chip-button ${currentLabel === label ? "labeling-chip-button-active" : ""}`}
                              onClick={() => void handleSetCandidateLabel(activeCase, candidate, label)}
                              disabled={isSaving}
                            >
                              {labelButtonText(label)}
                            </button>
                          )
                        )}
                        <button
                          type="button"
                          className="report-ghost-action"
                          onClick={() => void handleSetCandidateLabel(activeCase, candidate, null)}
                          disabled={isSaving}
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}
