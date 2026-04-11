# Tik-Tok-Recommendation-System

An explainable, multimodal recommendation system for TikTok that predicts content performance before publication. Combines data scraping, NLP, computer vision, audio analysis, and retrieval-based recommendations to empower creators with actionable, transparent insights.

# Predictive Social Core (Skeleton)

This repo is a lightweight scaffold for TikTok-style recommendation experiments. No scraping is included; data is assumed to come from an external source.

## Layout

- `data/mock/` � small mocked JSONL dataset for experiments.
- `src/common/` � shared schemas, validation utilities, constants.
- `src/data/` � stubs for data generation/ingestion helpers (no scrapers).
- `src/retrieval/` � retrieval skeleton (index + search abstractions).
- `src/baseline/` � simple baseline stats and reporting.
- `src/research/` � notes and TODOs for comparing retrieval approaches.
- `scripts/` � CLI entrypoints for validation, baselines, and retrieval.
- `tests/` � smoke tests to keep the scaffold wired up.
- `.github/workflows/ci.yml` � CI skeleton for lint + tests on PRs.
- `Makefile` � convenience targets.


