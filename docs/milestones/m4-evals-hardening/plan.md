# M4 — Evaluation & hardening: plan

**Status:** planned  
**Target release:** v0.4.0

## Goal
Make quality measurable and protect it in CI, so future changes to chunking, retrieval or prompts are judged by numbers, not impressions.

## Scope
1. Evaluation corpus and labeled question set, including unanswerable questions.
2. Harness measuring:
   - retrieval: hit@k and MRR@k against evidence spans
   - answers: keyword accuracy, grounded citation rate, abstention accuracy, false abstention rate
3. Offline mode (no API key) for retrieval metrics; optional real embeddings and LLM.
4. CI pipeline: lint, strict typing, unit + Postgres integration tests on two Python versions, retrieval quality gate, Docker build.
5. Documentation: README, architecture guide, changelog.

## Deliverables
- `evals/run_evals.py`, `evals/data/`
- `.github/workflows/ci.yml`, pull request template
- `README.md`, `docs/architecture.md`, `CHANGELOG.md`

## Acceptance criteria
- [ ] One command produces a comparison table across retrieval modes
- [ ] Results are saved as JSON for comparison between runs
- [ ] CI fails if retrieval hit rate drops below a threshold
- [ ] Harness itself is covered by tests
- [ ] README explains setup, API, configuration, evaluation, design decisions and limitations

## Out of scope
- LLM-as-judge faithfulness scoring
- Load testing

## Risks
| Risk | Mitigation |
|---|---|
| Small eval set overstates quality | Report it honestly as a baseline; grow the set with real user questions |
| Offline embedder results don't reflect production | Label them clearly; support `--embedder openai` for real numbers |
