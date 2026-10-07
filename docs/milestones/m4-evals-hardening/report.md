# M4 — Evaluation & hardening: report

**Status:** delivered  
**Release:** v0.4.0

## Delivered
| Area | Files |
|---|---|
| Evaluation | `evals/run_evals.py`, `evals/data/corpus/` (6 documents), `evals/data/questions.jsonl` (24 questions) |
| CI | `.github/workflows/ci.yml`, `.github/pull_request_template.md` |
| Docs | `README.md`, `docs/architecture.md`, `CHANGELOG.md` |

## Acceptance criteria
- [x] One command (`make evals`) produces a comparison table across retrieval modes
- [x] Results are saved as JSON in `evals/results/`
- [x] CI fails if retrieval hit@3 drops below 0.85
- [x] Harness covered by 3 tests (offline retrieval run, generation scoring with a scripted LLM, CLI quality gate)
- [x] README covers setup, API, configuration, evaluation, design decisions and limitations

## Results
Offline baseline, hashing embedder, k = 3, chunk size 300, overlap 50:

| mode | hit@3 | MRR@3 |
|---|---|---|
| vector | 0.905 | 0.810 |
| keyword | 0.952 | 0.881 |
| hybrid | 0.952 | 0.833 |

At k = 5, keyword and hybrid reach 1.000 hit rate.

**Interpretation.** The offline embedder is lexical, so this run cannot show the benefit of hybrid search over keyword search; both legs see the same signal. The value of hybrid retrieval shows with semantic embeddings, where the vector leg catches paraphrases that share no words with the document. The one consistent miss (q03: "submit an expense claim" vs. "Expenses must be submitted") is a stemming gap in the in-memory BM25.

## Verification
56 tests passing (53 unit/API + 3 Postgres integration), `ruff` clean, `mypy --strict` clean.

## Decisions and deviations from the plan
- The quality gate runs the **offline** retrieval eval so CI needs no secrets and stays deterministic.
- The eval corpus is a fictional company handbook written for this project, so it can be published and extended freely.

## Known limitations
- 24 questions is a small set; numbers are a baseline, not a benchmark.
- Answer accuracy is keyword-based and can miss subtle errors.

## Next steps
- Run `python -m evals.run_evals --embedder openai --generate --reranker llm` and record production-model numbers here.
- Add stemming to the in-memory BM25 and confirm q03 is fixed.
- Add an LLM-as-judge faithfulness metric.
- OCR for scanned PDFs.
- Grow the eval set to 100+ questions with paraphrased and multi-hop questions.
