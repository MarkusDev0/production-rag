"""Evaluation harness for retrieval and (optionally) answer quality.

Retrieval metrics run fully offline by default (hashing embedder, in-memory
store), so they work in CI and without an API key:

    python -m evals.run_evals                      # compare vector / keyword / hybrid
    python -m evals.run_evals --embedder openai    # same, with real embeddings

Answer metrics call the configured LLM (see .env):

    python -m evals.run_evals --generate --modes hybrid --reranker llm

Metrics
- hit@k: share of answerable questions where a top-k chunk contains the evidence span
- MRR@k: mean reciprocal rank of the first chunk containing the evidence span
- answer accuracy: answer contains all expected keywords
- grounded citation rate: at least one cited chunk contains the evidence span
- abstention accuracy: unanswerable questions correctly refused
- false abstention rate: answerable questions wrongly refused
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from production_rag.config import Settings
from production_rag.embeddings import Embedder, HashingEmbedder
from production_rag.factory import build_cost_fn, build_embedder, build_llm
from production_rag.generation.answer import AnswerGenerator
from production_rag.ingestion.loaders import load_text
from production_rag.models import ScoredChunk
from production_rag.pipeline import RAGPipeline
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import LLMReranker, NoopReranker, Reranker
from production_rag.store.memory import InMemoryStore

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_evidence(chunk_text: str, evidence: str) -> bool:
    return _norm(evidence) in _norm(chunk_text)


@dataclass
class Question:
    id: str
    question: str
    evidence: str | None
    answer_keywords: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.evidence is not None


@dataclass
class RunResult:
    config: dict[str, Any]
    hit_at_k: float
    mrr_at_k: float
    answer_accuracy: float | None = None
    grounded_citation_rate: float | None = None
    abstention_accuracy: float | None = None
    false_abstention_rate: float | None = None
    total_cost_usd: float | None = None
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)


def load_questions() -> list[Question]:
    lines = (DATA_DIR / "questions.jsonl").read_text().splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def build_pipeline(
    settings: Settings, embedder: Embedder, mode: str, reranker_name: str, need_llm: bool
) -> tuple[RAGPipeline, InMemoryStore]:
    store = InMemoryStore()
    llm = build_llm(settings) if (need_llm or reranker_name == "llm") else None
    cost_fn = build_cost_fn(settings)
    reranker: Reranker = (
        LLMReranker(llm, cost_fn=cost_fn) if llm and reranker_name == "llm" else NoopReranker()
    )
    retriever = Retriever(
        store, embedder, mode=mode, reranker=reranker, candidate_k=settings.candidate_k
    )
    generator = AnswerGenerator(llm, cost_fn=cost_fn) if llm else None
    pipeline = RAGPipeline(
        store=store,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    for path in sorted((DATA_DIR / "corpus").glob("*.md")):
        pipeline.ingest(load_text(path.read_text(), source=path.name))
    return pipeline, store


def first_hit_rank(results: list[ScoredChunk], evidence: str) -> int | None:
    for rank, item in enumerate(results, start=1):
        if contains_evidence(item.chunk.text, evidence):
            return rank
    return None


def evaluate(
    settings: Settings, embedder: Embedder, mode: str, reranker: str, k: int, generate: bool
) -> RunResult:
    started = time.perf_counter()
    pipeline, store = build_pipeline(settings, embedder, mode, reranker, need_llm=generate)
    questions = load_questions()
    answerable = [q for q in questions if q.answerable]

    hits, reciprocal_ranks, failures = 0, [], []
    for q in answerable:
        rank = first_hit_rank(pipeline.retrieve(q.question, top_k=k), q.evidence or "")
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank is None:
            failures.append(f"retrieval miss: {q.id} {q.question}")

    result = RunResult(
        config={
            "mode": mode,
            "reranker": reranker,
            "k": k,
            "embedder": type(embedder).__name__,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        hit_at_k=round(hits / len(answerable), 3),
        mrr_at_k=round(sum(reciprocal_ranks) / len(answerable), 3),
    )

    if generate:
        correct = grounded = false_abstain = abstain_ok = 0
        cost = 0.0
        unanswerable = [q for q in questions if not q.answerable]
        for q in questions:
            answer = pipeline.query(q.question, top_k=k)
            cost += answer.usage.cost_usd or 0.0
            if not q.answerable:
                abstain_ok += answer.abstained
                if not answer.abstained:
                    failures.append(f"should abstain: {q.id} -> {answer.answer[:80]}")
                continue
            if answer.abstained:
                false_abstain += 1
                failures.append(f"false abstention: {q.id} {q.question}")
                continue
            text = answer.answer.lower()
            if all(kw.lower() in text for kw in q.answer_keywords):
                correct += 1
            else:
                failures.append(f"wrong answer: {q.id} -> {answer.answer[:80]}")
            cited_texts = [
                chunk.text
                for cit in answer.citations
                if (chunk := store.get_chunk(cit.chunk_id)) is not None
            ]
            if any(contains_evidence(text, q.evidence or "") for text in cited_texts):
                grounded += 1
        result.answer_accuracy = round(correct / len(answerable), 3)
        result.grounded_citation_rate = round(grounded / len(answerable), 3)
        result.false_abstention_rate = round(false_abstain / len(answerable), 3)
        result.abstention_accuracy = (
            round(abstain_ok / len(unanswerable), 3) if unanswerable else None
        )
        result.total_cost_usd = round(cost, 4) if settings.price_input_per_mtok else None

    result.seconds = round(time.perf_counter() - started, 2)
    result.failures = failures
    return result


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def to_markdown(results: list[RunResult]) -> str:
    header = (
        "| mode | reranker | embedder | hit@k | MRR@k | answer acc. | grounded cit. "
        "| abstention acc. | false abstention |\n|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.config['mode']} | {r.config['reranker']} | {r.config['embedder']} "
        f"| {_fmt(r.hit_at_k)} | {_fmt(r.mrr_at_k)} | {_fmt(r.answer_accuracy)} "
        f"| {_fmt(r.grounded_citation_rate)} | {_fmt(r.abstention_accuracy)} "
        f"| {_fmt(r.false_abstention_rate)} |"
        for r in results
    ]
    return "\n".join([header, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--modes", default="vector,keyword,hybrid")
    parser.add_argument("--reranker", choices=["none", "llm"], default="none")
    parser.add_argument("--embedder", choices=["hashing", "openai"], default="hashing")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--generate", action="store_true", help="also evaluate answers (uses LLM)")
    parser.add_argument(
        "--min-hit-rate", type=float, default=None, help="exit 1 if below (CI gate)"
    )
    args = parser.parse_args(argv)

    settings = Settings(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=args.embedder,
        embedding_dim=512 if args.embedder == "hashing" else 1536,
    )
    embedder = build_embedder(settings) if args.embedder == "openai" else HashingEmbedder(512)

    results = [
        evaluate(settings, embedder, mode.strip(), args.reranker, args.k, args.generate)
        for mode in args.modes.split(",")
    ]

    print(to_markdown(results))
    for r in results:
        if r.failures:
            print(f"\n{r.config['mode']} failures:")
            for failure in r.failures:
                print(f"  - {failure}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")

    if args.min_hit_rate is not None and min(r.hit_at_k for r in results) < args.min_hit_rate:
        print(f"FAIL: hit@k below {args.min_hit_rate}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
