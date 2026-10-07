"""Grounded answer generation with validated citations.

The model sees numbered sources and must cite them as [n]. After generation,
citations are checked against the sources that were actually provided; any
reference to a source that doesn't exist is stripped and logged. If the sources
don't contain the answer, the model is instructed to abstain with a sentinel
token, which is converted into an explicit `abstained=True` response instead of
a guess.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from production_rag.generation.llm import LLM
from production_rag.models import Answer, Citation, CostFn, ScoredChunk, Usage

logger = logging.getLogger(__name__)

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"
ABSTAIN_MESSAGE = "I couldn't find the answer to this in the provided documents."

ANSWER_SYSTEM = f"""You answer questions using only the numbered sources provided.

Rules:
- Every factual sentence must cite at least one source, like [1] or [2][3].
- Use only information stated in the sources. Do not add outside knowledge.
- If the sources do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}
- If the sources only partially answer, answer the part they support and say what is missing.
- Be concise and direct."""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def format_sources(chunks: Sequence[ScoredChunk]) -> str:
    blocks = []
    for ref, item in enumerate(chunks, start=1):
        meta = item.chunk.metadata
        label = str(meta.get("title") or meta.get("source") or item.chunk.document_id)
        if meta.get("page") is not None:
            label += f", page {meta['page']}"
        blocks.append(f"[{ref}] ({label})\n{item.chunk.text}")
    return "\n\n".join(blocks)


def _make_citation(ref: int, item: ScoredChunk) -> Citation:
    meta = item.chunk.metadata
    snippet = item.chunk.text[:240].rstrip()
    if len(item.chunk.text) > 240:
        snippet += "..."
    page = meta.get("page")
    return Citation(
        ref=ref,
        chunk_id=item.chunk.id,
        document_id=item.chunk.document_id,
        source=str(meta.get("source", "")),
        title=str(meta.get("title", "")),
        page=int(page) if page is not None else None,
        snippet=snippet,
    )


class AnswerGenerator:
    def __init__(self, llm: LLM, *, cost_fn: CostFn | None = None, max_tokens: int = 1024):
        self.llm = llm
        self.cost_fn = cost_fn
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> Answer:
        if not chunks:
            # Nothing retrieved: abstain without paying for an LLM call.
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, citations=[])

        prompt = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
        response = self.llm.complete(ANSWER_SYSTEM, prompt, max_tokens=self.max_tokens)
        usage = Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=self.cost_fn(response.input_tokens, response.output_tokens)
            if self.cost_fn
            else None,
        )

        text = response.text.strip()
        if not text or ABSTAIN_TOKEN in text:
            return Answer(answer=ABSTAIN_MESSAGE, abstained=True, citations=[], usage=usage)

        valid_refs = set(range(1, len(chunks) + 1))
        cited = [int(m) for m in _CITATION_RE.findall(text)]
        invalid = sorted({r for r in cited if r not in valid_refs})
        if invalid:
            logger.warning("model cited non-existent sources", extra={"invalid_refs": invalid})
            text = _CITATION_RE.sub(
                lambda m: m.group(0) if int(m.group(1)) in valid_refs else "", text
            ).strip()

        refs_in_order = list(dict.fromkeys(r for r in cited if r in valid_refs))
        if not refs_in_order:
            logger.warning("answer has no valid citations")
        citations = [_make_citation(ref, chunks[ref - 1]) for ref in refs_in_order]
        return Answer(answer=text, abstained=False, citations=citations, usage=usage)
