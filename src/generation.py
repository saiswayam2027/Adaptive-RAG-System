"""
Generation & Hallucination Guardrails
--------------------------------------
Two responsibilities live here:

1. Generator — produces the final answer from retrieved context. This is a
   PLUG-IN POINT for a real LLM (Groq/Together free-tier Llama-3-8B, OpenAI,
   or a local model). No API keys exist in this sandbox, so a
   TemplateGenerator is provided that extracts and stitches the most
   relevant sentences from the top chunks with explicit [source] citations,
   so the whole pipeline still runs end-to-end and every claim is
   traceable. Swap in an LLMGenerator once you have API access — one class,
   one method, no other file changes.

2. HallucinationChecker — after generation, verify each generated claim is
   actually entailed by the retrieved context. A NliHallucinationChecker
   stub shows the real approach (NLI entailment model); the working
   fallback here uses token-overlap entailment, which is a real if weaker
   signal for "is this sentence supported by the source text."
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import re

from .reranker import RankedChunk


@dataclass
class GeneratedAnswer:
    text: str
    citations: List[str]  # chunk_ids used


@dataclass
class HallucinationReport:
    supported_sentences: int
    total_sentences: int
    unsupported: List[str]

    @property
    def faithfulness_score(self) -> float:
        if self.total_sentences == 0:
            return 1.0
        return self.supported_sentences / self.total_sentences


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: List[RankedChunk]) -> GeneratedAnswer:
        ...


class TemplateGenerator(Generator):
    """Working, zero-dependency generator: picks the sentence(s) in each top
    chunk most relevant to the query (by token overlap, reusing the same
    signal as the reranker) and stitches them into an answer with inline
    [source] tags. This guarantees 100% grounding by construction, which is
    a real (if unambitious) baseline — the honest comparison point for
    "how much does the LLM generator actually add."
    """

    def generate(self, query: str, context_chunks: List[RankedChunk]) -> GeneratedAnswer:
        if not context_chunks:
            return GeneratedAnswer(
                text="I don't have information about this in the available documents.",
                citations=[],
            )

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        pieces = []
        citations = []

        for rc in context_chunks:
            chunk = rc.retrieved.chunk
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            best_sentence, best_score = None, -1.0
            for s in sentences:
                s_tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
                overlap = len(q_tokens & s_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = s
            if best_sentence and best_score > 0:
                pieces.append(f"{best_sentence.strip()} [source: {chunk.source}, {chunk.section_id}]")
                citations.append(chunk.chunk_id)

        if not pieces:
            return GeneratedAnswer(
                text="I don't have information about this in the available documents.",
                citations=[],
            )

        return GeneratedAnswer(text=" ".join(pieces), citations=citations)


class LLMGenerator(Generator):
    """PLUG-IN POINT. Example using an OpenAI-compatible free-tier API
    (Groq, Together, etc.) once you have a key and network access:

        context = "\\n\\n".join(
            f"[{c.retrieved.chunk.chunk_id}] {c.retrieved.chunk.text}" for c in context_chunks
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "Cite the chunk_id in brackets after every claim. "
            "If the context doesn't contain the answer, say so explicitly.\\n\\n"
            f"Context:\\n{context}\\n\\nQuestion: {query}"
        )
        response = client.chat.completions.create(model="llama-3.1-8b-instant",
                                                    messages=[{"role": "user", "content": prompt}])

    Report faithfulness/answer-relevancy from RAGAS comparing this vs.
    TemplateGenerator — that delta is a genuine, defensible resume metric.
    """

    def generate(self, query: str, context_chunks: List[RankedChunk]) -> GeneratedAnswer:
        raise NotImplementedError("Wire up an LLM API client here.")


class HallucinationChecker(ABC):
    @abstractmethod
    def check(self, answer: GeneratedAnswer, context_chunks: List[RankedChunk]) -> HallucinationReport:
        ...


class LexicalEntailmentChecker(HallucinationChecker):
    """Working fallback: a generated sentence is 'supported' if a large
    fraction of its content tokens appear in SOME retrieved chunk. This is a
    real, if blunt, entailment proxy — it will pass sentences that reuse
    source vocabulary even if logic is subtly wrong, which is exactly the
    kind of limitation an NLI model fixes (see NliHallucinationChecker)."""

    def __init__(self, support_threshold: float = 0.5):
        self.support_threshold = support_threshold

    def check(self, answer: GeneratedAnswer, context_chunks: List[RankedChunk]) -> HallucinationReport:
        context_text = " ".join(c.retrieved.chunk.text.lower() for c in context_chunks)
        context_tokens = set(re.findall(r"[a-z0-9]+", context_text))

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.text) if s.strip()]
        supported, unsupported = 0, []

        for s in sentences:
            s_clean = re.sub(r"\[source:.*?\]", "", s)
            s_tokens = set(re.findall(r"[a-z0-9]+", s_clean.lower()))
            if not s_tokens:
                continue
            coverage = len(s_tokens & context_tokens) / len(s_tokens)
            if coverage >= self.support_threshold:
                supported += 1
            else:
                unsupported.append(s)

        return HallucinationReport(
            supported_sentences=supported,
            total_sentences=len(sentences),
            unsupported=unsupported,
        )


class NliHallucinationChecker(HallucinationChecker):
    """PLUG-IN POINT for a real NLI model:

        from sentence_transformers import CrossEncoder
        nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        scores = nli_model.predict([(context_text, sentence)])
        # label 'entailment' -> supported, 'contradiction'/'neutral' -> flagged

    This catches cases the lexical checker misses: a sentence can reuse all
    the right vocabulary while still asserting the wrong relationship
    between them (e.g. swapping which team approves what).
    """

    def check(self, answer: GeneratedAnswer, context_chunks: List[RankedChunk]) -> HallucinationReport:
        raise NotImplementedError("Load a fine-tuned/pretrained NLI cross-encoder here.")
