"""
Pipeline
--------
Wires every stage together and is the single place you'd swap fallback
components for real ones (TfidfEmbedder -> TransformerEmbedder,
LexicalCrossScorer -> FineTunedCrossEncoder, TemplateGenerator ->
LLMGenerator, LexicalEntailmentChecker -> NliHallucinationChecker) without
touching any other file, because every stage talks to the interfaces
defined in its own module.

Flow for a single query:
  1. Router decides: SIMPLE / MULTI_HOP / OUT_OF_SCOPE
  2. OUT_OF_SCOPE -> refuse immediately, skip retrieval+generation entirely
  3. SIMPLE -> one retrieval pass
     MULTI_HOP -> decompose into sub-questions, retrieve per sub-question,
     merge candidate pools
  4. Re-rank merged candidates
  5. Generate answer with citations
  6. Check generated answer for hallucination against retrieved context
"""

import os
from dataclasses import dataclass, field
from typing import List

from .ingestion import load_corpus, Chunk
from .embeddings import get_embedder
from .retrieval import HybridRetriever, RetrievedChunk
from .router import RuleBasedRouter, Route, decompose_multi_hop, default_oos_threshold
from .reranker import LexicalCrossScorer, RankedChunk
from .generation import TemplateGenerator, LexicalEntailmentChecker, GeneratedAnswer, HallucinationReport


@dataclass
class PipelineResult:
    query: str
    route: str
    route_reason: str
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    reranked: List[RankedChunk] = field(default_factory=list)
    answer: GeneratedAnswer = None
    hallucination: HallucinationReport = None


class RagPipeline:
    def __init__(self, corpus_dir: str, top_k_retrieve: int = 6, top_k_rerank: int = 3,
                 embedder_backend: str = "tfidf"):
        """embedder_backend: 'tfidf' (default, runs anywhere) or 'transformer'
        (real bge-small-en-v1.5 — needs network + `pip install sentence-transformers`,
        e.g. on Colab). See src/embeddings.py get_embedder()."""
        self.chunks: List[Chunk] = load_corpus(corpus_dir)
        self.embedder = get_embedder(embedder_backend)
        self.retriever = HybridRetriever(self.chunks, self.embedder)
        self.router = RuleBasedRouter(self.embedder, self.retriever.chunk_vecs,
                                       oos_threshold=default_oos_threshold(embedder_backend))
        self.reranker = LexicalCrossScorer()
        self.generator = TemplateGenerator()
        self.hallucination_checker = LexicalEntailmentChecker()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    def answer(self, query: str) -> PipelineResult:
        decision = self.router.route(query)

        if decision.route == Route.OUT_OF_SCOPE:
            return PipelineResult(
                query=query,
                route=decision.route.value,
                route_reason=decision.reason,
                answer=GeneratedAnswer(
                    text="I don't have information about this in the available documents.",
                    citations=[],
                ),
                hallucination=HallucinationReport(supported_sentences=0, total_sentences=0, unsupported=[]),
            )

        if decision.route == Route.MULTI_HOP:
            sub_questions = decompose_multi_hop(query)
            candidates: List[RetrievedChunk] = []
            seen_chunk_ids = set()
            for sq in sub_questions:
                for rc in self.retriever.retrieve(sq, top_k=self.top_k_retrieve):
                    if rc.chunk.chunk_id not in seen_chunk_ids:
                        candidates.append(rc)
                        seen_chunk_ids.add(rc.chunk.chunk_id)
        else:
            candidates = self.retriever.retrieve(query, top_k=self.top_k_retrieve)

        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
        generated = self.generator.generate(query, reranked)
        hallucination = self.hallucination_checker.check(generated, reranked)

        return PipelineResult(
            query=query,
            route=decision.route.value,
            route_reason=decision.reason,
            retrieved=candidates,
            reranked=reranked,
            answer=generated,
            hallucination=hallucination,
        )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(here, "..", "data", "corpus")
    pipeline = RagPipeline(corpus_dir)

    test_queries = [
        "What error rate triggers an automatic rollback?",
        "If a canary release for a high-traffic service breaches the error-rate threshold, what threshold applies and why is it different from the general rollback threshold?",
        "What's the weather like today?",
    ]
    for q in test_queries:
        result = pipeline.answer(q)
        print(f"\nQ: {q}")
        print(f"Route: {result.route} ({result.route_reason})")
        print(f"A: {result.answer.text}")
        print(f"Faithfulness: {result.hallucination.faithfulness_score:.2f}")
