"""
Re-ranker
---------
Retrieval (BM25 + dense) is optimized for recall over the whole corpus, so
top-k often contains near-misses: chunks that share vocabulary/topic with
the query but don't actually answer it. A re-ranker looks at each
(query, chunk) pair jointly (not as two separate vectors) and re-orders the
candidates by how well each one actually answers the query.

Two implementations:
1. LexicalCrossScorer — a real, working cross-scorer based on token overlap,
   query-term coverage, and length-normalized overlap. It is deliberately
   simple and will be outperformed by a real cross-encoder, but it runs with
   zero downloads and demonstrates the re-ranking STAGE and interface
   correctly.
2. FineTunedCrossEncoder — a stub for `cross-encoder/ms-marco-MiniLM-L-6-v2`
   fine-tuned on hard negatives mined from this corpus (chunks that a
   dense/BM25 retriever ranks highly but that don't answer the gold
   question in eval_set.json). This is the actual "I fine-tuned a
   re-ranker on my own hard negatives" resume claim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import re

from .retrieval import RetrievedChunk


@dataclass
class RankedChunk:
    retrieved: RetrievedChunk
    rerank_score: float


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RankedChunk]:
        ...


def _tokens(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class LexicalCrossScorer(Reranker):
    """Scores each (query, chunk) pair by:
    - query-term coverage (fraction of query tokens present in the chunk)
    - a bonus for exact numeric/entity matches (numbers, section ids), since
      those are usually load-bearing for factual QA ("2%", "90 days", "5 business days")
    """

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 3) -> List[RankedChunk]:
        q_tokens = _tokens(query)
        q_numbers = set(re.findall(r"\d+%?", query))

        scored = []
        for cand in candidates:
            c_tokens = _tokens(cand.chunk.text)
            if not q_tokens:
                coverage = 0.0
            else:
                coverage = len(q_tokens & c_tokens) / len(q_tokens)

            c_numbers = set(re.findall(r"\d+%?", cand.chunk.text))
            number_bonus = 0.15 * len(q_numbers & c_numbers)

            score = coverage + number_bonus
            scored.append(RankedChunk(retrieved=cand, rerank_score=score))

        scored.sort(key=lambda r: r.rerank_score, reverse=True)
        return scored[:top_k]


class FineTunedCrossEncoder(Reranker):
    """PLUG-IN POINT. On your own machine with GPU/network access:

        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Mine hard negatives: for each eval_set.json question, take chunks
        # the HybridRetriever ranks in top-5 that are NOT the gold_doc/section.
        # Fine-tune on (query, gold_chunk, label=1) and (query, hard_negative, label=0)
        # pairs using model.fit(...).

        scores = model.predict([(query, c.chunk.text) for c in candidates])

    Report nDCG@10 before vs. after fine-tuning on your own hard negatives —
    that comparison is the actual resume metric.
    """

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RankedChunk]:
        raise NotImplementedError("Load a fine-tuned CrossEncoder here.")
