"""
Hybrid Retrieval
----------------
Dense embeddings alone miss exact-term matches (a query for "PCI compliance
training" can retrieve something about "security awareness training" instead
because they're semantically close, even though only one chunk actually
mentions PCI). BM25 alone misses paraphrases with no shared vocabulary.

This module runs BOTH and fuses the two ranked lists with Reciprocal Rank
Fusion (RRF), which is simple, has no score-scale mismatch issues (unlike
naively averaging BM25 and cosine scores, which live on different scales),
and is what most production hybrid-search systems (Elasticsearch, Weaviate,
Azure AI Search) use under the hood.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi

from .ingestion import Chunk
from .embeddings import Embedder


@dataclass
class RetrievedChunk:
    chunk: Chunk
    bm25_rank: int
    dense_rank: int
    fused_score: float


def _tokenize(text: str):
    return text.lower().split()


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, rrf_k: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.rrf_k = rrf_k  # standard RRF damping constant

        self.texts = [c.text for c in chunks]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        self.embedder.fit(self.texts)
        self.chunk_vecs = self.embedder.encode(self.texts)

    def _bm25_ranked_indices(self, query: str, top_k: int):
        scores = self.bm25.get_scores(_tokenize(query))
        return np.argsort(scores)[::-1][:top_k]

    def _dense_ranked_indices(self, query: str, top_k: int):
        q_vec = self.embedder.encode([query])[0]
        sims = self.chunk_vecs @ q_vec
        return np.argsort(sims)[::-1][:top_k]

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> List[RetrievedChunk]:
        bm25_idx = list(self._bm25_ranked_indices(query, candidate_pool))
        dense_idx = list(self._dense_ranked_indices(query, candidate_pool))

        bm25_rank_of = {idx: rank + 1 for rank, idx in enumerate(bm25_idx)}
        dense_rank_of = {idx: rank + 1 for rank, idx in enumerate(dense_idx)}

        all_idx = set(bm25_idx) | set(dense_idx)
        fused = []
        for idx in all_idx:
            r_bm25 = bm25_rank_of.get(idx, candidate_pool + 1)
            r_dense = dense_rank_of.get(idx, candidate_pool + 1)
            score = 1.0 / (self.rrf_k + r_bm25) + 1.0 / (self.rrf_k + r_dense)
            fused.append(RetrievedChunk(
                chunk=self.chunks[idx],
                bm25_rank=r_bm25 if idx in bm25_rank_of else -1,
                dense_rank=r_dense if idx in dense_rank_of else -1,
                fused_score=score,
            ))

        fused.sort(key=lambda r: r.fused_score, reverse=True)
        return fused[:top_k]

    def retrieve_dense_only(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        idx = self._dense_ranked_indices(query, top_k)
        return [RetrievedChunk(chunk=self.chunks[i], bm25_rank=-1, dense_rank=r + 1, fused_score=0.0)
                for r, i in enumerate(idx)]

    def retrieve_bm25_only(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        idx = self._bm25_ranked_indices(query, top_k)
        return [RetrievedChunk(chunk=self.chunks[i], bm25_rank=r + 1, dense_rank=-1, fused_score=0.0)
                for r, i in enumerate(idx)]
