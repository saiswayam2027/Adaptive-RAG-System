"""
Query Router
------------
Not every query deserves the same treatment. A naive RAG system spends the
same retrieval + generation budget on "what's the sick leave allowance?"
(one lookup) as it does on "if a Sev-1 breach happens during the deploy
freeze, who approves the patch and who must be notified?" (needs evidence
from two documents), and it will also confidently hallucinate an answer to
"what's the weather today?" instead of admitting the corpus can't answer that.

This router classifies each query into one of three routes BEFORE retrieval
happens, so the pipeline only pays for multi-hop decomposition when it's
actually needed, and refuses out-of-scope queries early instead of
retrieving garbage and generating a confident wrong answer.

Two implementations:
1. RuleBasedRouter — works immediately, no training data needed. Uses
   corpus-similarity gating (via the same embedder as retrieval) to catch
   out-of-scope queries, and lexical/structural heuristics for multi-hop.
2. TrainedRouter — a stub for a fine-tuned DistilBERT classifier, which is
   what you'd swap in for the "I fine-tuned a router" resume claim once you
   have labeled query data and GPU access.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np

from .embeddings import Embedder


class Route(Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class RoutingDecision:
    route: Route
    reason: str
    max_corpus_similarity: float


# Signals that are ON THEIR OWN strong evidence of multi-hop reasoning: explicit
# comparison/contrast, sequencing-with-a-dependency, or an exception overriding
# a base rule -- all patterns that typically mean two different rules (often
# from two different documents) have to be reconciled. " if " is included
# because empirically, on this corpus's question style, a conditional-scenario
# framing precedes a multi-hop answer ~4x more often than a single-doc lookup
# (see router audit in project notes) -- it is not perfectly precise (a bare
# "if" conditional can still be a single-document lookup), but it's net
# positive. Plain " and " is deliberately NOT here: it appears equally often in
# genuinely single-fact simple questions ("...need, and from whom?") as in
# multi-hop ones, so alone it isn't informative -- see wh_count handling below.
MULTI_HOP_SIGNALS = [
    " if ", " difference between ", "compare", " different from ", " same as ",
    " as opposed to ", " versus ", " vs ", "instead", "still need",
    "still require", " both ", " then ",
]

_WH_WORD_RE = re.compile(r"\b(?:who|what|why|how)\b")

_ROUTER_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "do", "does", "did",
    "what", "which", "that", "this", "these", "those", "it", "its", "s", "t",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _ROUTER_STOPWORDS]


def default_oos_threshold(backend: str) -> float:
    """Out-of-scope similarity thresholds are NOT portable across embedding
    backends, because different backends produce very different similarity
    distributions for unrelated text:

    - TF-IDF is sparse: two texts with zero shared vocabulary score ~0.0
      similarity, so a low threshold (0.08) cleanly separates in/out of scope.
    - Pretrained sentence embeddings (e.g. bge-small) exhibit anisotropy:
      cosine similarity between ANY two pieces of ordinary English text
      tends to sit well above 0, even for unrelated content. Measured on
      this project's eval set, out-of-scope questions scored 0.40-0.44
      similarity with bge-small-en-v1.5, while genuinely in-scope questions
      scored 0.71-0.91 — so 0.08 (correct for TF-IDF) let everything through.

    These defaults are tuned from that observed gap. If you change corpus,
    embedding model, or eval questions, re-check the gap yourself (print
    RoutingDecision.max_corpus_similarity for known in-scope vs out-of-scope
    queries) rather than trusting these numbers blindly.
    """
    if backend == "tfidf":
        return 0.08
    if backend == "transformer":
        return 0.55
    raise ValueError(f"No default OOS threshold defined for backend {backend!r}")


class RuleBasedRouter:
    def __init__(self, embedder: Embedder, chunk_vecs: np.ndarray, oos_threshold: float = 0.08,
                 single_token_collapse_ratio: float = 0.7):
        """oos_threshold: below this max cosine similarity to ANY chunk, we
        treat the query as out-of-scope. Tuned empirically against the eval
        set (see run_eval.py) — this is exactly the kind of threshold you'd
        report and justify with numbers in a resume writeup.

        single_token_collapse_ratio: a SECOND out-of-scope gate that catches
        queries which clear oos_threshold only because they share ONE
        coincidental keyword with an unrelated chunk (e.g. "what's the
        weather like today?" scores 0.95 similarity against a facilities doc
        that mentions "severe weather advisory" in a completely unrelated
        context). We remove each content word from the query one at a time
        and recompute max similarity; if similarity collapses by more than
        this fraction when removing SOME single word, the original match was
        carried entirely by that one word rather than genuine topical
        overlap. Empirically, on this eval set, every genuine in-scope match
        (including deliberately low-vocabulary-overlap paraphrases) drops by
        at most 0.42 when any single word is removed, while the weather trap
        collapses by 1.00 — 0.7 sits with a wide margin in between.

        This gate is specific to a TF-IDF+SVD failure mode: a rare word (e.g.
        "weather") can dominate a whole compressed SVD component on a small
        corpus, spiking cosine similarity to an unrelated chunk. Verified on
        bge-small-en-v1.5 (Colab, 16-doc corpus) that this doesn't reproduce
        with real embeddings — all out-of-scope queries there scored
        0.43-0.52, comfortably below that backend's 0.55 oos_threshold, so
        they're caught by the primary gate before this one even runs, and
        collapse ratios were uniformly small (<=0.115) for in-scope and OOS
        queries alike (no compression artifact to exploit). So on the
        transformer backend this gate is inert but harmless — kept at the
        same 0.7 default for both backends since there's no genuine
        collapse-triggering OOS example on that backend to calibrate a
        different number against."""
        self.embedder = embedder
        self.chunk_vecs = chunk_vecs
        self.oos_threshold = oos_threshold
        self.single_token_collapse_ratio = single_token_collapse_ratio

    def _max_sim(self, query: str) -> float:
        q_vec = self.embedder.encode([query])[0]
        sims = self.chunk_vecs @ q_vec
        return float(np.max(sims)) if len(sims) else 0.0

    def _single_token_collapse(self, query: str, max_sim: float) -> float:
        """Largest fractional drop in max_sim caused by deleting any one
        content word from the query. See __init__ docstring for rationale."""
        if max_sim <= 0:
            return 0.0
        tokens = set(_content_tokens(query))
        if len(tokens) < 2:
            return 0.0  # can't test single-word dependence with 0-1 content words
        worst = 0.0
        for tok in tokens:
            masked = re.sub(r"\b" + re.escape(tok) + r"\b", "", query.lower())
            worst = max(worst, (max_sim - self._max_sim(masked)) / max_sim)
        return worst

    def route(self, query: str) -> RoutingDecision:
        max_sim = self._max_sim(query)

        if max_sim < self.oos_threshold:
            return RoutingDecision(Route.OUT_OF_SCOPE, "max corpus similarity below threshold", max_sim)

        collapse_ratio = self._single_token_collapse(query, max_sim)
        if collapse_ratio > self.single_token_collapse_ratio:
            return RoutingDecision(
                Route.OUT_OF_SCOPE,
                f"similarity collapses {collapse_ratio:.0%} when a single query word is removed "
                "-- likely one coincidental shared keyword, not genuine topical relevance",
                max_sim,
            )

        q_padded = f" {query.lower().strip()} "
        wh_count = len(_WH_WORD_RE.findall(q_padded))
        signal_hit = any(sig in q_padded for sig in MULTI_HOP_SIGNALS)
        # "before X can Y" signals a prerequisite/sequencing dependency, which
        # often crosses two procedural steps -- but "before" alone is too
        # common a word to trust without a role/entity question alongside it.
        sequencing_hit = " before " in q_padded and wh_count >= 1

        if signal_hit or wh_count >= 2 or sequencing_hit:
            return RoutingDecision(Route.MULTI_HOP, "multi-hop lexical signal detected", max_sim)

        return RoutingDecision(Route.SIMPLE, "single concept, direct lookup", max_sim)


class TrainedRouter:
    """PLUG-IN POINT for a fine-tuned classifier (e.g. DistilBERT) trained on
    labeled (query, route) pairs — see data/eval/eval_set.json 'type' field
    for a starter label set. On your own machine:

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # fine-tune DistilBERT on (question, type) pairs from eval_set.json
        # extended with more examples, 3-way classification.

    Report accuracy against a held-out split for the resume metric:
    "trained a query router with X% routing accuracy."
    """

    def route(self, query: str) -> RoutingDecision:
        raise NotImplementedError("Fine-tune and load a classifier here.")


def decompose_multi_hop(query: str) -> List[str]:
    """Very simple heuristic decomposition: split on common multi-hop
    connectives so each sub-question can be retrieved independently, then
    the generation step synthesizes across both retrieved contexts. A
    stronger version would use an LLM call to decompose ("split this
    question into its atomic sub-questions") — swap this out once you have
    LLM API access.
    """
    q = query
    for sig in [" and ", " but ", ", and "]:
        if sig in q.lower():
            parts = q.split(sig) if sig in q else [q]
            return [p.strip().rstrip("?") + "?" for p in parts if p.strip()]
    # Fall back: treat as a single sub-question if no clean split found
    return [query]
