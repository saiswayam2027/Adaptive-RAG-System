"""
Evaluation Harness
------------------
Runs the eval set (data/eval/eval_set.json) against different retrieval
configurations and reports real, computed metrics:

- Retrieval hit rate@k: did the correct source document appear in the
  top-k retrieved chunks?
- Routing accuracy: did the router assign the label matching the eval set's
  'type' field?
- Faithfulness: fraction of generated sentences supported by retrieved
  context (from generation.HallucinationChecker).

This is the script that produces the "v1 (dense-only) vs v2 (hybrid) vs v3
(hybrid + rerank)" comparison table for a resume/interview writeup, with
real numbers computed from actual retrieval runs on your corpus.
"""

import argparse
import json
import os
from typing import List, Dict

from .ingestion import load_corpus
from .embeddings import get_embedder
from .retrieval import HybridRetriever
from .router import RuleBasedRouter, Route, default_oos_threshold
from .reranker import LexicalCrossScorer
from .generation import TemplateGenerator, LexicalEntailmentChecker
from .pipeline import RagPipeline


def _load_eval_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gold_docs(item: Dict) -> set:
    g = item["gold_doc"]
    if g is None:
        return set()
    if isinstance(g, str):
        return {g}
    return set(g)


def evaluate_retrieval(retriever_fn, eval_items: List[Dict], top_k: int = 5, label: str = ""):
    """retriever_fn: callable(query, top_k) -> list of RetrievedChunk-like with .chunk.source"""
    hits, total_scored = 0, 0
    for item in eval_items:
        gold = _gold_docs(item)
        if not gold:
            continue  # out-of-scope items have no retrieval gold
        total_scored += 1
        results = retriever_fn(item["question"], top_k)
        retrieved_sources = {r.chunk.source for r in results}
        if gold & retrieved_sources:
            hits += 1

    hit_rate = hits / total_scored if total_scored else 0.0
    print(f"[{label}] Retrieval hit_rate@{top_k}: {hit_rate:.2%} ({hits}/{total_scored})")
    return hit_rate


def evaluate_routing(router: RuleBasedRouter, eval_items: List[Dict]):
    correct = 0
    for item in eval_items:
        decision = router.route(item["question"])
        expected = item["type"]
        got = decision.route.value
        match = (got == expected)
        correct += int(match)
        flag = "OK " if match else "MISS"
        print(f"  [{flag}] '{item['question'][:60]}...' expected={expected} got={got} (sim={decision.max_corpus_similarity:.3f})")
    acc = correct / len(eval_items)
    print(f"Routing accuracy: {acc:.2%} ({correct}/{len(eval_items)})")
    return acc


def evaluate_end_to_end(pipeline: RagPipeline, eval_items: List[Dict]):
    faithfulness_scores = []
    for item in eval_items:
        result = pipeline.answer(item["question"])
        faithfulness_scores.append(result.hallucination.faithfulness_score)
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    print(f"Average faithfulness across eval set: {avg_faithfulness:.2%}")
    return avg_faithfulness


def run_suite(embedder_backend: str, corpus_dir: str, eval_items: List[Dict]):
    print("#" * 70)
    print(f"# EMBEDDER BACKEND: {embedder_backend}")
    print("#" * 70)

    chunks = load_corpus(corpus_dir)
    embedder = get_embedder(embedder_backend)
    retriever = HybridRetriever(chunks, embedder)

    print("=" * 70)
    print("RETRIEVAL ABLATION: dense-only vs BM25-only vs hybrid (RRF)")
    print("=" * 70)
    dense_hr = evaluate_retrieval(retriever.retrieve_dense_only, eval_items, top_k=5, label="dense-only")
    bm25_hr = evaluate_retrieval(retriever.retrieve_bm25_only, eval_items, top_k=5, label="bm25-only")
    hybrid_hr = evaluate_retrieval(retriever.retrieve, eval_items, top_k=5, label="hybrid-rrf")

    print()
    print("=" * 70)
    print("QUERY ROUTING")
    print("=" * 70)
    router = RuleBasedRouter(embedder, retriever.chunk_vecs, oos_threshold=default_oos_threshold(embedder_backend))
    routing_acc = evaluate_routing(router, eval_items)

    print()
    print("=" * 70)
    print("END-TO-END PIPELINE (faithfulness)")
    print("=" * 70)
    pipeline = RagPipeline(corpus_dir, embedder_backend=embedder_backend)
    faithfulness = evaluate_end_to_end(pipeline, eval_items)

    return {
        "backend": embedder_backend,
        "dense_hit_rate": dense_hr,
        "bm25_hit_rate": bm25_hr,
        "hybrid_hit_rate": hybrid_hr,
        "routing_accuracy": routing_acc,
        "faithfulness": faithfulness,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument(
        "--backend", choices=["tfidf", "transformer", "both"], default="tfidf",
        help="Embedder backend: 'tfidf' (default, runs anywhere), 'transformer' "
             "(real bge-small-en-v1.5, needs `pip install sentence-transformers` "
             "+ network — use on Colab), or 'both' to print a side-by-side comparison.",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(here, "..", "data", "corpus")
    eval_path = os.path.join(here, "..", "data", "eval", "eval_set.json")
    eval_items = _load_eval_set(eval_path)

    backends = ["tfidf", "transformer"] if args.backend == "both" else [args.backend]
    results = [run_suite(b, corpus_dir, eval_items) for b in backends]

    if len(results) > 1:
        print()
        print("=" * 70)
        print("SIDE-BY-SIDE COMPARISON")
        print("=" * 70)
        header = f"{'metric':<20}" + "".join(f"{r['backend']:>15}" for r in results)
        print(header)
        for key, label in [
            ("dense_hit_rate", "dense hit_rate@5"),
            ("bm25_hit_rate", "bm25 hit_rate@5"),
            ("hybrid_hit_rate", "hybrid hit_rate@5"),
            ("routing_accuracy", "routing accuracy"),
            ("faithfulness", "faithfulness"),
        ]:
            row = f"{label:<20}" + "".join(f"{r[key]:>14.2%} " for r in results)
            print(row)


if __name__ == "__main__":
    main()
