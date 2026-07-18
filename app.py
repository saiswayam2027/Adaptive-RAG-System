"""
Streamlit demo for the Adaptive RAG pipeline.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from src.pipeline import RagPipeline

st.set_page_config(page_title="Adaptive RAG Demo", layout="wide")

st.title("Adaptive RAG System")
st.caption(
    "Hybrid retrieval (BM25 + dense) → query routing → re-ranking → "
    "cited generation → hallucination check. Corpus: sample engineering/HR policy docs."
)


@st.cache_resource
def load_pipeline():
    corpus_dir = os.path.join(os.path.dirname(__file__), "data", "corpus")
    return RagPipeline(corpus_dir)


pipeline = load_pipeline()

with st.sidebar:
    st.header("Corpus")
    for c in sorted({c.source for c in pipeline.chunks}):
        st.write(f"- {c}")
    st.markdown("---")
    st.caption(
        "Try an out-of-scope question ('what's the weather?') to see the "
        "router refuse instead of hallucinating, or a multi-hop question "
        "('If X happens, who approves it and who gets notified?') to see "
        "sub-question decomposition."
    )

query = st.text_input("Ask a question about the sample corpus:",
                       value="If a canary release for a high-traffic service breaches the error-rate threshold, what threshold applies and why?")

if st.button("Ask", type="primary") and query.strip():
    result = pipeline.answer(query)

    route_colors = {"simple": "blue", "multi_hop": "orange", "out_of_scope": "red"}
    color = route_colors.get(result.route, "gray")
    st.markdown(f"**Route:** :{color}[{result.route}] — {result.route_reason}")

    st.subheader("Answer")
    st.write(result.answer.text)

    faith = result.hallucination.faithfulness_score
    st.subheader("Faithfulness Check")
    st.progress(faith, text=f"{faith:.0%} of generated sentences supported by retrieved context")
    if result.hallucination.unsupported:
        st.warning("Unsupported sentence(s) flagged:")
        for s in result.hallucination.unsupported:
            st.write(f"- {s}")

    if result.retrieved:
        st.subheader("Retrieved & Re-ranked Chunks")
        for rc in result.reranked:
            chunk = rc.retrieved.chunk
            with st.expander(f"[{chunk.source} §{chunk.section_id}] {chunk.section_title} (rerank score: {rc.rerank_score:.2f})"):
                st.write(chunk.text)
                st.caption(f"BM25 rank: {rc.retrieved.bm25_rank} | Dense rank: {rc.retrieved.dense_rank} | RRF score: {rc.retrieved.fused_score:.4f}")
