"""
Embeddings
----------
Defines an Embedder interface so the rest of the pipeline never cares HOW
a vector is produced. Two implementations are provided:

1. TfidfEmbedder — a real, working, CPU-only "dense-ish" embedder based on
   TF-IDF + SVD (latent semantic indexing). This has none of the network
   dependencies, but has genuine limitations vs. a real bi-encoder
   (bge/e5): it captures term co-occurrence, not deep semantics, so it will
   miss paraphrases with no shared vocabulary. This is a real, documented
   limitation you can discuss in an interview.

2. TransformerEmbedder — a stub showing exactly where to plug in
   sentence-transformers (bge-small-en / e5-base) once you have GPU/network
   access (e.g. on Colab). Swapping this in is a one-line change in
   pipeline.py: `Embedder = TransformerEmbedder()`.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: List[str]):
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class TfidfEmbedder(Embedder):
    """Real, working embedder: TF-IDF -> L2-normalized truncated SVD vectors.
    Cosine similarity between these vectors approximates semantic similarity
    based on shared/co-occurring vocabulary."""

    def __init__(self, n_components: int = 128):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=1
        )
        self.n_components = n_components
        self.svd = None
        self._fitted = False

    def fit(self, texts: List[str]):
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .encode()")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms


class TransformerEmbedder(Embedder):
    """Real bi-encoder embedder using sentence-transformers. Requires network
    access to huggingface.co, which the original build sandbox did not have —
    this is meant to run on Colab or any machine with internet access.

        pip install sentence-transformers

    Uses BAAI/bge-small-en-v1.5 by default: a strong, small (~130MB), CPU-friendly
    open embedding model. `fit()` is a no-op here (no fitting needed for a
    pretrained bi-encoder, unlike TF-IDF/SVD which must fit vocab+components
    on your corpus) — it exists only to satisfy the shared Embedder interface.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: List[str]):
        self._load()  # just ensures the model is downloaded/loaded
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs)


def get_embedder(backend: str = "tfidf") -> Embedder:
    """Factory so the rest of the codebase (pipeline.py, eval.py) can switch
    embedding backends with one string instead of editing multiple files.

    backend: "tfidf" (default, runs anywhere, no downloads) or
             "transformer" (real bge-small-en-v1.5, needs network + sentence-transformers).
    """
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "transformer":
        return TransformerEmbedder()
    raise ValueError(f"Unknown embedder backend: {backend!r}. Use 'tfidf' or 'transformer'.")
