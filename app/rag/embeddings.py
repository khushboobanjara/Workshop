"""Local embeddings. Free: nothing here calls a paid or remote API.

`SentenceTransformerEmbedder` runs a Sentence Transformers model on this machine's CPU (default all-MiniLM-L6-v2, Apache-2.0,
384 dimensions, ~90 MB). The first use downloads the model from Hugging Face's free public hub and caches it; set
OPSPILOT_RAG_EMBEDDING_LOCAL_FILES_ONLY=true (or point OPSPILOT_RAG_EMBEDDING_MODEL at a local folder) to never go online.

`HashingEmbedder` is a deterministic bag-of-words stand-in with no model and no downloads. It matches WORDS, not meaning, so it
is only for tests and quick local trials; the app refuses to start with it in production.

Vectors are float32 and L2-normalised, so the inner product FAISS computes is the cosine similarity.
"""
import hashlib
import re
import threading
from typing import Protocol

import numpy as np


class EmbedderUnavailable(RuntimeError):
    """The embedding model could not be loaded (package missing, model not downloadable, ...). The message is safe to show."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(a / norms, dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, query_prefix: str = "", document_prefix: str = "",
                 local_files_only: bool = False):
        self.name, self.batch_size = model_name, batch_size
        self.query_prefix, self.document_prefix, self.local_files_only = query_prefix, document_prefix, local_files_only
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()                  # one encode at a time: the model object is shared by all requests

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EmbedderUnavailable("The 'sentence-transformers' package is not installed (pip install -r requirements.txt).") from None
            try:
                kwargs = {"local_files_only": True} if self.local_files_only else {}
                self._model = SentenceTransformer(self.name, device="cpu", **kwargs)
                size = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension   # renamed in newer versions
                self._dim = int(size())
            except Exception as exc:
                hint = " (offline mode is on: the model must already be cached)" if self.local_files_only else ""
                raise EmbedderUnavailable(f"The embedding model '{self.name}' could not be loaded{hint}: {type(exc).__name__}") from None
        return self._model

    @property
    def dim(self) -> int:
        with self._lock:
            self._load()
        return self._dim or 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            model = self._load()
            vecs = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False)
        return _normalise(vecs)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])


_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were will with".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


class HashingEmbedder:
    """Signed feature hashing of stemmed words and word pairs. Deterministic across processes (no Python hash())."""

    def __init__(self, dim: int = 384):
        self.dim, self.name = dim, f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]
        feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return np.sign(v) * np.sqrt(np.abs(v))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vec(text))


def build_embedder(settings) -> Embedder:
    kind = (settings.rag_embedder or "").strip().lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(settings.rag_embedding_model, settings.rag_embedding_batch_size,
                                           settings.rag_query_prefix, settings.rag_document_prefix,
                                           settings.rag_embedding_local_files_only)
    raise EmbedderUnavailable(f"Unknown OPSPILOT_RAG_EMBEDDER '{settings.rag_embedder}' (use 'sentence-transformers' or 'hashing').")
