"""FAISS vector index: exact cosine search (IndexFlatIP over L2-normalised vectors) wrapped in an ID map, so every vector
carries the id of its rag_chunks row. Exact search is the right choice at this scale (tens of thousands of chunks answer in
milliseconds), supports deletion, and has no tuning knobs that could silently lower recall.

The index is DERIVED data. MySQL (rag_documents / rag_chunks) is the source of truth: if the file is lost, damaged or was
built with another embedding model, `RagService.reindex()` rebuilds it from the stored chunk text. Files are written to a
temporary name and renamed, so a crash never leaves a half-written index. All methods are thread-safe.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

META_VERSION = 1


class IndexMismatch(RuntimeError):
    """The saved index was built with a different model/dimension (or is unreadable): it must be rebuilt."""


class FaissStore:
    def __init__(self, directory: str | Path, dim: int, model_name: str):
        import faiss                                           # imported here so the app starts without faiss when RAG is off
        self._faiss = faiss
        self.dir, self.dim, self.model_name = Path(directory), dim, model_name
        self.index_path, self.meta_path = self.dir / "faiss.index", self.dir / "faiss.meta.json"
        self._lock = threading.RLock()
        self.index = self._new()

    def _new(self):
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dim))

    # ------------------------------------------------------------ persistence
    def load(self) -> bool:
        """True if a compatible index was loaded; False if there is none yet. Raises IndexMismatch if it cannot be used."""
        with self._lock:
            if not self.index_path.exists() or not self.meta_path.exists():
                self.index = self._new()
                return False
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                loaded = self._faiss.read_index(str(self.index_path))
            except Exception:
                raise IndexMismatch("The saved vector index is unreadable.") from None
            if meta.get("version") != META_VERSION or meta.get("model") != self.model_name or meta.get("dim") != self.dim or loaded.d != self.dim:
                raise IndexMismatch("The saved vector index was built with a different embedding model.")
            self.index = loaded
            return True

    def save(self) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp_index, tmp_meta = self.index_path.with_suffix(".index.tmp"), self.meta_path.with_suffix(".json.tmp")
            self._faiss.write_index(self.index, str(tmp_index))
            tmp_meta.write_text(json.dumps({"version": META_VERSION, "model": self.model_name, "dim": self.dim,
                                            "metric": "cosine (inner product of unit vectors)", "vectors": int(self.index.ntotal)}),
                                encoding="utf-8")
            os.replace(tmp_index, self.index_path)
            os.replace(tmp_meta, self.meta_path)

    def reset(self) -> None:
        with self._lock:
            self.index = self._new()

    # ------------------------------------------------------------ content
    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def ids(self) -> set[int]:
        with self._lock:
            if self.index.ntotal == 0:
                return set()
            return {int(i) for i in self._faiss.vector_to_array(self.index.id_map)}

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        if len(ids) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError("vector dimension does not match the index")
        with self._lock:
            self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            return int(self.index.remove_ids(np.asarray(ids, dtype=np.int64)))

    def search(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        """The `k` nearest chunk ids and their cosine scores, best first."""
        with self._lock:
            if self.index.ntotal == 0 or k <= 0:
                return [], []
            scores, ids = self.index.search(np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32), min(k, self.index.ntotal))
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        return [i for i, _ in pairs], [s for _, s in pairs]
