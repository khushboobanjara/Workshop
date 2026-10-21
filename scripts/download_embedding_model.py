"""Download (and cache) the free embedding model so document search works offline afterwards.   python -m scripts.download_embedding_model
Uses OPSPILOT_RAG_EMBEDDING_MODEL (default sentence-transformers/all-MiniLM-L6-v2). Cache location: HF_HOME (default ~/.cache/huggingface)."""
import sys

from app.config import settings


def main() -> int:
    from sentence_transformers import SentenceTransformer
    name = settings.rag_embedding_model
    model = SentenceTransformer(name, device="cpu")
    vec = model.encode(["download check"], normalize_embeddings=True)
    print(f"Model '{name}' is ready ({vec.shape[1]} dimensions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
