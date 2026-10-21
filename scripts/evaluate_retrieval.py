"""Measure retrieval quality and pick OPSPILOT_RAG_MIN_SCORE with evidence instead of a guess.

  python -m scripts.evaluate_retrieval docs/sample_documents/eval_questions.jsonl [--embedder sentence-transformers|hashing]

The JSONL file has one object per line:  {"question": "...", "expect": "AP policy"}  (a substring of the expected document's title)
or {"question": "...", "expect": null} for a question the documents must NOT answer. It indexes the documents found next to
the questions file (*.md/*.txt/...) in a throw-away index (your real index and database are untouched) and reports
  hit@1 / hit@3 / MRR for answerable questions, the cosine scores of right and wrong top hits, and for several thresholds how many
  answerable questions would be kept and how many unanswerable ones would wrongly return a passage.
Choose the lowest threshold that keeps unanswerable questions at ~0 returned. Re-run it after changing model, chunk size or documents."""
import argparse
import json
import statistics as st
import sys
import tempfile
from pathlib import Path

from app.config import settings
from app.rag.chunking import Chunk, chunk_segments, embedding_text
from app.rag.embeddings import HashingEmbedder, build_embedder
from app.rag.extract import SUPPORTED, extract
from app.rag.index import FaissStore
from app.rag.sanitize import clean_extracted
from app.rag.extract import Segment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--embedder", default=settings.rag_embedder)
    a = ap.parse_args()
    qfile = Path(a.questions)
    cases = [json.loads(l) for l in qfile.read_text(encoding="utf-8").splitlines() if l.strip()]
    settings.rag_embedder = a.embedder
    emb = build_embedder(settings)
    docs = sorted(p for p in qfile.parent.iterdir() if p.suffix.lower() in SUPPORTED)
    texts, meta = [], []
    for d in docs:
        ex = extract(d.name, d.read_bytes())
        chunks = chunk_segments([Segment(clean_extracted(s.text), s.page) for s in ex.segments], settings.rag_chunk_chars,
                                settings.rag_chunk_overlap_chars, settings.rag_min_chunk_chars)
        for c in chunks:
            texts.append(embedding_text(c, d.stem.replace("_", " ")))
            meta.append(d.stem.replace("_", " "))
    print(f"{len(docs)} documents, {len(texts)} passages, model {emb.name}")
    with tempfile.TemporaryDirectory() as tmp:
        store = FaissStore(tmp, emb.dim, emb.name)
        store.add(list(range(len(texts))), emb.embed_documents(texts))
        ranks, good, bad, none_top = [], [], [], []
        for c in cases:
            ids, scores = store.search(emb.embed_query(c["question"]), 10)
            titles = [meta[i] for i in ids]
            if c.get("expect") is None:
                none_top.append(scores[0] if scores else 0.0)
                continue
            rank = next((i + 1 for i, t in enumerate(titles) if c["expect"].lower() in t.lower()), None)
            ranks.append(rank)
            (good if rank == 1 else bad).append(scores[0])
    answerable = [r for r in ranks]
    if answerable:
        print(f"answerable questions: {len(answerable)}   hit@1 {sum(r == 1 for r in answerable) / len(answerable):.0%}   "
              f"hit@3 {sum(r is not None and r <= 3 for r in answerable) / len(answerable):.0%}   "
              f"MRR {sum(1 / r for r in answerable if r) / len(answerable):.2f}")
        print(f"  top score when right: median {st.median(good):.2f}" if good else "  (no correct top hits)")
    if none_top:
        print(f"unanswerable questions: {len(none_top)}   top score: median {st.median(none_top):.2f}  max {max(none_top):.2f}")
    print("\nthreshold  answerable kept   unanswerable wrongly answered")
    tops = [(s) for s in good + bad]
    for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        kept = sum(s >= t for s in tops)
        wrong = sum(s >= t for s in none_top)
        print(f"  {t:.2f}     {kept:>3}/{len(tops):<3}            {wrong:>3}/{len(none_top)}")
    print(f"\ncurrent OPSPILOT_RAG_MIN_SCORE = {settings.rag_min_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
