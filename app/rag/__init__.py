"""Retrieval-augmented generation (RAG) over uploaded company documents. Free and local end to end.

  access.py      who may retrieve which document (min_role levels; pure)
  extract.py     file bytes -> text segments: PDF (pdfplumber), DOCX (python-docx), TXT / MD / CSV / HTML (standard library)
  sanitize.py    cleaning of extracted text, and the quarantine of passages that look like instructions to an AI (pure)
  chunking.py    structure-aware chunker: headings, pages, sentence-aware packing with overlap (pure)
  embeddings.py  Sentence Transformers (local, CPU) behind a small interface; a lexical "hashing" stand-in for tests
  index.py       FAISS vector index: exact cosine search, delete by id, atomic persistence
  service.py     ingest / delete / search / reindex: MySQL is the source of truth, FAISS is derived and rebuildable
  grounding.py   citation parsing and the deterministic answer checks (pure)
  bridge.py      gives the chat model a `search_documents` tool next to the MCP tools, with RBAC, rate limit and audit
  audit.py       audit_log rows for uploads, deletions, re-indexing and searches
  routes.py      /rag/* HTTP API and start-up safety checks

Live invoice and operational data does NOT come from here: it still comes from the MCP tools. Documents answer "what does
our policy / contract / procedure say"; MCP answers "what is in the database right now". Off unless OPSPILOT_RAG_ENABLED=true.
"""
