# Document search (RAG) for the AI assistant

The assistant ([CHAT.md](CHAT.md)) answers from two kinds of evidence, and never from memory:

| Question | Source | How |
|---|---|---|
| "What is invoice INV-1001? Which vendors cost most? What's in the review queue?" | **Live database** | MCP tools (unchanged, read-only, audited) |
| "What does our policy / contract / procedure say?" | **Uploaded documents** | `search_documents` tool: FAISS retrieval over chunks stored in MySQL |
| "Is INV-1001 within the approval limit in our AP policy?" | **Both** | The model calls both; the answer shows the invoice value from the records and the rule with its citation |

Everything is **free and local**: embeddings run on your CPU (Sentence Transformers), the vector index is the FAISS library, text extraction uses pdfplumber / python-docx / the standard library, and the only external call is the Groq chat model the assistant already used. No paid API, subscription, or billing-required service is involved. (Groq's free tier is rate limited; the first run also downloads the ~90 MB embedding model from Hugging Face's free public hub, once.)

## Architecture

```
 upload (admin)                                                      question (any signed-in user)
      │                                                                       │
      ▼                                                                       ▼
 extract text ──► clean + redact ──► chunk ──► quarantine ──► embed        POST /chat
 (PDF/DOCX/TXT/    (invisible chars,  (headings, pages,  instruction-   (Sentence      │ Groq model, with tools:
  MD/CSV/HTML)      card/IBAN/SSN)     overlap)          like passages    Transformers)  ├─ MCP tools  → live invoice / ops data
      │                                                                       │          └─ search_documents ─┐
      ▼                                                                       ▼                               │
 MySQL: rag_documents + rag_chunks (+ audit row, one transaction)      FAISS (vectors only)                   │
      ▲                         ▲                                           │ ids                             │
      │                         └───────── resolved through MySQL, filtered by status + the user's role ◄──────┘
      │
      └─ source of truth; FAISS is derived and rebuildable (POST /rag/reindex)

 answer checks:  figures ▸ citations ▸ word support ▸ (optional) model fact check ▸ fixed text when nothing was found
```

Code: `app/rag/` (see its `__init__.py` for a map). The chat loop is `app/chat/service.py`; documents are one more tool in it (`app/rag/bridge.py`), so the existing limits, verifier, audit and repair logic apply unchanged.

## Turning it on

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU build (skip if you already have torch)
pip install -r requirements.txt
python -m scripts.download_embedding_model                            # optional: fetch the model now instead of on first use
```

`.env` (in addition to the auth and chat settings the assistant already needs):

```
OPSPILOT_AUTH_ENABLED=true
OPSPILOT_CHAT_ENABLED=true
OPSPILOT_GROQ_API_KEY=gsk_...
OPSPILOT_RAG_ENABLED=true
```

Start the app, sign in as an admin, open **Documents**, upload a file and choose who may read it. Or from the command line:

```bash
python -m scripts.ingest_documents docs/sample_documents --min-role viewer
```

Then ask the assistant, for example *"How many approvers does a 15,000 USD invoice need?"*. The answer carries a citation such as `[DOC-1-C2]` and a "Checked against cited documents" badge; expand it to read the quoted passage.

The app **refuses to start** with RAG enabled if auth is off, demo mode is on, the embedder name is unknown, the data folder is not writable, FAISS is missing, or (in production) the test-only `hashing` embedder is selected.

## Supported documents

PDF (text-based), DOCX (headings and tables kept), TXT, Markdown, CSV (each row becomes a self-describing line), HTML (scripts and styles dropped). Not supported, and refused with a clear message rather than indexed as empty: scanned/image PDFs (no OCR), legacy `.doc`/`.xls`, password-protected files. The file type is checked against the file's *content*, not just its name.

Limits (all configurable, see below): 10 MB per file, 300 PDF pages, 2,000,000 characters of text, 500 documents, DOCX unpacked size 100 MB (zip-bomb guard). The same file content cannot be uploaded twice. The original file is **not kept**; only the cleaned text, as chunks.

## Chunking and retrieval

* Chunks are ~900 characters (about 220 tokens, inside MiniLM's 256-token window) built from whole paragraphs; long paragraphs split at sentence ends, then words. Neighbouring chunks of one section overlap by ~150 characters. A chunk therefore holds at most `chunk_chars + overlap` characters, and no text is dropped.
* Headings (Markdown, numbered, ALL-CAPS, Word heading styles) become the chunk's *section*; the PDF page is recorded. What is embedded is `title + section + text`; what is shown is the plain text.
* Search is exact cosine similarity (FAISS `IndexFlatIP` on unit vectors). A passage is only returned if its cosine similarity is at least `OPSPILOT_RAG_MIN_SCORE` (default 0.30); weaker matches are *not evidence*. Results are ranked with a small bonus for query words found verbatim, at most 3 per document, top 5.
* Search results are filtered by the document's visibility **after** the FAISS lookup, by resolving every hit through MySQL (see Security). The search keeps widening until it has enough visible hits or the ranking drops below the floor, so documents you may not see cannot crowd out ones you may.

### Choosing the relevance floor

0.30 is a sensible starting point for `all-MiniLM-L6-v2` but **has not been tuned on your documents**. Measure it:

```bash
python -m scripts.evaluate_retrieval docs/sample_documents/eval_questions.jsonl
```

It indexes the documents next to the question file in a throw-away index and prints hit@1, hit@3, MRR and, per threshold, how many answerable questions would be kept versus how many *unanswerable* questions would wrongly return a passage. Pick the lowest threshold that keeps the latter at about zero; write your own questions file for your own documents. A floor that is too high makes the assistant say "not in the documents" for things that are; too low lets weak matches through (the answer checks below still apply).

## How hallucinations are prevented

Layers, from the retrieval step to the answer. None is the only defence.

1. **No evidence, no answer.** If the document search finds nothing relevant and no database record backs the question, the model's text is *discarded* and a fixed message is returned ("I couldn't find anything about that in the uploaded documents you have access to…"). This does not depend on the model behaving. If the search itself is unavailable (model not loaded, audit write failed, rate limit), a different fixed message says so.
2. **Must retrieve first.** As before, a question about OpsPilot data with no tool lookup is sent back once and then refused. The prompt extends the scope to uploaded documents and forbids answering a document question from general knowledge.
3. **Citations.** Every statement from a passage must carry its server-issued ref (`[DOC-<document id>-C<passage number>]`). A ref that was not retrieved in this turn is rejected, and an answer that rests only on documents must cite at least one. Documents cannot plant refs: anything that looks like one is defused when the text is ingested.
4. **Figures, dates and identifiers.** The existing verifier requires every number, date, identifier and percentage in the answer to occur in what the model was actually shown, which now includes the passages.
   *What-if questions.* The verifier normally refuses amounts that appear only in the user's own message (a user cannot make "$5,000" true by typing it). For a what-if question such as "an invoice worth 12,000 with risk 0.10: may it be approved automatically?" the answer has to repeat those values, so in a turn that consulted **only documents and no database record**, numbers the user typed are accepted as *their premise*. Numbers the user did not type are still checked, and any turn that reads a database record keeps the strict rule. The prompt tells the model to call such values "the values you gave", not records.
5. **Word support** (new, deterministic, no model call). Each substantial sentence must be mostly made of words found in the passages and records the model was shown (`OPSPILOT_RAG_MIN_SUPPORT`, default 0.40; 0 turns it off). It catches answers that wander into content nobody provided. It is a blunt filter: a sentence can reuse the right words and still be wrong, and a faithful paraphrase with unusual wording can be flagged (then the model gets one chance to repair it).
6. **Optional model fact check** (`OPSPILOT_CHAT_FACT_CHECK=true`) now also sees the passages, so it can judge document claims.
7. **Show the source.** Cited passages are returned with the answer and displayed in the dashboard, so a person can check the claim.

A failed check gives the model one repair attempt (a missing citation is retried with the list of passage refs to choose from); if the repaired answer still fails, the user gets a "could not verify" message, never the unverified text. The message says *which kind* of check failed (figures not in the sources, a missing or invented citation, unsupported wording, a fact-check mismatch), and the chat audit row records the same kinds as `failed_checks` (kinds only, no text), so a withheld answer can be diagnosed and the settings tuned.

What this does **not** guarantee: that a retrieved passage is itself correct or current; that a cited passage *entails* the claim made (layers 5 and 6 reduce, not remove, that risk); that the right passage was retrieved (recall depends on the embedding model, chunking and the floor). Check important decisions against the cited source.

## Security

* **Authentication and RBAC.** All `/rag/*` routes and the `search_documents` tool use the application's own roles. Each document has `min_role`: `viewer` (every signed-in user), `reviewer` (reviewers and admins) or `admin`. Upload, delete and re-index are admin-only. The MCP system key and any unknown role see nothing.
* **Server-side filtering.** FAISS only supplies candidate ids. Each is resolved through MySQL, which checks the document's status and `min_role` against the caller's role. A stale, copied or tampered index can at worst miss passages; it cannot reveal a deleted or restricted document. Restricted passages never reach Groq for users who may not read them.
* **Audit.** Every upload (including refusals), delete, re-index and search writes an `audit_log` row (`entity_type=rag`), and each chat turn's row records the document searches and citations. Rows hold ids, counts, roles, the (shortened) query and a similarity score, **never document text or passages**. Search is **fail-closed**: if the audit row cannot be written, no passages are returned. Upload/delete audit rows commit in the same transaction as the change.
* **Untrusted text.** Documents are treated like inbound e-mail. Invisible/control characters are stripped; card numbers, IBANs and national-id patterns are redacted before storage; passages that read like instructions to an AI ("ignore previous instructions…") are *quarantined* (not indexed; the count is shown to admins); the tool result labels passage text as untrusted data and the prompt tells the model to ignore instructions inside it. Pattern matching will not catch every attack; the citation, support and figure checks and the fixed no-evidence answer still apply, and the assistant has no tools that change data.
* **Uploads.** Admin-only, size-limited per request (the global 1 MB body limit still applies everywhere else), type checked by content, file names never used as paths, files never written to disk (only parsed in memory), zip-bomb and page limits, rate limit on searches (`OPSPILOT_RAG_SEARCH_RATE_LIMIT_PER_MINUTE`, per user).
* **Data leaving the server.** When a user asks a question, the question and the passages their role may read that match it are sent to Groq, as database records already were. Do not upload documents you may not send to your LLM provider. Embeddings and the index never leave the machine.

## Operations

* **Hugging Face notice at start-up** ("unauthenticated requests to the HF Hub"): harmless. It only appears while the embedding model is loaded; for fewer network calls set `OPSPILOT_RAG_EMBEDDING_LOCAL_FILES_ONLY=true` once the model has been downloaded (a free `HF_TOKEN` is optional).

* **Backups:** MySQL is the source of truth; back it up as before. The FAISS folder (`OPSPILOT_RAG_DATA_DIR`, default `data/rag`) is derived data. If it is lost, corrupted, or built with a different embedding model, it is rebuilt automatically from the stored text on first use (or on demand with `POST /rag/reindex` / the **Rebuild index** button). On first use after a restart the index is also reconciled with MySQL: vectors with no row are dropped, rows with no vector are embedded and added.
* **Changing the model** (`OPSPILOT_RAG_EMBEDDING_MODEL`): restart; the mismatch is detected and the index is rebuilt. Models that expect prefixes (e5, bge) use `OPSPILOT_RAG_QUERY_PREFIX` / `OPSPILOT_RAG_DOCUMENT_PREFIX`. Re-run the evaluation script and re-tune the floor.
* **Offline / air-gapped:** run `python -m scripts.download_embedding_model` once while online (the Docker image does this at build time), then set `OPSPILOT_RAG_EMBEDDING_LOCAL_FILES_ONLY=true`, or point `OPSPILOT_RAG_EMBEDDING_MODEL` at a local folder.
* **Memory/CPU:** the model is loaded on first use (~0.5 GB RAM, a few seconds); `OPSPILOT_RAG_PRELOAD=true` loads it at start-up. Embedding a 50-page PDF takes seconds on a laptop CPU. Embedding calls are serialised per process.
* **Scale:** exact search is instantaneous up to tens of thousands of chunks (a few hundred documents). Beyond that, move to an approximate FAISS index (which cannot delete in place) or a vector server.
* **Several worker processes:** each worker keeps its own in-memory index; they notice each other's changes through the index file's timestamp and reconcile with MySQL, so an upload becomes searchable in the other workers within seconds. Access control never depends on this.
* **Docker:** the image uses CPU-only PyTorch and pre-downloads the model; `docker-compose.yml` keeps the index on a volume and passes the `OPSPILOT_RAG_*` variables. Auth and chat settings must be passed through the same way (the compose file only forwards the variables it lists).

## When the AI model is (and is not) used

The model is the most expensive and least predictable part, so the assistant uses it **last**. A question is handled by the first tier that can answer it (`app/chat/rules.py`, `app/rag/direct.py`, `app/chat/service.py`):

| Tier | Examples | Model calls |
|---|---|---|
| **1. Canned** | "hi", "thanks", "help" | 0 |
| **2. Fixed rules + MCP** | "how many invoices are pending review?", "show the review queue", "invoices from Acme", "rejected invoices", "top 5 vendors", "metrics", "which models are active?", "system status", "what can I access?", "show the audit log", "list users", "invoice 12", "audit trail for invoice 12" | 0: the server calls the right MCP tool and fills a template with the tool's own values |
| **3a. Document quote** | "What is the standard payment term?", "Can the AI Assistant approve an invoice?" | 0: the server searches the documents and quotes the sentences that answer, verbatim, with citations |
| **3b. Document + reasoning** | a what-if with numbers, "explain…", "compare…", "why…", "should…", or a quote that doesn't cover the question | 1: passages already in the prompt, no tool definitions, a short prompt (about 1/4 of the tokens) |
| **4. Model with tools** | anything mixing records and documents, ambiguous or multi-part questions, an invoice identifier plus a policy | 2 or more, as before |
| Declined | a question with nothing to do with OpsPilot (and, with documents, nothing found) | 0: a fixed message |

Rules are **conservative on purpose**: a question is routed only if every word of it is understood (known filler, a known keyword, a number or name in a known slot). A clause that is not understood ("last month", "over 5000", "and why") sends it to the model instead, so a wrongly routed question is rarer than an unrouted one. Role limits still apply: a role that may not ask gets a fixed refusal, still without a model call. Requests for the assistant to *change* something ("approve invoice 12", "can you delete invoices?") get a fixed read-only reply. If the model, forced to call a tool, answers in text instead (Groq's "Tool choice is required, but model did not call a tool"), the user gets a short "I couldn't turn that into a lookup" message instead of a 502. Quoted answers carry a footer saying how to get an AI-written one ("explain" or "summarise").

Settings: `OPSPILOT_CHAT_HARDCODE=false` turns tiers 1-3 off (everything goes to the model, as in the first version); `OPSPILOT_RAG_ANSWER_MODE=llm` keeps the server-side search but always uses one model call for documents instead of quoting; `OPSPILOT_RAG_DIRECT_MIN_OVERLAP` (0.5) is how much of the question's content words a quoted sentence must cover. Each answer shows how it was produced ("Instant answer", "Quoted from your documents", "1 AI call") and the chat audit row records `route`.

What this costs in quality: a quote is the document's own wording, so it can be terse or include a neighbouring clause, and it never reasons. If a question could be a data question you phrased loosely, the fixed rules may not recognise it, and it simply goes to the model.

## Groq's free-tier token limit ("HTTP 429 ... tokens per minute")

Groq's free tier caps **tokens per minute (TPM) per model**, and every model call re-sends the instructions, the tool definitions, the conversation so far and (for documents) the retrieved passages. Before the fixed-rules tiers existed, a document question made two or more calls, so with an 8,000 TPM model one question could use most of a minute, and a second question straight afterwards is refused with `HTTP 429`. What the application does about it:

* **Waits and retries.** Groq's 429 says how long to wait ("try again in 3.21s"). The client waits that long (plus a moment) and asks again, up to `OPSPILOT_CHAT_RATE_LIMIT_RETRIES` (2) times, for waits of at most `OPSPILOT_CHAT_RATE_LIMIT_MAX_WAIT_SECONDS` (12). Longer waits are reported to the user with the number of seconds instead of a vague "try again".
* **Sends fewer tokens.** Only the last `OPSPILOT_CHAT_HISTORY_TURNS` (6) messages are re-sent, each cut to `OPSPILOT_CHAT_HISTORY_CHARS` (500) characters; the document instructions are short; the search returns 4 passages of at most 1,000 characters by default.
* **What you can still do.** Click **New chat** between unrelated questions (history is re-sent every call); set `OPSPILOT_CHAT_FACT_CHECK=false` to save one model call per question (the figure, citation and word-support checks still run); choose a model whose free limits are higher (limits differ per model and change over time: see the Limits page of your Groq console); or move to Groq's paid tier, which is optional and not required by this project.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPSPILOT_RAG_ENABLED` | `false` | Master switch (needs auth, not demo mode) |
| `OPSPILOT_RAG_DATA_DIR` | `data/rag` | FAISS index folder |
| `OPSPILOT_RAG_EMBEDDER` | `sentence-transformers` | `hashing` = word-matching stand-in for tests; refused in production |
| `OPSPILOT_RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any Sentence Transformers model or local folder |
| `OPSPILOT_RAG_EMBEDDING_LOCAL_FILES_ONLY` | `false` | Never contact Hugging Face |
| `OPSPILOT_RAG_MIN_SCORE` | `0.30` | Cosine floor for a passage to count as evidence |
| `OPSPILOT_RAG_TOP_K` | `5` | Passages per search (max 8) |
| `OPSPILOT_RAG_MAX_CHUNKS_PER_DOCUMENT` | `3` | Diversity cap |
| `OPSPILOT_RAG_CHUNK_CHARS` / `_OVERLAP_CHARS` / `_MIN_CHUNK_CHARS` | `900` / `150` / `40` | Chunking |
| `OPSPILOT_RAG_MIN_SUPPORT` | `0.40` | Word-support check (0 = off) |
| `OPSPILOT_RAG_MAX_UPLOAD_BYTES` | `10000000` | Per file |
| `OPSPILOT_RAG_MAX_DOCUMENTS` / `_MAX_PDF_PAGES` / `_MAX_EXTRACTED_CHARS` | `500` / `300` / `2000000` | Limits |
| `OPSPILOT_RAG_SEARCH_RATE_LIMIT_PER_MINUTE` | `30` | Per user, chat and `/rag/search` together |
| `OPSPILOT_RAG_PRELOAD` | `false` | Load the model at start-up |

## HTTP API

| Route | Role | |
|---|---|---|
| `GET /rag/status` | any | enabled, visible document count (admins: model, passages, index health) |
| `GET /rag/documents` | any | documents your role may read |
| `POST /rag/documents` | admin | multipart `file`, `min_role` (required), optional `title` → 201, or 409 duplicate, 413 too large, 415 wrong type, 422 empty/scanned/quarantined, 503 model unavailable |
| `DELETE /rag/documents/{id}` | admin | removes text and vectors |
| `POST /rag/reindex` | admin | rebuild the vector index from stored text |
| `POST /rag/search` | any | `{query, limit}` → the passages the assistant would receive (audited, rate limited) |

## Tests

`tests/test_rag_*.py` (about 100 tests) run with no network and no model download: the vector index is real FAISS, documents and audit rows go through the real database layer, Groq is scripted, and embeddings use a deterministic word-hashing stand-in. They cover extraction (real generated PDF/DOCX), chunking invariants, role visibility, stale-index safety, quarantine, forged citations, fixed no-evidence answers, fail-closed audit, mixed database+document answers, and start-up refusals. **What they cannot show** is the semantic quality of the real Sentence Transformers model on your documents; use `scripts/evaluate_retrieval.py` for that.
