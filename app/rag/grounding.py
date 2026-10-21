"""Deterministic checks on an answer that uses document passages. Pure Python, no model calls, no I/O.

  * citations   every [DOC-3-C2] in the answer must name a passage that was actually retrieved in this turn; an answer
                that rests only on documents must cite at least one
  * support     every substantial sentence must be mostly made of words that occur in the data the model was shown
                (passages and database records). It catches an answer that wanders off into things nobody provided.

These are cheap, strict-but-blunt filters, not proof of truth: a sentence can reuse the right words and still say the wrong
thing. The model-based fact check (OPSPILOT_CHAT_FACT_CHECK) covers that, and every answer still shows its citations so a
person can open the passage. Tune or disable the support check with OPSPILOT_RAG_MIN_SUPPORT (0 = off).
"""
import re

from app.rag.embeddings import _STOP, _stem

_REF = re.compile(r"\bDOC-\d+-C\d+\b")
_ABSTAIN = re.compile(
    r"\b(?:do(?:es)? not|don't|doesn't|did not|didn't|cannot|can't|couldn't|unable|no (?:information|mention|record|passage|document|matching|relevant)|"
    r"not (?:in|stated|specified|available|found|provided|mentioned|covered|contain|include))", re.IGNORECASE)
_FILLER = _STOP | frozenset(
    "also been being both could does doing done each either even ever have having here into just like made make many more most much "
    "must only other over same shall should some such than that their them then there these they this those through under very "
    "what when where which while will with within without would your about above after again against because before below between "
    "can may might per via any all not but its our out use used uses using".split())
NO_DOCUMENT_EVIDENCE_TEXT = ("I couldn't find anything about that in the uploaded documents you have access to, so I can't answer it. "
                             "Try different keywords, or ask an administrator whether the relevant document has been uploaded.")


def cited_refs(text: str) -> list[str]:
    """Distinct citations in order of first appearance (bracketed or not)."""
    out: list[str] = []
    for m in _REF.finditer(text or ""):
        if m.group() not in out:
            out.append(m.group())
    return out


def invalid_citations(text: str, valid: set[str]) -> list[str]:
    return [r for r in cited_refs(text) if r not in valid]


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()) if w not in _FILLER]


def support_corpus(payloads) -> set[str]:
    """Stemmed words of every string AND key in the `data` of each tool payload the model was shown."""
    words: set[str] = set()

    def walk(node):
        if isinstance(node, str):
            words.update(_words(node))
        elif isinstance(node, dict):
            for k, v in node.items():
                words.update(_words(str(k).replace("_", " ")))
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    for p in payloads:
        walk((p or {}).get("data"))
    return words


def unsupported_sentences(answer: str, corpus: set[str], min_support: float, min_words: int = 5) -> list[str]:
    """Sentences whose content words are mostly absent from the corpus. Short sentences, abstentions ("the documents do not
    say...") and citation-only fragments are skipped; negative claims are left to the model-based fact check."""
    if min_support <= 0:
        return []
    bad: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+|\n+", _REF.sub(" ", answer or "")):
        sentence = re.sub(r"[\[\]]|^\s*(?:[-*•]|\d+[.)])\s+", " ", raw).strip()
        if not sentence or _ABSTAIN.search(sentence):
            continue
        words = _words(sentence)
        if len(words) < min_words:
            continue
        support = sum(1 for w in words if w in corpus) / len(words)
        if support < min_support:
            bad.append(sentence[:160])
    return bad[:5]
