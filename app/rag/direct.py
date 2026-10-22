"""Answer a document question WITHOUT a language model, by quoting. Pure Python.

Given the passages retrieved for a question, pick the sentences that contain most of the question's content words and return
them verbatim, each with its citation. Because the text is copied from the documents, it cannot hallucinate; and it costs no
tokens. It is deliberately modest: when no sentence covers enough of the question, `pick` returns None and the caller falls back
to one model call (see app/chat/service.py). Questions that need reasoning (a what-if, "explain", "compare") never come here.
"""
import re
from dataclasses import dataclass

from app.rag.chunking import _SENTENCE_END
from app.rag.embeddings import _STOP, _stem

_GENERIC = frozenset(
    "what which who whom when where how many much does did can could would should will please tell show give about policy policies "
    "company document documents say says said state states stated mention mentions according our your their any there here need needs "
    "want know like get have has had been being also".split())
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+")
INTRO = "According to the company documents:"
FOOTER = "(Quoted directly from the documents. Ask me to \"explain\" or \"summarise\" if you want an AI-written answer.)"


@dataclass
class Direct:
    text: str
    refs: list


def terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()) if len(w) > 2 and w not in _STOP and w not in _GENERIC}


MAX_QUOTE_CHARS = 260
_WINDOW_WORDS, _WINDOW_STEP = 32, 16


def _windows(s: str) -> list[str]:
    """A run of text with no sentence end (typically a table flattened by PDF extraction) is cut into overlapping windows of about
    32 words, each still a verbatim stretch of the document, marked with an ellipsis where it was cut."""
    words = s.split()
    out = []
    for i in range(0, max(1, len(words) - _WINDOW_WORDS + _WINDOW_STEP), _WINDOW_STEP):
        piece = " ".join(words[i:i + _WINDOW_WORDS])
        out.append(("\u2026 " if i else "") + piece + (" \u2026" if i + _WINDOW_WORDS < len(words) else ""))
    return out


def sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = _BULLET.sub("", line).strip()
        for s in _SENTENCE_END.split(line):
            s = s.strip()
            if len(s) < 25:
                continue
            out.extend(_windows(s) if len(s) > MAX_QUOTE_CHARS else [s])
    return out


def pick(question: str, passages: list[dict], min_overlap: float = 0.5, max_sentences: int = 3) -> Direct | None:
    """passages: dicts with ref, section, text, best first. None when nothing covers enough of the question."""
    q = terms(question)
    if len(q) < 2:                                   # too vague to match on words: let the model (or the user) refine it
        return None
    scored = []                                       # (score, own-word hits, -passage rank, -position, sentence, ref)
    for rank, p in enumerate(passages):
        context = terms(p.get("section") or "")
        for pos, s in enumerate(sentences(p.get("text") or "")):
            own = q & terms(s)
            hit = q & (terms(s) | context)
            if len(hit) >= 2:
                # ties (a heading can cover every term for every sentence under it) go to the sentence that itself uses more of the words
                scored.append((len(hit) / len(q), len(own), -rank, -pos, s, p["ref"]))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0][0]
    if best < min_overlap:
        return None
    chosen, seen = [], set()
    for sc, own, rank, pos, s, ref in scored:
        if sc < max(min_overlap, best - 0.15) or s in seen:
            continue
        seen.add(s)
        chosen.append((-rank, -pos, s, ref))
        if len(chosen) >= max_sentences:
            break
    chosen.sort()                                     # passage order, then position: reads like the document
    lines = [f"- {s} [{ref}]" for _, _, s, ref in chosen]
    return Direct("\n".join([INTRO, *lines, "", FOOTER]), list(dict.fromkeys(ref for *_, ref in chosen)))
