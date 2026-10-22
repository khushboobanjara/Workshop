"""Structure-aware chunker. Pure functions: text segments in, `Chunk`s out.

  * Headings (Markdown "#", numbered "3.2 Approval limits", ALL-CAPS lines, Word heading styles that extract.py turned into
    "#") become the chunk's `section`, and stay inside the chunk text so a retrieved passage says what it belongs to.
  * Paragraphs are packed whole into chunks of about `size` characters. A paragraph longer than `size` is split at sentence
    ends, and a "sentence" longer than `size` at word boundaries, so no chunk runs away in size and no text is dropped.
  * Consecutive chunks of one section overlap by about `overlap` characters, so an answer that straddles a boundary is
    still contained in one chunk. A chunk is therefore at most `size + overlap + min_chars` characters (a piece shorter
    than `min_chars` is folded into its neighbour instead of being indexed as noise).
  * Each chunk records the page it starts on (PDFs).
Nothing is summarised or rewritten: every character of the input (apart from whitespace) lands in at least one chunk.
"""
import re
from dataclasses import dataclass

from app.rag.extract import Segment

_MD_HEADING = re.compile(r"^#{1,6}\s+(\S.*?)\s*#*\s*$")
_NUMBERED = re.compile(r"^(?:(?:\d+(?:\.\d+){0,3}[.)]?)|(?:[A-Z][.)])|(?:(?:Section|Article|Clause|Chapter|Part)\s+\w+[.:]?))\s+(\S.*)$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,3}[.)]|[a-z][.)])\s+\S")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[•\-])")
_BLANK = re.compile(r"\n\s*\n")
MAX_HEADING_CHARS = 100


@dataclass
class Chunk:
    index: int                      # 1-based position in the document
    text: str
    page: int | None
    section: str | None


@dataclass
class _Unit:
    text: str
    page: int | None
    section: str | None
    heading: bool = False


def heading_of(line: str) -> str | None:
    """The heading text if `line` is a heading, else None. Deliberately conservative: a numbered LIST ITEM that is a full
    sentence ("1. Submit invoices within 30 days.") is not a heading."""
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS:
        return None
    m = _MD_HEADING.match(s)
    if m:
        return m.group(1).strip()
    if s[-1] in ".,;:!?" or len(s.split()) > 12:
        return None
    if _NUMBERED.match(s):
        return s
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and s.upper() == s and len(s) <= 80:
        return s
    return None


def _paragraph_text(lines: list[str]) -> str:
    """Soft-wrapped lines (typical of PDF text) are joined with a space; list items and table rows keep their line breaks."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif _LIST_ITEM.match(ln) or " | " in ln or " | " in out.rsplit("\n", 1)[-1]:
            out += "\n" + ln
        else:
            out += " " + ln
    return out


def _units(segments: list[Segment]) -> list[_Unit]:
    units: list[_Unit] = []
    section: str | None = None
    for seg in segments:
        for block in _BLANK.split(seg.text):
            pending: list[str] = []
            lines = block.split("\n")

            def flush():
                text = _paragraph_text(pending)
                pending.clear()
                if text:
                    units.append(_Unit(text, seg.page, section))

            for line in lines:
                h = heading_of(line)
                if h:
                    flush()
                    section = h[:200]
                    units.append(_Unit(h, seg.page, section, heading=True))
                else:
                    pending.append(line)
            flush()
    # a heading is glued to the paragraph that follows it, so it never ends a chunk on its own
    merged: list[_Unit] = []
    carry: _Unit | None = None
    for u in units:
        if u.heading:
            carry = _Unit((carry.text + "\n" + u.text) if carry else u.text, carry.page if carry else u.page, u.section, True)
            continue
        if carry:
            u = _Unit(carry.text + "\n" + u.text, carry.page, u.section)
            carry = None
        merged.append(u)
    if carry:
        merged.append(carry)
    return merged


def _hard_split(text: str, size: int) -> list[str]:
    """Words packed up to `size`; a single token longer than `size` is sliced."""
    out, cur = [], ""
    for word in text.split():
        while len(word) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:size])
            word = word[size:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= size:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _split_long(text: str, size: int) -> list[str]:
    """Pieces of at most `size` characters, split at line ends first, then sentence ends, then words."""
    if len(text) <= size:
        return [text]
    pieces, cur = [], ""
    for part in (p for line in text.split("\n") for p in _SENTENCE_END.split(line)):
        part = part.strip()
        if not part:
            continue
        if len(part) > size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_hard_split(part, size))
        elif not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= size:
            cur += " " + part
        else:
            pieces.append(cur)
            cur = part
    if cur:
        pieces.append(cur)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """The last ~`overlap` characters of `text`, starting at a sentence or word boundary (never mid-word)."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:]
    m = _SENTENCE_END.search(tail)
    if m:
        return tail[m.end():].strip()
    sp = tail.find(" ")
    return tail[sp + 1:].strip() if sp != -1 else ""


def chunk_segments(segments: list[Segment], size: int = 900, overlap: int = 150, min_chars: int = 40) -> list[Chunk]:
    size, overlap = max(200, size), max(0, min(overlap, size // 3))
    units: list[_Unit] = []
    for u in _units(segments):
        for piece in _split_long(u.text, size):
            units.append(_Unit(piece, u.page, u.section, u.heading))

    # (text, page, label, ended_by_size, first section, last section). The label lists EVERY section the chunk touches, so a
    # chunk that spans "1. Purpose" and "2. Approval limits" is not cited as just "1. Purpose".
    chunks: list[tuple[str, int | None, str | None, bool, str | None, str | None]] = []
    body: list[str] = []
    secs: list[str] = []
    body_len, page, section = 0, None, None

    def emit(by_size: bool):
        nonlocal body, body_len, secs
        text = "\n\n".join(body).strip()
        if text:
            chunks.append((text, page, (" / ".join(secs)[:200] or None), by_size, secs[0] if secs else None, secs[-1] if secs else None))
        body, body_len, secs = [], 0, []

    for u in units:
        sep = 2 if body else 0
        if body and (body_len + sep + len(u.text) > size or (u.section != section and body_len >= size * 0.35)):
            by_size = u.section == section
            emit(by_size)
            sep = 0
        if not body:
            page, section = u.page, u.section
        body.append(u.text)
        body_len += sep + len(u.text)
        if u.section and u.section not in secs:
            secs.append(u.section)
    if body:
        emit(False)

    # tiny trailing/leading chunks are folded into a neighbour instead of being indexed as noise
    folded: list[list] = []
    for text, pg, label, by_size, first, last in chunks:
        if folded and len(text) < min_chars:
            folded[-1][0] += "\n\n" + text
            continue
        folded.append([text, pg, label, by_size, first, last])

    out: list[Chunk] = []
    for i, (text, pg, label, _, first, _last) in enumerate(folded):
        prefix = ""
        if i > 0 and overlap and folded[i - 1][3] and folded[i - 1][5] == first:
            prefix = _overlap_tail(folded[i - 1][0], overlap)
        out.append(Chunk(len(out) + 1, (prefix + "\n" + text) if prefix else text, pg, label))
    return out


def embedding_text(chunk: Chunk, title: str = "") -> str:
    """What is embedded: the passage with its document title and section as context. The stored/displayed text is unchanged."""
    parts = [title.strip()] if title and title.strip() else []
    if chunk.section and not chunk.text.startswith(chunk.section):
        parts.append(chunk.section)
    parts.append(chunk.text)
    return "\n".join(parts)
