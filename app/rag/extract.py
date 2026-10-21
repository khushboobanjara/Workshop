"""File bytes -> text segments. Free libraries only: pdfplumber (already used for invoice PDFs), python-docx, and the
standard library for text, Markdown, CSV and HTML.

The file's NAME is only used to pick a parser and is then checked against its CONTENT (a PDF must start with %PDF, a DOCX
must be a ZIP holding word/document.xml, text must not be binary), so renaming a file cannot route it to the wrong parser.
Errors are `ExtractionError`s with a stable code, a message safe to show, and an HTTP status; they never echo file contents.

Not supported on purpose: scanned PDFs / images (OCR needs a heavyweight engine; the upload is refused with a clear message
instead of being indexed as empty), legacy .doc / .xls, password-protected files, macros and embedded objects (ignored).
"""
import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePath

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md", ".csv": "csv", ".html": "html", ".htm": "html"}
SUPPORTED_TEXT = "PDF, DOCX, TXT, MD, CSV and HTML"
MAX_DOCX_UNPACKED_BYTES = 100_000_000
MAX_DOCX_PART_BYTES = 50_000_000


class ExtractionError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 422):
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status


@dataclass
class Segment:
    text: str
    page: int | None = None


@dataclass
class Extracted:
    file_type: str
    segments: list[Segment] = field(default_factory=list)
    pages: int | None = None


def display_name(filename: str | None) -> str:
    """The upload's name reduced to a harmless label. It is never used as a path."""
    name = PurePath((filename or "").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:255] or "document"


def detect_type(filename: str, data: bytes) -> str:
    ext = PurePath(display_name(filename)).suffix.lower()
    kind = SUPPORTED.get(ext)
    if kind is None:
        hint = " Convert legacy .doc files to .docx first." if ext == ".doc" else ""
        raise ExtractionError("unsupported_type", f"Unsupported file type. Supported: {SUPPORTED_TEXT}.{hint}", 415)
    if not data:
        raise ExtractionError("empty", "The file is empty.", 422)
    head = data[:1024]
    mismatch = ExtractionError("type_mismatch", "The file's content does not match its extension.", 415)
    if kind == "pdf":
        if b"%PDF-" not in head:
            raise mismatch
    elif kind == "docx":
        if not head.startswith(b"PK"):
            raise mismatch
    else:
        if head.startswith((b"%PDF-", b"PK\x03\x04")) or b"\x00" in data[:8192]:
            raise mismatch
    return kind


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("cp1252", errors="replace")


def _check_size(total: int, max_chars: int) -> int:
    if total > max_chars:
        raise ExtractionError("too_large", f"The document has more than {max_chars:,} characters of text; split it into smaller files.", 413)
    return total


# ---------------------------------------------------------------- PDF
def _pdf(data: bytes, max_pages: int, max_chars: int) -> Extracted:
    import pdfplumber
    segments, total = [], 0
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            n = len(pdf.pages)
            if n > max_pages:
                raise ExtractionError("too_many_pages", f"The PDF has {n} pages; the limit is {max_pages}.", 413)
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                if hasattr(page, "flush_cache"):
                    page.flush_cache()
                if text:
                    total = _check_size(total + len(text), max_chars)
                    segments.append(Segment(text, i))
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("corrupt", "The PDF could not be read (it may be damaged or password-protected).") from None
    if not segments:
        raise ExtractionError("scanned", "No text could be extracted. This looks like a scanned PDF; OCR is not supported, "
                                         "so upload a text-based PDF or the original document.")
    return Extracted("pdf", segments, n)


# ---------------------------------------------------------------- DOCX
def _docx(data: bytes, max_chars: int) -> Extracted:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if "word/document.xml" not in {i.filename for i in infos}:
                raise ExtractionError("corrupt", "This is not a valid .docx file.", 415)
            if sum(i.file_size for i in infos) > MAX_DOCX_UNPACKED_BYTES or any(i.file_size > MAX_DOCX_PART_BYTES for i in infos):
                raise ExtractionError("too_large", "The .docx is unusually large once unpacked and was refused.", 413)
    except zipfile.BadZipFile:
        raise ExtractionError("corrupt", "This is not a valid .docx file.", 415) from None
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        document = docx.Document(io.BytesIO(data))
        blocks = []
        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, document)
                text = para.text.strip()
                if not text:
                    continue
                try:
                    style = para.style.name or ""
                except Exception:
                    style = ""
                m = re.match(r"Heading (\d)", style)
                if m:
                    text = "#" * min(int(m.group(1)), 6) + " " + text
                elif style == "Title":
                    text = "# " + text
                blocks.append(text)
            elif tag == "tbl":
                rows = []
                for row in Table(child, document).rows:
                    seen, cells = set(), []
                    for cell in row.cells:
                        if id(cell._tc) in seen:                       # merged cells repeat; keep one
                            continue
                        seen.add(id(cell._tc))
                        if cell.text.strip():
                            cells.append(cell.text.strip().replace("\n", " "))
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    blocks.append("\n".join(rows))
        _check_size(sum(len(b) for b in blocks), max_chars)
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("corrupt", "The .docx could not be read.") from None
    if not blocks:
        raise ExtractionError("empty", "The document contains no text.")
    return Extracted("docx", [Segment("\n\n".join(blocks))])


# ---------------------------------------------------------------- HTML
class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "noscript", "head", "template", "svg", "iframe"}
    BLOCK = {"p", "div", "br", "tr", "section", "article", "table", "ul", "ol", "blockquote", "pre", "header", "footer", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _html(data: bytes) -> Extracted:
    parser = _HtmlText()
    try:
        parser.feed(_decode(data))
        parser.close()
    except Exception:
        raise ExtractionError("corrupt", "The HTML could not be read.") from None
    text = re.sub(r"(?m)^\s*\|\s*", "", "".join(parser.out))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    if not text:
        raise ExtractionError("empty", "The document contains no text.")
    return Extracted("html", [Segment(text)])


# ---------------------------------------------------------------- CSV
def _csv(data: bytes, max_chars: int) -> Extracted:
    try:
        rows = [r for r in csv.reader(io.StringIO(_decode(data))) if any(c.strip() for c in r)]
    except csv.Error:
        raise ExtractionError("corrupt", "The CSV could not be read.") from None
    if not rows:
        raise ExtractionError("empty", "The document contains no text.")
    header, lines, total = [h.strip() for h in rows[0]], [], 0
    if len(rows) == 1:
        lines.append("; ".join(header))
    for r in rows[1:]:
        # Each row becomes one self-describing line ("column: value; ..."), so a retrieved row still says what its values mean.
        parts = [f"{h or f'column {i + 1}'}: {v.strip()}" for i, (h, v) in enumerate(zip(header, r)) if v.strip()]
        if parts:
            lines.append("; ".join(parts))
            total = _check_size(total + len(lines[-1]), max_chars)
    return Extracted("csv", [Segment("\n\n".join(lines))])


# ---------------------------------------------------------------- entry point
def extract(filename: str, data: bytes, *, max_pdf_pages: int = 300, max_chars: int = 2_000_000) -> Extracted:
    kind = detect_type(filename, data)
    if kind == "pdf":
        return _pdf(data, max_pdf_pages, max_chars)
    if kind == "docx":
        return _docx(data, max_chars)
    if kind == "html":
        return _html(data)
    if kind == "csv":
        return _csv(data, max_chars)
    text = _decode(data).strip()
    if not text:
        raise ExtractionError("empty", "The document contains no text.")
    _check_size(len(text), max_chars)
    return Extracted(kind, [Segment(text)])
