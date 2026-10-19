from pathlib import Path
from typing import BinaryIO

import pdfplumber


def read_pdf_text(source: str | Path | BinaryIO, max_pages: int | None = None) -> str:
    """Extract text from a text-based PDF (a path or an open binary stream).
    Scanned PDFs need OCR (a later upgrade); they raise ValueError so callers can flag them."""
    with pdfplumber.open(source) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        text = "\n".join((page.extract_text() or "") for page in pages)
    if not text.strip():
        raise ValueError("No extractable text in PDF (scanned PDF? OCR needed)")
    return text
