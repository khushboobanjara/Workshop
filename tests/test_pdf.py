import pytest

reportlab = pytest.importorskip("reportlab")

from reportlab.pdfgen import canvas

from app.extraction.regex_extractor import extract_fields
from app.ingestion.pdf_reader import read_pdf_text


def test_pdf_text_flows_into_extractor(tmp_path):
    p = tmp_path / "inv.pdf"
    c = canvas.Canvas(str(p))
    y = 800
    for line in ["Vendor: Acme Supplies Ltd", "Invoice INV-55555", "Invoice Date: 2026-04-01",
                 "Due Date: 2026-04-30", "Total Amount: USD 980.00"]:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    f = extract_fields("", read_pdf_text(p))
    assert f["invoice_number"].value == "INV-55555" and f["total_amount"].value == 980.0
