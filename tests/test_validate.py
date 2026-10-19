from datetime import date

from app.validation.rules import validate_invoice, validation_risk

TODAY = date(2026, 10, 1)
GOOD = {"invoice_number": "INV-12345", "invoice_date": "2026-03-01", "due_date": "2026-03-31",
        "currency": "USD", "total_amount": 1250.5}


def test_clean_invoice_has_no_issues():
    assert validate_invoice(GOOD, TODAY) == []
    assert validation_risk([]) == 0.0


def test_due_before_invoice_date():
    issues = validate_invoice({**GOOD, "due_date": "2026-02-01"}, TODAY)
    assert any("before invoice date" in i.message for i in issues)


def test_future_date_negative_amount_bad_currency():
    msgs = " ".join(i.message for i in validate_invoice(
        {**GOOD, "invoice_date": "2027-01-01", "total_amount": -5, "currency": "XYZ"}, TODAY))
    assert "future" in msgs and "positive" in msgs and "Unsupported currency" in msgs


def test_invalid_date_string():
    assert any(i.field == "due_date" for i in validate_invoice({**GOOD, "due_date": "31/03/2026"}, TODAY))


def test_missing_fields_are_not_double_penalised():
    assert validate_invoice({"invoice_number": None, "total_amount": None}, TODAY) == []
