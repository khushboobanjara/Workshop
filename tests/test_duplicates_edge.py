from app.duplicates.matcher import find_duplicate


def test_missing_vendor_never_matches_missing_vendor():
    hist = [{"id": 1, "vendor_name": None, "invoice_number": None, "invoice_date": "2026-03-01", "total_amount": 100.0}]
    inv = {"vendor_name": None, "invoice_number": None, "invoice_date": "2026-03-01", "total_amount": 100.0}
    assert find_duplicate(inv, hist) is None
