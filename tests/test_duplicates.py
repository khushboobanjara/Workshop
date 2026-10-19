from app.duplicates.matcher import find_duplicate

HIST = [{"id": 1, "vendor_name": "Acme Supplies Ltd", "invoice_number": "INV-12345",
         "invoice_date": "2026-03-01", "total_amount": 1250.50}]
BASE = {"vendor_name": "Acme Supplies Ltd", "invoice_number": "INV-12345",
        "invoice_date": "2026-03-01", "total_amount": 1250.50}


def test_exact_duplicate():
    m = find_duplicate(BASE, HIST)
    assert m and m.kind == "exact" and m.score == 1.0


def test_vendor_spelling_variation_still_matches():
    m = find_duplicate({**BASE, "vendor_name": "ACME Supplies Limited", "invoice_number": "inv 12345"}, HIST)
    assert m and m.kind == "exact"


def test_same_number_different_amount_is_fuzzy():
    m = find_duplicate({**BASE, "total_amount": 1500.0}, HIST)
    assert m and m.kind == "fuzzy"


def test_resubmitted_with_new_number():
    m = find_duplicate({**BASE, "invoice_number": "INV-99999"}, HIST)
    assert m and m.kind == "resubmitted"


def test_different_invoice_not_flagged():
    assert find_duplicate({**BASE, "invoice_number": "INV-55555", "total_amount": 99.0,
                           "invoice_date": "2026-07-01"}, HIST) is None
    assert find_duplicate({**BASE, "vendor_name": "Pixel Print Co"}, []) is None
