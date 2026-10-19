import csv
import io

from pydantic import SecretStr

from app import demo
from app.config import settings
from app.services.processing import _csv_safe
from tests.test_security import make_client  # noqa: F401  (fixture)


def seeded(make_client):
    c = make_client()
    demo.seed(c.app.state.pipeline)
    return c


def test_vendor_rollup_matches_the_invoices(make_client):
    c = seeded(make_client)
    vs = {v["vendor"]: v for v in c.get("/vendors").json()}
    invoices = c.get("/invoices?limit=500").json()
    assert sum(v["invoices"] for v in vs.values()) == len(invoices)
    assert vs["Acme Supplies Ltd"]["invoices"] == 6
    bright = vs["BrightCloud Hosting"]
    assert bright["duplicates"] == 1
    assert bright["total_spend"] < sum(i["total_amount"] for i in invoices if i["vendor_name"] == "BrightCloud Hosting")  # duplicate not counted as spend
    spends = [v["total_spend"] for v in vs.values()]
    assert spends == sorted(spends, reverse=True)


def test_audit_feed_is_newest_first_and_filterable(make_client):
    c = seeded(make_client)
    feed = c.get("/audit?limit=50").json()
    assert len(feed) == 50 and [f["id"] for f in feed] == sorted((f["id"] for f in feed), reverse=True)
    only = c.get("/audit?stage=duplicate_check&limit=500").json()
    assert only and {f["stage"] for f in only} == {"duplicate_check"}
    assert any(f["invoice_id"] for f in feed)
    assert c.get("/audit?limit=0").status_code == 422


def test_csv_export_has_one_row_per_invoice(make_client):
    c = seeded(make_client)
    r = c.get("/invoices.csv")
    assert r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == len(c.get("/invoices?limit=500").json()) and "invoice_number" in rows[0]


def test_csv_neutralises_spreadsheet_formulas(make_client):
    assert [_csv_safe(v) for v in ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "Acme", 5.0, None]] == ["'=1+1", "'+cmd", "'-2+3", "'@SUM(A1)", "Acme", 5.0, None]
    c = make_client()
    body = ("Invoice INV-77777\nVendor: =HYPERLINK(\"http://evil.example\",\"x\")\nInvoice Date: 2026-10-01\n"
            "Due Date: 2026-10-31\nTotal Amount: USD 100.00\n")
    assert c.post("/emails", json={"message_id": "evil-1", "subject": "Invoice INV-77777", "body": body}).status_code == 200
    row = next(csv.DictReader(io.StringIO(c.get("/invoices.csv").text)))
    assert row["vendor_name"].startswith("'=")


def test_settings_show_rules_but_never_secrets(make_client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr("s3cret-key"))
    monkeypatch.setattr(settings, "imap_password", SecretStr("mail-pass"))
    c = make_client()
    r = c.get("/settings", headers={"X-API-Key": "s3cret-key"})
    assert r.status_code == 200 and r.json()["auth_required"] is True
    assert "s3cret-key" not in r.text and "mail-pass" not in r.text and "imap" not in r.text.lower()
    t = r.json()["thresholds"]
    assert t["auto_approve_max_risk"] == settings.auto_approve_max_risk and "anomaly" in r.json()["risk_weights"]
    assert r.json()["validation_rules"]["max_payment_terms_days"] > 0


def test_new_endpoints_require_the_api_key_when_one_is_set(make_client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr("k"))
    c = make_client()
    for path in ["/vendors", "/audit", "/invoices.csv", "/settings"]:
        assert c.get(path).status_code == 401, path
        assert c.get(path, headers={"X-API-Key": "k"}).status_code == 200, path
