"""Mailbox ingestion: MIME parsing, the IMAP polling loop (against a fake server), and one
end-to-end pass through the real API."""
from collections import defaultdict
from datetime import datetime, timezone
from email.message import EmailMessage

import pytest
from fastapi.testclient import TestClient

from app.db import Base, Email, Invoice, SessionLocal, engine
from app.extraction.regex_extractor import extract_fields
from app.ingestion.mailbox import (MailboxPoller, Rejected, Unavailable, build_payload, html_to_text,
                                   parse_internaldate)
from app.main import create_app
from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import Pipeline
from app.pipeline.decision import DecisionStage
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage
from scripts.send_test_emails import render_pdf

INVOICE_TEXT = ("Vendor: Acme Supplies Ltd\nInvoice INV-55555\nInvoice Date: 2026-04-01\n"
                "Due Date: 2026-04-30\nTotal Amount: USD 980.00")


def make_raw(subject="Invoice INV-55555", body="Hello", message_id="<abc@mail.example>", pdf: bytes | None = None,
             html: str | None = None, extra_headers: dict | None = None) -> bytes:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "Billing <billing@acme.example>", "ap@me.example", subject
    if message_id:
        msg["Message-ID"] = message_id
    for k, v in (extra_headers or {}).items():
        msg[k] = v
    if html is not None:
        msg.set_content("plain fallback")
        msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(body)
    if pdf is not None:
        msg.add_attachment(pdf, maintype="application", subtype="pdf", filename="invoice.pdf")
    return msg.as_bytes()


# ------------------------------------------------------------------ parsing
def test_plain_text_email():
    p = build_payload(make_raw(body="Total Amount: USD 10.00"))
    assert p["message_id"] == "abc@mail.example"
    assert p["sender"] == "billing@acme.example"
    assert p["subject"] == "Invoice INV-55555"
    assert "Total Amount: USD 10.00" in p["body"]


def test_html_only_email_is_converted_and_scripts_dropped():
    html = ("<html><style>p{color:red}</style><script>alert(1)</script><body><p>Invoice INV-55555</p>"
            "<table><tr><td>Total Amount:</td><td>USD 980.00</td></tr></table></body></html>")
    body = build_payload(make_raw(html=html))["body"]
    assert "alert" not in body and "color:red" not in body
    assert "Total Amount: USD 980.00" in body.replace("\n", " ")


def test_pdf_attachment_text_reaches_the_extractor():
    p = build_payload(make_raw(body="Please find the attached invoice.", pdf=render_pdf(INVOICE_TEXT)))
    assert "--- attachment: invoice.pdf ---" in p["body"]
    f = extract_fields(p["subject"], p["body"])
    assert f["total_amount"].value == 980.0 and f["invoice_number"].value == "INV-55555"


def test_unreadable_attachments_are_noted_not_fatal():
    blank_pdf = render_pdf("")                       # a PDF with no text, like a scan
    p = build_payload(make_raw(body="see attached", pdf=blank_pdf))
    assert "attachment unreadable: invoice.pdf" in p["body"] and "see attached" in p["body"]

    msg = EmailMessage()
    msg["From"], msg["Subject"], msg["Message-ID"] = "a@b.example", "scan", "<x@y>"
    msg.set_content("scan attached")
    msg.add_attachment(b"\x89PNG....", maintype="image", subtype="png", filename="scan.png")
    assert "attachment not read: scan.png" in build_payload(msg.as_bytes())["body"]


def test_missing_message_id_gets_a_stable_derived_one():
    raw = make_raw(message_id=None)
    a, b = build_payload(raw)["message_id"], build_payload(raw)["message_id"]
    assert a == b and a.startswith("sha256-")
    assert a != build_payload(make_raw(message_id=None, body="different"))["message_id"]


def test_folded_subject_is_collapsed():
    raw = make_raw(subject="Invoice INV-55555 due").replace(b"Subject: Invoice INV", b"Subject: Invoice\n INV")
    assert b"Invoice\n INV" in raw
    assert build_payload(raw)["subject"] == "Invoice INV-55555 due"


def test_useless_plain_stub_does_not_hide_the_html_body():
    html = "<p>Invoice INV-55555</p><p>Total Amount: USD 980.00</p>"
    assert "980.00" in build_payload(make_raw(html=html))["body"]


# ------------------------------------------------------------------ arrival time
def test_parse_internaldate_converts_to_utc():
    meta = b'12 (UID 12 INTERNALDATE " 4-Oct-2026 10:15:00 +0530" BODY[] {99}'
    assert parse_internaldate(meta) == datetime(2026, 10, 4, 4, 45, tzinfo=timezone.utc)
    assert parse_internaldate(b"nothing here") is None
    assert parse_internaldate(b'INTERNALDATE "garbage"') is None


def test_server_arrival_time_beats_a_forged_date_header():
    raw = make_raw(extra_headers={"Date": "Mon, 01 Jan 2001 09:00:00 +0000"})
    server_time = datetime(2026, 10, 4, 4, 45, tzinfo=timezone.utc)
    assert build_payload(raw, server_time)["received_at"].startswith("2026-10-04T04:45")
    assert build_payload(raw)["received_at"].startswith("2001-01-01")      # only a fallback


# ------------------------------------------------------------------ polling (fake IMAP server)
class FakeIMAP:
    """Implements just the imaplib calls the poller uses."""

    def __init__(self, messages: dict[bytes, bytes], date_after_literal=False):
        self.messages, self.flags, self.after = messages, defaultdict(set), date_after_literal

    def select(self, folder):
        return "OK", [b"1"]

    def uid(self, cmd, *args):
        if cmd == "search":
            hit = [u for u in self.messages if not self.flags[u]]
            return "OK", [b" ".join(hit)]
        if cmd == "fetch":
            uid, raw = args[0], self.messages[args[0]]
            date = b'INTERNALDATE "04-Oct-2026 10:15:00 +0000"'
            if self.after:
                return "OK", [(b"1 (UID %b BODY[] {%d}" % (uid, len(raw)), raw), b" " + date + b")"]
            return "OK", [(b"1 (UID %b %b BODY[] {%d}" % (uid, date, len(raw)), raw), b")"]
        if cmd == "store":
            self.flags[args[0]].add(args[2].strip("()"))
            return "OK", [b""]
        raise AssertionError(cmd)


def two_messages():
    return {b"1": make_raw(message_id="<one@x>"), b"2": make_raw(message_id="<two@x>", subject="Other")}


def test_processed_messages_are_marked_seen_and_not_fetched_again():
    imap, seen = FakeIMAP(two_messages()), []
    poller = MailboxPoller(imap, lambda p: seen.append(p) or {"decision": "auto_approved"}, log=lambda *_: None)
    s1 = poller.poll_once()
    assert s1.processed == 2 and s1.decisions["auto_approved"] == 2
    assert all("\\Seen" in imap.flags[u] for u in imap.messages)
    assert poller.poll_once().processed == 0 and len(seen) == 2


def test_internaldate_after_the_literal_is_still_found():
    got = []
    MailboxPoller(FakeIMAP(two_messages(), date_after_literal=True),
                  lambda p: got.append(p) or {"decision": "x"}, log=lambda *_: None).poll_once()
    assert got[0]["received_at"].startswith("2026-10-04T10:15")


def test_api_down_leaves_mail_unread_and_the_next_cycle_retries():
    imap, state = FakeIMAP(two_messages()), {"up": False}

    def submit(p):
        if not state["up"]:
            raise Unavailable("connection refused")
        return {"decision": "auto_approved"}

    poller = MailboxPoller(imap, submit, log=lambda *_: None)
    s1 = poller.poll_once()
    assert s1.stopped_early and s1.processed == 0 and not any(imap.flags.values())
    state["up"] = True
    assert poller.poll_once().processed == 2


def test_rejected_message_is_flagged_and_does_not_block_the_rest():
    imap = FakeIMAP(two_messages())

    def submit(p):
        if p["subject"] == "Other":
            raise Rejected("422")
        return {"decision": "human_review"}

    s = MailboxPoller(imap, submit, log=lambda *_: None).poll_once()
    assert s.processed == 1 and s.flagged == 1
    assert "\\Flagged" in imap.flags[b"2"] and "\\Seen" in imap.flags[b"1"]
    assert MailboxPoller(imap, submit, log=lambda *_: None).poll_once().processed == 0


def test_unexpected_error_flags_the_message_instead_of_crashing_the_cycle():
    imap = FakeIMAP(two_messages())

    def submit(p):
        if p["subject"] == "Other":
            raise ValueError("bug while handling this email")
        return {"decision": "auto_approved"}

    s = MailboxPoller(imap, submit, log=lambda *_: None).poll_once()
    assert s.flagged == 1 and s.processed == 1 and "\\Flagged" in imap.flags[b"2"]


# ------------------------------------------------------------------ end to end through the real API
def light_pipeline():
    return Pipeline([ExtractStage(), ValidateStage(), DuplicateStage(),
                     AnomalyStage(model_path="no_such_model.joblib"), RiskStage(), DecisionStage()])


@pytest.fixture
def api():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(create_app(light_pipeline)) as c:
        yield c


def api_submit(client):
    def submit(payload):
        r = client.post("/emails", json=payload)
        if r.status_code >= 500:
            raise Unavailable(r.text)
        if r.status_code >= 400:
            raise Rejected(r.text)
        return r.json()
    return submit


def test_pdf_invoice_flows_from_mailbox_to_database_once(api):
    imap = FakeIMAP({b"7": make_raw(message_id="<pdf1@x>", body="Please find the attached invoice.",
                                    pdf=render_pdf(INVOICE_TEXT))})
    stats = MailboxPoller(imap, api_submit(api), log=lambda *_: None).poll_once()
    assert stats.processed == 1

    with SessionLocal() as s:
        inv = s.query(Invoice).one()
        email = s.query(Email).one()
    assert inv.total_amount == 980.0 and inv.invoice_number == "INV-55555" and inv.vendor_name == "Acme Supplies Ltd"
    assert email.received_at == datetime(2026, 10, 4, 10, 15)       # server arrival time, stored as naive UTC

    # Same message delivered again (e.g. crash before it was marked seen): no second invoice.
    imap.flags.clear()
    MailboxPoller(imap, api_submit(api), log=lambda *_: None).poll_once()
    with SessionLocal() as s:
        assert s.query(Invoice).count() == 1


def test_api_converts_offset_received_at_to_utc(api):
    r = api.post("/emails", json={"message_id": "tz-1", "subject": "hi", "body": "hello",
                                  "received_at": "2026-01-05T10:00:00+05:30"})
    assert r.status_code == 200
    with SessionLocal() as s:
        assert s.query(Email).filter_by(message_id="tz-1").one().received_at == datetime(2026, 1, 5, 4, 30)


def test_html_to_text_keeps_table_cells_apart():
    assert "Total Amount: USD 5" in html_to_text("<table><tr><td>Total Amount:</td><td>USD 5</td></tr></table>")
