import pytest
from fastapi.testclient import TestClient

from app.db import Base, Invoice, SessionLocal, engine
from app.ingestion.folder import ingest_folder
from app.ingestion.mailbox import Rejected, Unavailable, build_payload
from app.main import create_app
from scripts.send_test_emails import build_email
from tests.test_mailbox import api_submit, light_pipeline  # noqa: F401  (reuse the light pipeline helper)

ROW = {"subject": "Invoice INV-55555",
       "body": ("Vendor: Acme Supplies Ltd\nInvoice INV-55555\nInvoice Date: 2026-04-01\n"
                "Due Date: 2026-04-30\nTotal Amount: USD 980.00")}


def write(tmp_path, name, pdf=False):
    (tmp_path / name).write_bytes(build_email(ROW, "billing@vendor.example", pdf).as_bytes())


def test_generated_eml_round_trips_through_the_parser(tmp_path):
    write(tmp_path, "a.eml", pdf=True)
    p = build_payload((tmp_path / "a.eml").read_bytes())
    assert p["message_id"].endswith("@opspilot.test") and p["received_at"] is not None
    assert "Total Amount: USD 980.00" in p["body"]


def test_processed_files_move_and_are_not_read_twice(tmp_path):
    write(tmp_path, "a.eml")
    write(tmp_path, "b.eml", pdf=True)
    seen = []
    stats = ingest_folder(tmp_path, lambda p: seen.append(p) or {"decision": "auto_approved"}, log=lambda *_: None)
    assert stats.processed == 2 and len(seen) == 2
    assert sorted(f.name for f in (tmp_path / "processed").iterdir()) == ["a.eml", "b.eml"]
    assert ingest_folder(tmp_path, lambda p: {"decision": "x"}, log=lambda *_: None).processed == 0


def test_api_down_leaves_files_in_place(tmp_path):
    write(tmp_path, "a.eml")

    def down(p):
        raise Unavailable("refused")

    stats = ingest_folder(tmp_path, down, log=lambda *_: None)
    assert stats.stopped_early and (tmp_path / "a.eml").exists()


def test_rejected_and_corrupt_files_go_to_rejected(tmp_path):
    write(tmp_path, "a.eml")
    (tmp_path / "b.eml").write_bytes(b"")

    def picky(p):
        if p["message_id"].startswith("sha256-"):      # the empty file: no Message-ID header
            raise Rejected("422")
        return {"decision": "human_review"}

    stats = ingest_folder(tmp_path, picky, log=lambda *_: None)
    assert stats.processed == 1 and stats.flagged == 1
    assert (tmp_path / "rejected" / "b.eml").exists()


def test_folder_to_database_end_to_end(tmp_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    write(tmp_path, "a.eml", pdf=True)
    with TestClient(create_app(light_pipeline)) as client:
        stats = ingest_folder(tmp_path, api_submit(client), log=lambda *_: None)
    assert stats.processed == 1
    with SessionLocal() as s:
        inv = s.query(Invoice).one()
    assert inv.total_amount == 980.0 and inv.invoice_number == "INV-55555"
