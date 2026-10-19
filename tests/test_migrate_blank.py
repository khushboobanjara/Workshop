from app.db import AuditLog, Base, Email, Invoice, SessionLocal, engine
from scripts.migrate_blank_invoices import migrate

LOW = ["Low classification confidence (0.50) - needs human check"]


def add(mid, flags, **inv):
    with SessionLocal() as s:
        e = Email(message_id=mid, sender="s", subject=mid, body="b")
        s.add(e)
        s.flush()
        fields = dict(status="pending_review", decision="human_review", flags=flags)
        fields.update(inv)
        s.add(Invoice(email_id=e.id, **fields))
        s.commit()


def setup_rows():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    add("blank-unsure", LOW)                                                              # should move
    add("blank-but-extraction-failed", ["Missing required fields: vendor_name"])           # really an invoice: keep
    add("blank-already-reviewed", LOW, reviewed_by="asha")                                 # a person touched it: keep
    add("real", [], vendor_name="Acme", invoice_number="INV-1", total_amount=10.0)         # keep


def test_dry_run_changes_nothing():
    setup_rows()
    assert [m["subject"] for m in migrate(apply=False)] == ["blank-unsure"]
    with SessionLocal() as s:
        assert s.query(Invoice).count() == 4 and s.query(Email).filter(Email.triage_status.is_not(None)).count() == 0


def test_apply_moves_only_the_unsure_blank_and_is_audited_and_repeatable():
    setup_rows()
    assert len(migrate(apply=True)) == 1
    with SessionLocal() as s:
        assert s.query(Invoice).count() == 3
        assert s.query(Email).filter_by(triage_status="needs_triage").one().subject == "blank-unsure"
        assert s.query(AuditLog).filter_by(stage="migration").count() == 1
    assert migrate(apply=True) == []                                                       # second run finds nothing
