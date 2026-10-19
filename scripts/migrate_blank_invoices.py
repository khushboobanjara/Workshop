"""Move old blank 'invoices' (created before the Triage queue existed) into Triage.

A blank invoice is one that nobody has reviewed, has no vendor, number or amount, AND was sent to review because
the classifier was unsure. Invoices whose extraction merely failed are left alone: those really are invoices.

    python -m scripts.migrate_blank_invoices            # dry run: lists what would move, changes nothing
    python -m scripts.migrate_blank_invoices --apply    # moves them (SQLite: copies the file to a .bak first; MySQL: take a mysqldump first)
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.db import AuditLog, Email, Invoice, SessionLocal, init_db


def is_blank(inv: Invoice) -> bool:
    return (inv.vendor_name is None and inv.invoice_number is None and inv.total_amount is None
            and inv.status == "pending_review" and inv.reviewed_by is None and inv.duplicate_of is None
            and any("classification confidence" in f.lower() for f in (inv.flags or [])))


def migrate(apply: bool) -> list[dict]:
    init_db()                                         # makes sure the triage_status column exists
    moved = []
    with SessionLocal() as s:
        for inv in s.query(Invoice).order_by(Invoice.id).all():
            if not is_blank(inv):
                continue
            email = s.get(Email, inv.email_id)
            moved.append({"invoice_id": inv.id, "email_id": email.id, "subject": email.subject, "sender": email.sender})
            if apply:
                email.triage_status = "needs_triage"
                s.add(AuditLog(entity_type="email", entity_id=email.id, stage="migration", action="moved_to_triage",
                               actor="system", details={"was_invoice_id": inv.id}))
                s.delete(inv)
        if apply:
            s.commit()
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually make the change (default is a dry run)")
    a = ap.parse_args()

    if a.apply and settings.database_url.startswith("sqlite:///"):
        db = Path(settings.database_url.removeprefix("sqlite:///"))
        if db.exists():
            backup = db.with_name(db.name + datetime.now().strftime(".bak-%Y%m%d-%H%M%S"))
            shutil.copy2(db, backup)
            print(f"Backup written: {backup}")
    elif a.apply:
        print("Reminder: this changes data. Make sure you have a backup (for example: mysqldump --single-transaction <db> > backup.sql).")

    moved = migrate(a.apply)
    for m in moved:
        print(f"  invoice #{m['invoice_id']:<4} <- {m['sender'] or '?':<32} {m['subject'][:50]}")
    verb = "Moved" if a.apply else "Would move"
    print(f"{verb} {len(moved)} blank invoice(s) to Triage." + ("" if a.apply else "  Re-run with --apply to do it."))


if __name__ == "__main__":
    main()
