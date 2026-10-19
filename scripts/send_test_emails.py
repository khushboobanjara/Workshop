"""Create synthetic invoice emails, either mailed to your test mailbox or written as .eml files.
    python -m scripts.send_test_emails --count 5 --pdf --out-dir inbox   # no account needed
    python -m scripts.send_test_emails --count 5                          # email it to OPSPILOT_IMAP_USER
    --pdf puts the invoice details only inside a PDF attachment.
Mailing uses OPSPILOT_IMAP_USER and its password (for Gmail: an App Password). USE A THROWAWAY MAILBOX."""
import argparse
import io
import json
import smtplib
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from app.config import settings

DATA = Path("data/synthetic_emails_hard.jsonl")


def render_pdf(text: str) -> bytes:
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in text.splitlines():
        c.drawString(60, y, line[:110])
        y -= 16
    c.save()
    return buf.getvalue()


def build_email(row: dict, to: str, as_pdf: bool) -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = to, to, row["subject"]
    msg["Date"], msg["Message-ID"] = formatdate(localtime=True), make_msgid(domain="opspilot.test")
    if as_pdf:
        msg.set_content("Hello,\n\nPlease find the attached invoice.\n\nThank you.")
        msg.add_attachment(render_pdf(row["body"]), maintype="application", subtype="pdf", filename="invoice.pdf")
    else:
        msg.set_content(row["body"])
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--out-dir", help="write .eml files here instead of sending mail")
    a = ap.parse_args()
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    invoices = [r for r in rows if r["category"] == "invoice" and not r["is_duplicate"]][: a.count]
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(invoices, 1):
            (out / f"invoice_{i:03d}.eml").write_bytes(build_email(r, "billing@vendor.example", a.pdf).as_bytes())
        print(f"Wrote {len(invoices)} .eml files to {out}")
        return

    user, password = settings.imap_user, settings.imap_password.get_secret_value()
    if not (user and password):
        raise SystemExit("Set OPSPILOT_IMAP_USER and OPSPILOT_IMAP_PASSWORD first (see .env.example).")

    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port) as smtp:
        smtp.login(user, password)
        for r in invoices:
            smtp.send_message(build_email(r, user, a.pdf))
    print(f"Sent {len(invoices)} test emails to {user}")


if __name__ == "__main__":
    main()
