"""Generate labeled synthetic emails for training and evaluation.
Labels: category, is_duplicate, is_anomaly (ground truth for Phases 2-5)."""
import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

VENDORS = {  # vendor -> typical amount (mean, std)
    "Acme Supplies Ltd": (1200, 250), "BrightCloud Hosting": (450, 40),
    "Northwind Logistics": (3200, 600), "Pixel Print Co": (180, 30),
    "GreenLeaf Catering": (650, 120), "Summit Legal LLP": (5200, 900),
    "Orbit Office Rentals": (2400, 100), "Delta Cleaning Services": (320, 45),
}
OTHER = {
    "receipt": ["Your payment receipt", "Receipt for your order", "Payment received - thank you"],
    "purchase_order": ["Purchase Order {n} attached", "New PO {n} for approval"],
    "newsletter": ["Our monthly newsletter", "Product updates you'll love", "Webinar invitation"],
    "support": ["Question about my account", "Need help with login", "Meeting reschedule request"],
}


def make_invoice(vendor, anomalous=False):
    mean, std = VENDORS[vendor]
    amount = max(20, random.gauss(mean, std))
    if anomalous:
        amount *= random.choice([4, 6, 10])
    inv_date = date(2026, 1, 1) + timedelta(days=random.randint(0, 250))
    return {
        "vendor_name": vendor,
        "invoice_number": f"INV-{random.randint(10000, 99999)}",
        "invoice_date": inv_date.isoformat(),
        "due_date": (inv_date + timedelta(days=random.choice([14, 30, 45]))).isoformat(),
        "currency": "USD",
        "total_amount": round(amount, 2),
        "po_number": f"PO-{random.randint(1000, 9999)}" if random.random() < 0.6 else None,
    }


def invoice_email(inv, i):
    po = f"PO Number: {inv['po_number']}\n" if inv["po_number"] else ""
    body = (f"Hello,\n\nPlease find invoice {inv['invoice_number']} attached.\n"
            f"Vendor: {inv['vendor_name']}\nInvoice Date: {inv['invoice_date']}\n"
            f"Due Date: {inv['due_date']}\n{po}"
            f"Total Amount: {inv['currency']} {inv['total_amount']:,.2f}\n\nRegards,\n{inv['vendor_name']}")
    subject = random.choice([f"Invoice {inv['invoice_number']}", f"Invoice {inv['invoice_number']} - payment due",
                             f"{inv['vendor_name']} invoice"])
    if random.random() < 0.15:  # hard case: vague subject, terse body
        subject = random.choice(["Document attached", "Re: payment", "Fwd: details", "Monthly statement", "Please review"])
        body = (f"Hi,\n\nAttached is the bill for this month.\nRef {inv['invoice_number']}, "
                f"amount due {inv['currency']} {inv['total_amount']:,.2f} by {inv['due_date']}.\n\nThanks,\n{inv['vendor_name']}")
    return {
        "message_id": f"msg-{i:05d}",
        "sender": f"billing@{inv['vendor_name'].split()[0].lower()}.com",
        "subject": subject,
        "body": body,
    }


def main(n=1500):
    rows, invoices_seen = [], []
    for i in range(n):
        r = random.random()
        if r < 0.55:  # invoice
            vendor = random.choice(list(VENDORS))
            if invoices_seen and random.random() < 0.06:  # duplicate
                inv = dict(random.choice(invoices_seen))
                row = invoice_email(inv, i)
                row.update(category="invoice", is_duplicate=True, is_anomaly=False, truth=inv)
            else:
                anomalous = random.random() < 0.05
                inv = make_invoice(vendor, anomalous)
                invoices_seen.append(inv)
                row = invoice_email(inv, i)
                row.update(category="invoice", is_duplicate=False, is_anomaly=anomalous, truth=inv)
        else:
            cat = random.choice(list(OTHER))
            subj = random.choice(OTHER[cat]).format(n=random.randint(1000, 9999))
            body = f"{subj}\n\nThis is an automated or routine message."
            if cat in ("receipt", "purchase_order") and random.random() < 0.4:  # hard case: mentions an invoice
                body += f"\nReference: relates to invoice INV-{random.randint(10000, 99999)}, no payment action needed."
            row = {"message_id": f"msg-{i:05d}", "sender": f"user{random.randint(1,99)}@example.com",
                   "subject": subj, "body": body,
                   "category": cat, "is_duplicate": False, "is_anomaly": False, "truth": None}
        rows.append(row)
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic_emails.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"Wrote {len(rows)} emails to {out}")


if __name__ == "__main__":
    main()
