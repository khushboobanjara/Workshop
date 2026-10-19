"""Harder synthetic dataset -> data/synthetic_emails_hard.jsonl   Run: python -m scripts.generate_hard_data

What is harder than the original dataset (scripts/generate_synthetic_data.py):
  * Chronological: emails arrive in date order, so 'days since last invoice' style features are possible.
  * Messy layouts: label wording, date formats, amount formats ($, 1.234,56, suffix currency), vendor only in
    the signature, subtotal/tax lines, 'Net 30' instead of a due date, vendor-specific invoice-number schemes.
  * Noisy duplicates: reformatted invoice numbers, vendor name variants/typos, corrected amounts, resubmission
    under a new number. Plus legitimate look-alikes (monthly fixed rent, same-day identical twin orders).
  * Subtler anomalies: mild spikes inside natural variation, amounts just under the 10,000 approval limit,
    bursts of invoices from one vendor, first-ever invoice from a brand-new vendor.
  * Heavy-tailed vendors and slow price drift, so 'unusual' is not always 'wrong'.
  * Confusable non-invoices: receipts/POs that look like invoices, support mails that quote an invoice number.

Run with --shifted for a test set whose labels and date formats differ from the training data (robustness check).

Every row keeps the original keys (category, is_duplicate, is_anomaly, truth) plus dup_kind / anomaly_kind /
received so the evaluation scripts can report WHICH kinds of cases fail."""
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(7)
START, DAYS, AVG_PER_DAY = date(2025, 10, 1), 360, 7.4
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

VENDORS = {
    "Acme Supplies Ltd": dict(dist="normal", a=1200, b=250, cur="USD", num="INV"),
    "BrightCloud Hosting": dict(dist="normal", a=450, b=40, cur="USD", num="INV"),
    "Northwind Logistics": dict(dist="lognormal", a=3000, b=0.45, cur="USD", num="INV"),
    "Pixel Print Co": dict(dist="normal", a=180, b=30, cur="GBP", num="HASH"),
    "GreenLeaf Catering": dict(dist="lognormal", a=600, b=0.5, cur="USD", num="INV"),
    "Summit Legal LLP": dict(dist="normal", a=5200, b=900, cur="USD", num="SL"),
    "Orbit Office Rentals": dict(dist="fixed", a=2400.00, b=0, cur="USD", num="INV"),   # one fixed invoice per month
    "Delta Cleaning Services": dict(dist="normal", a=320, b=45, cur="USD", num="INV"),
    "Kestrel Software GmbH": dict(dist="lognormal", a=2800, b=0.3, cur="EUR", num="RE"),
}
RANDOM_VENDORS = [v for v in VENDORS if v != "Orbit Office Rentals"]
NEW_VENDORS = ["Zenith Consulting Group", "Maple Freight Co", "Quantum IT Solutions",
               "Harbor Print Works", "Redwood Advisory LLP", "Falcon Security Services"]
SYMBOL = {"USD": "$", "GBP": "£", "EUR": "€"}
GENERIC_SUBJECTS = ["Document attached", "Re: payment", "Fwd: details", "Monthly statement", "Please review", "Billing"]


# Wording the renderer draws from. SHIFTED uses labels/formats the extractor never saw in training, to test robustness.
DEFAULT_VOCAB = dict(
    intros=["Please find invoice {n} attached.", "Attached is invoice {n} for services rendered.", "Please find our invoice attached.", "Invoice No: {n}"],
    inv_date=[("Invoice Date", .6), ("Date", .2), ("Issued", .2)], due_label="Due Date", payment_due_label="Payment Due",
    amount=[("Total Amount", .45), ("Amount Due", .2), ("Total", .15), ("Balance Due", .1), ("Amount payable", .1)],
    vendor_label="Vendor", from_label="From", date_styles=[("iso", .55), ("slash", .15), ("long", .15), ("short", .15)],
    po=[("PO Number: {po}", .6), ("PO #{digits}", .2), ("Your ref: {po}", .2)])
SHIFTED_VOCAB = dict(
    intros=["Kindly process the enclosed invoice {n}.", "Invoice reference {n} is enclosed for your attention.", "Enclosed is our invoice {n}.", "Invoice No: {n}"],
    inv_date=[("Invoice dated", .4), ("Billing date", .3), ("Document date", .3)], due_label="Due on", payment_due_label="Settle by",
    amount=[("Grand total", .35), ("Net payable", .25), ("Invoice total", .2), ("Sum due", .2)],
    vendor_label="Supplier", from_label="Billed by", date_styles=[("iso", .35), ("slash", .1), ("long", .2), ("short", .2), ("dash", .15)],
    po=[("Order ref: {po}", .5), ("PO No. {digits}", .25), ("Purchase order {po}", .25)])
# A second unseen vocabulary, generated only AFTER the extractor was finalised, so it is a clean test set.
SHIFTED_TEST_VOCAB = dict(
    intros=["Here is invoice {n} for your records.", "Billing document {n} follows below.", "Invoice No: {n}", "Our invoice {n} is attached."],
    inv_date=[("Invoice issued", .4), ("Dated", .3), ("Bill date", .3)], due_label="Payment deadline", payment_due_label="Pay by",
    amount=[("Total payable", .3), ("Amount owed", .3), ("Invoice amount", .2), ("Final total", .2)],
    vendor_label="Billed from", from_label="Company", date_styles=[("iso", .35), ("slash", .1), ("long", .2), ("short", .2), ("dash", .15)],
    po=[("Customer PO: {po}", .5), ("PO ref {digits}", .25), ("P.O. {po}", .25)])
V = DEFAULT_VOCAB


def pick(options):
    """options: list of (value, weight)"""
    return random.choices([o[0] for o in options], weights=[o[1] for o in options])[0]


# ------------------------------------------------------------------ invoice generation
def new_number(scheme):
    return {"INV": lambda: f"INV-{random.randint(10000, 99999)}",
            "SL": lambda: f"SL-2026-{random.randint(1000, 9999)}",
            "RE": lambda: f"RE-{random.randint(100000, 999999)}",
            "HASH": lambda: f"#{random.randint(10000, 99999)}"}[scheme]()


def base_amount(vendor, day):
    d = VENDORS[vendor]
    if d["dist"] == "fixed":
        return float(d["a"])
    x = random.gauss(d["a"], d["b"]) if d["dist"] == "normal" else d["a"] * math.exp(random.gauss(0, d["b"]))
    return round(max(20.0, x * (1 + 0.015 * day / 30)), 2)      # slow price drift (~1.5% per month)


def make_invoice(vendor, day, amount=None, currency=None, scheme=None):
    d = VENDORS.get(vendor, dict(cur="USD", num="INV"))
    inv_date = START + timedelta(days=day) - timedelta(days=random.randint(0, 3))
    po = random.random() < 0.4
    return {
        "vendor_name": vendor, "invoice_number": new_number(scheme or d["num"]),
        "invoice_date": inv_date.isoformat(),
        "due_date": (inv_date + timedelta(days=random.choice([14, 30, 45]))).isoformat(),
        "currency": currency or d["cur"],
        "total_amount": amount if amount is not None else base_amount(vendor, day),
        "po_number": f"PO-{random.randint(1000, 9999)}" if po else None,
    }


def vendor_variant(name):
    options = [name.upper(), name.replace(" Ltd", " Limited"), name.replace(" Co", " Company"),
               name.rsplit(" ", 1)[0] if len(name.split()) > 2 else name, name.replace("Bright", "Bright ")]
    words = name.split()
    longest = max(range(len(words)), key=lambda i: len(words[i]))
    w = words[longest]
    if len(w) > 5:                                              # typo: drop one letter
        k = random.randint(1, len(w) - 2)
        words2 = words[:]; words2[longest] = w[:k] + w[k + 1:]
        options.append(" ".join(words2))
    options = [o for o in options if o != name]
    return random.choice(options) if options else name.upper()


# ------------------------------------------------------------------ email rendering
def fmt_date(iso, style):
    d = date.fromisoformat(iso)
    return {"iso": iso, "slash": d.strftime("%d/%m/%Y"), "long": f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
            "short": f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
            "dash": f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year}"}[style]


def fmt_amount(amount, cur, style):
    if style == "euro" and cur == "EUR":
        return f"EUR {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return {"code": f"{cur} {amount:,.2f}", "plain": f"{cur} {amount:.2f}",
            "symbol": f"{SYMBOL.get(cur, '')}{amount:,.2f}", "suffix": f"{amount:,.2f} {cur}"}.get(style, f"{cur} {amount:,.2f}")


def render_invoice(inv, vendor_text=None, number_text=None):
    vendor, number = vendor_text or inv["vendor_name"], number_text or inv["invoice_number"]
    cur, amt = inv["currency"], inv["total_amount"]
    ds = pick(V["date_styles"])
    a_style = pick([("code", .55), ("plain", .08), ("symbol", .15), ("suffix", .10), ("euro", .12)])
    d = lambda iso: fmt_date(iso, ds)

    if random.random() < .08 and number.upper().startswith("INV-"):
        number = "Inv #" + number[4:]
    lines = [random.choice(["Hello,", "Hi,", "Dear Accounts Payable,"]), ""]
    intro = random.choice(V["intros"]).format(n=number)
    lines.append(intro)
    if number not in intro and random.random() < .85:
        lines.append(f"Invoice No: {number}")
    vstyle = pick([("label", .5), ("from", .2), ("none", .3)])
    if vstyle == "label":
        lines.append(f"{V['vendor_label']}: {vendor}")
    elif vstyle == "from":
        lines.append(f"{V['from_label']}: {vendor}")
    lines.append(f"{pick(V['inv_date'])}: {d(inv['invoice_date'])}")
    due = pick([("due", .5), ("payment_due", .15), ("by", .1), ("terms", .15), ("none", .1)])
    if due == "due":
        lines.append(f"{V['due_label']}: {d(inv['due_date'])}")
    elif due == "payment_due":
        lines.append(f"{V['payment_due_label']}: {d(inv['due_date'])}")
    elif due == "terms":
        lines.append(f"Payment terms: Net {(date.fromisoformat(inv['due_date']) - date.fromisoformat(inv['invoice_date'])).days}")
    if inv["po_number"]:
        lines.append(pick(V["po"]).format(po=inv["po_number"], digits=inv["po_number"][3:]))
    if random.random() < .15:
        sub = round(amt / 1.08, 2)
        lines += [f"Subtotal: {fmt_amount(sub, cur, a_style)}", f"Tax (8%): {fmt_amount(round(amt - sub, 2), cur, a_style)}"]
    label = pick(V["amount"])
    amount_line = f"{label}: {fmt_amount(amt, cur, a_style)}"
    if due == "by":
        amount_line = f"Amount due {fmt_amount(amt, cur, a_style)} by {d(inv['due_date'])}"
    lines.append(amount_line)
    sig = pick([("vendor", .6), ("ar", .2), ("person", .2)])
    lines += ["", "Regards,", {"vendor": vendor, "ar": "Accounts Receivable Team", "person": random.choice(["John", "Priya", "Maria"])}[sig]]

    subject = pick([(f"Invoice {number}", .3), (f"Invoice {number} - payment due", .2), (f"{vendor} invoice", .2),
                    (random.choice(GENERIC_SUBJECTS), .3)])
    return subject, "\n".join(lines)


def sender_for(vendor):
    return f"billing@{vendor.split()[0].lower()}.com"


# ------------------------------------------------------------------ non-invoice emails
def other_email(cat, day):
    vendor = random.choice(RANDOM_VENDORS); cur = VENDORS[vendor]["cur"]
    amt = base_amount(vendor, day); when = (START + timedelta(days=day)).isoformat()
    ref = f"INV-{random.randint(10000, 99999)}"
    if cat == "receipt":
        subj = random.choice(["Your payment receipt", "Receipt for your order", "Payment received - thank you"])
        body = f"{subj}\n\nThis is an automated or routine message."
        if random.random() < .35:      # looks like an invoice
            body = (f"Receipt RCT-{random.randint(1000, 9999)}\nVendor: {vendor}\nDate: {when}\n"
                    f"Total Amount: {cur} {amt:,.2f}\nPaid in full. No action required.")
            if random.random() < .6:
                subj = random.choice(GENERIC_SUBJECTS)         # same vague subjects real invoices use
        elif random.random() < .3:
            body += f"\nReference: relates to invoice {ref}, no payment action needed."
    elif cat == "purchase_order":
        subj = random.choice([f"Purchase Order {random.randint(1000, 9999)} attached", f"New PO {random.randint(1000, 9999)} for approval"])
        body = f"{subj}\n\nThis is an automated or routine message."
        if random.random() < .4:
            body = (f"Purchase order PO-{random.randint(1000, 9999)}\nVendor: {vendor}\nTotal Amount: {cur} {amt:,.2f}\n"
                    f"Delivery by {when}. Please confirm.")
            if random.random() < .6:
                subj = random.choice(GENERIC_SUBJECTS)
    elif cat == "newsletter":
        subj = random.choice(["Our monthly newsletter", "Product updates you'll love", "Webinar invitation", "Save time on invoicing"])
        body = f"{subj}\n\nThis is an automated or routine message."
        if random.random() < .2:
            body += "\nTip: automate your invoice approvals and cut month-end work in half."
    else:
        subj = random.choice(["Question about my account", "Need help with login", "Meeting reschedule request", "Missing invoice copy"])
        body = f"{subj}\n\nThis is an automated or routine message."
        if random.random() < .25:
            body = f"Hi, could you resend invoice {ref}? I was charged {cur} {amt:,.2f} and cannot find the document.\nThanks"
    return subj, body


# ------------------------------------------------------------------ dataset assembly
def main(out_name="synthetic_emails_hard.jsonl", shifted=False, shifted_test=False):
    global V
    if shifted:
        V = SHIFTED_VOCAB
        random.seed(11)
    elif shifted_test:
        V = SHIFTED_TEST_VOCAB
        random.seed(23)
    rows, seen, used_new = [], [], set()      # seen: invoices already emitted (dicts with scheme)

    def emit(day, subject, body, sender, category, truth, dup_kind=None, anomaly_kind=None):
        rows.append(dict(day=day, subject=subject, body=body, sender=sender, category=category, truth=truth,
                         is_duplicate=dup_kind is not None and dup_kind != "twin",
                         is_anomaly=anomaly_kind is not None, dup_kind=dup_kind if dup_kind != "twin" else None,
                         anomaly_kind=anomaly_kind, twin=(dup_kind == "twin"),
                         received=(START + timedelta(days=day)).isoformat()))

    def emit_invoice(day, inv, vendor_text=None, number_text=None, dup_kind=None, anomaly_kind=None):
        s, b = render_invoice(inv, vendor_text, number_text)
        emit(day, s, b, sender_for(inv["vendor_name"]), "invoice", inv, dup_kind, anomaly_kind)

    for day in range(DAYS):
        today = START + timedelta(days=day)
        events = ["orbit"] if today.day == 1 else []
        events += [None] * max(0, round(random.gauss(AVG_PER_DAY, 2.5)))
        random.shuffle(events)
        for ev in events:
            if ev == "orbit":
                inv = make_invoice("Orbit Office Rentals", day); seen.append(inv)
                emit_invoice(day, inv)
                continue
            if random.random() >= .55:                                   # non-invoice
                cat = random.choice(["receipt", "purchase_order", "newsletter", "support"])
                subj, body = other_email(cat, day)
                emit(day, subj, body, f"user{random.randint(1, 99)}@example.com", cat, None)
                continue

            r = random.random()
            recent = seen[-60:]
            if recent and r < .06:                                       # ---- duplicate
                src = random.choice(recent)
                kind = pick([("exact", .35), ("number_fmt", .15), ("vendor_variant", .2), ("amount_rounded", .1), ("resubmitted", .2)])
                inv, vtext, ntext = dict(src), None, None
                if kind == "number_fmt":
                    if src["invoice_number"].startswith("INV-"):
                        ntext = random.choice([f"INV {src['invoice_number'][4:]}", f"inv-{src['invoice_number'][4:]}", f"INV{src['invoice_number'][4:]}"])
                    else:
                        kind = "exact"
                elif kind == "vendor_variant":
                    vtext = vendor_variant(src["vendor_name"]); inv["vendor_name"] = vtext
                elif kind == "amount_rounded":
                    a = src["total_amount"]; new = float(round(a))
                    inv["total_amount"] = new if abs(new - a) >= .01 else a + 0.5
                elif kind == "resubmitted":
                    scheme = next(v["num"] for n, v in VENDORS.items() if n == src["vendor_name"]) if src["vendor_name"] in VENDORS else "INV"
                    shift = random.randint(-2, 2)
                    nd = date.fromisoformat(src["invoice_date"]) + timedelta(days=shift)
                    inv.update(invoice_number=new_number(scheme), invoice_date=nd.isoformat(),
                               due_date=(nd + timedelta(days=30)).isoformat())
                emit_invoice(day, inv, vtext, ntext, dup_kind=kind)
                continue
            if recent and r < .07:                                       # ---- legitimate identical twin (NOT a duplicate)
                src = random.choice(recent)
                scheme = VENDORS.get(src["vendor_name"], {}).get("num", "INV")
                nd = date.fromisoformat(src["invoice_date"]) + timedelta(days=random.randint(-2, 2))
                inv = dict(src, invoice_number=new_number(scheme), invoice_date=nd.isoformat(), due_date=(nd + timedelta(days=30)).isoformat())
                seen.append(inv)
                emit_invoice(day, inv, dup_kind="twin")
                continue

            vendor = random.choice(RANDOM_VENDORS)
            if random.random() < .055:                                   # ---- anomaly
                kind = pick([("spike_mild", .30), ("spike_big", .30), ("near_limit", .10), ("burst", .15), ("new_vendor", .15)])
                if kind == "new_vendor":
                    free = [v for v in NEW_VENDORS if v not in used_new]
                    if not free:
                        kind = "spike_big"
                    else:
                        vendor = random.choice(free); used_new.add(vendor)
                        inv = make_invoice(vendor, day, amount=round(random.uniform(2500, 9000), 2)); seen.append(inv)
                        emit_invoice(day, inv, anomaly_kind=kind); continue
                if kind == "burst":
                    inv = make_invoice(vendor, day); seen.append(inv); emit_invoice(day, inv)
                    for _ in range(2):
                        extra = make_invoice(vendor, day); seen.append(extra)
                        emit_invoice(day, extra, anomaly_kind="burst")
                    continue
                amt = base_amount(vendor, day)
                if kind == "spike_mild":
                    amt = round(amt * random.uniform(1.8, 2.6), 2)
                elif kind == "near_limit" and VENDORS[vendor]["cur"] == "USD" and VENDORS[vendor]["a"] < 4000:
                    amt = round(random.uniform(9200, 9990), 2)
                else:
                    kind = "spike_big" if kind == "near_limit" else kind
                    amt = round(amt * random.choice([4, 6, 10]), 2)
                inv = make_invoice(vendor, day, amount=amt); seen.append(inv)
                emit_invoice(day, inv, anomaly_kind=kind)
                continue

            inv = make_invoice(vendor, day); seen.append(inv)           # ---- ordinary invoice
            emit_invoice(day, inv)

    for i, row in enumerate(rows):
        row.pop("day")
        row["message_id"] = f"msg-{i:05d}"
    ordered = [{k: r[k] for k in ("message_id", "sender", "subject", "body", "category", "is_duplicate", "is_anomaly",
                                   "truth", "dup_kind", "anomaly_kind", "twin", "received")} for r in rows]
    out = Path(__file__).resolve().parent.parent / "data" / out_name
    out.write_text("\n".join(json.dumps(r) for r in ordered))
    inv_rows = [r for r in ordered if r["category"] == "invoice"]
    print(f"Wrote {len(ordered)} emails ({len(inv_rows)} invoices, {sum(r['is_duplicate'] for r in ordered)} duplicates, "
          f"{sum(r['is_anomaly'] for r in ordered)} anomalies) to {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifted", action="store_true", help="write data/synthetic_emails_shifted.jsonl with unseen wording/formats")
    ap.add_argument("--shifted-test", action="store_true", help="write data/synthetic_emails_shifted_test.jsonl (second unseen vocabulary)")
    a = ap.parse_args()
    if a.shifted:
        main("synthetic_emails_shifted.jsonl", shifted=True)
    elif a.shifted_test:
        main("synthetic_emails_shifted_test.jsonl", shifted_test=True)
    else:
        main()
