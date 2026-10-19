"""Mailbox ingestion: turns raw RFC 822 emails into POST /emails payloads and polls an IMAP inbox.

Design notes
- The poller never trusts the sender's clock: arrival time is the IMAP INTERNALDATE (when YOUR mail
  server received the message), because burst detection depends on it and a Date header is forgeable.
- Messages are fetched with BODY.PEEK so they are only marked \\Seen after the API accepted them.
  If the API is down they stay unread and are retried next cycle. The API is idempotent on message_id,
  so retries and crashes can never create a second invoice.
- A message the API rejects (4xx) is flagged \\Flagged and skipped from then on, so one bad email cannot
  block the queue, and a person can see it in the mailbox.
- Attachments: text PDFs are read and appended to the body. Anything unreadable (scans, images, other
  types) is noted in the body so the missing data is visible to the reviewer instead of silently ignored."""
import hashlib
import html
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import message_from_bytes, policy
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from typing import Callable

from app.ingestion.pdf_reader import read_pdf_text

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_BODY_CHARS = 100_000


# ------------------------------------------------------------------ parsing
class _TextExtractor(HTMLParser):
    _BLOCKS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "table"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in ("td", "th"):
            self.parts.append(" ")
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    text = html.unescape("".join(parser.parts))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text)).strip()


def parse_internaldate(fetch_response: bytes) -> datetime | None:
    """Pull INTERNALDATE out of an IMAP FETCH response, as UTC. None if absent or malformed."""
    m = re.search(rb'INTERNALDATE "([^"]+)"', fetch_response or b"")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1).decode().strip(), "%d-%b-%Y %H:%M:%S %z").astimezone(timezone.utc)
    except ValueError:
        return None


def _message_id(msg, raw: bytes) -> str:
    mid = (msg.get("Message-ID") or "").strip().strip("<>").strip()
    if not mid or len(mid) > 180:                      # missing or absurd: derive a stable id from content
        return "sha256-" + hashlib.sha256(raw).hexdigest()[:40]
    return mid


def _body_text(msg) -> str:
    """Plain and HTML versions are often both present, and the plain one is sometimes a useless stub
    ('your client cannot show HTML'). Use whichever carries more text."""
    candidates = []
    for kind in ("plain", "html"):
        part = msg.get_body(preferencelist=(kind,))
        if part is not None:
            text = part.get_content()
            candidates.append(html_to_text(text) if kind == "html" else text.strip())
    return max(candidates, key=len, default="")


def _attachment_texts(msg) -> list[str]:
    out = []
    for part in msg.iter_attachments():
        name = part.get_filename() or "unnamed"
        ctype = part.get_content_type()
        is_pdf = ctype == "application/pdf" or name.lower().endswith(".pdf")
        if not is_pdf:
            out.append(f"[attachment not read: {name} ({ctype}) - needs manual check]")
            continue
        try:
            data = part.get_content()
            if len(data) > MAX_ATTACHMENT_BYTES:
                out.append(f"[attachment not read: {name} is larger than {MAX_ATTACHMENT_BYTES // 1_000_000} MB]")
                continue
            out.append(f"--- attachment: {name} ---\n{read_pdf_text(io.BytesIO(data), max_pages=MAX_PDF_PAGES)}")
        except Exception as exc:                       # one bad attachment must not lose the email
            out.append(f"[attachment unreadable: {name} - {type(exc).__name__}; may be a scan, needs manual check]")
    return out


def build_payload(raw: bytes, internal_date: datetime | None = None) -> dict:
    """Raw RFC 822 bytes -> the JSON body for POST /emails."""
    msg = message_from_bytes(raw, policy=policy.default)
    received = internal_date
    if received is None and msg.get("Date"):           # fallback only: the sender controls this header
        try:
            received = parsedate_to_datetime(str(msg["Date"])).astimezone(timezone.utc)
        except (TypeError, ValueError):
            received = None

    body = "\n\n".join(filter(None, [_body_text(msg), *_attachment_texts(msg)]))[:MAX_BODY_CHARS]
    return {
        "message_id": _message_id(msg, raw),
        "sender": parseaddr(str(msg.get("From", "")))[1][:200],
        "subject": re.sub(r"\s+", " ", str(msg.get("Subject", ""))).strip()[:500],
        "body": body,
        "received_at": received.isoformat() if received else None,
    }


# ------------------------------------------------------------------ submitting
class Unavailable(Exception):
    """The API could not be reached or failed (5xx): leave the email unread and retry later."""


class Rejected(Exception):
    """The API refused this particular email (4xx): flag it and move on."""


def http_submit(base_url: str, timeout: float = 60.0, api_key: str | None = None) -> Callable[[dict], dict]:
    import httpx
    client = httpx.Client(base_url=base_url, timeout=timeout, headers={"X-API-Key": api_key} if api_key else {})

    def submit(payload: dict) -> dict:
        try:
            r = client.post("/emails", json=payload)
        except httpx.TransportError as exc:
            raise Unavailable(str(exc)) from exc
        if r.status_code >= 500:
            raise Unavailable(f"API returned {r.status_code}")
        if r.status_code >= 400:
            raise Rejected(f"API returned {r.status_code}: {r.text[:200]}")
        return r.json()

    return submit


# ------------------------------------------------------------------ polling
@dataclass
class PollStats:
    processed: int = 0
    flagged: int = 0
    stopped_early: bool = False
    decisions: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        parts = [f"{self.processed} processed", f"{self.flagged} flagged"]
        if self.decisions:
            parts.append(", ".join(f"{k}: {v}" for k, v in sorted(self.decisions.items())))
        if self.stopped_early:
            parts.append("STOPPED EARLY: API unavailable, will retry")
        return " | ".join(parts)


class MailboxPoller:
    def __init__(self, imap, submit: Callable[[dict], dict], folder: str = "INBOX", batch: int = 50, log=print):
        self.imap, self.submit, self.folder, self.batch, self.log = imap, submit, folder, batch, log

    def poll_once(self) -> PollStats:
        stats = PollStats()
        self.imap.select(self.folder)
        _, data = self.imap.uid("search", None, "UNSEEN", "UNFLAGGED")
        uids = (data[0] or b"").split()[: self.batch]

        for uid in uids:
            _, parts = self.imap.uid("fetch", uid, "(INTERNALDATE BODY.PEEK[])")
            literal = next((p for p in parts or [] if isinstance(p, tuple)), None)
            if literal is None:                        # message disappeared between search and fetch
                continue
            # INTERNALDATE can sit before or after the message literal, so look at every non-literal part.
            meta = b" ".join(p if isinstance(p, bytes) else p[0] for p in parts)
            try:
                payload = build_payload(literal[1], parse_internaldate(meta))
                result = self.submit(payload)
            except Unavailable as exc:
                self.log(f"API unavailable ({exc}); leaving {len(uids)} message(s) unread for the next cycle")
                stats.stopped_early = True
                break
            except Exception as exc:                   # Rejected, or an email we could not even parse
                self.log(f"Flagging message uid={uid.decode()}: {type(exc).__name__}: {exc}")
                self.imap.uid("store", uid, "+FLAGS", "(\\Flagged)")
                stats.flagged += 1
                continue
            self.imap.uid("store", uid, "+FLAGS", "(\\Seen)")
            stats.processed += 1
            stats.decisions[result.get("decision") or "unknown"] += 1
        return stats
