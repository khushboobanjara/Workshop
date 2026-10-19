"""Run the mailbox poller.
    python -m app.ingestion.poll --once          # one pass, then exit
    python -m app.ingestion.poll                 # keep polling every OPSPILOT_POLL_INTERVAL_SECONDS
The API must be running (uvicorn app.main:app). Settings come from OPSPILOT_* environment variables or .env."""
import argparse
import imaplib
import sys
import time

from app.config import settings
from app.ingestion.mailbox import MailboxPoller, http_submit


def cycle(submit) -> bool:
    """One connect-poll-disconnect pass (a fresh connection each time avoids stale idle sessions)."""
    try:
        imap = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
        imap.login(settings.imap_user, settings.imap_password.get_secret_value())
    except (imaplib.IMAP4.error, OSError) as exc:
        print(f"Could not connect/login to {settings.imap_host}: {type(exc).__name__}: {exc}")
        return False
    try:
        stats = MailboxPoller(imap, submit, settings.imap_folder).poll_once()
        print(f"[{time.strftime('%H:%M:%S')}] {stats.summary()}")
        return not stats.stopped_early
    except (imaplib.IMAP4.error, OSError) as exc:
        print(f"Mailbox error: {type(exc).__name__}: {exc}")
        return False
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="poll one time and exit")
    ap.add_argument("--interval", type=int, default=settings.poll_interval_seconds)
    a = ap.parse_args()

    if not (settings.imap_host and settings.imap_user and settings.imap_password.get_secret_value()):
        print("Set OPSPILOT_IMAP_HOST, OPSPILOT_IMAP_USER and OPSPILOT_IMAP_PASSWORD (see .env.example).")
        return 2
    submit = http_submit(settings.api_url, api_key=settings.api_key.get_secret_value() or None)
    while True:
        ok = cycle(submit)
        if a.once:
            return 0 if ok else 1
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
