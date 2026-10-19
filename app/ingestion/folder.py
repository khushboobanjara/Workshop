"""Ingest saved emails (.eml files) from a folder: the same parsing and API path as the IMAP poller,
but no mail account or password needed. Useful for demos, tests, and for emails exported from any
mail client (in most clients: open the message, 'Save as' / 'Download message' gives an .eml).

    python -m app.ingestion.folder inbox            # process every inbox/*.eml once
    python -m app.ingestion.folder inbox --watch 10 # keep checking every 10 seconds (used by docker compose)
Processed files move to inbox/processed/, files the API rejected to inbox/rejected/.
Files stay where they are if the API is down, so just run it again."""
import argparse
import sys
import time
from pathlib import Path

from app.config import settings
from app.ingestion.mailbox import PollStats, Rejected, Unavailable, build_payload, http_submit


def ingest_folder(folder: str | Path, submit, log=print) -> PollStats:
    root = Path(folder)
    stats = PollStats()
    for path in sorted(root.glob("*.eml")):
        try:
            result = submit(build_payload(path.read_bytes()))
        except Unavailable as exc:
            log(f"API unavailable ({exc}); leaving the remaining files for the next run")
            stats.stopped_early = True
            break
        except Exception as exc:                       # Rejected, or a file we cannot parse
            log(f"Rejected {path.name}: {type(exc).__name__}: {exc}")
            _move(path, root / "rejected")
            stats.flagged += 1
            continue
        _move(path, root / "processed")
        stats.processed += 1
        stats.decisions[result.get("decision") or "unknown"] += 1
    return stats


def _move(path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(exist_ok=True)
    path.replace(dest_dir / path.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing .eml files")
    ap.add_argument("--api-url", default=settings.api_url)
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="keep running, checking the folder this often")
    a = ap.parse_args()
    if not Path(a.folder).is_dir():
        print(f"No such folder: {a.folder}")
        return 2
    submit = http_submit(a.api_url, api_key=settings.api_key.get_secret_value() or None)
    while True:
        stats = ingest_folder(a.folder, submit)
        if stats.processed or stats.flagged or stats.stopped_early or not a.watch:
            print(f"[{time.strftime('%H:%M:%S')}] {stats.summary()}", flush=True)
        if not a.watch:
            return 1 if stats.stopped_early else 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
