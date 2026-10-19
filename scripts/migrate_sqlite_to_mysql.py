"""One-off, NON-destructive copy of an existing SQLite database into the configured MySQL database.

    python -m scripts.migrate_sqlite_to_mysql opspilot.db             # dry run: counts rows, changes nothing
    python -m scripts.migrate_sqlite_to_mysql opspilot.db --apply     # copies the rows

* The SQLite file is only ever read. Nothing is deleted or modified in it, and it is your backup.
* Rows keep their ids, so foreign keys, `duplicate_of` links and audit-log entity ids stay valid.
* It refuses to run if any target table already has rows, so it can never merge into or overwrite MySQL data.
* The audit log is copied with plain INSERTs (the append-only triggers only block UPDATE and DELETE), in id order.
* Everything is copied in one transaction: either all tables arrive or none do.

The MySQL target is whatever OPSPILOT_DB_* / OPSPILOT_DATABASE_URL in your .env point to."""
import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select

from app.config import settings
from app.db import Base, engine, init_db

CHUNK = 500


def copy(sqlite_path: Path, apply: bool) -> dict[str, int]:
    src = create_engine(f"sqlite:///{sqlite_path.resolve().as_posix()}")
    src_tables = set(inspect(src).get_table_names())
    counts: dict[str, int] = {}

    if apply:
        init_db()                                                   # create missing tables + audit triggers; never drops anything
    target_tables = set(inspect(engine).get_table_names())

    with src.connect() as sc:
        for table in Base.metadata.sorted_tables:                   # parents before children (emails before invoices)
            if table.name not in src_tables:
                counts[table.name] = 0
                continue
            counts[table.name] = sc.execute(select(func.count()).select_from(table)).scalar_one()
            if table.name in target_tables:
                with engine.connect() as tc:
                    existing = tc.execute(select(func.count()).select_from(table)).scalar_one()
                if existing:
                    sys.exit(f"Refusing to continue: MySQL table {table.name!r} already has {existing} row(s). "
                             "Point OPSPILOT_DB_NAME at an empty database.")
        if not apply:
            return counts

        with engine.begin() as tc:
            for table in Base.metadata.sorted_tables:
                if table.name not in src_tables or not counts[table.name]:
                    continue
                have = {c["name"] for c in inspect(src).get_columns(table.name)}
                cols = [c for c in table.columns if c.name in have]     # an old SQLite file may lack newer columns
                query = select(*cols).order_by(table.c.id)
                batch = []
                for row in sc.execute(query).mappings():
                    batch.append(dict(row))
                    if len(batch) == CHUNK:
                        tc.execute(table.insert(), batch)
                        batch = []
                if batch:
                    tc.execute(table.insert(), batch)
            # Verify before committing: every table must have exactly as many rows as the source.
            for table in Base.metadata.sorted_tables:
                copied = tc.execute(select(func.count()).select_from(table)).scalar_one()
                if copied != counts[table.name]:
                    raise RuntimeError(f"{table.name}: copied {copied} of {counts[table.name]} rows; rolled back")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite_file", type=Path)
    ap.add_argument("--apply", action="store_true", help="actually copy (default is a dry run)")
    a = ap.parse_args()
    if not a.sqlite_file.is_file():
        sys.exit(f"No such SQLite file: {a.sqlite_file}")
    if not settings.database_url or settings.database_url.startswith("sqlite"):
        sys.exit("The target must be MySQL: set OPSPILOT_DB_* in .env first.")
    counts = copy(a.sqlite_file, a.apply)
    for name, n in counts.items():
        print(f"  {name:<16} {n:>8} row(s)")
    print(("Copied. " if a.apply else "Dry run only. ") + f"Total {sum(counts.values())} row(s)." +
          ("" if a.apply else "  Re-run with --apply to copy them."))


if __name__ == "__main__":
    main()
