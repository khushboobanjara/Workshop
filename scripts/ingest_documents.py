"""Add documents to the assistant's knowledge from the command line.

  python -m scripts.ingest_documents docs/sample_documents --min-role viewer
  python -m scripts.ingest_documents contracts/acme.pdf --min-role admin --title "Acme master agreement"

PATH is a file or a folder (searched recursively for PDF, DOCX, TXT, MD, CSV, HTML). --min-role is REQUIRED: who may retrieve the
document (viewer = every signed-in user, reviewer = reviewers and admins, admin = admins only). Run it where the application's
database and OPSPILOT_RAG_DATA_DIR are reachable. Each upload is audited with the actor name given by --actor (default: cli)."""
import argparse
import sys
from pathlib import Path

from app.db import init_db
from app.rag.extract import SUPPORTED
from app.rag.service import RagError, get_service


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--min-role", required=True, choices=["viewer", "reviewer", "admin"])
    ap.add_argument("--title", help="only for a single file; default is the file name")
    ap.add_argument("--actor", default="cli")
    a = ap.parse_args()
    root = Path(a.path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print("No supported files found.")
        return 1
    init_db()
    try:
        svc = get_service()
    except RagError as e:
        print(e.message)
        return 1
    failed = 0
    for f in files:
        try:
            doc = svc.ingest(data=f.read_bytes(), filename=f.name, title=a.title if len(files) == 1 else None, min_role=a.min_role, actor=a.actor)
            note = f", {doc['quarantined_chunks']} passage(s) held back as instruction-like" if doc["quarantined_chunks"] else ""
            print(f"  stored  {f.name}: document {doc['id']}, {doc['chunks']} passages{note}")
        except RagError as e:
            failed += 1
            print(f"  skipped {f.name}: {e.message}")
    print(f"{len(files) - failed} stored, {failed} skipped.")
    return 1 if failed and failed == len(files) else 0


if __name__ == "__main__":
    sys.exit(main())
