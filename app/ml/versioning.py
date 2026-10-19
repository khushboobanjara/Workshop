import hashlib
from pathlib import Path


def file_version(path: str | Path | None) -> str | None:
    """Identify a model by the content of its file (first 12 hex of SHA-256), so an audit row says exactly
    which model produced a decision, even if the file was later retrained or replaced."""
    if not path or not Path(path).is_file():
        return None
    return "sha-" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
