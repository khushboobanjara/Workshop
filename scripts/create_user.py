"""Create a user.   python -m scripts.create_user --username alice --role reviewer
The password is typed at a prompt (never put on the command line, where shell history would keep it)."""
import argparse
import getpass
import sys

from app import auth
from app.db import init_db


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--role", required=True, choices=auth.ROLES)
    a = ap.parse_args()
    pw = getpass.getpass("Password (min 10 characters): ")
    if pw != getpass.getpass("Repeat password: "):
        print("Passwords do not match.")
        return 1
    init_db()
    try:
        user = auth.create_user(a.username, pw, a.role, actor="cli")
    except (ValueError, LookupError) as e:
        print(f"Could not create user: {e}")
        return 1
    print(f"Created {user.role} '{user.username}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
