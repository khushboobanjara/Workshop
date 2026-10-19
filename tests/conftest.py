"""Runs before any test module, so the tests see a known, clean configuration.

Three things are handled here:
  * the developer's .env file (for example OPSPILOT_AUTH_ENABLED=true) and OPSPILOT_* shell variables are removed so
    they cannot leak into the tests;
  * the tests run against a dedicated MySQL TEST database, taken from OPSPILOT_TEST_DATABASE_URL or from the
    OPSPILOT_TEST_DB_* parts (shell or .env), never from the application's own OPSPILOT_DB_* settings;
  * because tests drop and recreate every table, the test database name must contain "test". Anything else aborts
    the run, so a mistake can never wipe real data.

An explicit sqlite:/// URL in OPSPILOT_TEST_DATABASE_URL is still accepted, for a quick run without a MySQL server;
the MySQL-only tests are then skipped. See the README ("Running the tests")."""
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import URL, make_url

_ENV_FILE = Path(os.environ.get("OPSPILOT_TEST_ENV_FILE", ".env"))
_file_vars = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}


def _test_var(name: str, default: str = "") -> str:
    return os.environ.get(name) or _file_vars.get(name) or default


def _test_database_url() -> str:
    explicit = _test_var("OPSPILOT_TEST_DATABASE_URL")
    if explicit:
        return explicit
    user = _test_var("OPSPILOT_TEST_DB_USER")
    if not user:
        pytest.exit("No test database configured. Set OPSPILOT_TEST_DB_USER, OPSPILOT_TEST_DB_PASSWORD, OPSPILOT_TEST_DB_HOST, "
                    "OPSPILOT_TEST_DB_PORT and OPSPILOT_TEST_DB_NAME (a database whose name contains 'test') in .env "
                    "or the shell, or OPSPILOT_TEST_DATABASE_URL. See README.md, 'Running the tests'.", returncode=2)
    return URL.create("mysql+pymysql", username=user, password=_test_var("OPSPILOT_TEST_DB_PASSWORD") or None,
                      host=_test_var("OPSPILOT_TEST_DB_HOST", "127.0.0.1"), port=int(_test_var("OPSPILOT_TEST_DB_PORT", "3306")),
                      database=_test_var("OPSPILOT_TEST_DB_NAME", "opspilot_test"),
                      query={"charset": "utf8mb4"}).render_as_string(hide_password=False)


TEST_DATABASE_URL = _test_database_url()
_parsed = make_url(TEST_DATABASE_URL)
if _parsed.get_backend_name() != "sqlite" and "test" not in (_parsed.database or "").lower():
    pytest.exit(f"Refusing to run: the test database is called {_parsed.database!r}. The tests drop and recreate every "
                "table, so its name must contain 'test'.", returncode=2)

for _name in [k for k in os.environ if k.startswith("OPSPILOT_")]:
    del os.environ[_name]

_tmp = Path(tempfile.mkdtemp(prefix="opspilot_test_"))
os.environ["OPSPILOT_ENV_FILE"] = str(_tmp / "no-such.env")          # a path that does not exist: ignore any real .env
if _parsed.get_backend_name() == "sqlite" and not _parsed.database:      # bare "sqlite://": use a throwaway file
    TEST_DATABASE_URL = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
os.environ["OPSPILOT_DATABASE_URL"] = TEST_DATABASE_URL


@pytest.fixture(scope="session")
def is_mysql() -> bool:
    from app.db import engine
    return engine.dialect.name == "mysql"
