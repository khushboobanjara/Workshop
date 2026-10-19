"""MySQL-specific guarantees. Skipped when the test database is not MySQL (see tests/conftest.py).

These cover what SQLite never needed: exact column types, long text, microsecond timestamps, case-sensitive
uniqueness, the database-level append-only triggers, and upgrading older tables in place."""
from datetime import datetime

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app import db
from app.db import AuditLog, Base, Email, Invoice, SessionLocal, engine, init_db

pytestmark = pytest.mark.skipif(engine.dialect.name != "mysql", reason="needs a MySQL test database")


def fresh():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@pytest.fixture(autouse=True)
def clean_schema():
    fresh()
    yield
    fresh()                                   # tests below that build legacy tables must leave a proper schema behind


def _col(table, column):
    with engine.connect() as c:
        return c.execute(text("SELECT DATA_TYPE, DATETIME_PRECISION, COLLATION_NAME FROM information_schema.COLUMNS "
                              "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c"),
                         {"t": table, "c": column}).one()


def test_column_types_are_precise_enough_for_money_time_and_email_bodies():
    assert _col("invoices", "total_amount")[0] == "double"           # FLOAT would round 1234567.89
    assert _col("emails", "body")[0] == "longtext"                   # TEXT would cut 200 000 characters at 64 KB
    assert _col("audit_log", "timestamp")[:2] == ("datetime", 6)     # keeps microseconds, so audit ordering is exact
    assert _col("emails", "message_id")[2] == "utf8mb4_bin"          # case-sensitive, like SQLite was


def test_values_survive_a_round_trip_unchanged():
    when, body = datetime(2026, 3, 1, 12, 30, 45, 123456), "x" * 200_000
    with SessionLocal() as s:
        e = Email(message_id="m1", sender="s", subject="sub", body=body, received_at=when)
        s.add(e)
        s.flush()
        s.add(Invoice(email_id=e.id, total_amount=1234567.891, risk_score=0.123456789, flags=["a", "b"],
                      risk_breakdown={"k": 1.5}, created_at=when))
        s.commit()
    with SessionLocal() as s:
        e, inv = s.query(Email).one(), s.query(Invoice).one()
        assert e.body == body and e.received_at == when
        assert inv.total_amount == 1234567.891 and inv.risk_score == 0.123456789
        assert inv.flags == ["a", "b"] and inv.risk_breakdown == {"k": 1.5} and inv.created_at == when


def test_message_ids_differing_only_by_case_are_distinct_but_exact_duplicates_are_refused():
    with SessionLocal() as s:
        s.add_all([Email(message_id="Abc", sender="", subject="", body=""), Email(message_id="abc", sender="", subject="", body="")])
        s.commit()
        s.add(Email(message_id="abc", sender="", subject="", body=""))
        with pytest.raises(IntegrityError):
            s.commit()


def test_foreign_key_and_role_check_are_enforced():
    with SessionLocal() as s:
        s.add(Invoice(email_id=999_999))
        with pytest.raises(IntegrityError):
            s.commit()
    with engine.begin() as conn:
        with pytest.raises(DBAPIError):
            conn.execute(text("INSERT INTO users (username, password_hash, role, is_active, token_version, created_at) "
                              "VALUES ('u', 'h', 'superuser', 1, 0, NOW())"))


def test_database_triggers_make_audit_log_append_only_and_inserts_still_work():
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="x", stage="t", action="a", actor="a" * 100))      # a 100-character reviewer name fits
        s.commit()
    with engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA = DATABASE() "
                                 "AND EVENT_OBJECT_TABLE = 'audit_log'")).scalar() == 2
    for sql in ("UPDATE audit_log SET actor='x'", "DELETE FROM audit_log"):
        with engine.begin() as conn:
            with pytest.raises(DBAPIError, match="append-only"):
                conn.execute(text(sql))
    with SessionLocal() as s:
        assert s.query(AuditLog).count() == 1


def test_init_db_is_idempotent_and_never_loses_rows():
    with SessionLocal() as s:
        s.add(AuditLog(entity_type="x", stage="t", action="a"))
        s.add(Email(message_id="keep", sender="", subject="", body=""))
        s.commit()
    init_db()
    init_db()
    with SessionLocal() as s:
        assert s.query(AuditLog).count() == 1 and s.query(Email).one().message_id == "keep"


def test_old_audit_table_gets_new_columns_and_triggers_on_mysql():
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTO_INCREMENT, entity_type VARCHAR(30), entity_id INTEGER, "
                          "stage VARCHAR(50), action VARCHAR(100), actor VARCHAR(50), details JSON, duration_ms DOUBLE, "
                          "timestamp DATETIME(6))"))
        conn.execute(text("INSERT INTO audit_log (entity_type, stage, action, actor) VALUES ('email','s','a','system')"))
    db._add_missing_columns()
    db._ensure_audit_triggers()
    db._ensure_audit_triggers()                                                       # idempotent
    assert {"request_id", "model_version"} <= {c["name"] for c in inspect(engine).get_columns("audit_log")}
    with engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM audit_log")).scalar() == 1     # history kept
    with engine.begin() as conn:
        with pytest.raises(DBAPIError, match="append-only"):
            conn.execute(text("DELETE FROM audit_log"))


def test_old_emails_table_gets_triage_column_on_mysql_without_losing_rows():
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE emails (id INTEGER PRIMARY KEY AUTO_INCREMENT, message_id VARCHAR(200), sender VARCHAR(200), "
                          "subject VARCHAR(500), body TEXT, received_at DATETIME, category VARCHAR(50), category_confidence DOUBLE)"))
        conn.execute(text("INSERT INTO emails (id, message_id, sender, subject, body) VALUES (1, 'm', 's', 'sub', 'b')"))
    db._add_missing_columns()
    db._add_missing_columns()
    assert "triage_status" in {c["name"] for c in inspect(engine).get_columns("emails")}
    with engine.connect() as conn:
        assert tuple(conn.execute(text("SELECT message_id, triage_status FROM emails")).one()) == ("m", None)
