from datetime import datetime, timezone

from sqlalchemy import UniqueConstraint, DDL, Boolean, CheckConstraint, event, JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings
from app.context import request_id_var


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# MySQL-specific column types. Each is a plain SQLAlchemy type everywhere else, so behaviour on other databases is unchanged.
#  * Float is single precision (FLOAT) on MySQL, which would silently corrupt money amounts -> use DOUBLE.
#  * DATETIME has no fractional seconds by default and ROUNDS them; the audit log is ordered by timestamp -> keep microseconds.
#  * TEXT holds 64 KB on MySQL, but an email body may be 200 000 characters -> LONGTEXT.
Double = Float().with_variant(mysql.DOUBLE(asdecimal=False), "mysql")   # asdecimal=False: hand back Python floats, not Decimal
Timestamp = DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql")
LongText = Text().with_variant(mysql.LONGTEXT(), "mysql")

# utf8mb4 + a binary collation: full Unicode, and string comparisons stay case- and accent-sensitive exactly as they were
# on SQLite (MySQL's default collation is case-insensitive, which would merge vendors "Acme" and "ACME" and make
# message_ids / usernames collide). Ignored by other databases.
_MYSQL = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin"}


class Email(Base):
    __tablename__ = "emails"
    __table_args__ = (_MYSQL,)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[str] = mapped_column(String(200), unique=True)
    sender: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(LongText)
    received_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    category: Mapped[str | None] = mapped_column(String(50))
    category_confidence: Mapped[float | None] = mapped_column(Double)
    # None = not applicable. "needs_triage" = the system could not tell what this email is, so a person must.
    # Then "dismissed" (not an invoice) or "converted" (a person confirmed it is one).
    triage_status: Mapped[str | None] = mapped_column(String(20), index=True)


class Invoice(Base):
    __tablename__ = "invoices"
    __table_args__ = (_MYSQL,)

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    vendor_name: Mapped[str | None] = mapped_column(String(200))
    invoice_number: Mapped[str | None] = mapped_column(String(100))
    invoice_date: Mapped[str | None] = mapped_column(String(20))
    due_date: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str | None] = mapped_column(String(3))
    total_amount: Mapped[float | None] = mapped_column(Double)
    po_number: Mapped[str | None] = mapped_column(String(100))
    extraction_confidence: Mapped[float | None] = mapped_column(Double)
    risk_score: Mapped[float | None] = mapped_column(Double)
    decision: Mapped[str | None] = mapped_column(String(30))  # auto_approved | human_review | rejected
    # `decision` is the SYSTEM's outcome and never changes.
    # `status` is the lifecycle: approved | pending_review | rejected (humans move pending_review -> approved/rejected)
    status: Mapped[str] = mapped_column(String(30), default="new")
    duplicate_of: Mapped[int | None] = mapped_column(Integer)
    decision_rule: Mapped[str | None] = mapped_column(String(60))
    decision_reasons: Mapped[list | None] = mapped_column(JSON)
    flags: Mapped[list | None] = mapped_column(JSON)
    risk_breakdown: Mapped[dict | None] = mapped_column(JSON)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    review_comment: Mapped[str | None] = mapped_column(LongText)
    reviewed_at: Mapped[datetime | None] = mapped_column(Timestamp)
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin','reviewer','viewer')", name="ck_users_role"), _MYSQL)

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))          # argon2id; the password itself is never stored
    role: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    token_version: Mapped[int] = mapped_column(Integer, default=0)   # bumping it invalidates every issued token
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(Timestamp)


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (_MYSQL,)

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(30))
    entity_id: Mapped[int | None]
    stage: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(100))
    actor: Mapped[str] = mapped_column(String(100), default="system")      # as wide as a reviewer name can be
    details: Mapped[dict | None] = mapped_column(JSON)
    duration_ms: Mapped[float | None] = mapped_column(Double)
    timestamp: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)    # ties every row to one API request
    model_version: Mapped[str | None] = mapped_column(String(40))            # content hash of the model file that ran


class ModelVersion(Base):
    """One row per model FILE (identified by content hash). Status says whether it is the one serving traffic."""
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_version"),
                      CheckConstraint("status IN ('candidate','production','archived')", name="ck_model_status"), _MYSQL)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(40), index=True)      # classifier | anomaly | extractor
    version: Mapped[str] = mapped_column(String(40))                      # sha-<12 hex>, same id the audit log records
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    algorithm: Mapped[str | None] = mapped_column(String(200))
    params: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)                    # includes how and on what data it was measured
    dataset: Mapped[dict | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(LongText)
    file_path: Mapped[str | None] = mapped_column(String(300))
    trained_at: Mapped[datetime | None] = mapped_column(Timestamp)
    registered_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class RagDocument(Base):
    """An uploaded document the AI assistant may quote (policies, contracts, procedures). The original file is NOT kept:
    only its cleaned text, as chunks (rag_chunks). `min_role` is who may retrieve it: viewer = every signed-in user,
    reviewer = reviewers and admins, admin = admins only."""
    __tablename__ = "rag_documents"
    __table_args__ = (CheckConstraint("min_role IN ('admin','reviewer','viewer')", name="ck_rag_documents_min_role"), _MYSQL)

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(10))                       # pdf | docx | txt | md | csv | html
    sha256: Mapped[str] = mapped_column(String(64), unique=True)             # the same content cannot be uploaded twice
    size_bytes: Mapped[int] = mapped_column(Integer)
    min_role: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    pages: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    quarantined_chunks: Mapped[int] = mapped_column(Integer, default=0)      # chunks held back because they look like instructions to an AI
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    uploaded_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(Timestamp, default=utcnow)


class RagChunk(Base):
    """One retrievable passage. Its id is also its id in the FAISS index, so a search hit is always resolved through
    this table (and the document's min_role) before anything is shown: a stale index can never leak a deleted document."""
    __tablename__ = "rag_chunks"
    __table_args__ = (_MYSQL,)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("rag_documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)                        # 1-based position inside the document
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(200))
    text: Mapped[str] = mapped_column(LongText)


class AuditImmutableError(RuntimeError):
    pass


@event.listens_for(AuditLog, "before_insert")
def _stamp_request_id(mapper, connection, target):
    if target.request_id is None:
        target.request_id = request_id_var.get()


@event.listens_for(AuditLog, "before_update")
@event.listens_for(AuditLog, "before_delete")
def _audit_is_append_only(mapper, connection, target):
    raise AuditImmutableError("audit_log is append-only: rows cannot be changed or deleted")


# Second line of defence: the database itself refuses UPDATE/DELETE on audit_log, even from raw SQL or other tools.
# (name, dialect, DDL). MySQL has no RAISE(), so a BEFORE trigger raises SQLSTATE 45000 with the same message.
_AUDIT_TRIGGERS = [(f"audit_log_no_{op.lower()}", "sqlite",
                    DDL(f"CREATE TRIGGER IF NOT EXISTS audit_log_no_{op.lower()} BEFORE {op} ON audit_log "
                        f"BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END"))
                   for op in ("UPDATE", "DELETE")] + [
                  (f"audit_log_no_{op.lower()}", "mysql",
                   DDL(f"CREATE TRIGGER audit_log_no_{op.lower()} BEFORE {op} ON audit_log FOR EACH ROW "
                       f"SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'audit_log is append-only'"))
                  for op in ("UPDATE", "DELETE")]
for _name, _dialect, _ddl in _AUDIT_TRIGGERS:
    event.listen(AuditLog.__table__, "after_create", _ddl.execute_if(dialect=_dialect))


class DatabaseNotConfigured(RuntimeError):
    pass


# With no database configured the engine still has to exist (the model-training scripts import this module at Docker
# build time, where there is no database). The URL is only used when something actually connects.
_UNCONFIGURED_URL = "mysql+pymysql://not-configured@invalid.invalid/not_configured"


def _engine_options(url: str) -> dict:
    if url.startswith("sqlite"):                      # kept only so the test-suite can run without a MySQL server
        return {"connect_args": {"check_same_thread": False}}
    opts: dict = {"pool_pre_ping": True,                                  # drop dead connections instead of failing a request
                  "pool_recycle": settings.db_pool_recycle_seconds,
                  "pool_size": settings.db_pool_size, "max_overflow": settings.db_max_overflow,
                  # Every statement sees the latest committed data (what SQLite effectively gave us) and takes fewer locks.
                  "isolation_level": "READ COMMITTED"}
    if settings.db_ssl_ca:
        opts["connect_args"] = {"ssl": {"ca": settings.db_ssl_ca}}
    return opts


engine = create_engine(settings.database_url or _UNCONFIGURED_URL, future=True, **_engine_options(settings.database_url))
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# (table, column, DDL type). Names are constants in this file, never user input. Alembic is the next step.
_NEW_COLUMNS = [("emails", "triage_status", "VARCHAR(20)"), ("audit_log", "request_id", "VARCHAR(64)"),
                ("audit_log", "model_version", "VARCHAR(40)")]


def init_db() -> None:
    if not settings.database_url:
        raise DatabaseNotConfigured("No database configured. Set OPSPILOT_DB_USER, OPSPILOT_DB_PASSWORD, OPSPILOT_DB_HOST, "
                                    "OPSPILOT_DB_PORT and OPSPILOT_DB_NAME in .env (see .env.example and the README).")
    Base.metadata.create_all(engine)              # creates only what is missing; never alters or drops existing tables
    _add_missing_columns()
    _ensure_audit_triggers()


def _add_missing_columns() -> None:
    """create_all never alters existing tables, so databases created before a column existed would break."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    for table, column, ddl in _NEW_COLUMNS:
        if table in tables and column not in {c["name"] for c in insp.get_columns(table)}:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def _ensure_audit_triggers() -> None:
    """Idempotent: installs whichever append-only triggers are missing (also on audit tables that predate them)."""
    from sqlalchemy import text
    dialect = engine.dialect.name
    triggers = [(n, ddl) for n, d, ddl in _AUDIT_TRIGGERS if d == dialect]
    if not triggers:
        return
    with engine.begin() as conn:
        if dialect == "mysql":
            # MySQL < 8.0.29 has no CREATE TRIGGER IF NOT EXISTS, so look first.
            existing = {r[0] for r in conn.execute(text(
                "SELECT TRIGGER_NAME FROM information_schema.TRIGGERS "
                "WHERE TRIGGER_SCHEMA = DATABASE() AND EVENT_OBJECT_TABLE = 'audit_log'"))}
            triggers = [(n, ddl) for n, ddl in triggers if n not in existing]
        for _, ddl in triggers:
            conn.execute(ddl)
