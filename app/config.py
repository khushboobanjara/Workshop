import os
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    # Database: MySQL 8.0.16+ (or MariaDB 10.5+). Set the OPSPILOT_DB_* parts in .env and the URL is assembled
    # for you (passwords with special characters are escaped correctly). OPSPILOT_DATABASE_URL, when set, wins.
    database_url: str = ""
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_name: str = "opspilot"
    db_user: str = ""
    db_password: SecretStr = SecretStr("")
    db_charset: str = "utf8mb4"
    db_ssl_ca: str = ""                            # optional path to a CA certificate to require TLS to the server
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 1800            # recycle before MySQL's wait_timeout closes idle connections
    classifier_model_path: str = "models/email_classifier.joblib"
    classifier_min_confidence: float = 0.70
    anomaly_model_path: str = "models/anomaly_detector.joblib"
    # Decision thresholds (tuned in Phase 6)
    auto_approve_max_risk: float = 0.30
    auto_reject_min_risk: float = 0.85
    auto_approve_max_amount: float = 10000.0
    new_vendor_review_amount: float = 2000.0

    env: str = "development"                      # "production" refuses to start without authentication and a strong secret
    # Authentication. Off by default so local development and the test suite stay simple; turn it on for anything shared.
    auth_enabled: bool = False
    jwt_secret: SecretStr = SecretStr("")          # >= 32 random characters, e.g. python -c "import secrets;print(secrets.token_urlsafe(48))"
    session_minutes: int = 480
    cookie_secure: bool = False                    # set true when served over HTTPS
    bootstrap_admin_user: str = ""                 # optional: creates this admin on first start if no users exist
    bootstrap_admin_password: SecretStr = SecretStr("")
    demo_password: str = "opspilot-demo"           # password of the public demo-viewer / demo-reviewer accounts (demo mode only)
    # Hardening / public demo
    api_key: SecretStr = SecretStr("")      # when set, every API call must send header X-API-Key
    max_request_bytes: int = 1_000_000      # larger request bodies get 413
    demo_mode: bool = False                 # public demo: seed sample data, rate-limit writes, reset on a schedule
    demo_writes_per_minute: int = 60
    demo_reset_minutes: int = 60
    # Mailbox integration (see .env.example). Credentials come from the environment or a git-ignored .env file.
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: SecretStr = SecretStr("")
    imap_folder: str = "INBOX"
    api_url: str = "http://127.0.0.1:8000"
    poll_interval_seconds: int = 60
    smtp_host: str = "smtp.gmail.com"      # only used by scripts/send_test_emails.py
    smtp_port: int = 465

    # Model Context Protocol server (see docs/MCP.md). Off by default; refuses to start without authentication.
    mcp_enabled: bool = False
    mcp_system_key: SecretStr = SecretStr("")      # >= 32 random chars; Bearer credential of the read-only 'system' role (separate from api_key)
    mcp_stdio_token: SecretStr = SecretStr("")     # credential used only by `python -m app.mcp_server` (a user JWT or the system key)
    mcp_allowed_origins: str = ""                  # comma-separated browser origins allowed to call POST /mcp; empty = none
    mcp_rate_limit_per_minute: int = 60            # tool calls per principal per minute
    mcp_max_message_bytes: int = 65_536

    # AI assistant (Groq) that answers questions by calling the MCP tools. See docs/CHAT.md. Off by default.
    chat_enabled: bool = False
    groq_api_key: SecretStr = SecretStr("")        # from https://console.groq.com/keys; never sent to the browser
    groq_model: str = "openai/gpt-oss-120b"    # any Groq model that supports tool calling
    groq_base_url: str = "https://api.groq.com/openai/v1"
    chat_timeout_seconds: float = 30.0
    chat_temperature: float = 0.1
    chat_fact_check: bool = True                   # second model call that checks every statement against the records (more accurate, slower)
    chat_rate_limit_per_minute: int = 20           # questions per user per minute
    # Groq's free tier caps TOKENS per minute per model (see your console's Limits page). A 429 carries "try again in N s":
    # wait that long (up to the cap) and retry instead of failing the question. Leaner prompts keep a question under the cap.
    chat_rate_limit_retries: int = 2
    chat_rate_limit_max_wait_seconds: float = 12.0
    chat_hardcode: bool = True                     # answer what fixed rules can (greetings, counts, lists, document quotes) WITHOUT the model
    chat_history_turns: int = 6                    # earlier messages re-sent with each question (each one costs tokens every call)
    chat_history_chars: int = 500                  # ... and how much of each one

    # Retrieval-augmented generation over uploaded documents (policies, contracts, procedures). See docs/RAG.md. Off by default.
    # Everything here is free and local: Sentence Transformers embeddings run on this machine, FAISS is an in-process library,
    # and the only external call is the chat model on Groq that the assistant already uses.
    rag_enabled: bool = False
    rag_data_dir: str = "data/rag"                  # FAISS index + metadata live here (derived data: it can be rebuilt from MySQL)
    rag_embedder: str = "sentence-transformers"     # "sentence-transformers" (real, semantic) | "hashing" (lexical stand-in for tests/dev; refused in production)
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # free, Apache-2.0, 384 dimensions, ~90 MB, runs on CPU
    rag_embedding_local_files_only: bool = False    # true = never contact Hugging Face (model must already be cached or be a local path)
    rag_embedding_batch_size: int = 32
    rag_query_prefix: str = ""                      # some models (e5, bge) want a prefix such as "query: " / "passage: "
    rag_document_prefix: str = ""
    rag_chunk_chars: int = 900                      # target chunk size (~220 tokens; fits MiniLM's 256-token window)
    rag_chunk_overlap_chars: int = 150
    rag_min_chunk_chars: int = 40
    rag_top_k: int = 5                              # passages given to the model
    rag_min_score: float = 0.30                     # cosine similarity floor: weaker matches are not evidence (tune with scripts/evaluate_retrieval.py)
    rag_max_chunks_per_document: int = 3
    rag_max_upload_bytes: int = 10_000_000
    rag_max_documents: int = 500
    rag_max_pdf_pages: int = 300
    rag_max_extracted_chars: int = 2_000_000
    rag_min_support: float = 0.40                   # share of an answer sentence's content words that must occur in the retrieved/returned data (0 = off)
    rag_search_rate_limit_per_minute: int = 30      # document searches per user per minute (chat and /rag/search together)
    rag_answer_mode: str = "auto"                   # auto = quote the documents when a sentence answers the question, else one model call; llm = always the model
    rag_direct_min_overlap: float = 0.5             # share of the question's content words a quoted sentence must cover
    rag_preload: bool = False                       # load the embedding model at start-up instead of on first use

    model_config = {"env_prefix": "OPSPILOT_", "env_file": os.environ.get("OPSPILOT_ENV_FILE", ".env"), "extra": "ignore"}

    @model_validator(mode="after")
    def _assemble_database_url(self):
        if not self.database_url and self.db_user:
            self.database_url = URL.create(
                "mysql+pymysql", username=self.db_user, password=self.db_password.get_secret_value() or None,
                host=self.db_host, port=self.db_port, database=self.db_name,
                query={"charset": self.db_charset}).render_as_string(hide_password=False)
        return self


settings = Settings()
