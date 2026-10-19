"""AI assistant for the dashboard, built on the MCP tools. Groq is the language model; the database is the only source of facts.

  errors.py      GroqError (pure)
  prompts.py     the system prompt and fixed fallback texts (pure)
  verifier.py    checks that every number, amount, date and identifier in an answer appears in the tool results (pure)
  bridge.py      runs the model's tool calls through the MCP core AS THE SIGNED-IN USER (RBAC, validation, redaction, audit)
  service.py     one chat turn: model <-> tools loop, verification, repair, fallback (pure given a model client and a server)
  groq_client.py thin httpx client for Groq's OpenAI-compatible chat API
  audit.py       one audit_log row per chat turn
  routes.py      POST /chat, GET /chat/status, start-up safety checks

The browser never talks to Groq and never sees the API key. Off unless OPSPILOT_CHAT_ENABLED=true.
"""
