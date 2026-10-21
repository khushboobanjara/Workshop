# Sanjeevani Clinic — FastAPI + FastMCP

Sanjeevani Clinic is a FastAPI healthcare assistance application with a FastMCP gateway. The MCP layer reuses the current doctor, pharmacy, and clinic database repositories and adds a dedicated operations console at `/mcp-console`.

## What was added

- **FastMCP Streamable HTTP endpoint:** `/mcp`, protected by a role-specific Bearer token.
- **10 MCP tools:** five USER tools, three ADMIN tools, and two SYSTEM tools.
- **Server-side RBAC:** permissions are enforced by Python policies and not by a role sent from the browser.
- **Database fail-closed control:** `/chat`, the image-analysis routes, and data-backed MCP tools block AI generation when the clinic database is unavailable.
- **Database-only chatbot answers:** general factual questions are answered only from matching, published rows in `clinic_knowledge`, returned verbatim from MySQL. No match or an unavailable knowledge table produces a fixed refusal; the chatbot no longer falls back to general LLM answers.
- **MCP Console page:** `/mcp-console` displays accessible tools, database health, LLM configuration, token usage, and optional cost estimates.
- **Token usage and pricing:** Groq token usage is recorded when the provider returns usage metadata. Per-token cost rates are configurable in `.env`; no unverified rate is hard-coded.

## MCP tool groups

| Group | Tools | Permission |
|---|---|---|
| USER — 5 tools | `user_search_doctors`, `user_list_specializations`, `user_get_doctor_availability`, `user_search_medicines`, `user_get_medicine_details` | `PATIENT`, `USER`, and `DOCTOR` application roles |
| ADMIN — 3 tools | `admin_list_pending_doctors`, `admin_approve_doctor`, `admin_clinic_metrics` | `ADMIN` and `SUPER_ADMIN`; inherits USER tools |
| SYSTEM — 2 tools | `system_database_health`, `system_mcp_status` | `SYSTEM`; `SUPER_ADMIN` can also inspect system status |
| SUPER_ADMIN | Access to all 10 tools | Must be assigned by a trusted administrator; never selectable on public registration |

The browser console lists tools allowed for the current authenticated session. External MCP clients authenticate separately with one Bearer token per role. The USER MCP tools intentionally do not expose any patient's appointment history or other user-specific records via a shared role token.

## Setup

1. Use Python 3.11 or newer. From the project directory, create and activate a virtual environment.

   **Windows PowerShell**

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. Configure `.env` with the existing application's settings. At minimum, set a strong `SESSION_SECRET`, plus `DB_HOST`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME`. Keep the real `.env` file out of version control.

3. In the selected clinic database, run `sql/create_clinic_knowledge.sql` once to create the approved knowledge table. On PowerShell, from the project root, one option is `Get-Content .\sql\create_clinic_knowledge.sql | mysql -u YOUR_DB_USER -p YOUR_DB_NAME`. Replace the placeholders with your actual database username and database name. The table is intentionally created empty and only published rows in an approved health/clinic category are searchable. Add only reviewed, clinic-approved material, for example:

   ```sql
   INSERT INTO clinic_knowledge (title, answer, keywords, category, is_published)
   VALUES (
     'Your reviewed clinic FAQ title',
     'The approved answer exactly as it should be shown to patients.',
     'relevant keyword one, relevant keyword two',
     'HEALTH',
     1
   );
   ```

   Do not add unverified diagnoses, prescriptions, clinic hours, fees, or other policies.

4. Generate four distinct MCP bearer tokens:

   ```bash
   python scripts/generate_mcp_tokens.py
   ```

   Copy the resulting values into `MCP_USER_TOKEN`, `MCP_ADMIN_TOKEN`, `MCP_SYSTEM_TOKEN`, and `MCP_SUPER_ADMIN_TOKEN` in `.env`. The script prints tokens to the terminal and does not write them to a file. Keep these secrets private and rotate them if exposed.

5. Configure `GROQ_API_KEY` for intent classification and image analysis. The chat model defaults to `openai/gpt-oss-120b` and can be changed with `GROQ_CHAT_MODEL`. Image analysis uses `GROQ_VISION_MODEL` as before.

6. Start the app:

   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

7. Sign in through the existing login page and open `http://127.0.0.1:8000/mcp-console`. The console's REST calls use the authenticated FastAPI session. External MCP clients connect to `http://127.0.0.1:8000/mcp/` with the matching `Authorization: Bearer <token>` header.

## Role assignment

Public registration permits only the existing `PATIENT` and `DOCTOR` roles. Do not enable admin access by adding a user-controlled role field. For trusted staff accounts, assign `ADMIN` or `SUPER_ADMIN` in the database using an audited administrative process. `SYSTEM` is intended for a service account/token rather than an ordinary interactive user. Ensure the existing `users.role` column supports the assigned values before creating such accounts.

## LLM usage and estimated pricing

Add verified rates from the current provider/model price schedule to `.env`:

```dotenv
LLM_INPUT_PRICE_PER_1M_USD=
LLM_OUTPUT_PRICE_PER_1M_USD=
```

The example file pre-fills a rate schedule for Groq `openai/gpt-oss-120b` (input $0.15, cached input $0.075, output $0.60 per million tokens at the time configured); verify current rates before using them in billing estimates. Vision-model rates are blank by default because its model and pricing can differ. For the default Groq chat model, the cost estimate is:

```text
((input_tokens − cached_input_tokens) × input_rate + cached_input_tokens × cached_input_rate + output_tokens × output_rate) / 1,000,000
```

The console reports actual input/output token counts when Groq returns usage metadata. The usage ledger is per-user and in-memory, so it resets when the application process restarts and does not replace the provider's billing dashboard. For multi-worker or production use, move the ledger to a shared database/Redis store. MCP data tools do not call the LLM and therefore consume no LLM tokens themselves.

## Database-only chatbot answer policy

- Doctor search, appointment workflows, patient history, pharmacy search, and other supported clinic actions continue to use their dedicated database repositories.
- Other factual questions search only `clinic_knowledge` rows where `is_published = 1` and the category is one of `HEALTH`, `MEDICAL`, `CLINIC`, `FAQ`, `PHARMACY`, `APPOINTMENT`, `DOCTOR`, `SCREENING`, `SAFETY`, or `POLICY`. The service returns the stored `answer` text verbatim and never asks the LLM to summarize or supplement it.
- If no sufficiently relevant row is found, the chatbot returns a fixed `knowledge_not_found` response. If the table/query is unavailable, it returns a fixed `knowledge_unavailable` response. Neither case falls back to a general answer.
- The LLM is used only for classifying likely clinic-action requests whose wording was not captured by deterministic routing, and for the separately configured image-analysis workflow. A classifier result is not shown as a factual answer.
- If MySQL is unavailable, `/chat` returns a static status response before chatbot routing. Image analysis and data-backed MCP tools also fail closed.
- The relevance matcher is conservative keyword overlap, not vector/semantic retrieval. Use clear titles and useful keywords on knowledge records, and publish only accurate, reviewed entries.

Keep the chatbot as an assistance tool, not a substitute for a clinician. Emergency guidance remains a fixed safety response and does not depend on a knowledge-base match.

## Security notes

- Enable HTTPS in production and set `SESSION_HTTPS_ONLY=true` behind HTTPS.
- Use strong, unique, rotated MCP role tokens; do not hard-code secrets in source files or expose them in the browser.
- Do not provide `ADMIN`, `SYSTEM`, or `SUPER_ADMIN` roles in the public registration flow.
- Keep admin approval tools restricted to trusted staff accounts and audit administrative actions in production.
- MCP usage counts/cost summaries are for development visibility. Use a durable shared store and access controls before scaling to multiple workers.

## Quick validation

```bash
python -m compileall -q main.py mcp_layer chatbot
python -m unittest discover -s tests -v
```

`pyproject.toml` is valid project metadata. The source archive intentionally does not include a lock file: the previous archive's lock was out of sync with its dependency list, and a fresh lock should be generated with `uv lock` on a machine that can reach the configured package index.

You can also check `/api/mcp/status` after signing in. The endpoint returns the database reachability state, tool count, model configuration, billing-rate configuration, and only the names of MCP roles whose tokens are configured—it never returns token values.
