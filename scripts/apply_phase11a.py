#!/usr/bin/env python3
"""Applies Phase 11 slice A. Two independent groups, each all-or-nothing:

  time-normalisation  mcp_layer/servers/_common.py   fixes MySQL TIME values arriving as "PT10H30M"
  chatbot-migration   chatbot/service.py, main.py    routes doctor search / history / personal questions via MCP

Safe by construction:
  * every anchor must match EXACTLY ONCE, otherwise NOTHING in that group is written;
  * the original files are copied to backup/phase11a/ first;
  * line endings (CRLF or LF) are preserved;
  * running it twice is harmless (an applied file is detected and skipped).

Usage (from the project root):   python scripts/apply_phase11a.py [--check]
"""
import shutil
import sys
from pathlib import Path

MARKER = "MCP-PHASE11A"

SERVICE = [
    # 1. imports
    ("from chatbot.llm import generate_response\nfrom chatbot.router import detect_intent, is_greeting\n",
     "from chatbot.llm import generate_response\nfrom chatbot.router import detect_intent, is_greeting\n"
     "from chatbot import mcp_bridge, mcp_handlers  # MCP-PHASE11A\n"),
    # 2. process_message accepts the verified principal
    ("async def process_message(\n    user_message,\n    conversation_history=None,\n    user=None\n):\n",
     "async def process_message(\n    user_message,\n    conversation_history=None,\n    user=None,\n    principal=None  # MCP-PHASE11A\n):\n"),
    # 3. safety screen on the raw text, before intent detection or any LLM call
    ("    if nearby_result is not None:\n        return nearby_result\n",
     "    if nearby_result is not None:\n        return nearby_result\n\n"
     "    # MCP-PHASE11A: block prompt-injection / SQL / unsafe text before it reaches NLU or the LLM.\n"
     "    if mcp_bridge.enabled():\n\n"
     "        blocked_reply = await mcp_bridge.screen_message(\n            message,\n            principal\n        )\n\n"
     "        if blocked_reply is not None:\n\n"
     "            return chatbot_response(\n                blocked_reply\n            )\n"),
    # 4. route the verified principal to the two migrated handlers
    ("        return await handle_doctor_search(\n            message,\n            intent_data,\n            user\n        )\n",
     "        return await handle_doctor_search(\n            message,\n            intent_data,\n            user,\n            principal  # MCP-PHASE11A\n        )\n"),
    ("        return await handle_patient_history(\n            user\n        )\n",
     "        return await handle_patient_history(\n            user,\n            principal  # MCP-PHASE11A\n        )\n"),
    # 5. GENERAL: questions about the user's OWN records are answered from verified data, never guessed
    ("    response = await _generate_llm_response(\n        message,\n        conversation_history\n    )\n",
     "    # MCP-PHASE11A: 'my appointment / my order / my profile' questions used to reach the LLM with no\n"
     "    # data. They now go MCP 05 -> grounding gate -> LLM, and are blocked if the data is unverified.\n"
     "    if mcp_bridge.enabled():\n\n"
     "        personal_topics = mcp_handlers.personal_topics(\n            message\n        )\n\n"
     "        if personal_topics:\n\n"
     "            return await mcp_handlers.answer_personal_question(\n"
     "                message,\n                personal_topics,\n                principal,\n                conversation_history\n            )\n\n"
     "    response = await _generate_llm_response(\n        message,\n        conversation_history\n    )\n"),
    # 6. doctor search: same output, same booking state machine, data now comes from MCP 02
    ("async def handle_doctor_search(\n    message,\n    intent_data,\n    user=None\n):\n",
     "async def handle_doctor_search(\n    message,\n    intent_data,\n    user=None,\n    principal=None  # MCP-PHASE11A\n):\n"),
    ("    doctors = find_doctors_by_specialization(\n        specialty\n    )\n\n    if not doctors:\n\n        return chatbot_response(\n            f\"I couldn't find any available \"",
     "    if mcp_bridge.enabled():\n\n"
     "        # MCP-PHASE11A: read through the MCP gateway (identity, RBAC, safety, audit), not the repository.\n"
     "        if not specialty:\n\n"
     "            doctors = []\n\n"
     "        else:\n\n"
     "            doctors, failure = await mcp_handlers.fetch_doctors(\n                specialty,\n                principal\n            )\n\n"
     "            if failure is not None:\n\n"
     "                return failure\n\n"
     "    else:\n\n"
     "        doctors = find_doctors_by_specialization(\n            specialty\n        )\n\n"
     "    if not doctors:\n\n        return chatbot_response(\n            f\"I couldn't find any available \""),
    # 7. appointment history
    ("async def handle_patient_history(\n    user\n):\n",
     "async def handle_patient_history(\n    user,\n    principal=None  # MCP-PHASE11A\n):\n"),
    ("    try:\n\n        appointments = get_user_appointments(\n            user[\"user_id\"]\n        )\n\n    except Exception:\n\n        return chatbot_response(\n            \"I could not retrieve your \"\n            \"appointment history right now.\"\n        )\n",
     "    if mcp_bridge.enabled():\n\n"
     "        # MCP-PHASE11A: the appointments of the VERIFIED principal, not of whatever the session dict says.\n"
     "        appointments, failure = await mcp_handlers.fetch_my_appointments(\n            principal\n        )\n\n"
     "        if failure is not None:\n\n"
     "            return failure\n\n"
     "    else:\n\n"
     "        try:\n\n            appointments = get_user_appointments(\n                user[\"user_id\"]\n            )\n\n"
     "        except Exception:\n\n            return chatbot_response(\n                \"I could not retrieve your \"\n                \"appointment history right now.\"\n            )\n"),
]

MAIN = [
    ("        from chatbot.service import process_message\n",
     "        from chatbot.service import process_message\n        from chatbot.mcp_bridge import principal_for_request  # MCP-PHASE11A\n"),
    ("        result = await process_message(\n            message,\n            user=user\n        )\n",
     "        # MCP-PHASE11A: the verified caller (None while MCP_CHATBOT_ENABLED is off).\n"
     "        principal = await principal_for_request(request)\n\n"
     "        result = await process_message(\n            message,\n            user=user,\n            principal=principal\n        )\n"),
]

COMMON = [
    ("from datetime import date, datetime, time\n",
     "from datetime import date, datetime, time, timedelta  # MCP-PHASE11A\n"),
    ("def ok(data: Any, source: str = SOURCE_DB, message: Optional[str] = None, **metadata) -> dict:\n"
     "    return MCPResult.ok(data, source=source, message=message, metadata=metadata).model_dump(mode=\"json\")\n",
     "def plain(value: Any) -> Any:\n"
     "    \"\"\"JSON-safe copy of a database value, applied to every tool result.  # MCP-PHASE11A\n\n"
     "    mysql-connector returns TIME columns as `timedelta`, and pydantic would serialise that as an\n"
     "    ISO-8601 DURATION (\"PT10H30M\"), which a chatbot or UI would show to the user verbatim. Times of\n"
     "    day are normalised to \"HH:MM:SS\" here, once, so no tool has to remember. Dates, numbers and\n"
     "    strings are left to pydantic (dates become \"YYYY-MM-DD\", Decimal becomes \"800.00\").\n"
     "    \"\"\"\n"
     "    if isinstance(value, timedelta):\n"
     "        total = int(value.total_seconds())\n"
     "        if total < 0:                       # not a time of day: leave it for pydantic rather than invent one\n"
     "            return value\n"
     "        return f\"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}\"\n"
     "    if isinstance(value, time):\n"
     "        return value.replace(microsecond=0).isoformat()\n"
     "    if isinstance(value, dict):\n"
     "        return {key: plain(item) for key, item in value.items()}\n"
     "    if isinstance(value, (list, tuple)):\n"
     "        return [plain(item) for item in value]\n"
     "    return value\n\n\n"
     "def ok(data: Any, source: str = SOURCE_DB, message: Optional[str] = None, **metadata) -> dict:\n"
     "    return MCPResult.ok(plain(data), source=source, message=message, metadata=metadata).model_dump(mode=\"json\")\n"),
    ("    items = list(rows or [])\n    return MCPResult.ok(items, source=source, message=None if items else empty_message,\n",
     "    items = plain(list(rows or []))\n    return MCPResult.ok(items, source=source, message=None if items else empty_message,\n"),
]

GROUPS = [
    ("time-normalisation", [("mcp_layer/servers/_common.py", COMMON)]),
    ("chatbot-migration", [("chatbot/service.py", SERVICE), ("main.py", MAIN)]),
]
TARGETS = [target for _, targets in GROUPS for target in targets]    # flat view (used by tests)


def plan(root: Path, targets=None):
    """Returns (new_contents, messages, ok). Reads only; writes nothing."""
    outputs, messages, ok = {}, [], True
    for rel, edits in (TARGETS if targets is None else targets):
        path = root / rel
        if not path.exists():
            messages.append(f"MISSING  {rel}")
            ok = False
            continue
        raw = path.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        if MARKER in text:
            messages.append(f"SKIPPED  {rel} (already applied)")
            continue
        for index, (old, new) in enumerate(edits, 1):
            count = text.count(old)
            if count != 1:
                messages.append(f"MISMATCH {rel} edit #{index}: expected the anchor once, found {count}x: {old.strip().splitlines()[0][:70]!r}")
                ok = False
            else:
                text = text.replace(old, new)
        outputs[rel] = text.replace("\n", newline)
        messages.append(f"READY    {rel} ({len(edits)} edits, {newline.replace(chr(13), 'CR').replace(chr(10), 'LF')})")
    return outputs, messages, ok


def main(argv) -> int:
    root = Path.cwd()
    failed = False
    for name, targets in GROUPS:
        print(f"== {name}")
        outputs, messages, ok = plan(root, targets)
        print("\n".join("   " + m for m in messages))
        if not ok:
            failed = True
            print("   NOTHING in this group was changed: your file differs from what the patch expects.\n"
                  "   Send me the current version of the file(s) named above.")
            continue
        if "--check" in argv:
            continue
        if not outputs:
            print("   already applied")
            continue
        for rel, content in outputs.items():
            backup = root / "backup" / "phase11a" / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, backup)
            (root / rel).write_bytes(content.encode("utf-8"))
        print("   applied (originals saved under backup/phase11a/)")
    if "--check" in argv:
        print("\nCheck only: nothing written.")
    elif not failed:
        print("\nDone. Set MCP_CHATBOT_ENABLED=true to switch the chatbot migration on.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
