"""Groq chat client with fail-closed database checks and token accounting."""

import os

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. Please add it to the .env file."
    )

client = AsyncGroq(api_key=GROQ_API_KEY)
MODEL_NAME = (os.getenv("GROQ_CHAT_MODEL") or "openai/gpt-oss-120b").strip()


async def generate_response(system_prompt: str, messages: list) -> str:
    """Generate a response only when the clinic database is reachable.

    The API route applies the same guard before chatbot routing. This second
    check is deliberate defense-in-depth for callers that invoke the LLM helper
    directly. A failed DB health check raises before any provider request occurs.
    """
    from mcp_layer.db_guard import check_database

    database_status = check_database()
    if not database_status.get("available"):
        raise RuntimeError("Clinic database unavailable; LLM generation blocked.")

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=0.2,
        max_completion_tokens=500,
    )

    # Store counts only. Prompt/response text is not retained by the accounting layer.
    usage = getattr(response, "usage", None)
    if usage is not None:
        from mcp_layer.usage import set_request_usage

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_input_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or prompt_tokens + completion_tokens
        )
        set_request_usage({
            "model": MODEL_NAME,
            "prompt_tokens": prompt_tokens,
            "cached_input_tokens": cached_input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })

    return response.choices[0].message.content or "I could not generate a response. Please try again."
