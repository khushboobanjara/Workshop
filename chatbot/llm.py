import os

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Please add it to the .env file."
    )

client = AsyncGroq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"


async def generate_response(
    system_prompt: str,
    messages: list
) -> str:

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ],
        temperature=0.3,
        max_completion_tokens=500
    )

    return response.choices[0].message.content