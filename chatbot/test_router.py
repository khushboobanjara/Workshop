import asyncio

from chatbot.router import detect_intent


async def main():

    test_messages = [
        "Hi",
        "I have fever and cough",
        "I need a neurologist",
        "Book an appointment with a cardiologist",
        "Show me my previous treatment",
        "Cancel my appointment",
        "I have severe chest pain and difficulty breathing"
    ]

    for message in test_messages:

        result = await detect_intent(message)

        print("\nUser:", message)
        print("Intent:", result)


if __name__ == "__main__":
    asyncio.run(main())