import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError

load_dotenv()

key = os.getenv("GROQ_API_KEY")
print("Key loaded:", bool(key))  # prints True/False only, never the key

client = OpenAI(
    api_key=key,
    base_url="https://api.groq.com/openai/v1",
)

for attempt in range(1, 4):
    try:
        r = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Reply with just: Groq works"}],
        )
        print("Reply:", r.choices[0].message.content)
        break
    except RateLimitError:
        print(f"Rate limited (attempt {attempt}), waiting 15s...")
        time.sleep(15)
    except APIStatusError as e:
        print("API error:", e.status_code, e.message)
        break