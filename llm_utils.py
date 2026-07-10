"""
llm_utils.py — Shared LLM calling utilities for the CAM pipeline.

Provides a single call_llm() function that tries three Groq API keys
in sequence, falling back to the next key on any failure. Used for
small structured calls (currency/unit detection, field mapping etc.)
across the pipeline.

Also provides call_gemini() / call_gemini_json() for tasks routed
through the Gemini API (google-genai SDK).
"""

import json
import time
from groq import Groq
import google.genai as genai

from config import GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_MODEL
from config import GEMINI_API_KEY, GEMINI_MODEL


def call_llm(prompt: str, expect_json: bool = False) -> str:
    """Call the Groq LLM with a 3-key fallback chain.

    Tries GROQ_API_KEY_1 first, falls back to GROQ_API_KEY_2, then
    GROQ_API_KEY_3 on any exception (rate limit, auth error, timeout etc.).
    Raises RuntimeError if all three keys fail.

    Args:
        prompt:      The user prompt to send to the LLM.
        expect_json: If True, instructs the model to return only valid
                     JSON with no preamble or markdown fences. The caller
                     is still responsible for parsing the returned string.

    Returns:
        The LLM response as a plain string.
    """
    keys = [GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3]

    # If JSON output is expected, append an explicit instruction so the
    # model doesn't wrap the response in markdown backticks or add prose.
    if expect_json:
        prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. "
            "No explanation, no markdown fences, no preamble."
        )

    last_error = None

    for i, key in enumerate(keys, start=1):
        if not key:
            # Key not set in .env — skip silently rather than trying
            # an empty string which would give a confusing auth error.
            continue
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,      # deterministic output for structured tasks
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Groq key {i} failed: {e}. Trying next key...")
            last_error = e

    raise RuntimeError(
        f"All Groq keys failed. Last error: {last_error}"
    )


def call_llm_json(prompt: str) -> dict:
    """Call the LLM and parse the response as JSON.

    Convenience wrapper around call_llm() for structured extraction tasks.
    Strips markdown fences if the model includes them despite instructions.
    Raises ValueError if the response cannot be parsed as JSON.
    """
    raw = call_llm(prompt, expect_json=True)

    # Strip markdown fences in case the model ignores the instruction
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response could not be parsed as JSON.\n"
            f"Raw response: {raw}\n"
            f"Error: {e}"
        )


# ------------------------------------------------------------------ #
#  Gemini LLM helpers  (google-genai SDK)                            #
# ------------------------------------------------------------------ #

def call_gemini(prompt: str, expect_json: bool = False) -> str:
    """Call Gemini with retry + automatic model fallback.

    Attempt order:
      1. gemini-2.5-flash — up to 5 attempts, 5s/10s/15s/20s/25s backoff
      2. gemini-3.1-flash-lite — up to 5 attempts, 5s/10s/15s/20s/25s backoff
         (triggered automatically if all 5 attempts on primary model fail)

    Raises RuntimeError only if both models exhaust all attempts.
    """
    if expect_json:
        prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. "
            "No explanation, no markdown fences, no preamble."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    models_to_try = [
        GEMINI_MODEL,             # gemini-2.5-flash (from config.py)
        "gemini-3.1-flash-lite",  # GA as of May 8 2026 — faster, cheaper fallback
    ]

    last_error = None

    for model in models_to_try:
        print(f"  Trying Gemini model: {model}")
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                if attempt == 4:
                    print(f"  {model} failed all 5 attempts. Last error: {e}")
                else:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s, 25s
                    print(f"  {model} attempt {attempt + 1} failed ({e}), "
                          f"retrying in {wait}s...")
                    time.sleep(wait)

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    )


def call_gemini_json(prompt: str) -> dict:
    """Call Gemini and parse the response as JSON.

    Convenience wrapper around call_gemini() for structured extraction
    tasks. Strips markdown fences if the model includes them despite
    instructions. Raises ValueError if the response cannot be parsed
    as JSON.
    """
    raw = call_gemini(prompt, expect_json=True)

    # Strip markdown fences in case the model ignores the instruction
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini response could not be parsed as JSON.\n"
            f"Raw response: {raw}\n"
            f"Error: {e}"
        )


if __name__ == "__main__":
    # Quick sanity check — run directly to test the Groq chain:
    # python llm_utils.py
    result = call_llm_json(
        'Return a JSON object with key "status" and value "ok".'
    )
    print("LLM chain working:", result)
