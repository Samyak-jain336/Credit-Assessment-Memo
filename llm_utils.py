"""
llm_utils.py — Shared LLM calling utilities for the CAM pipeline.

Supports three providers: Gemini, OpenAI, Azure OpenAI.
Provider is selected via LLM_PROVIDER in .env.

All pipeline code calls either:
  call_llm_primary(prompt)       — for prose / section generation
  call_llm_primary_json(prompt)  — for structured JSON extraction

call_llm() and call_llm_json() are aliases for backward compatibility
with ingestion modules that already import them by those names.
"""

import json
import time

import google.genai as genai
from openai import AzureOpenAI, OpenAI

from config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)


# ------------------------------------------------------------------ #
#  Gemini                                                             #
# ------------------------------------------------------------------ #

def call_gemini(prompt: str, expect_json: bool = False) -> str:
    """Call Gemini with retry + automatic model fallback.

    Attempt order:
      1. GEMINI_MODEL (gemini-2.5-flash) — up to 5 attempts
      2. gemini-2.0-flash                — up to 5 attempts
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
        GEMINI_MODEL,
        "gemini-2.0-flash",
    ]

    last_error = None

    for model in models_to_try:
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
                    wait = 5 * (attempt + 1)
                    print(f"  {model} attempt {attempt + 1} failed ({e}), "
                          f"retrying in {wait}s...")
                    time.sleep(wait)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def call_gemini_json(prompt: str) -> dict:
    """Call Gemini and parse response as JSON."""
    raw = call_gemini(prompt, expect_json=True)
    return _parse_json(raw)


# ------------------------------------------------------------------ #
#  OpenAI                                                             #
# ------------------------------------------------------------------ #

def call_openai(prompt: str, expect_json: bool = False) -> str:
    """Call OpenAI with 5-attempt retry and backoff."""
    if expect_json:
        prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. "
            "No explanation, no markdown fences, no preamble."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    last_error = None

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt == 4:
                print(f"  OpenAI failed all 5 attempts. Last error: {e}")
            else:
                wait = 5 * (attempt + 1)
                print(f"  OpenAI attempt {attempt + 1} failed ({e}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"OpenAI failed. Last error: {last_error}")


def call_openai_json(prompt: str) -> dict:
    """Call OpenAI and parse response as JSON."""
    raw = call_openai(prompt, expect_json=True)
    return _parse_json(raw)


# ------------------------------------------------------------------ #
#  Azure OpenAI                                                       #
# ------------------------------------------------------------------ #

def call_azure_openai(prompt: str, expect_json: bool = False) -> str:
    """Call Azure OpenAI with 5-attempt retry and backoff."""
    if expect_json:
        prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. "
            "No explanation, no markdown fences, no preamble."
        )

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    last_error = None

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt == 4:
                print(f"  Azure OpenAI failed all 5 attempts. Last error: {e}")
            else:
                wait = 5 * (attempt + 1)
                print(f"  Azure OpenAI attempt {attempt + 1} failed ({e}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"Azure OpenAI failed. Last error: {last_error}")


def call_azure_openai_json(prompt: str) -> dict:
    """Call Azure OpenAI and parse response as JSON."""
    raw = call_azure_openai(prompt, expect_json=True)
    return _parse_json(raw)


# ------------------------------------------------------------------ #
#  Unified router — all pipeline code should call these              #
# ------------------------------------------------------------------ #

def call_llm_primary(prompt: str, expect_json: bool = False) -> str:
    """Route to the configured provider based on LLM_PROVIDER in .env.

    This is the single entry point for all LLM calls in the pipeline.
    Switch providers by changing LLM_PROVIDER — no code changes needed.

    Valid values: "gemini", "openai", "azure_openai"
    """
    provider = LLM_PROVIDER.strip().lower()

    if provider == "gemini":
        return call_gemini(prompt, expect_json=expect_json)
    elif provider == "openai":
        return call_openai(prompt, expect_json=expect_json)
    elif provider == "azure_openai":
        return call_azure_openai(prompt, expect_json=expect_json)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Must be one of: gemini, openai, azure_openai"
        )


def call_llm_primary_json(prompt: str) -> dict:
    """Route to configured provider and parse response as JSON."""
    raw = call_llm_primary(prompt, expect_json=True)
    return _parse_json(raw)


# ------------------------------------------------------------------ #
#  Backward-compatibility aliases for ingestion modules              #
#  These modules import call_llm / call_llm_json by name.           #
#  Keeping these aliases means zero changes needed in ingestion.     #
# ------------------------------------------------------------------ #

def call_llm(prompt: str, expect_json: bool = False) -> str:
    """Alias for call_llm_primary(). Replaces Groq-backed call_llm()."""
    return call_llm_primary(prompt, expect_json=expect_json)


def call_llm_json(prompt: str) -> dict:
    """Alias for call_llm_primary_json(). Replaces Groq-backed call_llm_json()."""
    return call_llm_primary_json(prompt)


# ------------------------------------------------------------------ #
#  Internal JSON parser                                               #
# ------------------------------------------------------------------ #

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
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


if __name__ == "__main__":
    # Quick sanity check — run directly to test the active provider:
    # python llm_utils.py
    print(f"Active provider: {LLM_PROVIDER}")
    result = call_llm_primary("Say hello in one sentence.")
    print("Response:", result)
