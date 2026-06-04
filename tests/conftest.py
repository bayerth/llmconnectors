"""Shared fixtures and helpers for client integration tests.

Loads variables from ``.env`` in the project root (gitignored).
See README for the full list of supported environment variable names.
"""

import os
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Map .env aliases to names expected by tests
if not os.getenv("LMSTUDIO_API_KEY") and os.getenv("MAC_STUDIO_API_KEY"):
    os.environ["LMSTUDIO_API_KEY"] = os.environ["MAC_STUDIO_API_KEY"]
if not os.getenv("LMSTUDIO_BASE_URL"):
    os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:1234"
if not os.getenv("LMSTUDIO_MODEL"):
    os.environ["LMSTUDIO_MODEL"] = "qwen/qwen3-4b-2507"
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"].strip('"')

SYSTEM_PROMPT = "you are a helpful assistant"
USER_QUERY = "What is pypi"


def assert_llm_response(response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime):
    assert response_msg is not None, "response message must not be None"
    assert response_msg.strip(), "response message must not be empty"
    assert runtime is not None and runtime >= 0
    assert prompt_tokens is not None and prompt_tokens >= 0
    assert completion_tokens is not None and completion_tokens >= 0
    assert reasoning_tokens is not None and reasoning_tokens >= 0


def run_client_query(client):
    return client.send_request(
        USER_QUERY,
        system_message=SYSTEM_PROMPT,
        ignore_history=True,
    )


requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
requires_gemini = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
)
requires_ollama = pytest.mark.skipif(
    not os.getenv("OLLAMA_MODEL"),
    reason="OLLAMA_MODEL not set",
)


def lmstudio_server_reachable(base_url=None):
    """Return True if LM Studio responds on the v1 models endpoint."""
    url = (base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")).rstrip("/")
    try:
        response = requests.get(f"{url}/api/v1/models", timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False


requires_lmstudio = pytest.mark.skipif(
    not os.getenv("LMSTUDIO_MODEL")
    or not lmstudio_server_reachable(),
    reason="LM Studio not reachable (start server) or LMSTUDIO_MODEL not set",
)
