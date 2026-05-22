import os

import pytest
from google import genai

from rwu_llmconnector.gemini_client import GeminiClient
from tests.conftest import assert_llm_response, requires_gemini, run_client_query


@pytest.fixture
def gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    client = genai.Client(api_key=api_key)
    return GeminiClient(
        client,
        model=model,
        system_message="You are Gemini 3, a specialized assistant for Data Science and ontologies.",
        constraints="",
    )


@requires_gemini
def test_gemini_client_answers_pypi_question(gemini_client):
    response = run_client_query(gemini_client)
    assert_llm_response(*response)
