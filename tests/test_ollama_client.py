import os

import pytest

from rwu_llmconnector.ollama_client import OllamaClient
from tests.conftest import assert_llm_response, requires_ollama, run_client_query


@pytest.fixture
def ollama_client():
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ["OLLAMA_MODEL"]
    return OllamaClient(model=model, host=host)


@requires_ollama
def test_ollama_client_answers_pypi_question(ollama_client):
    response = run_client_query(ollama_client)
    assert_llm_response(*response)
