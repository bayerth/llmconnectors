import os

import pytest
from openai import OpenAI

from llmconnector_thomas_bayer.openai_client import OpenAIClient
from tests.conftest import assert_llm_response, requires_openai, run_client_query


@pytest.fixture
def openai_client():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return OpenAIClient(client, model=model)


@requires_openai
def test_openai_client_answers_pypi_question(openai_client):
    response = run_client_query(openai_client)
    assert_llm_response(*response)
