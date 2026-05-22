import os

import pytest

from rwu_llmconnector.lmstudio_client import LMStudioClient, StatefulLMStudioClient
from tests.conftest import assert_llm_response, requires_lmstudio, run_client_query


def _lmstudio_client_kwargs():
    """Build LM Studio client kwargs from environment variables."""
    kwargs = {
        "base_url": os.environ["LMSTUDIO_BASE_URL"],
        "model": os.environ["LMSTUDIO_MODEL"],
    }
    api_key = os.getenv("LMSTUDIO_API_KEY")
    if api_key is not None:
        kwargs["api_key"] = api_key
    else:
        kwargs["api_key"] = None
    return kwargs


@pytest.fixture
def lmstudio_client():
    return LMStudioClient(**_lmstudio_client_kwargs())


@pytest.fixture
def stateful_lmstudio_client():
    return StatefulLMStudioClient(**_lmstudio_client_kwargs())


@requires_lmstudio
def test_lmstudio_client_answers_pypi_question(lmstudio_client):
    response = run_client_query(lmstudio_client)
    assert_llm_response(*response)


@requires_lmstudio
def test_stateful_lmstudio_client_answers_pypi_question(stateful_lmstudio_client):
    response = run_client_query(stateful_lmstudio_client)
    assert_llm_response(*response)
