"""Tests for LM Studio client (unit + live integration)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from rwu_llmconnector.lmstudio_client import (
    LOCAL_HISTORY_CONTEXT,
    LMStudioClient,
    SERVER_STATE_CONTEXT,
    StatefulLMStudioClient,
)
from tests.conftest import assert_llm_response, lmstudio_server_reachable, requires_lmstudio, run_client_query
from tests.mcp_fixtures import load_mcp_hf_integration

LMSTUDIO_MCP_MODEL = os.getenv("LMSTUDIO_MCP_MODEL", "openai/gpt-oss-20b")

requires_lmstudio_mcp = pytest.mark.skipif(
    not lmstudio_server_reachable(os.environ["LMSTUDIO_BASE_URL"]),
    reason=f"LM Studio not reachable at {os.environ['LMSTUDIO_BASE_URL']}",
)


def _lmstudio_client_kwargs(context_mode=None):
    kwargs = {
        "base_url": os.environ["LMSTUDIO_BASE_URL"],
        "model": os.environ["LMSTUDIO_MODEL"],
        "api_key": os.getenv("LMSTUDIO_API_KEY"),
    }
    if context_mode is not None:
        kwargs["context_mode"] = context_mode
    return kwargs


@pytest.fixture
def lmstudio_client():
    return LMStudioClient(**_lmstudio_client_kwargs())


@pytest.fixture
def stateful_lmstudio_client():
    return StatefulLMStudioClient(**_lmstudio_client_kwargs())


def test_default_base_url_is_localhost_port_1234():
    client = LMStudioClient(model="test-model")
    assert client.base_url == "http://localhost:1234/api/v1/chat"


def test_invalid_context_mode_raises():
    with pytest.raises(ValueError, match="context_mode"):
        LMStudioClient(model="test", context_mode="invalid")


def test_stateful_client_uses_server_state_mode():
    client = StatefulLMStudioClient(model="test")
    assert client.context_mode == SERVER_STATE_CONTEXT
    assert client.uses_server_state is True


def test_local_history_builds_list_input():
    client = LMStudioClient(model="test", base_url="http://localhost:1234")
    client.history.append({"role": "user", "content": "prior"})
    payload = client._build_payload(
        user_message="hello",
        system_message="you are helpful",
        ignore_history=False,
        temperature=0,
        model=None,
        context_length=8000,
    )
    assert payload["store"] is False
    assert payload["input"] == [
        {"type": "text", "content": "prior"},
        {"type": "text", "content": "hello"},
    ]
    assert payload["integrations"] == []


def test_server_state_builds_string_input_and_store_true():
    client = LMStudioClient(
        model="test",
        base_url="http://localhost:1234",
        context_mode=SERVER_STATE_CONTEXT,
    )
    payload = client._build_payload(
        user_message="hello",
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        context_length=8000,
    )
    assert payload["store"] is True
    assert payload["input"] == "hello"
    assert "previous_response_id" not in payload


def test_server_state_chains_previous_response_id():
    client = LMStudioClient(
        model="test",
        context_mode=SERVER_STATE_CONTEXT,
    )
    client.last_response_id = "resp-123"
    payload = client._build_payload(
        user_message="follow-up",
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        context_length=8000,
    )
    assert payload["previous_response_id"] == "resp-123"


def test_integration_to_lmstudio_integration_hook():
    class MCPStub:
        def to_lmstudio_integration(self):
            return {"type": "mcp", "server": "demo"}

    client = LMStudioClient(model="test")
    client.add_integration(MCPStub())
    payload = client._build_payload(
        user_message="hi",
        system_message=None,
        ignore_history=True,
        temperature=0,
        model=None,
        context_length=8000,
    )
    assert payload["integrations"] == [{"type": "mcp", "server": "demo"}]


@patch("rwu_llmconnector.lmstudio_client.requests.post")
def test_call_client_parses_output_list(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "output": [{"type": "message", "content": "PyPI is the Python Package Index."}],
            "stats": {
                "input_tokens": 10,
                "total_output_tokens": 5,
                "reasoning_output_tokens": 0,
            },
        },
    )
    mock_post.return_value.raise_for_status = MagicMock()

    client = LMStudioClient(model="test", base_url="http://localhost:1234", api_key=None)
    message, prompt, completion, reasoning, runtime = client.call_client(
        "What is pypi?",
        system_message="you are a helpful assistant",
        ignore_history=True,
    )

    assert message == "PyPI is the Python Package Index."
    assert prompt == 10
    assert completion == 5
    assert reasoning == 0
    assert runtime >= 0
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["store"] is False


@requires_lmstudio
def test_lmstudio_local_history_live(lmstudio_client):
    assert lmstudio_client.context_mode == LOCAL_HISTORY_CONTEXT
    response = run_client_query(lmstudio_client)
    assert_llm_response(*response)


@requires_lmstudio
def test_stateful_lmstudio_server_state_live(stateful_lmstudio_client):
    assert stateful_lmstudio_client.uses_server_state is True
    response = run_client_query(stateful_lmstudio_client)
    assert_llm_response(*response)


@requires_lmstudio_mcp
def test_lmstudio_mcp_hf_lists_new_papers():
    """Live MCP test: Hugging Face integration via LM Studio."""
    mcp_hf = load_mcp_hf_integration()
    llm_client = LMStudioClient(
        base_url=os.environ["LMSTUDIO_BASE_URL"],
        model=LMSTUDIO_MCP_MODEL,
        temperature=0.0,
        integrations=[mcp_hf],
        api_key=os.getenv("LMSTUDIO_API_KEY"),
    )

    (
        response_msg,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        runtime,
    ) = llm_client.send_request("list 3 new papers", ignore_history=True, timeout=120)

    assert_llm_response(
        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime
    )
    assert llm_client.integrations == [mcp_hf]
