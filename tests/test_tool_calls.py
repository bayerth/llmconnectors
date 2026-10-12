"""Unit tests for shared tool-call helpers and OpenAI tool flow."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rwu_llmconnector.llmconnector import call_requested_tools, resolve_max_output_tokens
from rwu_llmconnector.ollama_client import OllamaClient
from rwu_llmconnector.openai_client import OpenAIClient


def _tool_call(name, arguments, tool_call_id="call-1"):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_resolve_max_output_tokens_prefers_explicit_and_alias():
    assert resolve_max_output_tokens(100, instance_default=10) == 100
    assert resolve_max_output_tokens(None, instance_default=10, max_tokens=50) == 50
    assert resolve_max_output_tokens(None, instance_default=10) == 10


@patch("rwu_llmconnector.openai_client.OpenAIClient._call_client")
def test_openai_call_client_maps_max_output_tokens(mock_call_client):
    mock_call_client.return_value = (
        MagicMock(message=MagicMock(content="ok")),
        1,
        1,
        0,
        0.1,
    )
    client = OpenAIClient(client=MagicMock(), model="gpt-4o-mini", max_output_tokens=64)
    client.call_client("hi", ignore_history=True, max_output_tokens=128)

    assert client.max_output_tokens == 64
    request_kwargs = mock_call_client.call_args.args[0]
    assert request_kwargs["max_completion_tokens"] == 128


def test_call_requested_tools_executes_handlers():
    message = SimpleNamespace(
        tool_calls=[_tool_call("add", '{"a": 1, "b": 2}', "call-1")]
    )
    result = call_requested_tools(
        message,
        tool_mapping={"add": lambda a, b: {"sum": a + b}},
    )
    assert result == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"sum": 3}',
        }
    ]


def test_call_requested_tools_supports_ollama_tool_messages():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Toronto"},
                }
            }
        ],
    }
    result = call_requested_tools(
        message,
        tool_mapping={"get_weather": lambda city: f"{city}: 11 degrees celsius"},
    )
    assert result == [
        {
            "role": "tool",
            "tool_name": "get_weather",
            "content": "Toronto: 11 degrees celsius",
        }
    ]


def test_call_requested_tools_requires_tool_mapping():
    message = SimpleNamespace(tool_calls=[_tool_call("add", '{"a": 1, "b": 2}')])
    with pytest.raises(ValueError, match="tool_mapping is required"):
        call_requested_tools(message, tool_mapping=None)


def test_call_requested_tools_raises_for_unknown_tool():
    message = SimpleNamespace(tool_calls=[_tool_call("missing", "{}")])
    with pytest.raises(KeyError, match="No tool handler configured"):
        call_requested_tools(message, tool_mapping={"add": lambda a, b: a + b})


@patch("rwu_llmconnector.openai_client.OpenAIClient._call_client")
def test_call_client_with_tools_follows_up_after_tool_calls(mock_call_client):
    tool_request_message = SimpleNamespace(
        content=None,
        tool_calls=[_tool_call("echo", '{"text": "hi"}', "call-42")],
    )
    final_message = SimpleNamespace(content="done", tool_calls=None)
    mock_call_client.side_effect = [
        (
            SimpleNamespace(
                message=tool_request_message,
                finish_reason="tool_calls",
            ),
            10,
            2,
            0,
            0.1,
        ),
        (
            SimpleNamespace(
                message=final_message,
                finish_reason="stop",
            ),
            5,
            3,
            0,
            0.2,
        ),
    ]

    client = OpenAIClient(client=MagicMock(), model="gpt-4o-mini")
    response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = (
        client.call_client_with_tools(
            "use echo",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            tool_mapping={"echo": lambda text: {"text": text}},
            ignore_history=True,
        )
    )

    assert response == "done"
    assert prompt_tokens == 15
    assert completion_tokens == 5
    assert reasoning_tokens == 0
    assert runtime == pytest.approx(0.3)
    assert mock_call_client.call_count == 2
    follow_up_messages = mock_call_client.call_args_list[1].args[0]["messages"]
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_call_id"] == "call-42"


@patch("rwu_llmconnector.openai_client.OpenAIClient.call_client_with_tools")
def test_send_request_routes_to_call_client_with_tools(mock_call_client_with_tools):
    mock_call_client_with_tools.return_value = ("done", 1, 2, 0, 0.5)
    client = OpenAIClient(client=MagicMock(), model="gpt-4o-mini")

    response, *_ = client.send_request(
        "use echo",
        tools=[{"type": "function", "function": {"name": "echo"}}],
        tool_mapping={"echo": lambda text: {"text": text}},
    )

    assert response == "done"
    mock_call_client_with_tools.assert_called_once()
    assert mock_call_client_with_tools.call_args.args[1] == [
        {"type": "function", "function": {"name": "echo"}}
    ]


@patch("rwu_llmconnector.ollama_client.OllamaClient._call_client")
def test_ollama_call_client_with_tools_follows_up_after_tool_calls(mock_call_client):
    mock_call_client.side_effect = [
        (
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 2,
            },
            10,
            2,
            0,
            0.1,
        ),
        (
            {
                "message": {
                    "role": "assistant",
                    "content": "It is 20 degrees in Tokyo.",
                },
                "prompt_eval_count": 8,
                "eval_count": 5,
            },
            8,
            5,
            0,
            0.2,
        ),
    ]

    client = OllamaClient(model="llama3.2")
    client.client = MagicMock()
    response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = (
        client.call_client_with_tools(
            "what is the weather in tokyo?",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_mapping={"get_weather": lambda city: f"{city}: 20 degrees"},
            ignore_history=True,
        )
    )

    assert response == "It is 20 degrees in Tokyo."
    assert prompt_tokens == 18
    assert completion_tokens == 7
    assert reasoning_tokens == 0
    assert runtime == pytest.approx(0.3)
    assert mock_call_client.call_count == 2
    follow_up_messages = mock_call_client.call_args_list[1].args[0]["messages"]
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_name"] == "get_weather"
