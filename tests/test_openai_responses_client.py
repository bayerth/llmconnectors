"""Unit tests for OpenAIResponsesClient."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rwu_llmconnector.llmconnector import (
    convert_chat_completion_tools_to_responses,
    extract_responses_function_calls,
)
from rwu_llmconnector.openai_responses_client import OpenAIResponsesClient


def test_convert_chat_completion_tools_to_responses():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                "strict": True,
            },
        }
    ]
    converted = convert_chat_completion_tools_to_responses(tools)
    assert converted[0]["name"] == "echo"
    assert converted[0]["type"] == "function"
    assert "function" not in converted[0]
    assert converted[0]["strict"] is True


def test_extract_responses_function_calls():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="message", content="hi"),
            SimpleNamespace(type="function_call", name="echo", call_id="call_1", arguments='{"text":"x"}'),
        ]
    )
    calls = extract_responses_function_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "echo"


@patch("rwu_llmconnector.openai_responses_client.OpenAIResponsesClient._call_responses")
def test_responses_client_maps_max_output_tokens(mock_call_responses):
    mock_call_responses.return_value = (
        SimpleNamespace(output_text="ok", output=[], id="resp_1", usage=None),
        1,
        2,
        0,
        0.5,
    )
    api = MagicMock()
    client = OpenAIResponsesClient(api, model="gpt-4o-mini", max_output_tokens=64)
    client.call_client("hi", ignore_history=True, max_output_tokens=128)

    request_kwargs = mock_call_responses.call_args.args[0]
    assert request_kwargs["max_output_tokens"] == 128


@patch("rwu_llmconnector.openai_responses_client.OpenAIResponsesClient._call_responses")
def test_responses_tool_loop_follows_up(mock_call_responses):
    first = SimpleNamespace(
        output_text="",
        id="resp_1",
        output=[
            SimpleNamespace(
                type="function_call",
                name="echo",
                call_id="call_abc",
                arguments='{"text":"hello"}',
            )
        ],
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=5,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    second = SimpleNamespace(
        output_text="hello",
        id="resp_2",
        output=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            output_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
    )
    mock_call_responses.side_effect = [
        (first, 3, 5, 0, 0.2),
        (second, 10, 3, 1, 0.3),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]
    api = MagicMock()
    client = OpenAIResponsesClient(api, model="gpt-4o-mini")
    text, prompt, completion, reasoning, runtime = client.call_client_with_tools(
        "echo hello",
        tools=tools,
        tool_mapping={"echo": lambda text: text},
        ignore_history=True,
    )

    assert text == "hello"
    assert mock_call_responses.call_count == 2
    second_input = mock_call_responses.call_args_list[1].args[0]["input"]
    assert any(item.get("type") == "function_call_output" for item in second_input)
    assert prompt == 13
    assert completion == 8
    assert reasoning == 1
