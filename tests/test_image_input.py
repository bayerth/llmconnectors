"""Unit tests for multimodal image helpers and client wiring."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from rwu_llmconnector.image_input import (
    build_json_schema_response_format,
    build_lmstudio_input_items,
    build_openai_user_content,
    resolve_response_format,
)
from rwu_llmconnector.lmstudio_client import LMStudioClient
from rwu_llmconnector.openai_client import OpenAIClient

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z5+BAQAG/wL1Cq0inwAAAABJRU5ErkJggg=="
)


def test_build_openai_user_content_with_bytes():
    content = build_openai_user_content("describe", [PNG_1X1])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_lmstudio_input_items_includes_image():
    items = build_lmstudio_input_items("what color?", [PNG_1X1], prior_text_contents=["hi"])
    assert items[0] == {"type": "text", "content": "hi"}
    assert items[1] == {"type": "text", "content": "what color?"}
    assert items[2]["type"] == "image"
    assert items[2]["data_url"].startswith("data:image/png;base64,")


def test_resolve_response_format_from_json_schema():
    kwargs = {
        "json_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        "json_schema_name": "demo",
        "timeout": 30,
    }
    fmt = resolve_response_format(kwargs)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "demo"
    assert kwargs == {"timeout": 30}


def test_build_json_schema_response_format_strict_default():
    fmt = build_json_schema_response_format("answer", {"type": "object"})
    assert fmt["json_schema"]["strict"] is True


def test_lmstudio_payload_with_images():
    client = LMStudioClient(model="test", base_url="http://localhost:1234")
    payload = client._build_payload(
        user_message="see image",
        system_message=None,
        ignore_history=True,
        temperature=0,
        model=None,
        images=[PNG_1X1],
    )
    assert payload["input"][-1]["type"] == "image"


@patch("rwu_llmconnector.openai_client.OpenAIClient._call_client")
def test_openai_call_client_sends_response_format(mock_call_client):
    mock_call_client.return_value = (
        MagicMock(message=MagicMock(content='{"a":1}', reasoning_content=None)),
        1,
        1,
        0,
        0.1,
        None,
        None,
    )
    client = OpenAIClient(client=MagicMock(), model="gpt-4o-mini")
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    client.call_client(
        "return json",
        ignore_history=True,
        json_schema=schema,
        json_schema_name="nums",
    )
    request_kwargs = mock_call_client.call_args.args[0]
    assert request_kwargs["response_format"]["json_schema"]["name"] == "nums"


@patch("rwu_llmconnector.openai_client.OpenAIClient._call_client")
def test_openai_call_client_multimodal_user_content(mock_call_client):
    mock_call_client.return_value = (
        MagicMock(message=MagicMock(content="red", reasoning_content=None)),
        1,
        1,
        0,
        0.1,
        None,
        None,
    )
    client = OpenAIClient(client=MagicMock(), model="gpt-4o-mini")
    client.call_client("color?", ignore_history=True, images=[PNG_1X1])
    request_kwargs = mock_call_client.call_args.args[0]
    user = request_kwargs["messages"][-1]["content"]
    assert user[0]["type"] == "text"
    assert user[1]["type"] == "image_url"
