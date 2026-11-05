"""Error response helpers and send_request failure behavior."""

from unittest.mock import MagicMock, patch

from rwu_llmconnector import is_error_response
from rwu_llmconnector.llmconnector import normalize_error_response
from rwu_llmconnector.lmstudio_client import LMStudioClient


def test_is_error_response():
    assert is_error_response(None)
    assert is_error_response("Error: boom")
    assert not is_error_response("hello")
    assert not is_error_response("")


def test_normalize_error_response():
    assert normalize_error_response(None).startswith("Error:")
    assert normalize_error_response("Error: x") == "Error: x"
    assert normalize_error_response("ok") == "ok"


@patch.object(LMStudioClient, "call_client")
def test_send_request_skips_success_metrics_on_error(mock_call_client):
    mock_call_client.return_value = ("Error: LM Studio API call failed: 401", 0, 0, 0, 0.1)
    client = LMStudioClient(model="test", base_url="http://localhost:1234", api_key=None)
    client.logger = MagicMock()

    response, prompt, completion, reasoning, runtime = client.send_request("hi")

    assert response.startswith("Error:")
    assert client.total_prompt_tokens == 0
    assert client.total_completion_tokens == 0
    assert client.total_runtime == 0
    client.logger.error.assert_any_call(
        f"Request finished with error: {response}",
    )
    summary_info = [
        call
        for call in client.logger.info.call_args_list
        if call.args and "t/s:" in call.args[0]
    ]
    assert not summary_info
