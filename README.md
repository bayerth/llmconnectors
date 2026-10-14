# rwu-llmconnector

Unified Python connectors for calling large language models through a common interface.

Supported backends:

- **OpenAI** (official API, tool/function calling)
- **OpenRouter** and other **OpenAI-compatible** endpoints
- **Google Gemini** (`google-genai`)
- **Ollama** (local server, tool/function calling)
- **LM Studio** (local or remote server, stateful conversations, MCP integrations)

## Installation

```bash
pip install rwu-llmconnector
```

With a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install rwu-llmconnector
```

## Quick start

All provider clients inherit from `LLMClient` and share `send_request()`:

```python
response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    user_message,
    system_message=None,
    ignore_history=False,
    temperature=None,
    model=None,
    tools=None,          # OpenAI-compatible tool definitions
    integrations=None,   # LM Studio MCP / integration specs (per request)
    max_output_tokens=None,  # generation cap (alias: max_tokens via kwargs)
    logger=None,         # per-request logger; default: client instance logger
    **kwargs,            # e.g. tool_mapping, timeout, context_length (LM Studio)
)
```

`send_request_perf(...)` accepts the same arguments and returns `(runtime, tokens_per_second, response_message)` instead of token breakdown.

**Logging:** Clients use `default_logger` at **INFO** (request start/end, token summary, errors). Pass `logger=` to the client constructor or to `send_request` / `send_request_perf` / `call_client` / `call_client_with_tools` to override for a single call. For a custom sink, use `get_print_logger(name, msg_producer=..., level=logging.DEBUG)`.

```python
import logging

from rwu_llmconnector import OpenAIClient, get_print_logger

verbose = get_print_logger("my-app", level=logging.DEBUG)
llm = OpenAIClient(client, model="gpt-4o-mini", logger=verbose)
llm.send_request("Hi", logger=verbose)  # same or different logger per call
```

**Output token cap (`max_output_tokens`):** limits **generated** tokens only (not prompt/context size). Per-call values override the client default; the instance attribute is not mutated. Mapping: OpenAI → `max_completion_tokens`; LM Studio → `max_output_tokens`; Ollama → `num_predict`. Alias: `max_tokens`.

**LM Studio `context_length`:** omit by default so the server uses the context of the **already loaded** model (avoid sending a mismatched value, e.g. 8000 vs 8192). Set on the client or per request when you need an explicit cap:

```python
llm.send_request("...", max_output_tokens=512)           # or max_tokens=512
llm.send_request("...", context_length=8192)             # LM Studio only
```

### Public exports

```python
from rwu_llmconnector import (
    LLMClient,
    OpenAIClient,
    GeminiClient,
    OllamaClient,
    LMStudioClient,
    StatefulLMStudioClient,
    call_requested_tools,
    default_logger,
    get_print_logger,
)
```

`list_models()` is implemented on OpenAI, Ollama, and LM Studio clients (Gemini raises `NotImplementedError`).

### OpenAI

```python
from openai import OpenAI
from rwu_llmconnector import OpenAIClient

client = OpenAI(api_key="your-openai-api-key")
llm = OpenAIClient(client, model="gpt-4o-mini")

response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

Pass `max_output_tokens` (or `max_tokens`) to cap the assistant reply; OpenAI receives `max_completion_tokens`:

```python
response, *_ = llm.send_request("What is PyPI?", max_output_tokens=256)
```

#### Tool calling

When `tools` is set, `OpenAIClient` runs the tool loop automatically: model requests tools → handlers execute → follow-up completion.

Pass `tool_mapping` as `{tool_name: callable}` (keyword args match the tool JSON schema):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text back",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]

response, *_ = llm.send_request(
    "echo hello",
    tools=tools,
    tool_mapping={"echo": lambda text: {"text": text}},
    ignore_history=True,
)
```

You can also call `llm.call_client_with_tools()` directly. `call_client_with_tool()` is a backward-compatible alias.

Optional: `slim_history=True` omits the assistant tool-call message from the follow-up context (keeps history lean).

To reuse tool execution outside `OpenAIClient`:

```python
from rwu_llmconnector import call_requested_tools

tool_messages = call_requested_tools(assistant_message, tool_mapping=handlers, logger=my_logger)
```

### OpenRouter

[OpenRouter](https://openrouter.ai/) is OpenAI-compatible — use `OpenAIClient` with a custom `base_url` (tools work the same way):

```python
import os

from openai import OpenAI
from rwu_llmconnector import OpenAIClient

client = OpenAI(
    base_url="https://openrouter.ai/api/v1/",
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
)
llm = OpenAIClient(client, model="gpt-4o-mini")

response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

Same limits as OpenAI: optional `max_output_tokens` on `OpenAIClient(..., max_output_tokens=256)` or on each `send_request(..., max_output_tokens=256)` (alias `max_tokens`); forwarded as `max_completion_tokens` to the API.

Set `OPEN_ROUTER_API_KEY` in your environment (API key from [openrouter.ai/keys](https://openrouter.ai/keys)).

### Google Gemini

```python
from google import genai
from rwu_llmconnector import GeminiClient

client = genai.Client(api_key="your-gemini-api-key")
llm = GeminiClient(client, model="gemini-2.5-flash-lite")

response, *_ = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

### Ollama

```python
from rwu_llmconnector import OllamaClient

llm = OllamaClient(model="llama3.2", host="http://localhost:11434")

response, *_ = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

`max_output_tokens` maps to Ollama `num_predict` in chat `options`.

Tool calling uses the same `tools` / `tool_mapping` pattern as OpenAI ([Ollama chat API](https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-history-with-tools)):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response, *_ = llm.send_request(
    "what is the weather in Toronto?",
    tools=tools,
    tool_mapping={"get_weather": lambda city: f"{city}: 11 degrees celsius"},
    ignore_history=True,
)
```

### LM Studio

Default server: `http://localhost:1234` (override with `LMSTUDIO_BASE_URL` or env var). If LM Studio **Developer → Authentication** is enabled, set `LMSTUDIO_API_KEY` in `.env` or pass `api_key=` (the client also reads `LMSTUDIO_API_KEY` / `MAC_STUDIO_API_KEY` from the environment when `api_key` is omitted).

```python
from rwu_llmconnector import LMStudioClient, StatefulLMStudioClient

# local_history: send conversation from self.history (default)
llm = LMStudioClient(
    model="qwen/qwen3-4b-2507",
    base_url="http://localhost:1234",
    # api_key=os.environ["LMSTUDIO_API_KEY"],  # when Bearer auth is required
    # Optional defaults (only sent to the API when set):
    # context_length=8192,
    # max_output_tokens=512,
)

response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

**Context and limits**

| Parameter | Constructor | `send_request` / `call_client` | Payload |
|-----------|-------------|----------------------------------|---------|
| `context_length` | `LMStudioClient(..., context_length=8192)` | same name; overrides for that call only | `context_length` (omitted if unset) |
| `max_output_tokens` | `max_output_tokens=512` | same; alias `max_tokens` | `max_output_tokens` |

Do not set `context_length` unless you mean to — a value that does not match the loaded model can force LM Studio to reload and fail (e.g. OOM). Per-request `context_length` / `max_output_tokens` do **not** change `self.context_length` / `self.max_output_tokens`.

For server-side context (`store=True`, `previous_response_id` chain), use `StatefulLMStudioClient`.

#### MCP / integrations

Pass integration specs in the constructor, per request via `send_request(integrations=...)`, or add them later with `add_integration()`. Each entry is either a dict (LM Studio integration JSON) or an object with `to_lmstudio_integration()`. Per-request `integrations` override the client-level list for that call only.

**Hugging Face MCP** (same pattern as the live test in `tests/test_lmstudio_client.py`):

```python
from rwu_llmconnector import LMStudioClient

mcp_hf = {
    "type": "ephemeral_mcp",
    "server_label": "huggingface",
    "server_url": "https://huggingface.co/mcp",
}

llm = LMStudioClient(
    base_url="http://localhost:1234",
    model="openai/gpt-oss-20b",
    temperature=0.0,
    integrations=[mcp_hf],
)

response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    "list 3 new papers",
    ignore_history=True,
    timeout=120,
)
print(response_msg)
print(
    f"prompt tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, "
    f"reasoning_tokens: {reasoning_tokens}, runtime: {runtime:.2f} seconds"
)
```

If Hugging Face is already configured in `~/.lmstudio/mcp.json`, use a plugin id instead:

```python
mcp_hf = {"type": "plugin", "id": "mcp/huggingface"}
```

**Custom integration wrapper** (same hook as `test_integration_to_lmstudio_integration_hook`):

```python
class MyMCP:
    def to_lmstudio_integration(self):
        return {"type": "mcp", "server": "demo"}

llm = LMStudioClient(model="your-model")
llm.add_integration(MyMCP())
response, *_ = llm.send_request("hi", ignore_history=True)
```

### Model loader helpers

```python
from rwu_llmconnector.model_loader import load_openai_model, load_ollama_model

llm = load_openai_model("gpt-4o-mini", env="openai")
ollama = load_ollama_model("llama3.2")
```

Set `OPENAI_API_KEY` in the environment when using `env="openai"`.

## Development

Clone the repository, then:

```bash
uv sync --group dev
cp .env.example .env   # add your API keys locally; never commit .env
```

Run tests (unit tests always; live tests skip if the backend is unreachable):

```bash
uv run pytest -v
```

Focused test runs:

```bash
uv run pytest tests/test_tool_calls.py -v
uv run pytest tests/test_lmstudio_client.py -v
uv run pytest tests/test_lmstudio_client.py::test_lmstudio_mcp_hf_lists_new_papers -v
```

Tests load variables from `.env` via `python-dotenv`. See `.env.example` for supported names.

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `OPEN_ROUTER_API_KEY` | OpenRouter (OpenAI-compatible) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini |
| `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL` | LM Studio live tests (defaults: `http://localhost:1234`, `qwen/qwen3-4b-2507`) |
| `LMSTUDIO_API_KEY` or `MAC_STUDIO_API_KEY` | LM Studio Bearer token (required when local auth is enabled) |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | Ollama (optional) |
| `LMSTUDIO_MCP_MODEL` | MCP live test model (default `openai/gpt-oss-20b`) |
| `LMSTUDIO_MCP_INTEGRATION_JSON` | Override MCP spec (e.g. plugin id from `mcp.json`) |

## Project layout

```
src/rwu_llmconnector/   # library code
tests/                  # unit + live integration tests
pyproject.toml
uv.lock                 # pinned dev dependencies
.env.example            # template for local secrets (commit this)
```

Local scratch scripts can live in `tmp/` (gitignored). Use `cp .env.example .env` and load secrets from `.env`; see `.vscode/launch.json` for `PYTHONPATH=src` when debugging scripts there.

Do **not** commit: `.env`, `.venv/`, `dist/`, `__pycache__/`, `.pytest_cache/`, `.idea/`, `tmp/`

## Publishing to PyPI

Build and upload (requires [PyPI account](https://pypi.org/account/register/) and API token):

```bash
uv build
uv publish --token pypi-AgEIcHlwaS...
```

Or set `UV_PUBLISH_TOKEN` in the environment and run `uv publish`. Upload only the files in `dist/`; do not add `dist/` to git.

## License

MIT — see [LICENSE](LICENSE).
