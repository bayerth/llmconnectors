# rwu-llmconnector

Unified Python connectors for calling large language models through a common interface.

Supported backends:

- **OpenAI** (official API and OpenAI-compatible endpoints)
- **Google Gemini** (`google-genai`)
- **Ollama** (local server)
- **LM Studio** (local or remote server, stateful conversations, MCP tool integrations)

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

### LM Studio

Default server: `http://localhost:1234` (override with `LMSTUDIO_BASE_URL` or env var).

```python
from rwu_llmconnector import LMStudioClient, StatefulLMStudioClient

# local_history: send conversation from self.history (default)
llm = LMStudioClient(
    model="qwen/qwen3-4b-2507",
    base_url="http://localhost:1234",
    api_key=None,
)

response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

For server-side context (`store=True`, `previous_response_id` chain), use `StatefulLMStudioClient`.

#### MCP / tool integrations

Pass integration specs in the constructor, or add them later with `add_integration()`. Each entry is either a dict (LM Studio integration JSON) or an object with `to_lmstudio_integration()`.

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

**Custom integration wrapper** (same hook as the unit test `test_integration_to_lmstudio_integration_hook`):

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

LM Studio only (including MCP):

```bash
uv run pytest tests/test_lmstudio_client.py -v
uv run pytest tests/test_lmstudio_client.py::test_lmstudio_mcp_hf_lists_new_papers -v
```

Tests load variables from `.env` via `python-dotenv`. See `.env.example` for supported names.

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini |
| `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL` | LM Studio live tests (defaults: `http://localhost:1234`, `qwen/qwen3-4b-2507`) |
| `LMSTUDIO_API_KEY` or `MAC_STUDIO_API_KEY` | LM Studio auth (optional locally) |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | Ollama (optional) |
| `LMSTUDIO_MCP_MODEL` | MCP live test model (default `openai/gpt-oss-20b`) |
| `LMSTUDIO_MCP_INTEGRATION_JSON` | Override MCP spec (e.g. plugin id from `mcp.json`) |

## Project layout

```
src/rwu_llmconnector/   # library code
tests/                           # integration tests
pyproject.toml
uv.lock                          # pinned dev dependencies
.env.example                     # template for local secrets (commit this)
```

Do **not** commit: `.env`, `.venv/`, `dist/`, `__pycache__/`, `.pytest_cache/`, `.idea/`

## Publishing to PyPI

Build and upload (requires [PyPI account](https://pypi.org/account/register/) and API token):

```bash
uv build
uv publish
```

Upload only the files in `dist/`; do not add `dist/` to git.

After creating the GitHub repository, uncomment and set `Repository` under `[project.urls]` in `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE).
