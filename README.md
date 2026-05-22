# llmconnector-thomas-bayer

Unified Python connectors for calling large language models through a common interface.

Supported backends:

- **OpenAI** (official API and OpenAI-compatible endpoints)
- **Google Gemini** (`google-genai`)
- **Ollama** (local server)
- **LM Studio** (local or remote server, including stateful conversations)

## Installation

```bash
pip install llmconnector-thomas-bayer
```

With a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install llmconnector-thomas-bayer
```

## Quick start

### OpenAI

```python
from openai import OpenAI
from llmconnector_thomas_bayer import OpenAIClient

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
from llmconnector_thomas_bayer import GeminiClient

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
from llmconnector_thomas_bayer import OllamaClient

llm = OllamaClient(model="llama3.2", host="http://localhost:11434")

response, *_ = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

### LM Studio

```python
from llmconnector_thomas_bayer import LMStudioClient

llm = LMStudioClient(
    model="qwen/qwen3-4b-2507",
    base_url="http://localhost:1234",
    api_key=None,  # optional; set LMSTUDIO_API_KEY when required
)

response, *_ = llm.send_request(
    "What is PyPI?",
    system_message="you are a helpful assistant",
)
print(response)
```

For server-side conversation state, use `StatefulLMStudioClient` instead of `LMStudioClient`.

### Model loader helpers

```python
from llmconnector_thomas_bayer.model_loader import load_openai_model, load_ollama_model

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

Run integration tests (live APIs; Ollama skipped without a running server):

```bash
uv run pytest -v
```

Tests load variables from `.env` via `python-dotenv`. See `.env.example` for supported names.

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini |
| `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL` | LM Studio (required for those tests) |
| `LMSTUDIO_API_KEY` or `MAC_STUDIO_API_KEY` | LM Studio auth (optional locally) |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | Ollama (optional) |

## Project layout

```
src/llmconnector_thomas_bayer/   # library code
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
