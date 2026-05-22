"""Factory helpers for constructing clients from named environments."""

import os
from openai import OpenAI
from rwu_llmconnector.openai_client import OpenAIClient
from rwu_llmconnector.lmstudio_client import LMStudioClient
from rwu_llmconnector.ollama_client import OllamaClient

# Named presets: mac_studio/local use LM Studio; openai uses the official API; ollama is separate
MODELS_CONFIG = {
    "mac_studio": {
        "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234"),
        "api_key": os.getenv("LMSTUDIO_API_KEY") or os.getenv("MAC_STUDIO_API_KEY"),
    },
    "local": {
        "base_url": "http://localhost:1234",
        "api_key": None,
    },
    "openai": {
        "base_url": None,
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "ollama": {
        "host": "http://localhost:11434",
    },
}


def load_openai_model(model_name, env="local", **kwargs):
    """Return a client for an OpenAI-compatible backend.

    Args:
        model_name: Model identifier passed to the client.
        env: Key from :data:`MODELS_CONFIG` (``mac_studio``, ``local``, ``openai``).
        **kwargs: Forwarded to :class:`OpenAIClient` or :class:`LMStudioClient`.

    Returns:
        :class:`LMStudioClient` when ``env == "mac_studio"``, else :class:`OpenAIClient`.
    """
    config = MODELS_CONFIG.get(env)
    if not config:
        raise ValueError(f"Unknown environment: {env}")

    if env in ("mac_studio", "local"):
        return LMStudioClient(
            model=model_name,
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            **kwargs,
        )

    client = OpenAI(
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
    )

    return OpenAIClient(client, model=model_name, **kwargs)


def load_ollama_model(model_name, host=None, **kwargs):
    """Return an :class:`OllamaClient` for the given model tag.

    Args:
        model_name: Ollama model name.
        host: Server URL; defaults to :data:`MODELS_CONFIG` ``ollama`` entry.
        **kwargs: Forwarded to :class:`OllamaClient`.
    """
    if host is None:
        host = MODELS_CONFIG["ollama"]["host"]

    return OllamaClient(model=model_name, host=host, **kwargs)


def get_available_environments():
    """Return configured environment names (keys of :data:`MODELS_CONFIG`)."""
    return list(MODELS_CONFIG.keys())
