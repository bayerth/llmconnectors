"""Unified LLM connectors for OpenAI, Gemini, Ollama, and LM Studio.

Quick start::

    from openai import OpenAI
    from rwu_llmconnector import OpenAIClient

    llm = OpenAIClient(OpenAI(), model="gpt-4o-mini")
    text, *_ = llm.send_request("Hello", system_message="you are a helpful assistant")
"""

from rwu_llmconnector.gemini_client import GeminiClient
from rwu_llmconnector.llmconnector import (
    DEFAULT_LOGGER_NAME,
    LLMCClient,
    default_logger,
    get_print_logger,
)
from rwu_llmconnector.lmstudio_client import LMStudioClient, StatefulLMStudioClient
from rwu_llmconnector.ollama_client import OllamaClient
from rwu_llmconnector.openai_client import OpenAIClient

__all__ = [
    "GeminiClient",
    "LLMCClient",
    "LMStudioClient",
    "OllamaClient",
    "OpenAIClient",
    "StatefulLMStudioClient",
    "DEFAULT_LOGGER_NAME",
    "default_logger",
    "get_print_logger",
]

__version__ = "0.1.0"
