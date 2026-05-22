"""Unified LLM connectors for OpenAI, Gemini, Ollama, and LM Studio.

Quick start::

    from openai import OpenAI
    from llmconnector_thomas_bayer import OpenAIClient

    llm = OpenAIClient(OpenAI(), model="gpt-4o-mini")
    text, *_ = llm.send_request("Hello", system_message="you are a helpful assistant")
"""

from llmconnector_thomas_bayer.gemini_client import GeminiClient
from llmconnector_thomas_bayer.llmconnector import (
    DEFAULT_LOGGER_NAME,
    LLMCClient,
    default_logger,
    get_print_logger,
)
from llmconnector_thomas_bayer.lmstudio_client import LMStudioClient, StatefulLMStudioClient
from llmconnector_thomas_bayer.ollama_client import OllamaClient
from llmconnector_thomas_bayer.openai_client import OpenAIClient

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
