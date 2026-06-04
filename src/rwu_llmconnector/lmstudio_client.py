"""LM Studio HTTP client with local history or server-side state."""

import os
import time

import requests

from rwu_llmconnector.llmconnector import LLMCClient

DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY") or os.getenv(
    "MAC_STUDIO_API_KEY"
)
LOCAL_HISTORY_CONTEXT = "local_history"
SERVER_STATE_CONTEXT = "server_state"


class LMStudioClient(LLMCClient):
    """LM Studio client with optional MCP integrations and explicit context modes.

    Context handling:
    - ``local_history`` (default): ``self.history`` is sent as prompt context; ``store=False``.
    - ``server_state``: LM Studio ``response_id`` chain holds context; ``self.history`` is transcript only.
    """

    def __init__(
        self,
        model,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        api_key=DEFAULT_LMSTUDIO_API_KEY,
        integrations=None,
        context_mode=LOCAL_HISTORY_CONTEXT,
        **kwargs,
    ):
        super().__init__(client=None, model=model, **kwargs)
        self.base_url = base_url.rstrip("/") + "/api/v1/chat"
        self.api_key = api_key
        self.logger.debug(f"URL: {self.base_url}")
        self.context_mode = context_mode
        self.integrations = list(integrations) if integrations is not None else []
        self.last_response_id = None
        self._validate_context_mode()

    def _validate_context_mode(self):
        if self.context_mode not in {LOCAL_HISTORY_CONTEXT, SERVER_STATE_CONTEXT}:
            raise ValueError(
                f"Unsupported LM Studio context_mode '{self.context_mode}'. "
                f"Use '{LOCAL_HISTORY_CONTEXT}' or '{SERVER_STATE_CONTEXT}'."
            )

    @property
    def uses_server_state(self):
        """Whether LM Studio's server-side response_id chain is used as model context."""
        return self.context_mode == SERVER_STATE_CONTEXT

    def add_integration(self, integration):
        """Add an LM Studio integration spec (e.g. an MCP server descriptor)."""
        self.integrations.append(integration)

    def set_integrations(self, integrations):
        """Replace integrations without changing request/state handling logic."""
        self.integrations = list(integrations) if integrations is not None else []

    def clear_integrations(self):
        """Remove all integrations while preserving conversation state/history."""
        self.integrations = []

    def clear_history(self, reset_counters=False):
        """Clear local transcript/history and reset the server-state pointer if used."""
        super().clear_history(reset_counters)
        self.last_response_id = None

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        **kwargs,
    ):
        if self.uses_server_state and ignore_history:
            self.last_response_id = None

        payload = self._build_payload(
            user_message=user_message,
            system_message=system_message,
            ignore_history=ignore_history,
            temperature=temperature,
            model=model,
            context_length=kwargs.get("context_length", 8000),
        )
        headers = self._build_headers()

        try:
            start = time.time()
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 60),
            )
            response.raise_for_status()
            data = response.json()
            runtime = time.time() - start
            self.logger.debug(f"LM Studio Request took {runtime} seconds")
        except Exception as e:
            self.logger.error(f"LM Studio API call failed: {e}")
            return None, 0, 0, 0, 0

        if self.uses_server_state:
            self.last_response_id = data.get("response_id")

        response_message = self._extract_response_message(data)
        prompt_tokens, completion_tokens, reasoning_tokens = self._extract_token_usage(
            data
        )

        response_id_log = (
            f" (ID: {self.last_response_id})" if self.uses_server_state else ""
        )
        self.logger.debug(f"LM Studio Response: {response_message}{response_id_log}")

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response_message})

        if write_to_file:
            self.write_json(self.history, filename=self.filename)

        return (
            response_message,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
        )

    def _build_payload(
        self,
        user_message,
        system_message,
        ignore_history,
        temperature,
        model,
        context_length,
    ):
        payload = {
            "model": model if model is not None else self.model,
            "input": self._build_input(user_message, ignore_history),
            "system_prompt": self._resolve_system_prompt(system_message),
            "integrations": self._resolve_integrations(),
            "context_length": context_length,
            "temperature": temperature,
            "store": self.uses_server_state,
        }

        if self.uses_server_state and self.last_response_id:
            payload["previous_response_id"] = self.last_response_id

        return payload

    def _build_input(self, user_message, ignore_history):
        if self.uses_server_state:
            return user_message

        input_data = []
        if not ignore_history:
            for entry in self.history:
                input_data.append({"type": "text", "content": entry["content"]})
        input_data.append({"type": "text", "content": user_message})
        return input_data

    def _resolve_system_prompt(self, system_message):
        if system_message is not None:
            self.system_message = system_message
            self.logger.debug("System message changed")
        system_prompt = (
            self.system_message
            if self.system_message is not None
            else "You are a helpful assistant."
        )
        self.logger.debug(f"System prompt: {system_prompt}")
        return system_prompt

    def _resolve_integrations(self):
        return [
            integration.to_lmstudio_integration()
            if hasattr(integration, "to_lmstudio_integration")
            else integration
            for integration in self.integrations
        ]

    def _build_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_response_message(self, data):
        response_message = ""

        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                response_message = choice["message"].get("content", "")
            elif "text" in choice:
                response_message = choice.get("text", "")
        elif "output" in data:
            if isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and item.get("content"):
                        response_message = item.get("content", "")
            else:
                response_message = data["output"]
        elif "content" in data:
            response_message = data["content"]

        return response_message

    def _extract_token_usage(self, data):
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0

        if "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens", 0)
            completion_tokens = data["usage"].get("completion_tokens", 0)
            reasoning_tokens = data["usage"].get("reasoning_tokens", 0)
        elif "stats" in data:
            prompt_tokens = data["stats"].get("input_tokens", 0)
            reasoning_tokens = data["stats"].get("reasoning_output_tokens", 0)
            total_output = data["stats"].get("total_output_tokens", 0)
            completion_tokens = total_output - reasoning_tokens
            tokens_per_second = data["stats"].get("tokens_per_second", 0)
            time_to_first_token = data["stats"].get("time_to_first_token_seconds", 0)
            self.logger.debug(
                f"tokens_per_second: {tokens_per_second:,.1f}, "
                f"time_to_first_token: {time_to_first_token:.3f}"
            )

        return prompt_tokens, completion_tokens, reasoning_tokens


class StatefulLMStudioClient(LMStudioClient):
    """LM Studio client using server-side state (``context_mode=server_state``)."""

    def __init__(
        self,
        model,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        api_key=DEFAULT_LMSTUDIO_API_KEY,
        integrations=None,
        **kwargs,
    ):
        kwargs["context_mode"] = SERVER_STATE_CONTEXT
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            integrations=integrations,
            **kwargs,
        )
