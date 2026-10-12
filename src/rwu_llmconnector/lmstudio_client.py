"""LM Studio HTTP client with local history or server-side state."""

import os
import time

import requests

from rwu_llmconnector.llmconnector import LLMClient

DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY") or os.getenv(
    "MAC_STUDIO_API_KEY"
)
LOCAL_HISTORY_CONTEXT = "local_history"
SERVER_STATE_CONTEXT = "server_state"


class LMStudioClient(LLMClient):
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
        context_length=None,
        max_output_tokens=None,
        **kwargs,
    ):
        super().__init__(
            client=None,
            model=model,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        self.base_url = base_url.rstrip("/") + "/api/v1/chat"
        self.api_key = api_key
        self.logger.debug(f"URL: {self.base_url}")
        self.context_mode = context_mode
        self.context_length = context_length
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

    def call_client_with_tools(
        self,
        user_message,
        tools,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        tool_mapping=None,
        integrations=None,
        slim_history=False,
        context_length=None,
        max_output_tokens=None,
        **kwargs,
    ):
        """LM Studio native tools/MCP via payload; no OpenAI-style tool_mapping loop."""
        return self.call_client(
            user_message,
            system_message=system_message,
            ignore_history=ignore_history,
            temperature=temperature,
            model=model,
            write_to_file=write_to_file,
            tools=tools,
            integrations=integrations,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        integrations=None,
        tools=None,
        context_length=None,
        max_output_tokens=None,
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
            context_length=self._resolve_context_length(context_length, **kwargs),
            max_output_tokens=self._resolve_max_output_tokens(max_output_tokens, **kwargs),
            tools=tools,
            integrations=integrations,
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
        except requests.RequestException as e:
            self._log_request_error(e)
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
        context_length=None,
        max_output_tokens=None,
        tools=None,
        integrations=None,
    ):
        payload = {
            "model": model if model is not None else self.model,
            "input": self._build_input(user_message, ignore_history),
            "system_prompt": self._resolve_system_prompt(system_message),
            "integrations": self._resolve_integrations(integrations),
            "temperature": temperature,
            "store": self.uses_server_state,
        }

        if context_length is not None:
            payload["context_length"] = context_length
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if tools is not None:
            payload["tools"] = tools

        if self.uses_server_state and self.last_response_id:
            payload["previous_response_id"] = self.last_response_id

        return payload

    def _resolve_context_length(self, context_length=None, **kwargs):
        """Resolve per-request context length without mutating ``self.context_length``."""
        if context_length is not None:
            return context_length
        kw_context_length = kwargs.get("context_length")
        if kw_context_length is not None:
            return kw_context_length
        return self.context_length

    def _log_request_error(self, error):
        message = str(error)
        response = getattr(error, "response", None)
        if response is not None:
            message = f"{message}; response body: {response.text}"
        self.logger.error(f"LM Studio API call failed: {message}")

    def list_models(self):
        """Return model ids from ``GET /api/v1/models``."""
        models_url = self.base_url.replace("/api/v1/chat", "/api/v1/models")
        try:
            response = requests.get(
                models_url, headers=self._build_headers(), timeout=60
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            self._log_request_error(e)
            return []

        entries = data.get("data") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return []

        names = []
        for item in entries:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("id") or item.get("name") or item.get("model")
                if name:
                    names.append(name)
        return names

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

    def _resolve_integrations(self, integrations=None):
        source = self.integrations if integrations is None else integrations
        return [
            integration.to_lmstudio_integration()
            if hasattr(integration, "to_lmstudio_integration")
            else integration
            for integration in source
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
        context_length=None,
        max_output_tokens=None,
        **kwargs,
    ):
        kwargs["context_mode"] = SERVER_STATE_CONTEXT
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            integrations=integrations,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
