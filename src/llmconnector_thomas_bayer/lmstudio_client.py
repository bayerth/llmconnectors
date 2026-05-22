"""LM Studio HTTP clients (stateless and stateful)."""

import os
import time
import requests
from llmconnector_thomas_bayer.llmconnector import LLMCClient

DEFAULT_LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
DEFAULT_LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY") or os.getenv("MAC_STUDIO_API_KEY")


class LMStudioClient(LLMCClient):
    """Stateless LM Studio client using ``POST /api/v1/chat``.

    Sends full conversation input on each request (``store=False``).
    """

    def __init__(
        self,
        model,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        api_key=DEFAULT_LMSTUDIO_API_KEY,
        integrations=None,
        **kwargs,
    ):
        """Configure HTTP endpoint and optional tool integrations.

        Args:
            model: Model id as exposed by LM Studio.
            base_url: Server root URL (``/api/v1/chat`` is appended).
            api_key: Bearer token; set ``None`` to omit the Authorization header.
            integrations: Optional integration list for the LM Studio payload.
        """
        # LLMCClient expects a client handle; HTTP calls use requests directly instead.
        super().__init__(client=None, model=model, **kwargs)
        self.base_url = base_url + "/api/v1/chat"
        self.api_key = api_key
        self.logger.debug(f"URL: {self.base_url}")
        self.integrations = integrations if integrations is not None else []

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
        """Post a chat request and parse the response from several possible JSON shapes."""
        input_data = []
        if not ignore_history:
            # Map role/content history to LM Studio's {"type": "text", "content": ...} input list.
            for entry in self.history:
                input_data.append({"type": "text", "content": entry["content"]})
        input_data.append({"type": "text", "content": user_message})
        if system_message is not None:
            self.system_message = system_message
            self.logger.debug("System message changed")
        system_prompt = (
            self.system_message
            if self.system_message is not None
            else "You are a helpful assistant."
        )
        self.logger.debug(f"System prompt: {system_prompt}")
        payload = {
            "model": model if model is not None else self.model,
            "input": input_data,
            "system_prompt": system_prompt,
            "integrations": self.integrations,
            "context_length": kwargs.get("context_length", 8000),
            "temperature": temperature,
            "store": False,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
            end = time.time()
            runtime = end - start
            self.logger.debug(f"LM Studio Request took {runtime} seconds")
        except Exception as e:
            self.logger.error(f"LM Studio API call failed: {e}")
            return None, 0, 0, 0, 0

        # LM Studio may return OpenAI-style "choices", a typed "output" list, or bare "content".
        response_message = ""
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0

        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                response_message = choice["message"].get("content", "")
            elif "text" in choice:
                response_message = choice.get("text", "")
        elif "output" in data:
            if isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message":
                        response_message = item.get("content", "")
            else:
                response_message = data["output"]
        elif "content" in data:
            response_message = data["content"]

        if "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens", 0)
            completion_tokens = data["usage"].get("completion_tokens", 0)
            reasoning_tokens = data["usage"].get("reasoning_tokens", 0)
        elif "stats" in data:
            prompt_tokens = data["stats"].get("input_tokens", 0)
            reasoning_tokens = data["stats"].get("reasoning_output_tokens", 0)
            # stats.total_output_tokens includes reasoning; split for LLMCClient's counters.
            total_output = data["stats"].get("total_output_tokens", 0)
            completion_tokens = total_output - reasoning_tokens
            tokens_per_second = data["stats"].get("tokens_per_second", 0)
            time_to_first_token = data["stats"].get("time_to_first_token_seconds", 0)
            self.logger.debug(
                f"tokens_per_second: {tokens_per_second:,.1f}, time_to_first_token: {time_to_first_token:.3f}"
            )

        self.logger.debug(f"LM Studio Respond: {response_message}")
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


class StatefulLMStudioClient(LMStudioClient):
    """LM Studio client with server-side conversation state.

    Sets ``store=True`` and chains requests via ``previous_response_id``.
    """

    def __init__(
        self,
        model,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        api_key=DEFAULT_LMSTUDIO_API_KEY,
        integrations=None,
        **kwargs,
    ):
        super().__init__(model, base_url, api_key, integrations, **kwargs)
        self.last_response_id = None

    def clear_history(self, reset_counters=False):
        """Clear local history and drop the server-side ``response_id`` chain."""
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
        """Send only the latest user text; context is held on the LM Studio server."""
        if ignore_history:
            self.last_response_id = None

        payload = {
            "model": model if model is not None else self.model,
            "input": user_message,
            "system_prompt": system_message
            if system_message is not None
            else "You are a helpful assistant.",
            "integrations": self.integrations,
            "context_length": kwargs.get("context_length", 8000),
            "temperature": temperature,
            "store": True,
        }

        if self.last_response_id:
            payload["previous_response_id"] = self.last_response_id

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
            end = time.time()
            runtime = end - start
            self.logger.debug(f"Stateful LM Studio Request took {runtime} seconds")
        except Exception as e:
            self.logger.error(f"Stateful LM Studio API call failed: {e}")
            return None, 0, 0, 0, 0

        response_message = ""
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0

        self.last_response_id = data.get("response_id")

        if "output" in data:
            if isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message":
                        response_message = item.get("content", "")
                    elif item.get("type") == "reasoning":
                        pass
            else:
                response_message = data["output"]
        elif "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                response_message = choice["message"].get("content", "")

        if "stats" in data:
            prompt_tokens = data["stats"].get("input_tokens", 0)
            reasoning_tokens = data["stats"].get("reasoning_output_tokens", 0)
            total_output = data["stats"].get("total_output_tokens", 0)
            completion_tokens = total_output - reasoning_tokens
        elif "usage" in data:
            prompt_tokens = data["usage"].get("prompt_tokens", 0)
            completion_tokens = data["usage"].get("completion_tokens", 0)
            reasoning_tokens = data["usage"].get("reasoning_tokens", 0)

        self.logger.debug(
            f"Stateful LM Studio Response: {response_message} (ID: {self.last_response_id})"
        )

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
