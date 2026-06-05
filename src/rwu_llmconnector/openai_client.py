"""OpenAI Chat Completions client."""

import time
from rwu_llmconnector.llmconnector import LLMClient


class OpenAIClient(LLMClient):
    """Client for the OpenAI Chat Completions API (or compatible endpoints).

    Expects ``client`` to be an :class:`openai.OpenAI` instance with
    ``client.chat.completions.create`` support.
    """

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        tools=None,
        integrations=None,
        **kwargs,
    ):
        """Build a message list, call the API, and append the assistant reply to history.

        Uses ``seed=42`` for reproducible outputs. Reasoning tokens are always ``0``.
        """
        message = [] if ignore_history else self.history
        self.logger.debug(f"history: {message}")

        if system_message is not None:
            message.append({"role": "system", "content": system_message})
        self.logger.debug(f"System message: {system_message}")
        message.append({"role": "user", "content": user_message})
        self.logger.debug(f"User message: {user_message}")
        try:
            model = self.model if model is None else model
            start = time.time()
            request_kwargs = {
                "temperature": temperature,
                "messages": message,
                "model": model,
                "seed": 42,
            }
            if tools is not None:
                request_kwargs["tools"] = tools
            response = self.client.chat.completions.create(**request_kwargs)
            end = time.time()
            runtime = end - start
            self.logger.debug(f"GPT Request took {runtime} seconds")
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return None, None, None, None, None

        prompt_tokens, completion_tokens = 0, 0
        if response and response.choices:
            try:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            except Exception as e:
                self.logger.error(f"Error while determining token count: {e}")
            response_message = response.choices[0].message.content
            self.logger.debug(f"GPT Respond: {response_message}")
            self.history.append({"role": "assistant", "content": response_message})
        else:
            response_message = "Error: No valid response from API."
            self.logger.error(response_message)
        if write_to_file:
            self.write_json(message, filename=self.filename)
        return response_message, prompt_tokens, completion_tokens, 0, runtime
