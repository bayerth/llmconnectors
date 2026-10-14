"""Ollama local/remote server client."""

import time

from ollama import Client

from rwu_llmconnector.llmconnector import (
    LLMClient,
    call_requested_tools,
    default_logger,
    message_to_dict,
)


class OllamaClient(LLMClient):
    """Client for an Ollama server via the official ``ollama`` Python library."""

    def __init__(
        self,
        model,
        host="http://localhost:11434",
        headers=None,
        filename=None,
        write_to_file=False,
        append=False,
        logger=default_logger,
    ):
        """Create an Ollama client.

        Args:
            model: Model tag as shown by ``ollama list``.
            host: Ollama server base URL.
            headers: Optional HTTP headers for the underlying client.
        """
        client = Client(host=host, headers=headers)
        super().__init__(client, model, filename, write_to_file, append, logger)

    def list_models(self):
        """Return model names from ``ollama list``."""
        try:
            response = self.client.list()
            return [model["name"] for model in response.get("models", [])]
        except Exception as e:
            self.logger.error(f"Failed to list Ollama models: {e}")
            return []

    def _build_ollama_options(self, temperature, max_output_tokens=None):
        options = {"temperature": temperature}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        return options

    def _build_chat_messages(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        retrieved_information=None,
        logger=None,
    ):
        log = self._resolve_logger(logger)
        messages = [] if ignore_history else list(self.history)
        log.debug(f"history: {messages}")

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        log.debug(f"System message: {system_message}")

        if retrieved_information is not None:
            log.debug(f"Retrieved information: {retrieved_information}")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{user_message}. Here is ontology retrieved information "
                        f"based on the topic: {retrieved_information}"
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": user_message})
        log.debug(f"User message: {user_message}")
        return messages

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
        retrieved_information=None,
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Call Ollama chat with tools and resolve requested tool calls."""
        log = self._resolve_logger(logger)
        log.debug("Calling OllamaClient with tool support")
        log.debug(f"Tools provided: {tools}")
        messages = self._build_chat_messages(
            user_message=user_message,
            system_message=system_message,
            ignore_history=ignore_history,
            retrieved_information=retrieved_information,
            logger=log,
        )
        resolved_max_output_tokens = self._resolve_max_output_tokens(
            max_output_tokens, **kwargs
        )
        resolved_model = self.model if model is None else model
        options = self._build_ollama_options(temperature, resolved_max_output_tokens)

        request_kwargs = {
            "model": resolved_model,
            "messages": messages,
            "tools": tools,
            "options": options,
        }
        response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self._call_client(
            request_kwargs, logger=log
        )
        if not response:
            response_message = "Error: No valid response from Ollama API."
            log.error(response_message)
            return response_message, prompt_tokens, completion_tokens, reasoning_tokens, runtime

        message = response["message"]
        if message.get("tool_calls"):
            log.debug(f"Tool calls requested: {message['tool_calls']}")
            if not slim_history:
                messages.append(message_to_dict(message))
            messages.extend(
                call_requested_tools(
                    message,
                    tool_mapping=tool_mapping,
                    logger=log,
                )
            )

            follow_up_kwargs = {
                "model": resolved_model,
                "messages": messages,
                "tools": tools,
                "options": options,
            }
            (
                response,
                follow_up_prompt_tokens,
                follow_up_completion_tokens,
                follow_up_reasoning_tokens,
                follow_up_runtime,
            ) = self._call_client(follow_up_kwargs, logger=log)
            prompt_tokens += follow_up_prompt_tokens
            completion_tokens += follow_up_completion_tokens
            reasoning_tokens += follow_up_reasoning_tokens
            runtime += follow_up_runtime
            if not response:
                response_message = "Error: No valid response from Ollama API after tool call."
                log.error(response_message)
                return (
                    response_message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                )
            message = response["message"]

        response_content = message.get("content") or ""
        log.debug(f"Ollama Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return response_content, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def _call_client(self, request_kwargs, logger=None):
        log = self._resolve_logger(logger)
        runtime = 0
        chat_kwargs = {
            "model": request_kwargs["model"],
            "messages": request_kwargs["messages"],
        }
        if request_kwargs.get("tools") is not None:
            chat_kwargs["tools"] = request_kwargs["tools"]
        if request_kwargs.get("options") is not None:
            chat_kwargs["options"] = request_kwargs["options"]

        try:
            start = time.time()
            response = self.client.chat(**chat_kwargs)
            runtime = time.time() - start
            log.debug(f"Ollama Request took {runtime:.2f} seconds")
        except Exception as e:
            log.error(f"Ollama API call failed: {e}")
            return None, 0, 0, 0, runtime

        prompt_tokens, completion_tokens = 0, 0
        if response and "message" in response:
            try:
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
            except Exception as e:
                log.error(f"Error while determining token count: {e}")
            log.debug(f"Ollama Respond: {response['message']}")
        else:
            response = None
            log.error("Error: No valid response from Ollama API.")
        return response, prompt_tokens, completion_tokens, 0, runtime

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        retrieved_information=None,
        model=None,
        write_to_file=False,
        integrations=None,
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Call ``client.chat`` with OpenAI-style roles.

        Args:
            retrieved_information: Optional RAG/ontology text appended to the user message.
        """
        log = self._resolve_logger(logger)
        messages = self._build_chat_messages(
            user_message=user_message,
            system_message=system_message,
            ignore_history=ignore_history,
            retrieved_information=retrieved_information,
            logger=log,
        )

        request_kwargs = {
            "model": model if model is not None else self.model,
            "messages": messages,
            "options": self._build_ollama_options(
                temperature,
                self._resolve_max_output_tokens(max_output_tokens, **kwargs),
            ),
        }
        response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self._call_client(
            request_kwargs, logger=log
        )

        if not response:
            response_message = "Error: No valid response from Ollama API."
            log.error(response_message)
            return response_message, prompt_tokens, completion_tokens, reasoning_tokens, runtime

        response_content = response["message"].get("content") or ""
        log.debug(f"Ollama Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return response_content, prompt_tokens, completion_tokens, reasoning_tokens, runtime
