"""OpenAI Chat Completions client."""

import time

from rwu_llmconnector.llmconnector import LLMClient, call_requested_tools, message_to_dict


class OpenAIClient(LLMClient):
    """Client for the OpenAI Chat Completions API (or compatible endpoints).

    Expects ``client`` to be an :class:`openai.OpenAI` instance with
    ``client.chat.completions.create`` support.
    """

    def list_models(self):
        """Return model ids from the OpenAI models API."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            self.logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def _build_chat_request_kwargs(
        self,
        messages,
        model,
        temperature,
        tools=None,
        max_output_tokens=None,
    ):
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": 42,
        }
        if tools is not None:
            request_kwargs["tools"] = tools
        if max_output_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_output_tokens
        return request_kwargs

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
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Call the API with tool support and resolve requested tool calls."""
        log = self._resolve_logger(logger)
        log.debug("Calling OpenAIClient with tool support")
        log.debug(f"Tools provided: {tools}")
        messages = [] if ignore_history else list(self.history)
        log.debug(f"history: {messages}")

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        log.debug(f"System message: {system_message}")
        messages.append({"role": "user", "content": user_message})
        log.debug(f"User message: {user_message}")

        resolved_model = self.model if model is None else model
        resolved_max_output_tokens = self._resolve_max_output_tokens(
            max_output_tokens, **kwargs
        )
        request_kwargs = self._build_chat_request_kwargs(
            messages=messages,
            model=resolved_model,
            temperature=temperature,
            tools=tools,
            max_output_tokens=resolved_max_output_tokens,
        )
        (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
        ) = self._call_client(request_kwargs, logger=log)
        if not response_choice:
            response_message = "Error: No valid response from API."
            log.error(response_message)
            return response_message, prompt_tokens, completion_tokens, reasoning_tokens, runtime

        response_message = response_choice.message
        if response_choice.finish_reason == "tool_calls" and response_message.tool_calls:
            log.debug(f"Tool calls requested: {response_message.tool_calls}")
            if not slim_history:
                messages.append(message_to_dict(response_message))
            messages.extend(
                call_requested_tools(
                    response_message,
                    tool_mapping=tool_mapping,
                    logger=log,
                )
            )

            follow_up_kwargs = self._build_chat_request_kwargs(
                messages=messages,
                model=resolved_model,
                temperature=temperature,
                tools=tools,
                max_output_tokens=resolved_max_output_tokens,
            )
            (
                response_choice,
                follow_up_prompt_tokens,
                follow_up_completion_tokens,
                follow_up_reasoning_tokens,
                follow_up_runtime,
            ) = self._call_client(follow_up_kwargs, logger=log)
            prompt_tokens += follow_up_prompt_tokens
            completion_tokens += follow_up_completion_tokens
            reasoning_tokens += follow_up_reasoning_tokens
            runtime += follow_up_runtime
            if not response_choice:
                response_message = "Error: No valid response from API after tool call."
                log.error(response_message)
                return (
                    response_message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                )
            response_message = response_choice.message

        response_content = response_message.content or ""
        log.debug(f"GPT Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return response_content, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def call_client_with_tool(self, *args, **kwargs):
        """Backward-compatible alias for callers using the old singular method name."""
        return self.call_client_with_tools(*args, **kwargs)

    def _call_client(self, request_kwargs, logger=None):
        log = self._resolve_logger(logger)
        runtime = 0
        try:
            start = time.time()
            response = self.client.chat.completions.create(**request_kwargs)
            runtime = time.time() - start
            log.debug(f"GPT Request took {runtime:,.2f} seconds")
        except Exception as e:
            log.error(f"API call failed: {e}")
            return None, 0, 0, 0, runtime

        prompt_tokens, completion_tokens, reasoning_tokens = 0, 0, 0
        if response and response.choices:
            try:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            except Exception as e:
                log.error(f"Error while determining token count: {e}")
            response_choice = response.choices[0]
            log.info(
                f"Token usage - Prompt: {prompt_tokens:,.0f}, "
                f"Completion: {completion_tokens:,.0f}, Reasoning: {reasoning_tokens:,.0f}"
            )
            if runtime > 0:
                log.info(f"Tokens per second: {completion_tokens / runtime:.2f}")
            log.debug(f"GPT Respond: {response_choice.message}")
        else:
            response_choice = None
            log.error("Error: No valid response from API.")
        return response_choice, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        integrations=None,
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Build a message list, call the API, and append the assistant reply to history.

        Uses ``seed=42`` for reproducible outputs. Reasoning tokens are always ``0``.
        Tool requests are handled by :meth:`call_client_with_tools` via :meth:`send_request`.
        """
        log = self._resolve_logger(logger)
        messages = [] if ignore_history else list(self.history)
        log.debug(f"history: {messages}")

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        log.debug(f"System message: {system_message}")
        messages.append({"role": "user", "content": user_message})
        log.debug(f"User message: {user_message}")

        request_kwargs = self._build_chat_request_kwargs(
            messages=messages,
            model=self.model if model is None else model,
            temperature=temperature,
            max_output_tokens=self._resolve_max_output_tokens(max_output_tokens, **kwargs),
        )
        (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
        ) = self._call_client(request_kwargs, logger=log)

        if not response_choice:
            response_message = "Error: No valid response from API."
            log.error(response_message)
            return response_message, prompt_tokens, completion_tokens, reasoning_tokens, runtime

        response_content = response_choice.message.content or ""
        log.debug(f"GPT Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return response_content, prompt_tokens, completion_tokens, reasoning_tokens, runtime
