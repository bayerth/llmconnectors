"""OpenAI Chat Completions client."""

import time
from types import SimpleNamespace

from rwu_llmconnector.image_input import (
    build_openai_user_content,
    resolve_response_format,
)
from rwu_llmconnector.llmconnector import (
    LLMClient,
    apply_chat_stream_kwargs,
    call_requested_tools,
    extract_chat_completion_stats,
    llm_call_result,
    log_api_exception,
    message_to_dict,
)


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

    @staticmethod
    def _assistant_text(message):
        content = getattr(message, "content", None) or ""
        if isinstance(content, str) and content.strip():
            return content
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        if isinstance(content, str):
            return content
        return str(content) if content is not None else ""

    def _build_chat_request_kwargs(
        self,
        messages,
        model,
        temperature,
        tools=None,
        max_output_tokens=None,
        response_format=None,
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
        if response_format is not None:
            request_kwargs["response_format"] = response_format
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
        log_metrics = kwargs.pop("log_metrics", True)
        log.debug("Calling OpenAIClient with tool support")
        log.debug(f"Tools provided: {tools}")
        messages = [] if ignore_history else list(self.history)
        log.debug(f"history: {messages}")

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        log.debug(f"System message: {system_message}")
        images = kwargs.pop("images", None)
        response_format = resolve_response_format(kwargs)
        user_content = build_openai_user_content(user_message, images)
        messages.append({"role": "user", "content": user_content})
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
            response_format=response_format,
        )
        if tools is not None:
            kwargs.pop("stream", None)
            kwargs.pop("stream_options", None)
        else:
            apply_chat_stream_kwargs(request_kwargs, kwargs)
        (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
            time_to_first_token,
            tokens_per_second,
        ) = self._call_client(request_kwargs, logger=log)
        if not response_choice:
            response_message = "Error: No valid response from API."
            log.error(response_message)
            return self._finish_call_result(
                log,
                llm_call_result(
                    response_message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                    time_to_first_token=time_to_first_token,
                    tokens_per_second=tokens_per_second,
                ),
                log_metrics=log_metrics,
            )

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
                response_format=response_format,
            )
            (
                response_choice,
                follow_up_prompt_tokens,
                follow_up_completion_tokens,
                follow_up_reasoning_tokens,
                follow_up_runtime,
                follow_up_ttft,
                follow_up_tps,
            ) = self._call_client(follow_up_kwargs, logger=log)
            prompt_tokens += follow_up_prompt_tokens
            completion_tokens += follow_up_completion_tokens
            reasoning_tokens += follow_up_reasoning_tokens
            runtime += follow_up_runtime
            if follow_up_ttft is not None:
                time_to_first_token = follow_up_ttft
            if follow_up_tps is not None:
                tokens_per_second = follow_up_tps
            if not response_choice:
                response_message = "Error: No valid response from API after tool call."
                log.error(response_message)
                return self._finish_call_result(
                    log,
                    llm_call_result(
                        response_message,
                        prompt_tokens,
                        completion_tokens,
                        reasoning_tokens,
                        runtime,
                        time_to_first_token=time_to_first_token,
                        tokens_per_second=tokens_per_second,
                    ),
                    log_metrics=log_metrics,
                )
            response_message = response_choice.message

        response_content = self._assistant_text(response_message)
        log.debug(f"GPT Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return self._finish_call_result(
            log,
            llm_call_result(
                response_content,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
                time_to_first_token=time_to_first_token,
                tokens_per_second=tokens_per_second,
            ),
            log_metrics=log_metrics,
        )

    def call_client_with_tool(self, *args, **kwargs):
        """Backward-compatible alias for callers using the old singular method name."""
        return self.call_client_with_tools(*args, **kwargs)

    def _log_completion_timing(
        self,
        log,
        response,
        runtime,
        completion_tokens,
        reasoning_tokens=0,
        ttft=None,
    ):
        stats = extract_chat_completion_stats(response) if response is not None else None
        server_ttft = None
        server_tps = None
        if stats:
            server_ttft = stats.get("time_to_first_token_seconds")
            server_tps = stats.get("tokens_per_second")
            if server_ttft is not None:
                log.debug(f"time_to_first_token: {server_ttft:.3f}s")
            if server_tps is not None:
                log.debug(f"tokens_per_second (server): {server_tps:,.1f}")
        elif ttft is not None:
            log.debug(f"time_to_first_token: {ttft:.3f}s (stream)")
        effective_ttft = server_ttft if server_ttft is not None else ttft
        output_tokens = completion_tokens + reasoning_tokens
        if runtime > 0 and output_tokens:
            if server_tps is not None:
                return
            if effective_ttft is not None and runtime > effective_ttft:
                log.debug(
                    f"Tokens per second (gen): {output_tokens / (runtime - effective_ttft):.2f}"
                )
            else:
                log.debug(f"Tokens per second (wall): {output_tokens / runtime:.2f}")

    def _call_client_stream(self, request_kwargs, logger=None):
        log = self._resolve_logger(logger)
        start = time.time()
        content_parts = []
        reasoning_parts = []
        ttft = None
        usage = None
        finish_reason = None
        try:
            stream = self.client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                for choice in getattr(chunk, "choices", None) or []:
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None) or ""
                    reasoning = getattr(delta, "reasoning_content", None) or ""
                    if ttft is None and (content or reasoning):
                        ttft = time.time() - start
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
        except Exception as e:
            runtime = time.time() - start
            log_api_exception(log, e, provider="OpenAI")
            return None, 0, 0, 0, runtime, None, None

        runtime = time.time() - start
        log.debug(f"GPT Request took {runtime:,.2f} seconds (stream)")
        message = SimpleNamespace(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=None,
        )
        response_choice = SimpleNamespace(message=message, finish_reason=finish_reason)

        prompt_tokens = completion_tokens = reasoning_tokens = 0
        if usage:
            try:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            except Exception as e:
                log.error(f"Error while determining token count: {e}")
        log.debug(
            f"Token usage - Prompt: {prompt_tokens:,.0f}, "
            f"Completion: {completion_tokens:,.0f}, Reasoning: {reasoning_tokens:,.0f}"
        )
        self._log_completion_timing(
            log, None, runtime, completion_tokens, reasoning_tokens, ttft=ttft
        )
        log.debug(f"GPT Respond: {message}")
        return (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
            ttft,
            None,
        )

    def _call_client(self, request_kwargs, logger=None):
        log = self._resolve_logger(logger)
        runtime = 0
        if request_kwargs.pop("stream", False):
            return self._call_client_stream(request_kwargs, logger=log)
        try:
            start = time.time()
            response = self.client.chat.completions.create(**request_kwargs)
            runtime = time.time() - start
            log.debug(f"GPT Request took {runtime:,.2f} seconds")
        except Exception as e:
            log_api_exception(log, e, provider="OpenAI")
            return None, 0, 0, 0, runtime, None, None

        prompt_tokens, completion_tokens, reasoning_tokens = 0, 0, 0
        time_to_first_token = None
        tokens_per_second = None
        if response and response.choices:
            try:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                details = getattr(response.usage, "completion_tokens_details", None)
                if details is not None:
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            except Exception as e:
                log.error(f"Error while determining token count: {e}")
            response_choice = response.choices[0]
            log.debug(
                f"Token usage - Prompt: {prompt_tokens:,.0f}, "
                f"Completion: {completion_tokens:,.0f}, Reasoning: {reasoning_tokens:,.0f}"
            )
            stats = extract_chat_completion_stats(response)
            if stats:
                time_to_first_token = stats.get("time_to_first_token_seconds")
                tokens_per_second = stats.get("tokens_per_second")
            self._log_completion_timing(
                log, response, runtime, completion_tokens, reasoning_tokens
            )
            log.debug(f"GPT Respond: {response_choice.message}")
        else:
            response_choice = None
            log.error("Error: No valid response from API.")
        return (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
            time_to_first_token,
            tokens_per_second,
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
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Build a message list, call the API, and append the assistant reply to history.

        Uses ``seed=42`` for reproducible outputs. Reasoning tokens are always ``0``.
        Tool requests are handled by :meth:`call_client_with_tools` via :meth:`send_request`.
        """
        log = self._resolve_logger(logger)
        log_metrics = kwargs.pop("log_metrics", True)
        messages = [] if ignore_history else list(self.history)
        log.debug(f"history: {messages}")

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        log.debug(f"System message: {system_message}")
        images = kwargs.pop("images", None)
        response_format = resolve_response_format(kwargs)
        user_content = build_openai_user_content(user_message, images)
        messages.append({"role": "user", "content": user_content})
        log.debug(f"User message: {user_message}")

        request_kwargs = self._build_chat_request_kwargs(
            messages=messages,
            model=self.model if model is None else model,
            temperature=temperature,
            max_output_tokens=self._resolve_max_output_tokens(max_output_tokens, **kwargs),
            response_format=response_format,
        )
        apply_chat_stream_kwargs(request_kwargs, kwargs)
        (
            response_choice,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            runtime,
            time_to_first_token,
            tokens_per_second,
        ) = self._call_client(request_kwargs, logger=log)

        if not response_choice:
            response_message = "Error: No valid response from API."
            log.error(response_message)
            return self._finish_call_result(
                log,
                llm_call_result(
                    response_message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                    time_to_first_token=time_to_first_token,
                    tokens_per_second=tokens_per_second,
                ),
                log_metrics=log_metrics,
            )

        response_content = self._assistant_text(response_choice.message)
        log.debug(f"GPT Respond: {response_content}")
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(messages, filename=self.filename)
        return self._finish_call_result(
            log,
            llm_call_result(
                response_content,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
                time_to_first_token=time_to_first_token,
                tokens_per_second=tokens_per_second,
            ),
            log_metrics=log_metrics,
        )
