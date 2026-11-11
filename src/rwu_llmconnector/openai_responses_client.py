"""OpenAI Responses API client (successor to Chat Completions for official OpenAI)."""

import time

from rwu_llmconnector.image_input import (
    build_responses_user_content,
    resolve_responses_text_config,
)
from rwu_llmconnector.llmconnector import (
    LLMClient,
    convert_chat_completion_tools_to_responses,
    execute_responses_function_calls,
    extract_responses_function_calls,
    llm_call_result,
    log_api_exception,
    responses_output_items_for_input,
)


class OpenAIResponsesClient(LLMClient):
    """Client for the OpenAI **Responses** API via ``client.responses.create``.

    Use this for new OpenAI projects (reasoning models, native tools, ``previous_response_id``).
    For OpenRouter and other Chat-Completions-compatible hosts, use :class:`OpenAIClient`.

    Expects ``client`` to be an :class:`openai.OpenAI` instance with ``responses.create``.
    """

    def __init__(
        self,
        client,
        model,
        use_previous_response_id=False,
        store=None,
        **kwargs,
    ):
        super().__init__(client, model, **kwargs)
        self.use_previous_response_id = use_previous_response_id
        self.store = store
        self.last_response_id = None

    def clear_history(self, reset_counters=False):
        super().clear_history(reset_counters=reset_counters)
        self.last_response_id = None

    def list_models(self):
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            self.logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def _resolve_instructions(self, system_message):
        if system_message is not None:
            return system_message
        return self.system_message

    def _build_message_input(self, user_message, ignore_history=False, images=None):
        user_content = (
            build_responses_user_content(user_message, images)
            if images
            else user_message
        )
        if self.use_previous_response_id:
            if images:
                return [{"role": "user", "content": user_content}]
            return user_message
        messages = []
        if not ignore_history:
            for entry in self.history:
                role = entry.get("role")
                if role in {"user", "assistant", "system"}:
                    messages.append({"role": role, "content": entry["content"]})
        messages.append({"role": "user", "content": user_content})
        if len(messages) == 1 and not images:
            return messages[0]["content"]
        return messages

    def _build_responses_request_kwargs(
        self,
        input_data,
        model,
        temperature,
        instructions=None,
        tools=None,
        max_output_tokens=None,
        previous_response_id=None,
        text=None,
    ):
        request_kwargs = {
            "model": model,
            "input": input_data,
            "temperature": temperature,
        }
        if instructions:
            request_kwargs["instructions"] = instructions
        if tools is not None:
            request_kwargs["tools"] = convert_chat_completion_tools_to_responses(tools)
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens
        if text is not None:
            request_kwargs["text"] = text
        if self.store is not None:
            request_kwargs["store"] = self.store
        if previous_response_id:
            request_kwargs["previous_response_id"] = previous_response_id
        return request_kwargs

    def _extract_usage(self, response):
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        usage = getattr(response, "usage", None)
        if not usage:
            return prompt_tokens, completion_tokens, reasoning_tokens
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        completion_tokens = max(output_tokens - reasoning_tokens, 0)
        return prompt_tokens, completion_tokens, reasoning_tokens

    def _call_responses(self, request_kwargs, logger=None):
        log = self._resolve_logger(logger)
        runtime = 0.0
        try:
            start = time.time()
            response = self.client.responses.create(**request_kwargs)
            runtime = time.time() - start
            log.debug(f"Responses API request took {runtime:,.2f} seconds")
        except Exception as e:
            log_api_exception(log, e, provider="OpenAI Responses")
            return None, 0, 0, 0, runtime

        if not response:
            log.error("Error: No valid response from Responses API.")
            return None, 0, 0, 0, runtime

        prompt_tokens, completion_tokens, reasoning_tokens = self._extract_usage(response)
        log.debug(
            f"Token usage - Prompt: {prompt_tokens:,.0f}, "
            f"Completion: {completion_tokens:,.0f}, Reasoning: {reasoning_tokens:,.0f}"
        )
        if runtime > 0 and completion_tokens:
            log.debug(f"Tokens per second: {completion_tokens / runtime:.2f}")
        return response, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def _response_text(self, response):
        if response is None:
            return ""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        parts = []
        for item in getattr(response, "output", None) or []:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "message":
                continue
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif content:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        parts.append(block.get("text", ""))
                    else:
                        text = getattr(block, "text", None)
                        if text:
                            parts.append(text)
        return "".join(parts)

    def _run_responses_tool_loop(
        self,
        request_kwargs,
        tool_mapping,
        slim_history,
        logger,
    ):
        log = self._resolve_logger(logger)
        input_data = request_kwargs["input"]
        tools = request_kwargs.get("tools")
        accumulated_input = input_data if isinstance(input_data, list) else None

        total_prompt = total_completion = total_reasoning = 0
        total_runtime = 0.0
        response = None

        while True:
            if accumulated_input is not None:
                request_kwargs["input"] = accumulated_input
            response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self._call_responses(
                request_kwargs, logger=log
            )
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_reasoning += reasoning_tokens
            total_runtime += runtime
            if response is None:
                return None, total_prompt, total_completion, total_reasoning, total_runtime

            function_calls = extract_responses_function_calls(response)
            if not function_calls:
                break

            log.debug(f"Tool calls requested: {function_calls}")
            if accumulated_input is None:
                accumulated_input = (
                    [{"role": "user", "content": input_data}]
                    if isinstance(input_data, str)
                    else list(input_data)
                )
            if not slim_history:
                accumulated_input.extend(responses_output_items_for_input(response))
            accumulated_input.extend(
                execute_responses_function_calls(
                    function_calls, tool_mapping=tool_mapping, logger=log
                )
            )

        return response, total_prompt, total_completion, total_reasoning, total_runtime

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
        log = self._resolve_logger(logger)
        log_metrics = kwargs.pop("log_metrics", True)
        log.debug("Calling OpenAIResponsesClient with tool support")
        images = kwargs.pop("images", None)
        text_config = resolve_responses_text_config(kwargs)
        resolved_model = self.model if model is None else model
        instructions = self._resolve_instructions(system_message)
        previous_response_id = None
        if self.use_previous_response_id:
            if ignore_history:
                self.last_response_id = None
            previous_response_id = self.last_response_id

        request_kwargs = self._build_responses_request_kwargs(
            input_data=self._build_message_input(
                user_message, ignore_history=ignore_history, images=images
            ),
            model=resolved_model,
            temperature=temperature,
            instructions=instructions,
            tools=tools,
            max_output_tokens=self._resolve_max_output_tokens(max_output_tokens, **kwargs),
            previous_response_id=previous_response_id,
            text=text_config,
        )

        response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self._run_responses_tool_loop(
            request_kwargs,
            tool_mapping=tool_mapping,
            slim_history=slim_history,
            logger=log,
        )
        if response is None:
            message = "Error: No valid response from Responses API."
            log.error(message)
            return self._finish_call_result(
                log,
                llm_call_result(
                    message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                ),
                log_metrics=log_metrics,
            )

        if getattr(response, "id", None):
            self.last_response_id = response.id

        response_content = self._response_text(response)
        log.debug(f"Responses output: {response_content}")
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(self.history, filename=self.filename)
        return self._finish_call_result(
            log,
            llm_call_result(
                response_content,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
                extras={"response_id": getattr(response, "id", None)},
            ),
            log_metrics=log_metrics,
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
        log = self._resolve_logger(logger)
        log_metrics = kwargs.pop("log_metrics", True)
        images = kwargs.pop("images", None)
        text_config = resolve_responses_text_config(kwargs)
        resolved_model = self.model if model is None else model
        instructions = self._resolve_instructions(system_message)
        previous_response_id = None
        if self.use_previous_response_id:
            if ignore_history:
                self.last_response_id = None
            previous_response_id = self.last_response_id

        request_kwargs = self._build_responses_request_kwargs(
            input_data=self._build_message_input(
                user_message, ignore_history=ignore_history, images=images
            ),
            model=resolved_model,
            temperature=temperature,
            instructions=instructions,
            max_output_tokens=self._resolve_max_output_tokens(max_output_tokens, **kwargs),
            previous_response_id=previous_response_id,
            text=text_config,
        )
        response, prompt_tokens, completion_tokens, reasoning_tokens, runtime = self._call_responses(
            request_kwargs, logger=log
        )
        if response is None:
            message = "Error: No valid response from Responses API."
            log.error(message)
            return self._finish_call_result(
                log,
                llm_call_result(
                    message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                ),
                log_metrics=log_metrics,
            )

        if getattr(response, "id", None):
            self.last_response_id = response.id

        response_content = self._response_text(response)
        if not response_content.strip():
            message = "Error: No valid response from Responses API."
            log.error(message)
            return self._finish_call_result(
                log,
                llm_call_result(
                    message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    runtime,
                ),
                log_metrics=log_metrics,
            )

        log.debug(f"Responses output: {response_content}")
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response_content})
        if write_to_file:
            self.write_json(self.history, filename=self.filename)
        return self._finish_call_result(
            log,
            llm_call_result(
                response_content,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
                extras={"response_id": getattr(response, "id", None)},
            ),
            log_metrics=log_metrics,
        )
