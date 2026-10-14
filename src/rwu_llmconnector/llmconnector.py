"""Base client, logging helpers, and shared request orchestration.

All provider clients inherit from :class:`LLMClient` and implement :meth:`LLMClient.call_client`.
Use :func:`send_request` for the full flow (logging, metrics, optional file export).
Use :func:`call_requested_tools` to execute model-requested tool calls.
"""

import copy
import json
import time
import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path


def default_log(log_string):
    """Default sink for :func:`get_print_logger`; prints formatted log lines to stdout."""
    print(f"------------------------ \n{log_string}")


class FunctionHandler(logging.Handler):
    """Logging handler that forwards formatted records to a custom callable."""

    def __init__(self, sink_function):
        super().__init__()
        self.sink_function = sink_function

    def emit(self, record):
        """Format ``record`` and pass the result to ``sink_function``."""
        try:
            msg = self.format(record)
            self.sink_function(msg)
        except Exception:
            self.handleError(record)


# Name used by logging.getLogger(); tune levels via logging.getLogger(DEFAULT_LOGGER_NAME)
DEFAULT_LOGGER_NAME = "rwu_llmconnector"


def get_print_logger(
    name=DEFAULT_LOGGER_NAME,
    msg_producer=default_log,
    level=logging.DEBUG,
):
    """Return a logger that forwards formatted records to ``msg_producer`` instead of stderr."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    handler = FunctionHandler(msg_producer)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Package-wide default; pass a custom logger into client constructors to override
default_logger = get_print_logger(DEFAULT_LOGGER_NAME)


def append_to_json_file(message, filename=None):
    """Append ``message`` to a JSON dict file, keyed by Unix timestamp string."""
    filename = filename if filename.endswith(".json") else filename + ".json"
    path = Path(filename)
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if content.strip():
            data = json.loads(content)
            if not isinstance(data, dict):
                data = {}
            data[str(time.time())] = message
            path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def _log_via_logger(logger, level, message):
    if logger is not None:
        getattr(logger, level)(message)


def model_to_dict(value):
    """Serialize an OpenAI SDK object (or similar) to a plain dict."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return value


def message_to_dict(message):
    """Serialize a chat message object for OpenAI-compatible history payloads."""
    message_dict = model_to_dict(message)
    if isinstance(message_dict, dict):
        return message_dict

    result = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [model_to_dict(tool_call) for tool_call in tool_calls]
    return result


def _extract_tool_calls(message):
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def _tool_call_name(tool_call):
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        return function.get("name") if isinstance(function, dict) else None
    function = getattr(tool_call, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _tool_call_arguments(tool_call):
    if isinstance(tool_call, dict):
        arguments = tool_call.get("function", {}).get("arguments", {})
    else:
        function = getattr(tool_call, "function", None)
        arguments = function.arguments if function is not None else {}
    if isinstance(arguments, dict):
        return arguments
    if arguments in (None, ""):
        return {}
    return json.loads(arguments)


def _tool_call_id(tool_call):
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _format_tool_result_message(tool_call, tool_response):
    tool_name = _tool_call_name(tool_call)
    tool_call_id = _tool_call_id(tool_call)
    if isinstance(tool_response, str):
        content = tool_response
    else:
        content = json.dumps(tool_response, default=str)

    if tool_call_id:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    return {"role": "tool", "tool_name": tool_name, "content": content}


def call_requested_tools(message, tool_mapping, logger=None):
    """Execute tool calls requested by the model and return ``role=tool`` messages.

    Supports OpenAI-style messages (``tool_call_id``) and Ollama-style messages
    (``tool_name`` when no call id is present).

    Args:
        message: Assistant message object or dict with ``tool_calls``.
        tool_mapping: ``{tool_name: callable}`` invoked with parsed JSON arguments.
        logger: Optional logger for debug/error output.

    Returns:
        List of tool result message dicts for the follow-up completion request.
    """
    if tool_mapping is None:
        raise ValueError("tool_mapping is required when the model requests tool calls.")

    messages = []
    tool_calls = _extract_tool_calls(message)
    total_start = time.time()
    _log_via_logger(
        logger, "debug", f"Processing {len(tool_calls)} requested tool call(s)."
    )
    try:
        for index, tool_call in enumerate(tool_calls, start=1):
            tool_name = _tool_call_name(tool_call)
            tool_call_id = _tool_call_id(tool_call)
            if tool_name not in tool_mapping:
                raise KeyError(f"No tool handler configured for '{tool_name}'.")

            tool_args = _tool_call_arguments(tool_call)
            tool_start = time.time()
            call_ref = tool_call_id or tool_name
            _log_via_logger(
                logger,
                "debug",
                f"Calling tool {index}/{len(tool_calls)}: {tool_name} "
                f"(ref={call_ref}, args={tool_args})",
            )
            try:
                tool_response = tool_mapping[tool_name](**tool_args)
            except Exception as e:
                tool_runtime = time.time() - tool_start
                _log_via_logger(
                    logger,
                    "error",
                    f"Tool call failed after {tool_runtime:.3f}s: {tool_name} "
                    f"(ref={call_ref}): {e}",
                )
                raise
            tool_runtime = time.time() - tool_start
            _log_via_logger(
                logger,
                "debug",
                f"Tool call completed in {tool_runtime:.3f}s: {tool_name} "
                f"(ref={call_ref}, response={tool_response})",
            )
            messages.append(_format_tool_result_message(tool_call, tool_response))
    finally:
        total_runtime = time.time() - total_start
        _log_via_logger(
            logger,
            "debug",
            f"Processed {len(messages)}/{len(tool_calls)} tool call(s) in {total_runtime:.2f}s.",
        )
    return messages


def resolve_max_output_tokens(max_output_tokens=None, instance_default=None, **kwargs):
    """Resolve generation token cap: explicit arg, kwargs, ``max_tokens`` alias, instance default."""
    if max_output_tokens is not None:
        return max_output_tokens
    kw_value = kwargs.get("max_output_tokens")
    if kw_value is not None:
        return kw_value
    alias = kwargs.get("max_tokens")
    if alias is not None:
        return alias
    return instance_default


class LLMClient(ABC):
    """Abstract base for provider-specific LLM clients with shared history and metrics.

    Attributes:
        history: List of ``{"role": ..., "content": ...}`` message dicts.
        total_prompt_tokens, total_completion_tokens, total_reasoning_tokens: Cumulative counters.
        total_runtime: Cumulative wall-clock seconds across :meth:`send_request` calls.
    """

    def __init__(
        self,
        client,
        model,
        system_messgage="You are a helpful assistant.",
        filename=None,
        write_to_file=False,
        append=False,
        temperature=0,
        max_output_tokens=None,
        logger=default_logger,
    ):
        """Initialize shared state.

        Args:
            client: Provider SDK client (may be ``None`` for HTTP-only backends).
            model: Default model identifier for API calls.
            system_messgage: Optional persistent system message (typo preserved for compatibility), default is "You are a helpful assistant.".
            filename: Target ``.json`` path when ``write_to_file`` is enabled.
            write_to_file: Persist conversation history after each :meth:`send_request`.
            append: Use timestamp-keyed append mode in :meth:`write_json`.
            temperature: Default sampling temperature.
            max_output_tokens: Default cap on generated tokens (provider-specific mapping).
            logger: Logger instance; falls back to :data:`default_logger`.
        """
        self.client = client
        self.model = model
        self.system_message = system_messgage
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.logger = (
            logger if logger is not None else get_print_logger(name=DEFAULT_LOGGER_NAME)
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_runtime = 0
        self.history = []
        self.filename = filename
        self.write_to_file = write_to_file
        self.append = append
        self.response = None

    def get_client(self):
        """Return the underlying provider client."""
        return self.client

    def get_model(self):
        """Return the default model name."""
        return self.model

    def get_history(self):
        """Return a deep copy of the conversation history."""
        return copy.deepcopy(self.history)

    def set_history(self, history):
        """Replace conversation history with a deep copy of ``history``."""
        self.history = copy.deepcopy(history)

    def clear_history(self, reset_counters=False):
        """Clear message history; optionally reset token/runtime counters."""
        self.history = []
        if reset_counters:
            self.reset_counters()

    def reset_counters(self):
        """Reset cumulative token and runtime counters to zero."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_runtime = 0

    def reset(self):
        """Clear history, counters, system message, and last response."""
        self.clear_history(reset_counters=True)
        self.system_message = None
        self.response = None

    def write_json(self, message, filename=None, append=False):
        """Write ``message`` to JSON; overwrite or append depending on flags."""
        filename = filename if filename is not None else self.filename
        if filename and not filename.endswith(".json"):
            filename += ".json"
        try:
            append = append or self.append
            if append:
                append_to_json_file(filename=filename, message=message)
            else:
                Path(filename).write_text(json.dumps(message), encoding="utf-8")
        except Exception as e:
            default_logger.error(f"Error writing JSON file: {e}")
            default_logger.debug(f"Stacktrace:\n{traceback.format_exc()}")

    def write_history(self, filename):
        """Persist :attr:`history` to ``filename``."""
        self.write_json(self.history, filename)

    def _resolve_max_output_tokens(self, max_output_tokens=None, **kwargs):
        """Resolve output token limit without mutating ``self.max_output_tokens``."""
        return resolve_max_output_tokens(
            max_output_tokens=max_output_tokens,
            instance_default=self.max_output_tokens,
            **kwargs,
        )

    def _resolve_logger(self, logger=None):
        """Use per-call ``logger`` when set, otherwise :attr:`logger`."""
        return logger if logger is not None else self.logger

    def send_request(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=None,
        model=None,
        write_to_file=False,
        append=False,
        tools=None,
        integrations=None,
        logger=None,
        **kwargs,
    ):
        """Send a chat request, update metrics/history, and optionally save to disk.

        Args:
            user_message: User prompt text.
            system_message: Per-request system prompt (provider-specific handling).
            ignore_history: Skip prior turns when building the provider payload.
            temperature: Override instance default temperature.
            model: Override instance default model.
            write_to_file: Save history after a successful call.
            append: Passed through to :meth:`write_json`.
            tools: Optional tool definitions for this request (provider-specific).
            integrations: Optional integration specs for this request (e.g. LM Studio MCP).
            logger: Optional logger for this request; defaults to the client instance logger.
            **kwargs: Forwarded to :meth:`call_client` or :meth:`call_client_with_tools`.
                Accepts ``max_output_tokens`` or alias ``max_tokens`` (generation cap).

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens, reasoning_tokens, runtime).
        """
        model = model if model is not None else self.model
        temperature = temperature if temperature is not None else self.temperature
        logger = self._resolve_logger(logger)
        logger.info(
            f"-- GPT Request started. Used model: {model}, temperature: {temperature} --"
        )
        logger.debug(f"Prompt: {user_message}")
        start = time.time()
        request_kwargs = dict(kwargs)
        tool_mapping = request_kwargs.pop("tool_mapping", None)
        slim_history = request_kwargs.pop("slim_history", False)
        max_output_tokens = request_kwargs.pop("max_output_tokens", None)
        if max_output_tokens is None:
            max_output_tokens = request_kwargs.pop("max_tokens", None)
        if tools is not None:
            (
                response_msg,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
            ) = self.call_client_with_tools(
                user_message,
                tools,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                tool_mapping=tool_mapping,
                integrations=integrations,
                slim_history=slim_history,
                max_output_tokens=max_output_tokens,
                logger=logger,
                **request_kwargs,
            )
        else:
            (
                response_msg,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
            ) = self.call_client(
                user_message,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                integrations=integrations,
                max_output_tokens=max_output_tokens,
                logger=logger,
                **request_kwargs,
            )
        end = time.time()
        runtime = end - start
        if prompt_tokens is None:
            prompt_tokens = 0
        if completion_tokens is None:
            completion_tokens = 0
        if reasoning_tokens is None:
            reasoning_tokens = 0
        output_tokens = completion_tokens + reasoning_tokens
        logger.info(
            f"runtime: {runtime:.3f}s, t/s: {output_tokens / runtime:.1f} , prompt_tokens: {prompt_tokens:,.0f}, completion_tokens: {completion_tokens:,.0f}, reasoning_tokens: {reasoning_tokens:,.0f}, output tokens: {output_tokens:,.0f}"
        )
        try:
            if write_to_file or self.write_to_file:
                self.write_json(self.history, filename=self.filename)
        except Exception as e:
            self.logger.error(f"Error writing history: {e}")
            self.logger.debug(f"Stacktrace:\n{traceback.format_exc()}")

        self.total_runtime += runtime
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_reasoning_tokens += reasoning_tokens
        return response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime

    def send_request_perf(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        model=None,
        write_to_file=False,
        append=False,
        tools=None,
        integrations=None,
        logger=None,
        **kwargs,
    ):
        """Like :meth:`send_request` but returns only performance summary.

        Args:
            logger: Optional logger for this request; defaults to the client instance logger.

        Returns:
            (runtime, tokens_per_second, response_message)
        """
        model = model if model is not None else self.model
        temperature = temperature if temperature is not None else self.temperature
        logger = self._resolve_logger(logger)
        logger.info(
            f"-- GPT Request started. Used model: {model}, temperature: {temperature} --"
        )
        request_kwargs = dict(kwargs)
        tool_mapping = request_kwargs.pop("tool_mapping", None)
        slim_history = request_kwargs.pop("slim_history", False)
        max_output_tokens = request_kwargs.pop("max_output_tokens", None)
        if max_output_tokens is None:
            max_output_tokens = request_kwargs.pop("max_tokens", None)
        if tools is not None:
            (
                response_msg,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
            ) = self.call_client_with_tools(
                user_message,
                tools,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                tool_mapping=tool_mapping,
                integrations=integrations,
                slim_history=slim_history,
                max_output_tokens=max_output_tokens,
                logger=logger,
                **request_kwargs,
            )
        else:
            (
                response_msg,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                runtime,
            ) = self.call_client(
                user_message,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                append=append,
                integrations=integrations,
                max_output_tokens=max_output_tokens,
                logger=logger,
                **request_kwargs,
            )
        output_tokens = (completion_tokens if completion_tokens is not None else 0) + (
            reasoning_tokens if reasoning_tokens is not None else 0
        )
        tps = output_tokens / runtime if runtime > 0 else 0
        return runtime, tps, response_msg

    @abstractmethod
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
        """Execute one provider request with tool support (implemented by subclasses).

        Args:
            user_message: User prompt text.
            tools: Tool definitions for this request (provider-specific).
            tool_mapping: ``{tool_name: callable}`` for OpenAI-style function execution.
            integrations: Optional integration specs (e.g. LM Studio MCP).
            slim_history: Provider-specific history handling during tool follow-up.
            max_output_tokens: Cap on generated tokens (``max_tokens`` alias via kwargs).

        Returns:
            (response_text, prompt_tokens, completion_tokens, reasoning_tokens, runtime)
            or ``(None, None, None, None, None)`` on failure.
        """
        pass

    @abstractmethod
    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        write_to_file=False,
        model=None,
        integrations=None,
        max_output_tokens=None,
        logger=None,
        **kwargs,
    ):
        """Execute one provider request (implemented by subclasses).

        Args:
            integrations: Optional integration specs for this request (e.g. LM Studio MCP).
            max_output_tokens: Cap on generated tokens (``max_tokens`` alias via kwargs).
            logger: Optional logger for this call; defaults to the client instance logger.

        Returns:
            (response_text, prompt_tokens, completion_tokens, reasoning_tokens, runtime)
            or ``(None, None, None, None, None)`` on failure.
        """
        pass

    @abstractmethod
    def list_models(self):
        """Return a list of available model names for the provider."""
        pass
