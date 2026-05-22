"""Base client, logging helpers, and shared request orchestration.

All provider clients inherit from :class:`LLMCClient` and implement :meth:`LLMCClient.call_client`.
Use :func:`send_request` for the full flow (logging, metrics, optional file export).
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


class LLMCClient(ABC):
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
        system_messgage=None,
        filename=None,
        write_to_file=False,
        append=False,
        temperature=0,
        logger=default_logger,
    ):
        """Initialize shared state.

        Args:
            client: Provider SDK client (may be ``None`` for HTTP-only backends).
            model: Default model identifier for API calls.
            system_messgage: Optional persistent system message (typo preserved for compatibility).
            filename: Target ``.json`` path when ``write_to_file`` is enabled.
            write_to_file: Persist conversation history after each :meth:`send_request`.
            append: Use timestamp-keyed append mode in :meth:`write_json`.
            temperature: Default sampling temperature.
            logger: Logger instance; falls back to :data:`default_logger`.
        """
        self.client = client
        self.model = model
        self.system_message = system_messgage
        self.temperature = temperature
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

    def send_request(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=None,
        model=None,
        write_to_file=False,
        append=False,
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
            **kwargs: Forwarded to :meth:`call_client`.

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens, reasoning_tokens, runtime).
        """
        model = model if model is not None else self.model
        temperature = temperature if temperature is not None else self.temperature
        default_logger.info(
            f"-- GPT Request started. Used model: {model}, temperature: {temperature} --"
        )
        self.logger.debug(f"Prompt: {user_message}")
        start = time.time()
        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = (
            self.call_client(
                user_message,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                **kwargs,
            )
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
        self.logger.info(
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
        **kwargs,
    ):
        """Like :meth:`send_request` but returns only performance summary.

        Returns:
            (runtime, tokens_per_second, response_message)
        """
        default_logger.info(f"-- GPT Request started. Used model: {model} --")
        response_msg, prompt_tokens, completion_tokens, reasoning_tokens, runtime = (
            self.call_client(
                user_message,
                system_message=system_message,
                ignore_history=ignore_history,
                temperature=temperature,
                model=model,
                write_to_file=write_to_file,
                append=append,
                **kwargs,
            )
        )
        output_tokens = (completion_tokens if completion_tokens is not None else 0) + (
            reasoning_tokens if reasoning_tokens is not None else 0
        )
        tps = output_tokens / runtime if runtime > 0 else 0
        return runtime, tps, response_msg

    @abstractmethod
    def call_client(
        self,
        user_message,
        system_message=None,
        history=None,
        temperature=0,
        write_to_file=False,
        model=None,
        **kwargs,
    ):
        """Execute one provider request (implemented by subclasses).

        Returns:
            (response_text, prompt_tokens, completion_tokens, reasoning_tokens, runtime)
            or ``(None, None, None, None, None)`` on failure.
        """
        pass
