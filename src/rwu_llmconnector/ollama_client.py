"""Ollama local/remote server client."""

import time
from ollama import Client
from rwu_llmconnector.llmconnector import LLMCClient, default_logger


class OllamaClient(LLMCClient):
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

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        retrieved_information=None,
        model=None,
        write_to_file=False,
        **kwargs,
    ):
        """Call ``client.chat`` with OpenAI-style roles.

        Args:
            retrieved_information: Optional RAG/ontology text appended to the user message.
        """
        message = [] if ignore_history else list(self.history)
        self.logger.debug(f"history: {message}")

        if system_message is not None:
            message.append({"role": "system", "content": system_message})
        self.logger.debug(f"System message: {system_message}")

        if retrieved_information is not None:
            self.logger.debug(f"Retrieved information: {retrieved_information}")
            message.append(
                {
                    "role": "user",
                    "content": (
                        f"{user_message}. Here is ontology retrieved information "
                        f"based on the topic: {retrieved_information}"
                    ),
                }
            )
        else:
            message.append({"role": "user", "content": user_message})
        self.logger.debug(f"Retrieved information: {retrieved_information}")

        try:
            model = model if model is not None else self.model
            start = time.time()
            response = self.client.chat(
                model=model,
                messages=message,
                options={"temperature": temperature},
            )
            end = time.time()
            runtime = end - start
            self.logger.debug(f"Ollama Request took {runtime} seconds")
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            return None, None, None, None, None

        prompt_tokens, completion_tokens = 0, 0
        if response and "message" in response:
            try:
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
            except Exception as e:
                self.logger.error(f"Error while determining token count: {e}")

            response_message = response["message"]["content"]
            self.logger.debug(f"Ollama Respond: {response_message}")
            self.history.append({"role": "assistant", "content": response_message})
        else:
            response_message = "Error: No valid response from Ollama API."
            self.logger.error(response_message)

        if write_to_file:
            self.write_json(message, filename=self.filename)

        return response_message, prompt_tokens, completion_tokens, 0, runtime
