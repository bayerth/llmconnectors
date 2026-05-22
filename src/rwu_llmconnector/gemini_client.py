"""Google Gemini client with structured XML prompts."""

import time
from google.genai import types
from rwu_llmconnector.llmconnector import LLMCClient, default_logger


class GeminiClient(LLMCClient):
    """Google Gemini client using a structured XML prompt template.

    Conversation history is folded into the template ``task`` field rather than
    sent as native multi-turn messages. Set ``ignore_history=True`` to skip prior turns.

    Expects ``client`` to be a :class:`google.genai.Client` with
    ``client.models.generate_content``.
    """

    def __init__(
        self,
        client,
        model,
        filename=None,
        system_message="You are Gemini 3, a specialized assistant for Data Science and ontologies. You are precise, analytical, and persistent.",
        constraints="""
                1. use general reasoning, but only consider data from context and user message
                2. if information is not provided, request specific information: <REQUEST> requested data </REQUEST>,
                3. Context may contain text and ontology-description, do not confuse them
                4. Ontology contains classes with attributes, relations, and instances - use them
                4. Output format is specified in the user message
                """,
        prompt_template=None,
        logger=default_logger,
    ):
        """Configure role, constraints, and optional custom ``prompt_template``.

        Template placeholders: ``role``, ``constraints``, ``context``, ``task``.
        """
        super().__init__(client, model, filename, logger)
        self.system_message = system_message
        self.constraints = constraints
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else """
    <role>
    {role}
    </role>
    <instructions>
    1. **Plan**: Analyze the task the context and user message.
    2. **Execute**: Identify objects, classes, relations from user message, use provided data (ontology) to get instances or connections
    3. **Validate**: Review your output against the user's task.
    4. **Format**: Present the final answer in the requested structure.
    </instructions>
    <constraints>
    {constraints}
    </constraints>
    <output_format>
    short answer, provide only objects as a list or, only if requestet, as json syntax. Embrace list or json syntax with square brackets. 
    </output_format>
    <context>
    {context}
    </context>
    
    <task>
    {task}
    </task>
    """
        )
        self.history.append(
            {"role": "system", "content": system_message + "\n" + constraints}
        )

    def call_client(
        self,
        user_message,
        system_message=None,
        ignore_history=False,
        temperature=0,
        retrieved_information=None,
        write_to_file=False,
        constraints=None,
        **kwargs,
    ):
        """Format the XML prompt, call Gemini, and update history.

        Args:
            system_message: Mapped to template ``context`` (per-request context).
            constraints: Override instance constraints for this call only.
            retrieved_information: Stored in history metadata alongside the user turn.
            temperature: Accepted for API compatibility; not passed to Gemini here.

        Note:
            The API model id is currently fixed to ``gemini-2.5-flash-lite`` in the request.
        """
        system_instruction = "" if system_message is None else system_message
        history = (
            ""
            if ignore_history
            else "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.history[1:-1]]
            )
        )
        self.logger.debug(f"history: {history}")
        msg = self.prompt_template.format(
            role=self.system_message,
            constraints=self.constraints if constraints is None else constraints,
            context=system_instruction,
            task=history + " " + user_message,
        )
        prompt = types.Content(role="user", parts=[types.Part.from_text(text=msg)])
        start = time.time()
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=[prompt]
        )
        end = time.time()
        runtime = end - start
        message = response.text
        self.logger.debug(f"history: {message}")
        user_history = "" if constraints is None else constraints
        user_history += (
            "\n" + str(retrieved_information)
            if retrieved_information is not None
            else ""
        )
        user_history += f"\n{system_instruction}\n{user_message}"
        self.history.append({"role": "user", "content": user_history})
        self.history.append({"role": "assistant", "content": message})
        prompt_tokens, completion_tokens, reasoning_tokens = 0, 0, 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            reasoning_tokens = response.usage_metadata.thoughts_token_count
            reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else 0
        return message, prompt_tokens, completion_tokens, reasoning_tokens, runtime
