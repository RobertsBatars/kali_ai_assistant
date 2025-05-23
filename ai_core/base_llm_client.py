# ai_core/base_llm_client.py
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.interrupted = False

    def set_interrupted(self, interrupted_status: bool):
        self.interrupted = interrupted_status

    @abstractmethod
    def get_response_stream(self, system_prompt: str, messages: list, max_tokens: int | None = None):
        """
        Yields responses from the LLM API using streaming.
        Expected yields:
            - ("text_chunk", str_chunk)
            - ("first_tool_call_details", preamble_text, tool_name, tool_args)
            - ("stream_complete", full_text_if_no_tool_call, stop_reason)
            - ("error", error_message_str, "error_type_str")
            - ("interrupted", accumulated_text, "interrupted_type_str")
        """
        pass

    @abstractmethod
    def summarize_conversation(self, conversation_history: list, target_token_count: int) -> str | None:
        """
        Summarizes a conversation history.
        """
        pass
