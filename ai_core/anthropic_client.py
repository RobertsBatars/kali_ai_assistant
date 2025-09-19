# ai_core/anthropic_llm_client.py
import anthropic
import config # Main config from project root
import logging
import json
from .base_llm_client import BaseLLMClient # Import from the same directory

logger = logging.getLogger(f"{config.SERVICE_NAME}.AnthropicLLMClient")

class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model_name=None):
        super().__init__(model_name or config.DEFAULT_ANTHROPIC_MODEL)
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        if not self.api_key:
            logger.error("Anthropic API key is not configured for AnthropicLLMClient.")
            raise ValueError("Anthropic API key is not configured.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"AnthropicLLMClient initialized with model: {self.model_name}")

    def get_response_stream(self, system_prompt, messages, max_tokens=None):
        if self.interrupted:
            logger.info("Anthropic stream interrupted before API call.")
            yield "interrupted", "", "interrupted_before_call"
            return
        
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        
        tag_detection_buffer = "" 
        all_text_chunks_this_segment = [] 
        tool_call_start_tag = "<tool_call>"
        tool_call_end_tag = "</tool_call>"
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                yield "error", "Invalid messages format for Anthropic.", "internal_error"
                return

            logger.debug(f"Opening Anthropic stream. Model: {self.model_name}, Max Tokens: {effective_max_tokens}")
            
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=effective_max_tokens,
                system=system_prompt,
                messages=messages
            ) as stream:
                for event in stream:
                    if self.interrupted:
                        logger.info("Anthropic stream processing interrupted by flag.")
                        yield "interrupted", "".join(all_text_chunks_this_segment), "interrupted_during_stream"
                        stream.close() 
                        return

                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        text_chunk = event.delta.text
                        yield "text_chunk", text_chunk 
                        all_text_chunks_this_segment.append(text_chunk)
                        tag_detection_buffer += text_chunk

                        start_idx = tag_detection_buffer.find(tool_call_start_tag)
                        if start_idx != -1:
                            end_idx = tag_detection_buffer.find(tool_call_end_tag, start_idx + len(tool_call_start_tag))
                            if end_idx != -1:
                                tool_json_str = tag_detection_buffer[start_idx + len(tool_call_start_tag) : end_idx]
                                try:
                                    tool_data = json.loads(tool_json_str)
                                    tool_name = tool_data.get("tool_name")
                                    tool_args = tool_data.get("arguments")

                                    if tool_name and isinstance(tool_args, dict):
                                        logger.info(f"Anthropic: First tool call detected: {tool_name}")
                                        preamble_text = "".join(all_text_chunks_this_segment).split(tool_call_start_tag)[0].strip()
                                        yield "first_tool_call_details", preamble_text, tool_name, tool_args
                                        stream.close() 
                                        return 
                                    # else: Malformed JSON structure, continue treating as text
                                except json.JSONDecodeError:
                                    pass # Not a valid JSON, continue treating as text
                                # If tool call was malformed/unparsable, it's text. Advance buffer.
                                tag_detection_buffer = tag_detection_buffer[end_idx + len(tool_call_end_tag):]
                    elif event.type == "message_stop":
                        final_message = stream.get_final_message()
                        final_stop_reason = final_message.stop_reason if final_message else "unknown_stop"
                        full_text = "".join(all_text_chunks_this_segment)
                        logger.info(f"Anthropic stream ended (message_stop). Stop Reason: {final_stop_reason}. Full text: {len(full_text)} chars")
                        yield "stream_complete", full_text, final_stop_reason
                        return
            
                final_message_obj_fallback = stream.get_final_message()
                final_stop_reason_fallback = final_message_obj_fallback.stop_reason if final_message_obj_fallback else "ended_unexpectedly"
                full_text_fallback = "".join(all_text_chunks_this_segment)
                logger.info(f"Anthropic stream loop exited. Fallback Stop Reason: {final_stop_reason_fallback}")
                yield "stream_complete", full_text_fallback, final_stop_reason_fallback

        except Exception as e:
            error_type = "api_error" if isinstance(e, anthropic.APIError) else "stream_processing_error"
            logger.error(f"Error during Anthropic stream ({error_type}): {e}", exc_info=True)
            yield "error", f"Anthropic stream error ({error_type}): {e}", error_type
    
    def summarize_conversation(self, conversation_history: list, target_token_count: int) -> str | None:
        # (Logic from your previous anthropic_client.py's summarize_conversation,
        # ensuring it uses self.get_response_stream correctly)
        if self.interrupted: logger.info("Summarization interrupted."); return None
        if not conversation_history: logger.info("No history to summarize."); return None

        summarization_system_prompt = (
            "You are an expert summarization AI. Your task is to summarize the provided conversation concisely. "
            "Focus on key facts, decisions, user requests, and important outcomes. "
            "The summary should be a coherent narrative text that captures the essence of the conversation. "
            f"Aim for a summary that is approximately {target_token_count} tokens long. "
            "Retain critical information, especially unresolved questions or ongoing tasks. "
            "Your output MUST be ONLY the summary text. Do NOT include any preambles, sign-offs, or structured data like JSON. "
            "Under NO circumstances should your summary output contain any tool calls (i.e., no `<tool_call>` or `</tool_call>` tags)."
        )
        max_summary_tokens = int(target_token_count * 1.5) 
        if max_summary_tokens < 200: max_summary_tokens = 200
        if max_summary_tokens > 4096: max_summary_tokens = 4096
        if target_token_count > 4096: max_summary_tokens = int(target_token_count * 1.2)
        
        logger.info(f"Anthropic: Requesting summarization. Max summary tokens: {max_summary_tokens}")
        
        accumulated_summary_text_chunks = []
        final_reason_for_summary = "error" 
        tool_call_detected_in_summary_attempt = False

        for event_type, data, *extra in self.get_response_stream(
            system_prompt=summarization_system_prompt,
            messages=conversation_history,
            max_tokens=max_summary_tokens
        ):
            if event_type == "text_chunk":
                accumulated_summary_text_chunks.append(data)
            elif event_type == "first_tool_call_details": 
                preamble_before_tool, tool_name, tool_args = data, extra[0], extra[1]
                logger.warning(f"Anthropic: Tool call ('{tool_name}') detected during summarization. Invalid.")
                if preamble_before_tool: accumulated_summary_text_chunks.append(preamble_before_tool)
                malformed_tool_text = f"<tool_call>{json.dumps({'tool_name': tool_name, 'arguments': tool_args})}</tool_call>"
                accumulated_summary_text_chunks.append(malformed_tool_text)
                tool_call_detected_in_summary_attempt = True
                final_reason_for_summary = "tool_call_in_summary_attempt"
                break 
            elif event_type == "stream_complete":
                final_reason_for_summary = extra[0] if extra else "unknown_end"
                if data: accumulated_summary_text_chunks = [data] 
                break
            elif event_type in ["error", "interrupted"]:
                logger.error(f"Anthropic: Summarization stream error/interrupt: {event_type} - {data}")
                return None 
        
        full_summary_text = "".join(accumulated_summary_text_chunks).strip()

        if tool_call_detected_in_summary_attempt or \
           ("<tool_call>" in full_summary_text or "</tool_call>" in full_summary_text):
            logger.warning(f"Anthropic: Tool call tags present in summary attempt. Original: '{full_summary_text[:300]}...'")
            import re # Local import for safety
            cleaned_summary_text = re.sub(r"<tool_call>.*?</tool_call>", "", full_summary_text, flags=re.DOTALL).strip()
            if not cleaned_summary_text: logger.error("Anthropic: Summary empty after cleaning tool calls."); return None
            logger.info(f"Anthropic: Summary cleaned. Length: {len(cleaned_summary_text)}.")
            return cleaned_summary_text
        
        if final_reason_for_summary not in ["error", "interrupted"] and full_summary_text:
            logger.info(f"Anthropic: Summarization successful. Length: {len(full_summary_text)}, Reason: {final_reason_for_summary}")
            return full_summary_text
        
        logger.warning(f"Anthropic: Summarization failed or no text. Reason: {final_reason_for_summary}, Text: '{full_summary_text[:200]}'")
        return None
