# ai_core/anthropic_client.py
import anthropic
import config # Assuming config.py is in the parent directory or accessible
import logging
import json
import re # Import regex for cleaning

logger = logging.getLogger(f"{config.SERVICE_NAME}.AnthropicClient")

class AnthropicClient:
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        if not self.api_key:
            logger.error("Anthropic API key is not configured.")
            raise ValueError("Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your .env file.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_name = model_name or config.DEFAULT_AI_MODEL
        self.interrupted = False
        logger.info(f"AnthropicClient initialized with model: {self.model_name}")

    def set_interrupted(self, interrupted_status):
        if self.interrupted != interrupted_status:
            logger.debug(f"Interruption status set to: {interrupted_status}")
        self.interrupted = interrupted_status

    def get_response_stream(self, system_prompt, messages, max_tokens=None):
        """
        Yields responses from the Anthropic API using streaming.
        - Yields ("text_chunk", str_chunk) for text parts.
        - Yields ("first_tool_call_details", preamble_text, tool_name, tool_args) when the *first* complete 
          tool call is found. The stream processing for this AI response then stops.
        - Yields ("stream_complete", full_text_if_no_tool_call, stop_reason) if stream ends 
          without a tool call being actioned.
        - Yields ("error", error_message_str, "error_type_str") on API or processing error.
        - Yields ("interrupted", accumulated_text_before_interrupt, "interrupted_type_str") if interrupted.
        """
        if self.interrupted:
            logger.info("AI interaction interrupted before API call.")
            yield "interrupted", "", "interrupted_before_call"
            return
        
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        
        tag_detection_buffer = "" 
        all_text_chunks_this_segment = [] 
        tool_call_start_tag = "<tool_call>"
        tool_call_end_tag = "</tool_call>"
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                yield "error", "Invalid messages format.", "internal_error"
                return

            logger.debug(f"Opening stream to Anthropic. Model: {self.model_name}, Max Tokens: {effective_max_tokens}")
            
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=effective_max_tokens,
                system=system_prompt,
                messages=messages
            ) as stream:
                for event in stream:
                    if self.interrupted:
                        logger.info("AI stream processing interrupted by flag.")
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
                                        logger.info(f"First tool call detected and parsed: {tool_name}")
                                        preamble_text = "".join(all_text_chunks_this_segment).split(tool_call_start_tag)[0].strip()
                                        yield "first_tool_call_details", preamble_text, tool_name, tool_args
                                        stream.close() 
                                        return 
                                    else:
                                        logger.warning(f"Malformed tool JSON (parsed but invalid structure): {tool_json_str}")
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSONDecodeError in tool call: {e}. Content: {tool_json_str}")
                                # If tool call was malformed or unparsable, it's treated as text.
                                # Advance buffer past this point to avoid re-parsing the same invalid block.
                                tag_detection_buffer = tag_detection_buffer[end_idx + len(tool_call_end_tag):]
                    elif event.type == "message_stop":
                        final_message = stream.get_final_message()
                        final_stop_reason = final_message.stop_reason if final_message else "unknown_stop"
                        full_text = "".join(all_text_chunks_this_segment)
                        logger.info(f"Stream ended by API (message_stop). Stop Reason: {final_stop_reason}. Full text length: {len(full_text)}")
                        yield "stream_complete", full_text, final_stop_reason
                        return
            
                # Fallback if loop finishes (e.g. stream closed by interrupt before message_stop)
                final_message_obj_fallback = stream.get_final_message()
                final_stop_reason_fallback = final_message_obj_fallback.stop_reason if final_message_obj_fallback else "ended_unexpectedly"
                full_text_fallback = "".join(all_text_chunks_this_segment)
                logger.info(f"Stream loop exited. Final text: '{full_text_fallback[:100]}...'. Fallback Stop Reason: {final_stop_reason_fallback}")
                yield "stream_complete", full_text_fallback, final_stop_reason_fallback

        except Exception as e:
            error_type = "api_error" if isinstance(e, anthropic.APIError) else "stream_processing_error"
            logger.error(f"Error during Anthropic stream ({error_type}): {e}", exc_info=True)
            yield "error", f"Stream error ({error_type}): {e}", error_type

    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
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
        
        max_summary_tokens = int(target_token_count * 1.5); 
        if max_summary_tokens < 200: max_summary_tokens = 200
        if max_summary_tokens > 4096: max_summary_tokens = 4096 
        if target_token_count > 4096 : max_summary_tokens = int(target_token_count * 1.2)
        
        logger.info(f"Requesting summarization. Max summary tokens: {max_summary_tokens}")
        
        accumulated_summary_text_chunks = []
        final_reason_for_summary = "error" 
        tool_call_was_detected_in_summary = False

        for event_type, data, *extra in self.get_response_stream(
            system_prompt=summarization_system_prompt,
            messages=conversation_history,
            max_tokens=max_summary_tokens
        ):
            if event_type == "text_chunk":
                accumulated_summary_text_chunks.append(data)
            elif event_type == "first_tool_call_details": 
                preamble_before_tool, tool_name, tool_args = data, extra[0], extra[1]
                logger.warning(f"Tool call ('{tool_name}') detected during summarization within text: '{preamble_before_tool}'. This is invalid for a summary.")
                # Also append the preamble text that came before the invalid tool call
                if preamble_before_tool:
                    accumulated_summary_text_chunks.append(preamble_before_tool)
                # And append the problematic tool call itself as text so it can be cleaned
                malformed_tool_text = f"<tool_call>{json.dumps({'tool_name': tool_name, 'arguments': tool_args})}</tool_call>"
                accumulated_summary_text_chunks.append(malformed_tool_text)
                tool_call_was_detected_in_summary = True
                # Don't break; let the stream finish to capture any text *after* the bad tool call.
                # The client side `get_response_stream` already stops after the first tool call.
                # This means the stream will end shortly after this event if the client worked as intended.
                # However, the client is designed to yield `first_tool_call_details` and then `return`.
                # This logic here might need to align with how `get_response_stream` behaves when it yields `first_tool_call_details`.
                # If `get_response_stream` truly stops and returns, this loop will break.
                # For now, let's assume the stream might continue and we clean up later.
                # The current `get_response_stream` *does* return after yielding `first_tool_call_details`.
                # So, this loop will break after this event.
                final_reason_for_summary = "tool_call_in_summary_attempt"
                break 
            elif event_type == "stream_complete":
                final_reason_for_summary = extra[0] if extra else "unknown_end"
                if data: # data is the full text from client's buffer
                    accumulated_summary_text_chunks = [data] # Prefer this complete text
                break
            elif event_type in ["error", "interrupted"]:
                logger.error(f"Summarization stream error/interrupt: {event_type} - {data}")
                return None 
        
        full_summary_text = "".join(accumulated_summary_text_chunks).strip()

        if tool_call_was_detected_in_summary or \
           ("<tool_call>" in full_summary_text or "</tool_call>" in full_summary_text):
            logger.warning(f"Tool call tags were present in the AI's summary attempt. Original text: '{full_summary_text[:300]}...'")
            # Failsafe: Remove tool calls using regex
            cleaned_summary_text = re.sub(r"<tool_call>.*?</tool_call>", "", full_summary_text, flags=re.DOTALL).strip()
            
            if not cleaned_summary_text:
                logger.error("Summary is empty after removing tool calls.")
                return None
            
            logger.info(f"Summary cleaned. Original length: {len(full_summary_text)}, Cleaned length: {len(cleaned_summary_text)}. Reason for original issue: {final_reason_for_summary}")
            return cleaned_summary_text
        
        # If no tool calls were detected and stream completed normally
        if final_reason_for_summary not in ["error", "interrupted"] and full_summary_text:
            logger.info(f"Summarization successful. Length: {len(full_summary_text)}, Reason: {final_reason_for_summary}")
            return full_summary_text
        
        logger.warning(f"Summarization resulted in no text or an unresolved issue. Reason: {final_reason_for_summary}, Final Text: '{full_summary_text[:200]}'")
        return None

# if __name__ == '__main__':
# ... (Test code from previous version, would need updates for the new 'first_tool_call_details' event structure)
