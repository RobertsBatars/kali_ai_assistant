# ai_core/anthropic_client.py
import anthropic
import config # Assuming config.py is in the parent directory or accessible
import logging
import json

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
        - Yields ("first_tool_call_details", tool_name, tool_args) when the *first* complete 
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
        
        # Buffer to accumulate text across chunks to reliably detect complete tool call tags
        tag_detection_buffer = "" 
        # List to store all text chunks yielded in this segment (for full message if no tool call)
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
                                        yield "first_tool_call_details", tool_name, tool_args
                                        stream.close() 
                                        return # Stop processing this stream; first tool found
                                    else:
                                        logger.warning(f"Malformed tool JSON (parsed but invalid structure): {tool_json_str}")
                                        # This malformed call will be treated as text by the main loop as it continues to stream.
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSONDecodeError in tool call: {e}. Content: {tool_json_str}")
                                    # This unparsable call will be treated as text.
                                
                                # If tool call was malformed or unparsable, it's treated as text.
                                # To prevent re-parsing this invalid block, advance buffer past it.
                                # This is tricky if we want to find *subsequent valid* tool calls.
                                # For "act on first *valid* tool call", this is fine.
                                tag_detection_buffer = tag_detection_buffer[end_idx + len(tool_call_end_tag):]

                    elif event.type == "message_stop":
                        final_message = stream.get_final_message()
                        final_stop_reason = final_message.stop_reason if final_message else "unknown_stop"
                        full_text = "".join(all_text_chunks_this_segment)
                        logger.info(f"Stream ended by API (message_stop). Stop Reason: {final_stop_reason}. Full text length: {len(full_text)}")
                        yield "stream_complete", full_text, final_stop_reason
                        return
            
                # Fallback if loop finishes (e.g. stream closed by interrupt before message_stop)
                final_message_obj_fallback = stream.get_final_message() # May be None if stream was hard closed
                final_stop_reason_fallback = final_message_obj_fallback.stop_reason if final_message_obj_fallback else "ended_unexpectedly"
                full_text_fallback = "".join(all_text_chunks_this_segment)
                logger.info(f"Stream loop exited. Final text: '{full_text_fallback[:100]}...'. Fallback Stop Reason: {final_stop_reason_fallback}")
                yield "stream_complete", full_text_fallback, final_stop_reason_fallback

        except Exception as e:
            error_type = "api_error" if isinstance(e, anthropic.APIError) else "stream_processing_error"
            logger.error(f"Error during Anthropic stream ({error_type}): {e}", exc_info=True)
            yield "error", f"Stream error ({error_type}): {e}", error_type

    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
        # (This method remains largely the same, but will now use the streaming client internally
        # and accumulate the summary text)
        if self.interrupted: logger.info("Summarization interrupted."); return None
        if not conversation_history: logger.info("No history to summarize."); return None

        summarization_system_prompt = (
            f"Summarize the following conversation concisely, aiming for {target_token_count} tokens. "
            "Focus on key facts, decisions, and outcomes. Output ONLY the summary text."
        )
        max_summary_tokens = int(target_token_count * 1.5); 
        if max_summary_tokens < 200: max_summary_tokens = 200
        if max_summary_tokens > 4096: max_summary_tokens = 4096
        if target_token_count > 4096 : max_summary_tokens = int(target_token_count * 1.2)
        
        logger.info(f"Requesting summarization. Max summary tokens: {max_summary_tokens}")
        
        accumulated_summary = []
        final_reason = "error"
        for event_type, data, *extra in self.get_response_stream(
            system_prompt=summarization_system_prompt,
            messages=conversation_history,
            max_tokens=max_summary_tokens
        ):
            if event_type == "text_chunk":
                accumulated_summary.append(data)
            elif event_type == "first_tool_call_details": 
                logger.warning("Tool call detected during summarization, ignoring.")
                # This implies the summary itself contained a tool call, which is bad.
                # We should probably use the text accumulated so far if any.
                final_reason = "tool_call_in_summary" 
                break 
            elif event_type == "stream_complete":
                final_reason = extra[0] if extra else "unknown_end"
                # 'data' here is the full_text from the client's buffer for stream_complete
                if data: # Prefer this fully formed text if available
                    accumulated_summary = [data] 
                break
            elif event_type in ["error", "interrupted"]:
                logger.error(f"Summarization stream error/interrupt: {event_type} - {data}")
                return None
        
        full_summary_text = "".join(accumulated_summary).strip()
        if final_reason not in ["error", "interrupted", "tool_call_in_summary"] and full_summary_text:
            logger.info(f"Summarization successful. Length: {len(full_summary_text)}, Reason: {final_reason}")
            return full_summary_text
        
        logger.warning(f"Summarization failed or no text. Reason: {final_reason}, Text: '{full_summary_text}'")
        return None

# if __name__ == '__main__':
# ... (Test code would need significant rework to handle the new event types)
