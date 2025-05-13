# ai_core/anthropic_client.py
import anthropic
import config
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
        Yields:
            - ("text_delta", str_chunk) for text parts.
            - ("tool_call", tool_name, tool_args) for the *first* complete tool call, then stops.
            - ("stream_end", full_text_before_tool_or_end, stop_reason) if stream ends without a tool call yielded.
            - ("error", error_message_str) on API or processing error.
            - ("interrupted", accumulated_text_before_interrupt) if interrupted.
        """
        if self.interrupted:
            logger.info("AI interaction interrupted before API call.")
            yield "interrupted", "", "interrupted_before_call"
            return
        
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        
        text_buffer = "" # Buffer to accumulate text and detect tool calls
        tool_call_start_tag = "<tool_call>"
        tool_call_end_tag = "</tool_call>"
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                logger.error(f"Invalid messages format: {type(messages)}. Expected list of dicts.")
                raise ValueError("Messages must be a list of dictionaries.")

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
                        yield "interrupted", text_buffer, "interrupted_during_stream"
                        stream.close()
                        return

                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            text_chunk = event.delta.text
                            yield "text_delta", text_chunk
                            text_buffer += text_chunk

                            # Check for complete tool call
                            start_idx = text_buffer.find(tool_call_start_tag)
                            if start_idx != -1:
                                end_idx = text_buffer.find(tool_call_end_tag, start_idx)
                                if end_idx != -1:
                                    # Found a complete tool call
                                    tool_json_str = text_buffer[start_idx + len(tool_call_start_tag) : end_idx]
                                    preamble_text = text_buffer[:start_idx].strip()
                                    
                                    try:
                                        tool_data = json.loads(tool_json_str)
                                        tool_name = tool_data.get("tool_name")
                                        tool_args = tool_data.get("arguments")

                                        if tool_name and isinstance(tool_args, dict):
                                            logger.info(f"First tool call detected and parsed: {tool_name}")
                                            # Yield preamble first if it exists and hasn't been fully yielded by chunks
                                            # The caller will accumulate preamble from text_delta events.
                                            # We just need to signal the tool call.
                                            yield "tool_call", tool_name, tool_args
                                            stream.close() # Stop processing further from this stream
                                            return
                                        else:
                                            logger.warning(f"Malformed tool JSON: {tool_json_str}")
                                            # Continue streaming, treating this malformed part as text
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSONDecodeError in tool call: {e}. Content: {tool_json_str}")
                                        # Continue streaming, treating this as text
                    elif event.type == "message_stop":
                        # This event signals the end of the message from the API
                        final_message = stream.get_final_message() # Get the full message object
                        final_stop_reason = final_message.stop_reason
                        logger.info(f"Stream ended by API. Stop Reason: {final_stop_reason}. Final text buffer length: {len(text_buffer)}")
                        yield "stream_end", text_buffer, final_stop_reason
                        return
                
                # If loop finishes without message_stop (e.g. stream closed by us after tool_call)
                # or if stream had no message_stop but ended.
                # This part might be redundant if message_stop is always guaranteed.
                final_message_obj = stream.get_final_message()
                final_stop_reason_fallback = final_message_obj.stop_reason if final_message_obj else "unknown"
                logger.info(f"Stream loop exited. Final text buffer: '{text_buffer[:100]}...'. Fallback Stop Reason: {final_stop_reason_fallback}")
                yield "stream_end", text_buffer, final_stop_reason_fallback
                return

        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {e}")
            yield "error", f"API connection error: {e}", "error"
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic API rate limit exceeded: {e}")
            yield "error", f"API rate limit exceeded: {e}", "error"
        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API status/response error: {e.status_code} - {e.response if hasattr(e, 'response') else e}", exc_info=True)
            yield "error", f"API status error {e.status_code if hasattr(e, 'status_code') else 'N/A'}: {e}", "error"
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            yield "error", f"API error: {e}", "error"
        except Exception as e:
            logger.error(f"An unexpected error occurred during Anthropic stream: {e}", exc_info=True)
            yield "error", f"Unexpected stream error: {e}", "error"

    # get_response is now an alias or wrapper if needed for non-streaming summarization,
    # but summarization can also use the stream.
    # For simplicity, let's assume summarization will also be adapted or uses a non-streaming path if that's simpler.
    # The main interaction loop will use get_response_stream.

    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
        # This still uses the old synchronous get_response logic for simplicity,
        # as summaries are usually not that long and don't need UI streaming.
        # It will internally use the new streaming get_response but accumulate.
        if self.interrupted:
            logger.info("Summarization interrupted before API call.")
            return None
        if not conversation_history:
            logger.info("No conversation history to summarize.")
            return None

        summarization_system_prompt = (
            "You are a summarization expert. Summarize the following conversation concisely, "
            f"aiming for approximately {target_token_count} tokens. Focus on key facts, decisions, and outcomes. "
            "Output ONLY the summary text."
        )
        max_summary_gen_tokens = int(target_token_count * 1.5)
        if max_summary_gen_tokens < 500: max_summary_gen_tokens = 500
        if max_summary_gen_tokens > 4096: max_summary_gen_tokens = 4096 
        if target_token_count > 4096 : max_summary_gen_tokens = int(target_token_count * 1.2)
        
        logger.info(f"Requesting summarization for {len(conversation_history)} messages. Target tokens: {target_token_count}, Max gen: {max_summary_gen_tokens}")

        # Accumulate response from stream for summarization
        full_summary_text = ""
        final_reason = "error"
        # Use a list for messages to summarize, not the main conversation_history directly
        # if it contains system roles not suitable for 'messages' param directly.
        # For summarization, the history *is* the messages.
        for event_type, data, *extra in self.get_response_stream(
            system_prompt=summarization_system_prompt,
            messages=conversation_history, # Pass the history to be summarized
            max_tokens=max_summary_gen_tokens
        ):
            if event_type == "text_delta":
                full_summary_text += data
            elif event_type == "tool_call": # Should not happen for summarization prompt
                logger.warning("Tool call received during summarization, ignoring.")
                break 
            elif event_type == "stream_end":
                # data here is the full_text, extra[0] is stop_reason
                final_reason = extra[0] if extra else "unknown_end"
                break
            elif event_type == "error" or event_type == "interrupted":
                final_reason = extra[0] if extra else event_type
                logger.error(f"Summarization failed or interrupted: {event_type} - {data} - Reason: {final_reason}")
                return None
        
        if final_reason not in ["error", "interrupted"] and full_summary_text:
            logger.info(f"Summarization successful. Length: {len(full_summary_text)}, Stop reason: {final_reason}")
            return full_summary_text.strip()
        logger.warning(f"Summarization resulted in no text or error. Reason: {final_reason}")
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("AnthropicClientTestStream")
    import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # If needed
    # import config

    try:
        client = AnthropicClient()
        prompt_file_path = "../system_prompt.txt"
        if not os.path.exists(prompt_file_path): prompt_file_path = "system_prompt.txt"
        try:
            with open(prompt_file_path, "r") as f: test_system_prompt = f.read()
        except FileNotFoundError: test_system_prompt = "You are a helpful assistant."

        test_logger.info("--- Testing Streamed Response with Text and Tool Call ---")
        # Simulate a response that has text then a tool call
        mock_messages_tool_call = [
            {"role": "user", "content": "Please list files in current directory and then tell me the weather."}
        ]
        
        # This test won't actually make Claude output a tool call reliably without a complex prompt.
        # The logic is more about how the client *would* process it if Claude did.
        # Instead, let's test the summarization which uses the stream internally.

        test_logger.info("\n--- Testing Summarization (which uses get_response_stream internally) ---")
        sample_history = [
            {"role": "user", "content": "First, I did X."},
            {"role": "assistant", "content": "Okay, X was done."},
            {"role": "user", "content": "Then I performed Y."},
            {"role": "assistant", "content": "Understood, Y was performed. What's next?"}
        ]
        summary = client.summarize_conversation(sample_history, 20)
        if summary:
            test_logger.info(f"Generated Summary:\n{summary}")
        else:
            test_logger.warning("Failed to generate summary.")

        test_logger.info("\n--- Testing direct stream consumption (conceptual) ---")
        test_messages_text_only = [{"role": "user", "content": "Explain quantum physics in one short paragraph."}]
        full_text_parts = []
        for event_type, data, *extra_data in client.get_response_stream(test_system_prompt, test_messages_text_only, max_tokens=100):
            test_logger.debug(f"Event: {event_type}, Data: '{str(data)[:50]}...', Extra: {extra_data}")
            if event_type == "text_delta":
                print(data, end="", flush=True) # Simulate real-time printing
                full_text_parts.append(data)
            elif event_type == "tool_call":
                print(f"\nTOOL CALL DETECTED: {data} with args {extra_data[0]}")
                break
            elif event_type == "stream_end":
                print(f"\nSTREAM ENDED. Reason: {extra_data[0] if extra_data else 'N/A'}")
                break
            elif event_type == "error" or event_type == "interrupted":
                print(f"\nSTREAM ERROR/INTERRUPT: {data}, Reason: {extra_data[0] if extra_data else 'N/A'}")
                break
        test_logger.info(f"\n--- End of direct stream test. Full accumulated text: {''.join(full_text_parts)[:200]}...")


    except Exception as e:
        test_logger.critical(f"An error occurred during testing: {e}", exc_info=True)

