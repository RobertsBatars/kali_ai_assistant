# ai_core/lm_studio_llm_client.py
import requests
import json
import config # Main config from project root
import logging
from .base_llm_client import BaseLLMClient # Import from the same directory

logger = logging.getLogger(f"{config.SERVICE_NAME}.LMStudioLLMClient")

class LMStudioLLMClient(BaseLLMClient):
    def __init__(self, api_base=None, model_name=None):
        super().__init__(model_name or config.LM_STUDIO_MODEL_NAME)
        self.api_base = api_base or config.LM_STUDIO_API_BASE
        if not self.api_base:
            logger.error("LM Studio API base URL is not configured.")
            raise ValueError("LM Studio API base URL is not configured.")
        logger.info(f"LMStudioLLMClient initialized. API Base: {self.api_base}, Model: {self.model_name}")

    def get_response_stream(self, system_prompt: str, messages: list, max_tokens: int | None = None):
        if self.interrupted:
            logger.info("LM Studio stream interrupted before API call.")
            yield "interrupted", "", "interrupted_before_call"
            return

        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        
        # Format messages for OpenAI-compatible API
        # System prompt goes as the first message with role "system"
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            # Ensure role is one of 'user', 'assistant' (or 'system' if not already added)
            role = msg.get("role")
            if role not in ["user", "assistant", "system"]:
                logger.warning(f"LMStudio: Invalid role '{role}' in message, defaulting to 'user'.")
                role = "user" 
            # LM Studio might not handle 'system' role well within the main messages list if already passed
            if role == "system" and any(m['role'] == 'system' for m in formatted_messages):
                # If system prompt already added, convert subsequent system messages to user or assistant context
                # This is a heuristic. Better to ensure history doesn't have extra system messages.
                logger.debug("LMStudio: Converting subsequent system message to user context for API.")
                role = "user" 
            formatted_messages.append({"role": role, "content": msg.get("content", "")})


        payload = {
            "model": self.model_name, # Model loaded in LM Studio
            "messages": formatted_messages,
            "stream": True,
            # "temperature": 0.7, # Optional: add other OpenAI params
        }
        if effective_max_tokens is not None and effective_max_tokens > 0 : # LM Studio might not always respect max_tokens strictly
             payload["max_tokens"] = effective_max_tokens


        logger.debug(f"Sending request to LM Studio: {self.api_base}/chat/completions. Payload (excluding messages): { {k:v for k,v in payload.items() if k != 'messages'} }")

        tag_detection_buffer = ""
        all_text_chunks_this_segment = []
        tool_call_start_tag = "<tool_call>"
        tool_call_end_tag = "</tool_call>"

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                stream=True,
                timeout=300 # Generous timeout for stream start, actual token generation can be long
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)

            for line in response.iter_lines():
                if self.interrupted:
                    logger.info("LM Studio stream processing interrupted by flag.")
                    yield "interrupted", "".join(all_text_chunks_this_segment), "interrupted_during_stream"
                    response.close()
                    return

                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_content_str = decoded_line[len("data: "):]
                        if json_content_str.strip() == "[DONE]":
                            logger.info("LM Studio stream: [DONE] received.")
                            break # Stream finished
                        
                        try:
                            json_content = json.loads(json_content_str)
                            if json_content.get("choices"):
                                delta = json_content["choices"][0].get("delta", {})
                                text_chunk = delta.get("content")
                                if text_chunk: # Can be None if it's just a role or finish_reason
                                    yield "text_chunk", text_chunk
                                    all_text_chunks_this_segment.append(text_chunk)
                                    tag_detection_buffer += text_chunk

                                    # Tool call detection (same logic as Anthropic client)
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
                                                    logger.info(f"LMStudio: First tool call detected: {tool_name}")
                                                    preamble_text = "".join(all_text_chunks_this_segment).split(tool_call_start_tag)[0].strip()
                                                    yield "first_tool_call_details", preamble_text, tool_name, tool_args
                                                    response.close()
                                                    return
                                            except json.JSONDecodeError: pass # Malformed JSON
                                            tag_detection_buffer = tag_detection_buffer[end_idx + len(tool_call_end_tag):]
                                
                                # Check for finish reason if present in delta or top level choice
                                finish_reason = json_content["choices"][0].get("finish_reason")
                                if finish_reason:
                                    logger.info(f"LM Studio stream: Choice finished with reason: {finish_reason}")
                                    # This might signal end of content for this choice, but stream might continue for other reasons
                                    # The [DONE] marker is more definitive for the whole stream.
                        except json.JSONDecodeError:
                            logger.warning(f"LM Studio stream: Could not decode JSON from line: {json_content_str}")
            
            # After loop (either by [DONE] or iter_lines finishing)
            full_text = "".join(all_text_chunks_this_segment)
            # LM Studio doesn't provide a clear stop_reason like Anthropic's message_stop event
            # We infer "stop" if [DONE] was received or loop finished.
            # If max_tokens was hit, finish_reason might indicate that.
            # For now, assume "stop" if not interrupted or errored.
            # A more robust way would be to check the last `finish_reason` if available.
            stop_reason_final = "stop" # Default for LM Studio successful completion
            logger.info(f"LM Studio stream processing finished. Full text length: {len(full_text)}")
            yield "stream_complete", full_text, stop_reason_final

        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio request error: {e}", exc_info=True)
            yield "error", f"LM Studio request error: {e}", "network_error"
        except Exception as e:
            logger.error(f"Unexpected error during LM Studio stream: {e}", exc_info=True)
            yield "error", f"Unexpected LM Studio stream error: {e}", "processing_error"

    def summarize_conversation(self, conversation_history: list, target_token_count: int) -> str | None:
        # (Similar to Anthropic's, uses self.get_response_stream and accumulates)
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
        
        logger.info(f"LMStudio: Requesting summarization. Max summary tokens: {max_summary_tokens}")
        
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
                logger.warning(f"LMStudio: Tool call ('{tool_name}') detected during summarization. Invalid.")
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
                logger.error(f"LMStudio: Summarization stream error/interrupt: {event_type} - {data}")
                return None
        
        full_summary_text = "".join(accumulated_summary_text_chunks).strip()

        if tool_call_detected_in_summary_attempt or \
           ("<tool_call>" in full_summary_text or "</tool_call>" in full_summary_text):
            logger.warning(f"LMStudio: Tool call tags present in summary attempt. Original: '{full_summary_text[:300]}...'")
            import re # Local import
            cleaned_summary_text = re.sub(r"<tool_call>.*?</tool_call>", "", full_summary_text, flags=re.DOTALL).strip()
            if not cleaned_summary_text: logger.error("LMStudio: Summary empty after cleaning tool calls."); return None
            logger.info(f"LMStudio: Summary cleaned. Length: {len(cleaned_summary_text)}.")
            return cleaned_summary_text
        
        if final_reason_for_summary not in ["error", "interrupted"] and full_summary_text:
            logger.info(f"LMStudio: Summarization successful. Length: {len(full_summary_text)}, Reason: {final_reason_for_summary}")
            return full_summary_text
        
        logger.warning(f"LMStudio: Summarization failed or no text. Reason: {final_reason_for_summary}, Text: '{full_summary_text[:200]}'")
        return None
