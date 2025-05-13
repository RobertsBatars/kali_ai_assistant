# ai_core/anthropic_client.py
import anthropic
import config
import logging

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

    def get_response(self, system_prompt, messages, max_tokens=None):
        """
        Gets a response from the Anthropic API using streaming.
        The method accumulates the streamed response and returns it as a whole.
        Args:
            system_prompt (str): The system prompt.
            messages (list): Conversation history.
            max_tokens (int, optional): Max tokens for generation. Defaults to config.MAX_AI_OUTPUT_TOKENS.
        Returns:
            tuple(str | None, str | None): (The AI's response content, stop_reason) or (None, "error" or "interrupted")
        """
        if self.interrupted:
            logger.info("AI interaction interrupted before API call.")
            return None, "interrupted"
        
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        accumulated_text = []
        final_stop_reason = None
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                logger.error(f"Invalid messages format: {type(messages)}. Expected list of dicts.")
                raise ValueError("Messages must be a list of dictionaries.")

            logger.debug(f"Opening stream to Anthropic. Model: {self.model_name}, System Prompt (len): {len(system_prompt)}, Messages (count): {len(messages)}, Max Tokens: {effective_max_tokens}")
            
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=effective_max_tokens,
                system=system_prompt,
                messages=messages
            ) as stream:
                for event in stream:
                    if self.interrupted:
                        logger.info("AI stream processing interrupted by flag.")
                        stream.close() # Attempt to close the stream
                        return "".join(accumulated_text) if accumulated_text else None, "interrupted"

                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            # logger.debug(f"Stream text delta: '{event.delta.text}'") # Can be very verbose
                            accumulated_text.append(event.delta.text)
                    # The stop_reason might also be available in message_delta or message_stop events
                    # but get_final_message() is the most robust way to get the complete message object.

                # After the loop, finalize the message
                final_message = stream.get_final_message()
                final_stop_reason = final_message.stop_reason
                # The accumulated_text should match final_message.content[0].text if there's only one text block
                # For safety, we use the accumulated text from deltas.
                
            full_response_text = "".join(accumulated_text)
            logger.info(f"Stream finished. Total content length: {len(full_response_text)}, Stop Reason: {final_stop_reason}")
            logger.debug(f"Anthropic final message object from stream: {final_message}")
            
            return full_response_text, final_stop_reason

        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {e}")
            print(f"Anthropic API connection error: {e}")
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic API rate limit exceeded: {e}")
            print(f"Anthropic API rate limit exceeded: {e}")
        except anthropic.APIStatusError as e: # Includes APIResponseValidationError for unexpected stream events
            logger.error(f"Anthropic API status/response error: {e.status_code} - {e.response if hasattr(e, 'response') else e}", exc_info=True)
            print(f"Anthropic API status/response error: {e.status_code if hasattr(e, 'status_code') else 'N/A'} - {e.response if hasattr(e, 'response') else e}")
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            print(f"Anthropic API error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Anthropic stream: {e}", exc_info=True)
            print(f"An unexpected error occurred during Anthropic stream: {e}")
        
        return "".join(accumulated_text) if accumulated_text else None, "error"


    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
        """
        Uses the Anthropic model to summarize a conversation history (also uses streaming internally).
        """
        if self.interrupted:
            logger.info("Summarization interrupted before API call.")
            return None

        if not conversation_history:
            logger.info("No conversation history to summarize.")
            return None

        summarization_system_prompt = (
            "You are a summarization expert. Summarize the following conversation concisely, "
            "focusing on key facts, decisions, user requests, and outcomes. "
            "The summary should be a coherent narrative that captures the essence of the conversation. "
            f"Aim for a summary that is approximately {target_token_count} tokens long. "
            "Retain critical information, especially unresolved questions or ongoing tasks. "
            "Output ONLY the summary text, without any preamble or sign-off."
        )
        
        logger.info(f"Requesting summarization for {len(conversation_history)} messages. Target tokens: {target_token_count}")
        
        try:
            # Determine max_tokens for summary generation
            # This should be related to target_token_count, but give it some leeway.
            max_summary_gen_tokens = int(target_token_count * 1.5) 
            if max_summary_gen_tokens < 500: max_summary_gen_tokens = 500 # Minimum reasonable summary length
            # Cap summary generation length to avoid excessive token use for summaries
            # This should be less than config.MAX_AI_OUTPUT_TOKENS if that is very large.
            # Let's use a fixed cap like 4096 for summaries unless target is larger.
            if max_summary_gen_tokens > 4096: max_summary_gen_tokens = 4096 
            if target_token_count > 4096 : max_summary_gen_tokens = int(target_token_count * 1.2) # If target is huge


            summary_text, stop_reason = self.get_response(
                system_prompt=summarization_system_prompt,
                messages=conversation_history,
                max_tokens=max_summary_gen_tokens
            )

            if summary_text and stop_reason not in ["error", "interrupted"]:
                logger.info(f"Summarization successful. Summary length: {len(summary_text)} chars. Stop reason: {stop_reason}")
                return summary_text
            
            logger.warning(f"Summarization response content is empty or an error occurred. Stop reason: {stop_reason}. Summary text: '{summary_text}'")
            return None
        except Exception as e: # Catch any other unexpected error from get_response or logic here
            logger.error(f"Error during summarization call: {e}", exc_info=True)
            return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("AnthropicClientTest")
    # Ensure config.py is in the parent directory or adjust path if running this file directly
    # For testing, you might need to temporarily add `sys.path.append('..')` if config is not found
    # import sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # import config # Now config should be found

    try:
        client = AnthropicClient() # Uses config for API key and model
        
        prompt_file_path = "../system_prompt.txt" # If in ai_core
        # Check if running from project root instead
        import os
        if not os.path.exists(prompt_file_path):
            prompt_file_path = "system_prompt.txt"

        try:
            with open(prompt_file_path, "r") as f: 
                test_system_prompt = f.read()
            test_logger.info(f"Loaded system prompt from {prompt_file_path}")
        except FileNotFoundError:
            test_logger.error(f"Error: system_prompt.txt not found at {prompt_file_path} or ./system_prompt.txt.")
            test_system_prompt = "You are a helpful assistant."


        test_messages_regular = [
            {"role": "user", "content": "Hello, who are you and what can you do in a short sentence?"}
        ]
        
        test_logger.info("--- Testing Regular Response (Streamed but Accumulated) ---")
        ai_response, stop_reason = client.get_response(system_prompt=test_system_prompt, messages=test_messages_regular)
        
        if ai_response:
            test_logger.info(f"AI Response:\n{ai_response}\nStop Reason: {stop_reason}")
        else:
            test_logger.warning(f"Failed to get response from AI. Stop Reason: {stop_reason}")

        test_logger.info("\n--- Testing Summarization (Streamed but Accumulated) ---")
        sample_history_for_summary = [
            {"role": "user", "content": "What was the first step we discussed for reconnaissance?"},
            {"role": "assistant", "content": "We talked about using nmap for initial port scanning."},
            {"role": "user", "content": "And after that, what tool did I mention for web enumeration?"},
            {"role": "assistant", "content": "You mentioned using Gobuster to find directories and files."},
            {"role": "user", "content": "Correct. And what was the target IP?"},
            {"role": "assistant", "content": "The target IP was 10.0.0.5."},
        ]
        
        summary = client.summarize_conversation(sample_history_for_summary, target_token_count=30) # Short summary
        if summary:
            test_logger.info("Generated Summary:\n" + summary)
        else:
            test_logger.warning("Failed to generate summary.")
            
        test_logger.info("\n--- Testing Interruption during stream (simulated) ---")
        # To truly test interruption, you'd need to set client.interrupted = True from another thread
        # during the stream. This is a conceptual test.
        client.set_interrupted(True)
        interrupted_response, interrupted_reason = client.get_response(
            system_prompt=test_system_prompt, 
            messages=[{"role": "user", "content": "Tell me a very long story that will take time to generate."}]
        )
        client.set_interrupted(False) # Reset for other tests
        if interrupted_reason == "interrupted":
            test_logger.info(f"Interruption test successful. Partial response: '{interrupted_response}', Reason: {interrupted_reason}")
        else:
            test_logger.error(f"Interruption test failed or completed too quickly. Response: '{interrupted_response}', Reason: {interrupted_reason}")


    except ValueError as ve:
        test_logger.critical(f"Configuration Error: {ve}")
    except Exception as e:
        test_logger.critical(f"An error occurred during testing: {e}", exc_info=True)

