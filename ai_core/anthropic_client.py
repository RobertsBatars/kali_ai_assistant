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
        if self.interrupted != interrupted_status: # Log only on change
            logger.debug(f"Interruption status set to: {interrupted_status}")
        self.interrupted = interrupted_status

    def get_response(self, system_prompt, messages, max_tokens=None):
        """
        Gets a response from the Anthropic API.
        Args:
            system_prompt (str): The system prompt defining the AI's role and instructions.
            messages (list): A list of message objects representing the conversation history.
            max_tokens (int, optional): The maximum number of tokens to generate. 
                                        Defaults to config.MAX_AI_OUTPUT_TOKENS.
        Returns:
            tuple(str | None, str | None): (The AI's response content, stop_reason) or (None, "error" or "interrupted")
        """
        if self.interrupted:
            logger.info("AI interaction interrupted before API call.")
            # User-facing message might be printed by the main loop based on interrupt flag
            return None, "interrupted"
        
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_AI_OUTPUT_TOKENS
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                logger.error(f"Invalid messages format: {type(messages)}. Expected list of dicts.")
                raise ValueError("Messages must be a list of dictionaries.")

            logger.debug(f"Sending request to Anthropic. Model: {self.model_name}, System Prompt (len): {len(system_prompt)}, Messages (count): {len(messages)}, Max Tokens: {effective_max_tokens}")
            
            response_obj = self.client.messages.create(
                model=self.model_name,
                max_tokens=effective_max_tokens,
                system=system_prompt,
                messages=messages
            )
            
            ai_text_response = None
            # Ensure stop_reason is always present, default to None if not in response_obj for some reason
            stop_reason = getattr(response_obj, 'stop_reason', None) 

            if response_obj.content and len(response_obj.content) > 0:
                ai_text_response = response_obj.content[0].text
                logger.info(f"Received response from Anthropic. Content length: {len(ai_text_response)}, Stop Reason: {stop_reason}")
                logger.debug(f"Anthropic raw response object: {response_obj}")
            else:
                logger.warning(f"Anthropic response content is empty or missing. Stop Reason: {stop_reason}")
            
            return ai_text_response, stop_reason

        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {e}")
            print(f"Anthropic API connection error: {e}")
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic API rate limit exceeded: {e}")
            print(f"Anthropic API rate limit exceeded: {e}")
        except anthropic.APIStatusError as e:
            logger.error(f"Anthropic API status error: {e.status_code} - {e.response}")
            print(f"Anthropic API status error: {e.status_code} - {e.response}")
        except anthropic.APIError as e: # Catch other Anthropic specific API errors
            logger.error(f"Anthropic API error: {e}")
            print(f"Anthropic API error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while contacting Anthropic API: {e}", exc_info=True)
            print(f"An unexpected error occurred while contacting Anthropic API: {e}")
        return None, "error" # Ensure a stop_reason is returned on error

    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
        """
        Uses the Anthropic model to summarize a conversation history.
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
            max_summary_gen_tokens = int(target_token_count * 1.5) 
            if max_summary_gen_tokens < 500: max_summary_gen_tokens = 500
            if max_summary_gen_tokens > 4000: max_summary_gen_tokens = 4000

            summary_text, stop_reason = self.get_response(
                system_prompt=summarization_system_prompt,
                messages=conversation_history,
                max_tokens=max_summary_gen_tokens
            )

            if summary_text and stop_reason not in ["error", "interrupted"]:
                logger.info(f"Summarization successful. Summary length: {len(summary_text)} chars. Stop reason: {stop_reason}")
                return summary_text
            logger.warning(f"Summarization response content is empty or an error occurred. Stop reason: {stop_reason}")
            return None
        except Exception as e:
            logger.error(f"Error during summarization API call: {e}", exc_info=True)
            # User-facing message might be handled by the caller
            return None

if __name__ == '__main__':
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("AnthropicClientTest")

    try:
        # Ensure config.py is accessible (e.g., run from project root)
        client = AnthropicClient()
        
        prompt_file_path = "system_prompt.txt" 
        try:
            with open(prompt_file_path, "r") as f: 
                test_system_prompt = f.read()
        except FileNotFoundError:
            test_logger.error(f"system_prompt.txt not found at {prompt_file_path}")
            test_system_prompt = "You are a helpful assistant."


        test_messages = [
            {"role": "user", "content": "Hello, who are you?"}
        ]
        
        test_logger.info("Sending request to Anthropic for regular response...")
        ai_response, stop_reason = client.get_response(system_prompt=test_system_prompt, messages=test_messages)
        
        if ai_response:
            test_logger.info(f"AI Response:\n{ai_response}\nStop Reason: {stop_reason}")
        else:
            test_logger.warning(f"Failed to get response from AI. Stop Reason: {stop_reason}")

    except ValueError as ve:
        test_logger.critical(f"Configuration Error: {ve}")
    except Exception as e:
        test_logger.critical(f"An error occurred during testing: {e}", exc_info=True)
