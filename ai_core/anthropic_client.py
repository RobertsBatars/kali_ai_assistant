# ai_core/anthropic_client.py
import anthropic
import config  # Import from the root directory's config.py
import logging

logger = logging.getLogger(f"{config.SERVICE_NAME}.AnthropicClient")

class AnthropicClient:
    def __init__(self, api_key=None, model_name=None):
        """
        Initializes the Anthropic client.
        Args:
            api_key (str, optional): Anthropic API key. Defaults to config.ANTHROPIC_API_KEY.
            model_name (str, optional): The model name to use. Defaults to config.DEFAULT_AI_MODEL.
        """
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        if not self.api_key:
            logger.error("Anthropic API key is not configured.")
            raise ValueError("Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your .env file.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_name = model_name or config.DEFAULT_AI_MODEL
        self.interrupted = False # Flag for interruption
        logger.info(f"AnthropicClient initialized with model: {self.model_name}")

    def set_interrupted(self, interrupted_status):
        """Sets the interruption status."""
        if self.interrupted != interrupted_status: # Log only on change
            logger.debug(f"Interruption status set to: {interrupted_status}")
        self.interrupted = interrupted_status

    def get_response(self, system_prompt, messages, max_tokens=2048):
        """
        Gets a response from the Anthropic API.
        Args:
            system_prompt (str): The system prompt defining the AI's role and instructions.
            messages (list): A list of message objects representing the conversation history.
            max_tokens (int): The maximum number of tokens to generate.
        Returns:
            str: The AI's response content, or None if an error occurs or interrupted.
        """
        if self.interrupted:
            logger.info("AI interaction interrupted before API call.")
            print("AI interaction interrupted.") # User-facing message
            return None
        
        try:
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                logger.error(f"Invalid messages format: {type(messages)}. Expected list of dicts.")
                raise ValueError("Messages must be a list of dictionaries.")

            logger.debug(f"Sending request to Anthropic. Model: {self.model_name}, System Prompt (len): {len(system_prompt)}, Messages (count): {len(messages)}, Max Tokens: {max_tokens}")
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )
            
            if response.content and len(response.content) > 0:
                # Assuming the first content block is the primary text response
                ai_text_response = response.content[0].text
                logger.info(f"Received response from Anthropic. Content length: {len(ai_text_response)}")
                logger.debug(f"Anthropic raw response: {response}")
                return ai_text_response
            logger.warning("Anthropic response content is empty or missing.")
            return None
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
        return None

    def summarize_conversation(self, conversation_history: list[dict], target_token_count: int) -> str | None:
        """
        Uses the Anthropic model to summarize a conversation history.
        Args:
            conversation_history (list[dict]): The conversation history to summarize.
            target_token_count (int): The desired token count for the summary (approximate).
        Returns:
            str: The summarized text, or None if an error occurs.
        """
        if self.interrupted:
            logger.info("Summarization interrupted before API call.")
            return None

        if not conversation_history:
            logger.info("No conversation history to summarize.")
            return None

        # Create a prompt for the summarization task
        # The summarization prompt itself should be concise.
        # We pass the history as messages, and a specific system prompt for summarization.
        summarization_system_prompt = (
            "You are a summarization expert. Summarize the following conversation concisely, "
            "focusing on key facts, decisions, user requests, and outcomes. "
            "The summary should be a coherent narrative that captures the essence of the conversation. "
            f"Aim for a summary that is approximately {target_token_count} tokens long. "
            "Retain critical information, especially unresolved questions or ongoing tasks. "
            "Output ONLY the summary text, without any preamble or sign-off."
        )
        
        # The conversation_history itself is the main content to be summarized.
        # Anthropic's `messages` API expects the history in the `messages` parameter.
        # The `system` parameter is for the overarching instruction (our summarization_system_prompt).

        logger.info(f"Requesting summarization for {len(conversation_history)} messages. Target tokens: {target_token_count}")
        
        try:
            # We can use a slightly different model or settings for summarization if needed,
            # but for now, we'll use the default model.
            # Max tokens for summary generation should be related to target_token_count, but give it some room.
            # A good summary might be a bit longer or shorter.
            max_summary_gen_tokens = int(target_token_count * 1.5) # Allow summary to be up to 1.5x target
            if max_summary_gen_tokens < 500: max_summary_gen_tokens = 500 # Minimum reasonable summary length
            if max_summary_gen_tokens > 4000: max_summary_gen_tokens = 4000 # Cap summary generation length

            response = self.client.messages.create(
                model=self.model_name, # Or a model specialized for summarization if available/preferred
                max_tokens=max_summary_gen_tokens,
                system=summarization_system_prompt,
                messages=conversation_history # Pass the actual history here
            )

            if response.content and len(response.content) > 0:
                summary_text = response.content[0].text
                logger.info(f"Summarization successful. Summary length: {len(summary_text)} chars.")
                return summary_text
            logger.warning("Summarization response content is empty.")
            return None
        except Exception as e:
            logger.error(f"Error during summarization API call: {e}", exc_info=True)
            print(f"Error during summarization: {e}")
            return None


if __name__ == '__main__':
    # Example usage (requires .env file with ANTHROPIC_API_KEY)
    # This part is for testing the client directly.
    # Ensure config.py and .env are correctly set up.
    
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # If running this file directly, config.SERVICE_NAME might not be set up by a main app logger.
    # So, we use the logger directly.
    test_logger = logging.getLogger("AnthropicClientTest")

    try:
        client = AnthropicClient() # Uses config for API key and model
        
        # Load system prompt from file for testing
        try:
            # Adjust path if running from ai_core directory vs project root
            prompt_file_path = "../system_prompt.txt" 
            if not os.path.exists(prompt_file_path):
                prompt_file_path = "system_prompt.txt" # If running from project root

            with open(prompt_file_path, "r") as f:
                test_system_prompt = f.read()
            test_logger.info(f"Loaded system prompt from {prompt_file_path}")
        except FileNotFoundError:
            test_logger.error(f"Error: system_prompt.txt not found at {prompt_file_path} or ./system_prompt.txt.")
            test_system_prompt = "You are a helpful assistant."


        test_messages = [
            {"role": "user", "content": "Hello, who are you?"}
        ]
        
        test_logger.info("Sending request to Anthropic for regular response...")
        ai_response = client.get_response(system_prompt=test_system_prompt, messages=test_messages)
        
        if ai_response:
            test_logger.info("AI Response:\n" + ai_response)
        else:
            test_logger.warning("Failed to get response from AI.")

        # Test summarization
        test_logger.info("\n--- Testing Summarization ---")
        sample_history_for_summary = [
            {"role": "user", "content": "What was the first step we discussed for reconnaissance?"},
            {"role": "assistant", "content": "We talked about using nmap for initial port scanning."},
            {"role": "user", "content": "And after that, what tool did I mention for web enumeration?"},
            {"role": "assistant", "content": "You mentioned using Gobuster to find directories and files."},
            {"role": "user", "content": "Correct. And what was the target IP?"},
            {"role": "assistant", "content": "The target IP was 10.0.0.5."},
            {"role": "user", "content": "Great, let's proceed with that nmap scan then."},
            {"role": "assistant", "content": "Okay, I will prepare the nmap command. Do you want to specify any particular nmap flags or use a default scan?"}
        ]
        
        summary = client.summarize_conversation(sample_history_for_summary, target_token_count=50)
        if summary:
            test_logger.info("Generated Summary:\n" + summary)
        else:
            test_logger.warning("Failed to generate summary.")


    except ValueError as ve:
        test_logger.critical(f"Configuration Error: {ve}")
    except Exception as e:
        test_logger.critical(f"An error occurred during testing: {e}", exc_info=True)

