# utils/token_estimator.py
import logging

# Get a logger instance for this module, prefixed by the service name if setup elsewhere
logger = logging.getLogger("KaliAIAssistant.TokenEstimator")

# Average characters per token (heuristic, varies by model and language)
# For many English models, it's roughly 4 characters per token.
# This is a very rough estimate.
CHARS_PER_TOKEN_ESTIMATE = 4

def estimate_token_count(text: str) -> int:
    """
    Estimates the token count of a given text.
    This is a heuristic and not a precise measure.
    Args:
        text (str): The text to estimate token count for.
    Returns:
        int: The estimated token count.
    """
    if not text:
        return 0
    # A simple heuristic: number of characters / average characters per token
    # Another common one is roughly num_words * 1.33
    estimated_tokens = len(text) / CHARS_PER_TOKEN_ESTIMATE
    # logger.debug(f"Estimated tokens for text (len {len(text)} chars): {int(estimated_tokens)}")
    return int(estimated_tokens)

def estimate_messages_token_count(messages: list[dict]) -> int:
    """
    Estimates the total token count for a list of message objects.
    Args:
        messages (list[dict]): A list of message objects,
                               each with a "content" key.
    Returns:
        int: The total estimated token count for all messages.
    """
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_tokens += estimate_token_count(content)
        elif isinstance(content, list): # Handle cases like Anthropic's multimodal content
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total_tokens += estimate_token_count(item.get("text", ""))
    # logger.debug(f"Total estimated tokens for messages list: {total_tokens}")
    return total_tokens

if __name__ == '__main__':
    test_text_short = "Hello world!"
    test_text_long = "This is a longer sentence to test the token estimation. It has several words and punctuation marks."

    print(f"'{test_text_short}' - Estimated tokens: {estimate_token_count(test_text_short)}")
    print(f"'{test_text_long}' - Estimated tokens: {estimate_token_count(test_text_long)}")

    test_messages_list = [
        {"role": "user", "content": "What is the weather like?"},
        {"role": "assistant", "content": "The weather is sunny today."},
        {"role": "user", "content": test_text_long}
    ]
    total_estimated = estimate_messages_token_count(test_messages_list)
    print(f"Total estimated tokens for messages list: {total_estimated}")

    # Example of multimodal content structure (simplified)
    multimodal_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            # {"type": "image", ...} # Image data not counted by this simple estimator
        ]}
    ]
    multimodal_tokens = estimate_messages_token_count(multimodal_messages)
    print(f"Total estimated tokens for multimodal messages list: {multimodal_tokens}")
