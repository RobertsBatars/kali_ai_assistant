# kali_ai_tool.py
import json
import readline # For better input experience
import sys
import logging # For logging
import time # For time related operations

# Project imports
import config
from ai_core.anthropic_client import AnthropicClient
from tools.base_tool import BaseTool
from tools.command_line_tool import CommandLineTool
from tools.web_search_tool import WebSearchTool
from tools.cve_search_tool import CVESearchTool
from utils.interrupt_handler import InterruptHandler
from utils.logger_setup import setup_logging # New import
from utils.token_estimator import estimate_messages_token_count # New import

# --- Global Variables & Setup ---

# Setup Logging (Call this early)
# The logger instance can be retrieved in other modules via logging.getLogger(config.SERVICE_NAME)
# or logging.getLogger(f"{config.SERVICE_NAME}.module_name")
logger = setup_logging(
    log_file_path=config.LOG_FILE_PATH,
    log_level_file=config.LOG_LEVEL_FILE,
    log_level_console=config.LOG_LEVEL_CONSOLE,
    service_name=config.SERVICE_NAME
)

# Load system prompt
try:
    with open("system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    logger.info("System prompt loaded successfully.")
except FileNotFoundError:
    logger.critical("CRITICAL: system_prompt.txt not found. Please create it.")
    print("Error: system_prompt.txt not found. Please create it in the same directory as this script.")
    sys.exit(1)
except Exception as e:
    logger.critical(f"CRITICAL: Error reading system_prompt.txt: {e}", exc_info=True)
    print(f"Error reading system_prompt.txt: {e}")
    sys.exit(1)

# Initialize AI Client
try:
    ai_client = AnthropicClient() # Uses settings from config.py
    logger.info("AnthropicClient initialized.")
except ValueError as e:
    logger.critical(f"CRITICAL: Error initializing AI Client: {e}. Check ANTHROPIC_API_KEY.", exc_info=True)
    print(f"Error initializing AI Client: {e}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"CRITICAL: Unexpected error initializing AI Client: {e}", exc_info=True)
    print(f"Unexpected error initializing AI Client: {e}")
    sys.exit(1)


# Initialize Tools
available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(),
    "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(),
}
logger.info(f"Available tools initialized: {list(available_tools.keys())}")

# Initialize Interrupt Handler
interrupt_handler = InterruptHandler()
logger.info("InterruptHandler initialized.")

# Conversation history
conversation_history = []
# Add initial system message to history if your AI model prefers it,
# or rely on the 'system' parameter in Anthropic's API.
# For Anthropic's messages API, the system prompt is separate.
# conversation_history.append({"role": "system", "content": SYSTEM_PROMPT}) # Not for Claude Messages API

# --- Helper Functions ---
def print_ai_message(message: str):
    logger.info(f"AI: {message[:200]}...") # Log snippet
    print(f"\n🤖 Assistant:\n{message}")

def print_user_message(message: str): # Used for logging user input
    logger.info(f"User: {message}")
    # print(f"\n👤 User: {message}") # Input already shows it

def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:300]}...") # Log snippet
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_message(message: str, level=logging.INFO):
    logger.log(level, f"System: {message}")
    # Only print to console for important system messages or errors not caught by specific prints
    if level >= logging.WARNING or "kali ai assistant started" in message.lower() or "exiting" in message.lower():
         print(f"\n⚙️ System:\n{message}")


def manage_conversation_history_and_summarize():
    """
    Checks token count and summarizes history if it exceeds soft limit.
    Returns True if summarization was attempted (successfully or not), False otherwise.
    """
    global conversation_history
    
    current_tokens = estimate_messages_token_count(conversation_history)
    logger.debug(f"Current estimated token count: {current_tokens}. Soft limit: {config.CONTEXT_TOKEN_SOFT_LIMIT}")

    if current_tokens > config.CONTEXT_TOKEN_SOFT_LIMIT and len(conversation_history) > config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:
        print_system_message(f"Context length ({current_tokens} tokens) nearing limit. Attempting summarization...", level=logging.INFO)
        
        # Decide how much to summarize. Keep the last few messages intact.
        messages_to_keep_suffix = conversation_history[-config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:]
        messages_to_summarize = conversation_history[:-config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY]

        if not messages_to_summarize:
            logger.info("Not enough messages to summarize after keeping suffix.")
            return False

        summary_text = ai_client.summarize_conversation(
            messages_to_summarize,
            config.SUMMARIZED_HISTORY_TARGET_TOKENS
        )

        if interrupt_handler.is_interrupted():
            print_system_message("Summarization interrupted by user.", level=logging.WARNING)
            return True # Attempted

        if summary_text:
            new_history = [{"role": "system", "content": f"Previous conversation summary: {summary_text}"}]
            new_history.extend(messages_to_keep_suffix)
            
            # Calculate potential token reduction
            old_tokens_summarized_part = estimate_messages_token_count(messages_to_summarize)
            new_tokens_summary_part = estimate_messages_token_count([new_history[0]])
            reduction = old_tokens_summarized_part - new_tokens_summary_part
            
            conversation_history = new_history
            new_total_tokens = estimate_messages_token_count(conversation_history)
            print_system_message(f"Conversation history summarized. Token count reduced by approx {reduction} to {new_total_tokens}.", level=logging.INFO)
            logger.info(f"History summarized. Old part tokens: {old_tokens_summarized_part}, New summary tokens: {new_tokens_summary_part}, New total: {new_total_tokens}")
            return True
        else:
            print_system_message("Failed to summarize conversation history. Proceeding with full history.", level=logging.WARNING)
            # Potentially add a warning if current_tokens > CONTEXT_TOKEN_HARD_LIMIT
            if current_tokens > config.CONTEXT_TOKEN_HARD_LIMIT:
                 print_system_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit ({config.CONTEXT_TOKEN_HARD_LIMIT}). AI may truncate.", level=logging.ERROR)
            return True # Attempted
    return False


def parse_tool_call(ai_response: str) -> tuple[str | None, dict | None]:
    try:
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        start_index = ai_response.find(start_tag)
        end_index = ai_response.find(end_tag)

        if start_index != -1 and end_index != -1:
            tool_call_json_str = ai_response[start_index + len(start_tag):end_index].strip()
            logger.debug(f"Raw tool call JSON string: {tool_call_json_str}")
            tool_call_data = json.loads(tool_call_json_str)
            
            tool_name = tool_call_data.get("tool_name")
            arguments = tool_call_data.get("arguments")

            if not tool_name or not isinstance(arguments, dict):
                logger.warning(f"AI tool call format error: Name or args missing. Data: {tool_call_data}")
                print_system_message("Error: AI tried to call a tool with invalid format (missing name or args).", level=logging.ERROR)
                return None, None
            logger.info(f"Parsed tool call: {tool_name}, Args: {arguments}")
            return tool_name, arguments
    except json.JSONDecodeError as e:
        logger.warning(f"JSONDecodeError parsing tool call: {e}. String was: '{tool_call_json_str}'")
    except Exception as e:
        logger.error(f"Error parsing tool call: {e}", exc_info=True)
    return None, None


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name in available_tools:
        tool = available_tools[tool_name]
        
        if tool_name == "command_line":
            command_to_run = arguments.get("command")
            # Confirmation for NEW commands. stdin_input or terminate_interactive don't need re-confirmation.
            if command_to_run and not arguments.get("stdin_input") and not arguments.get("terminate_interactive"):
                confirm_prompt = f"AI wants to execute: '{command_to_run}'. Allow? (yes/no): "
                try:
                    user_confirmation = input(confirm_prompt).strip().lower()
                    if user_confirmation != "yes":
                        logger.info(f"User declined execution of command: {command_to_run}")
                        return "User declined command execution."
                    logger.info(f"User approved execution of command: {command_to_run}")
                except EOFError:
                    logger.warning("EOF received during command confirmation.")
                    return "User declined command execution (EOF)."
                except KeyboardInterrupt:
                    interrupt_handler.handle_interrupt(None, None)
                    logger.warning("KeyboardInterrupt during command confirmation.")
                    return "User interrupted command confirmation."

        tool.set_interrupted(interrupt_handler.is_interrupted())
        try:
            return tool.execute(arguments)
        except Exception as e:
            logger.error(f"Exception during tool execution '{tool_name}': {str(e)}", exc_info=True)
            return f"Error during tool execution '{tool_name}': {str(e)}"
    else:
        logger.error(f"Tool '{tool_name}' not found.")
        return f"Error: Tool '{tool_name}' not found."

# --- Main Application Loop ---
def main():
    global conversation_history

    print_system_message(f"{config.SERVICE_NAME} started. Type 'exit' or 'quit' to end.", level=logging.INFO)
    logger.info(f"Application main loop started. Model: {config.DEFAULT_AI_MODEL}")
    
    while True:
        interrupt_handler.reset()
        ai_client.set_interrupted(False) # Reset AI client interruption status too
        for tool in available_tools.values(): # Reset tool interruption status
            tool.set_interrupted(False)

        try:
            user_input = input("\n👤 You: ").strip()
        except KeyboardInterrupt:
            if interrupt_handler.is_interrupted(): # Second Ctrl+C
                print_system_message("Exiting due to repeated Ctrl+C.", level=logging.INFO)
                break
            interrupt_handler.handle_interrupt(None, None) # Sets flag
            print_system_message("Input interrupted. Type 'exit' or press Ctrl+C again to quit.", level=logging.WARNING)
            continue
        except EOFError:
            print_system_message("EOF received. Exiting.", level=logging.INFO)
            break
        
        if interrupt_handler.is_interrupted(): # Check after input, before processing
            print_system_message("Operation cancelled by user after input.", level=logging.WARNING)
            continue

        if user_input.lower() in ["exit", "quit"]:
            print_system_message(f"Exiting {config.SERVICE_NAME}.", level=logging.INFO)
            break
        
        if not user_input:
            continue

        print_user_message(user_input) # Log the user input
        conversation_history.append({"role": "user", "content": user_input})

        # Manage context length BEFORE calling AI
        manage_conversation_history_and_summarize()
        if interrupt_handler.is_interrupted(): # Summarization could be interrupted
            print_system_message("Operation interrupted during context management.", level=logging.WARNING)
            if conversation_history and conversation_history[-1]["role"] == "user": # Clean up last user message
                conversation_history.pop()
            continue
        
        # Get response from AI
        ai_client.set_interrupted(interrupt_handler.is_interrupted())
        ai_response_text = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

        if interrupt_handler.is_interrupted():
            print_system_message("AI response generation interrupted.", level=logging.WARNING)
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop() # Remove user message that AI didn't get to process
            continue
        
        if not ai_response_text:
            print_system_message("AI did not provide a response. Please try again.", level=logging.ERROR)
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop()
            continue

        conversation_history.append({"role": "assistant", "content": ai_response_text})
        
        tool_name, tool_args = parse_tool_call(ai_response_text)

        # Loop for handling multiple tool calls or follow-ups from a single AI response (if designed for it)
        # For now, simple single tool call per AI response.
        if tool_name and tool_args:
            # The AI's textual response might have preamble before the tool call. Print it.
            # A more sophisticated approach would be to extract text before/after tool_call tags.
            # For now, the system prompt asks AI to ONLY output tool_call if using a tool.
            # If there's text AND a tool call, we might need to decide how to display it.
            # Let's assume for now if tool_call is present, the text in ai_response_text is primarily for the tool.
            # A better way: if system_prompt says "ONLY <tool_call>", then ai_response_text *is* the call.
            # If AI can talk *and* call tool, then parse_tool_call should return the surrounding text too.
            # Current system_prompt: "Do NOT add any text outside these tags if you are making a tool call."
            # So, if tool_name is found, ai_response_text *is* the tool call.
            # We should print a more generic "AI is using a tool" message.
            print_ai_message(f"Okay, I will use the '{tool_name}' tool.") # More natural than printing the JSON
            
            tool_output = execute_tool(tool_name, tool_args)
            print_tool_output(tool_name, tool_output)
            
            if interrupt_handler.is_interrupted():
                print_system_message(f"Tool '{tool_name}' execution or follow-up interrupted.", level=logging.WARNING)
                # Add observation about interruption for AI context
                observation_content = f"Observation: Tool {tool_name} execution was interrupted by the user. Output so far: {tool_output if tool_output else 'None'}"
                conversation_history.append({"role": "user", "content": observation_content}) # Represent tool output as user message
            else:
                observation_content = f"Observation: {tool_output}"
                conversation_history.append({"role": "user", "content": observation_content})

            # Manage context again AFTER tool use and before asking AI for follow-up
            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted():
                print_system_message("Operation interrupted during post-tool context management.", level=logging.WARNING)
                continue

            ai_client.set_interrupted(interrupt_handler.is_interrupted())
            final_ai_response = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

            if interrupt_handler.is_interrupted():
                print_system_message("AI response generation after tool use was interrupted.", level=logging.WARNING)
                continue

            if final_ai_response:
                print_ai_message(final_ai_response)
                conversation_history.append({"role": "assistant", "content": final_ai_response})
            else:
                print_system_message("AI did not provide a follow-up response after tool execution.", level=logging.ERROR)
                # Decide if to pop the observation or keep it for next user turn
        else:
            # No tool call, just a regular text response from AI
            print_ai_message(ai_response_text)

if __name__ == "__main__":
    try:
        main()
    except SystemExit: # Allow sys.exit() to terminate cleanly
        pass
    except Exception as e:
        logger.critical(f"--- An unexpected critical error occurred in main: {e} ---", exc_info=True)
        print(f"\n--- An unexpected critical error occurred: {e} ---")
    finally:
        logger.info("Application terminated.")
        print("\nApplication terminated.")
        # Restore original SIGINT handler (already handled in InterruptHandler.__del__)
