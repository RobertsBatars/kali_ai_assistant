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
from utils.logger_setup import setup_logging
from utils.token_estimator import estimate_messages_token_count

logger = setup_logging(
    log_file_path=config.LOG_FILE_PATH,
    log_level_file=config.LOG_LEVEL_FILE,
    log_level_console=config.LOG_LEVEL_CONSOLE, # Respects config's default of WARNING
    service_name=config.SERVICE_NAME
)

try:
    with open("system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    logger.info("System prompt loaded successfully.")
except FileNotFoundError:
    # Use print for critical startup errors as logger console might be WARNING
    print("CRITICAL: system_prompt.txt not found. Please create it.", file=sys.stderr)
    logger.critical("CRITICAL: system_prompt.txt not found.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Error reading system_prompt.txt: {e}", file=sys.stderr)
    logger.critical(f"CRITICAL: Error reading system_prompt.txt: {e}", exc_info=True)
    sys.exit(1)

try:
    ai_client = AnthropicClient()
    logger.info("AnthropicClient initialized.")
except ValueError as e:
    print(f"CRITICAL: Error initializing AI Client: {e}. Check ANTHROPIC_API_KEY.", file=sys.stderr)
    logger.critical(f"CRITICAL: Error initializing AI Client: {e}.", exc_info=True)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Unexpected error initializing AI Client: {e}", file=sys.stderr)
    logger.critical(f"CRITICAL: Unexpected error initializing AI Client: {e}", exc_info=True)
    sys.exit(1)

available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(),
    "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(),
}
logger.info(f"Available tools initialized: {list(available_tools.keys())}")

interrupt_handler = InterruptHandler()
logger.info("InterruptHandler initialized.")

conversation_history = []

# --- Helper Functions ---
def print_ai_message(message: str):
    """Prints messages from the AI. Logs the full message, prints to console."""
    logger.info(f"AI: {message[:1000]}{'...' if len(message) > 1000 else ''}") # Log snippet or full if short
    print(f"\n🤖 Assistant:\n{message}")

def print_user_message_log(message: str): # For logging user input
    logger.info(f"User: {message}")

def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:1000]}{'...' if len(output) > 1000 else ''}")
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_console_message(message: str):
    """Prints important system messages directly to console AND logs them as INFO."""
    # This is for user-facing system status like "started", "exiting".
    logger.info(f"SystemConsole: {message}")
    print(f"\n⚙️ System:\n{message}")


def manage_conversation_history_and_summarize():
    global conversation_history
    current_tokens = estimate_messages_token_count(conversation_history)
    logger.debug(f"Current estimated token count: {current_tokens}. Soft limit: {config.CONTEXT_TOKEN_SOFT_LIMIT}")

    if current_tokens > config.CONTEXT_TOKEN_SOFT_LIMIT and len(conversation_history) > config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:
        print_system_console_message(f"Context length ({current_tokens} tokens) nearing limit. Attempting summarization...")
        
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
            print_system_console_message("Summarization interrupted by user.")
            return True 

        if summary_text:
            new_history = [{"role": "system", "content": f"Previous conversation summary: {summary_text}"}]
            new_history.extend(messages_to_keep_suffix)
            
            old_tokens_summarized_part = estimate_messages_token_count(messages_to_summarize)
            new_tokens_summary_part = estimate_messages_token_count([new_history[0]])
            reduction = old_tokens_summarized_part - new_tokens_summary_part
            
            conversation_history = new_history
            new_total_tokens = estimate_messages_token_count(conversation_history)
            print_system_console_message(f"Conversation history summarized. Token count reduced by approx {reduction} to {new_total_tokens}.")
            return True
        else:
            # Log this as a warning, but also inform user if it's critical (e.g. over hard limit)
            logger.warning("Failed to summarize conversation history. Proceeding with full history.")
            print_system_console_message("Failed to summarize conversation history.")
            if current_tokens > config.CONTEXT_TOKEN_HARD_LIMIT:
                 logger.error(f"CRITICAL: Token count ({current_tokens}) exceeds hard limit ({config.CONTEXT_TOKEN_HARD_LIMIT}). AI may truncate.")
                 print_system_console_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit. AI may truncate or fail.")
            return True 
    return False


def parse_tool_call(ai_response: str) -> tuple[str | None, dict | None, str | None]:
    """
    Parses the AI's response to find a tool call and any text before it.
    Returns: (tool_name, arguments, text_before_tool_call)
    """
    text_before_tool_call = None
    try:
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        start_index = ai_response.find(start_tag)
        end_index = ai_response.find(end_tag)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            if start_index > 0:
                text_before_tool_call = ai_response[:start_index].strip()
            
            tool_call_json_str = ai_response[start_index + len(start_tag):end_index].strip()
            logger.debug(f"Raw tool call JSON string: {tool_call_json_str}")
            tool_call_data = json.loads(tool_call_json_str)
            
            tool_name = tool_call_data.get("tool_name")
            arguments = tool_call_data.get("arguments")

            if not tool_name or not isinstance(arguments, dict):
                logger.warning(f"AI tool call format error: Name or args missing. Data: {tool_call_data}")
                print_system_console_message("Error: AI tried to call a tool with invalid format.")
                return None, None, text_before_tool_call # Return text even if tool call is bad
            logger.info(f"Parsed tool call: {tool_name}, Args: {arguments}. Text before: '{text_before_tool_call}'")
            return tool_name, arguments, text_before_tool_call
    except json.JSONDecodeError as e:
        logger.warning(f"JSONDecodeError parsing tool call: {e}. String was: '{tool_call_json_str if 'tool_call_json_str' in locals() else 'N/A'}'")
    except Exception as e:
        logger.error(f"Error parsing tool call: {e}", exc_info=True)
    
    # If no valid tool call found, the whole response is considered text_before_tool_call (or just text)
    return None, None, ai_response.strip() if not text_before_tool_call else text_before_tool_call


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name in available_tools:
        tool = available_tools[tool_name]
        
        if tool_name == "command_line" and config.REQUIRE_COMMAND_CONFIRMATION:
            command_to_run = arguments.get("command")
            # Only ask for confirmation for NEW commands.
            if command_to_run and not arguments.get("stdin_input") and not arguments.get("terminate_interactive"):
                confirm_prompt = f"AI wants to execute: '{command_to_run}'. Allow? (yes/no): "
                try:
                    user_confirmation = input(confirm_prompt).strip().lower()
                    if user_confirmation != "yes":
                        logger.info(f"User declined execution of command: {command_to_run}")
                        return "User declined command execution."
                    logger.info(f"User approved execution of command: {command_to_run}")
                except EOFError: # Handle Ctrl+D during input
                    logger.warning("EOF received during command confirmation.")
                    return "User declined command execution (EOF)."
                except KeyboardInterrupt: # Handle Ctrl+C during input
                    interrupt_handler.handle_interrupt(None, None) # Trigger our handler
                    logger.warning("KeyboardInterrupt during command confirmation.")
                    # This state will be caught by the main loop's interrupt check
                    return "User interrupted command confirmation." 
                    # Or, raise an exception specific to interruption if preferred

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

    print_system_console_message(f"{config.SERVICE_NAME} started. Type 'exit' or 'quit' to end.")
    logger.info(f"Application main loop started. Model: {config.DEFAULT_AI_MODEL}")
    
    while True:
        interrupt_handler.reset()
        ai_client.set_interrupted(False)
        for tool in available_tools.values():
            tool.set_interrupted(False)

        try:
            user_input = input("\n👤 You: ").strip()
        except KeyboardInterrupt:
            if interrupt_handler.is_interrupted():
                print_system_console_message("Exiting due to repeated Ctrl+C.")
                break
            interrupt_handler.handle_interrupt(None, None)
            print_system_console_message("Input interrupted. Type 'exit' or press Ctrl+C again to quit.")
            continue
        except EOFError:
            print_system_console_message("EOF received. Exiting.")
            break
        
        if interrupt_handler.is_interrupted():
            print_system_console_message("Operation cancelled by user after input.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print_system_console_message(f"Exiting {config.SERVICE_NAME}.")
            break
        
        if not user_input:
            continue

        print_user_message_log(user_input)
        conversation_history.append({"role": "user", "content": user_input})

        manage_conversation_history_and_summarize()
        if interrupt_handler.is_interrupted():
            print_system_console_message("Operation interrupted during context management.")
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
            continue
        
        ai_client.set_interrupted(interrupt_handler.is_interrupted())
        ai_response_text = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

        if interrupt_handler.is_interrupted():
            print_system_console_message("AI response generation interrupted.")
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop()
            continue
        
        if not ai_response_text:
            print_system_console_message("AI did not provide a response. Please try again.")
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop()
            continue

        conversation_history.append({"role": "assistant", "content": ai_response_text})
        
        tool_name, tool_args, text_before_tool = parse_tool_call(ai_response_text)

        if text_before_tool: # If there's any text portion from AI (even if tool call is bad/missing)
            print_ai_message(text_before_tool)

        if tool_name and tool_args:
            # If text_before_tool was already printed, we might just add a small note about tool usage here,
            # or rely on the confirmation prompt (if active) to indicate tool usage.
            # For now, the confirmation prompt itself is a good indicator.
            # If no text_before_tool, and AI only sent tool call, this means print_ai_message wasn't called yet.
            if not text_before_tool: # AI only sent tool call
                 print_ai_message(f"Understood. Requesting to use tool: '{tool_name}'.")


            tool_output = execute_tool(tool_name, tool_args)
            
            if interrupt_handler.is_interrupted() and "User interrupted command confirmation" in tool_output:
                print_system_console_message("Command confirmation was interrupted.")
                # Don't proceed to AI with this specific message, let user try again.
                # Remove AI's last message (the tool call request) and user's "observation" of interruption.
                if conversation_history and conversation_history[-1]["role"] == "assistant": conversation_history.pop()
                # The user's input that led to this is still there.
                continue


            print_tool_output(tool_name, tool_output)
            
            observation_content = f"Observation: {tool_output}"
            if interrupt_handler.is_interrupted(): # General interruption during tool exec
                print_system_console_message(f"Tool '{tool_name}' execution or follow-up interrupted.")
                observation_content = f"Observation: Tool {tool_name} execution was interrupted by the user. Output so far: {tool_output if tool_output else 'None'}"
            
            conversation_history.append({"role": "user", "content": observation_content})

            manage_conversation_history_and_summarize() # After tool use
            if interrupt_handler.is_interrupted():
                print_system_console_message("Operation interrupted during post-tool context management.")
                continue

            ai_client.set_interrupted(interrupt_handler.is_interrupted())
            final_ai_response = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

            if interrupt_handler.is_interrupted():
                print_system_console_message("AI response generation after tool use was interrupted.")
                continue

            if final_ai_response:
                print_ai_message(final_ai_response)
                conversation_history.append({"role": "assistant", "content": final_ai_response})
            else:
                print_system_console_message("AI did not provide a follow-up response after tool execution.")
        
        # If no tool_name/tool_args, text_before_tool is the full AI response, already printed if it existed.

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass # Allow clean exit
    except Exception as e:
        logger.critical(f"--- An unexpected critical error occurred in main: {e} ---", exc_info=True)
        # Also print to stderr for visibility if console logging is off
        print(f"\n--- CRITICAL ERROR: {e} ---", file=sys.stderr)
    finally:
        logger.info("Application terminated.")
        # print_system_console_message("Application terminated.") # Avoid if already exited cleanly
        if not (sys.exc_info()[0] is None or sys.exc_info()[0] is SystemExit):
             print("\nApplication terminated due to an error.", file=sys.stderr)
        else:
             print("\nApplication terminated.")

