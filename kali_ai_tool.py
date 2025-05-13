# kali_ai_tool.py
import json
import readline
import sys
import logging
import time
import re # For parsing multiple tool calls

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
    log_level_console=config.LOG_LEVEL_CONSOLE,
    service_name=config.SERVICE_NAME
)

try:
    with open("system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    logger.info("System prompt loaded successfully.")
except FileNotFoundError:
    print("CRITICAL: system_prompt.txt not found. Please create it.", file=sys.stderr)
    logger.critical("CRITICAL: system_prompt.txt not found.")
    sys.exit(1)
# ... (rest of the initial setup: AI client, tools, interrupt handler, history) ...
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


def print_ai_message(message: str):
    logger.info(f"AI: {message[:1000]}{'...' if len(message) > 1000 else ''}")
    print(f"\n🤖 Assistant:\n{message}")

def print_user_message_log(message: str):
    logger.info(f"User: {message}")

def print_tool_being_used(tool_name: str, tool_args: dict):
    """Prints a notification that a specific tool is about to be used."""
    args_str = json.dumps(tool_args)
    # Truncate long args for display
    if len(args_str) > 100:
        args_str = args_str[:100] + "..."
    message = f"Attempting to use tool: '{tool_name}' with arguments: {args_str}"
    logger.info(message)
    print(f"\n⚙️ System: {message}")


def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:1000]}{'...' if len(output) > 1000 else ''}")
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_console_message(message: str, is_error=False):
    log_level = logging.ERROR if is_error else logging.INFO
    logger.log(log_level, f"SystemConsole: {message}")
    print(f"\n⚙️ System:\n{message}")

def manage_conversation_history_and_summarize():
    # (This function remains the same as the previous version)
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

        summary_text = ai_client.summarize_conversation(messages_to_summarize, config.SUMMARIZED_HISTORY_TARGET_TOKENS)

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
            logger.warning("Failed to summarize conversation history. Proceeding with full history.")
            print_system_console_message("Failed to summarize conversation history.")
            if current_tokens > config.CONTEXT_TOKEN_HARD_LIMIT:
                 logger.error(f"CRITICAL: Token count ({current_tokens}) exceeds hard limit ({config.CONTEXT_TOKEN_HARD_LIMIT}). AI may truncate.")
                 print_system_console_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit. AI may truncate or fail.", is_error=True)
            return True
    return False


def parse_ai_response_for_actions(ai_response: str) -> list:
    """
    Parses the AI's response into a sequence of actions (text or tool calls).
    Returns a list of tuples: e.g., [("text", "Some text"), ("tool", (tool_name, args)), ...].
    """
    actions = []
    if not ai_response:
        return actions

    # Regex to find <tool_call>...</tool_call> blocks and capture them along with preceding text.
    # This regex will find all tool calls.
    # It captures:
    # 1. Text before a tool call (optional)
    # 2. The content of the tool call (the JSON string)
    pattern = re.compile(r"(.*?)<tool_call>(.*?)</tool_call>", re.DOTALL)
    
    last_end = 0
    for match in pattern.finditer(ai_response):
        pre_text = match.group(1).strip()
        tool_json_str = match.group(2).strip()
        
        if pre_text:
            actions.append(("text", pre_text))
        
        try:
            tool_data = json.loads(tool_json_str)
            tool_name = tool_data.get("tool_name")
            arguments = tool_data.get("arguments")
            if tool_name and isinstance(arguments, dict):
                actions.append(("tool", (tool_name, arguments)))
                logger.info(f"Parsed tool action: {tool_name}, Args: {arguments}")
            else:
                logger.warning(f"Malformed tool JSON content: {tool_json_str}")
                actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>")) # Add malformed call as text
        except json.JSONDecodeError:
            logger.warning(f"JSONDecodeError for tool call content: {tool_json_str}")
            actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>")) # Add unparsable call as text
            
        last_end = match.end()
        
    # Add any remaining text after the last tool call
    remaining_text = ai_response[last_end:].strip()
    if remaining_text:
        actions.append(("text", remaining_text))
        
    # If no tool calls were found at all, the whole response is a single text action
    if not actions and ai_response:
        actions.append(("text", ai_response.strip()))
        
    return actions


def execute_tool(tool_name: str, arguments: dict) -> str:
    # (This function remains the same as the previous version)
    if tool_name in available_tools:
        tool = available_tools[tool_name]
        
        if tool_name == "command_line" and config.REQUIRE_COMMAND_CONFIRMATION:
            command_to_run = arguments.get("command")
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
    print_system_console_message(f"{config.SERVICE_NAME} started. Type 'exit' or 'quit' to end.")
    logger.info(f"Application main loop started. Model: {config.DEFAULT_AI_MODEL}, Max Output Tokens: {config.MAX_AI_OUTPUT_TOKENS}")
    
    while True: # Outer loop for user input
        interrupt_handler.reset()
        ai_client.set_interrupted(False)
        for tool in available_tools.values(): tool.set_interrupted(False)

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
        
        if user_input.lower() in ["exit", "quit"]:
            print_system_console_message(f"Exiting {config.SERVICE_NAME}.")
            break
        if not user_input: continue

        print_user_message_log(user_input)
        conversation_history.append({"role": "user", "content": user_input})

        # --- AI Turn Processing ---
        # This inner loop allows the AI to respond, potentially make tool calls,
        # and then respond again to the tool outputs, all before returning to the user for new input,
        # IF the AI structures its response to require immediate follow-up after an observation.
        # However, the primary design is: User -> AI -> Tools (if any) -> Observations -> AI -> User
        
        # We expect one consolidated AI response first.
        # Then we process actions from it.
        # Then we send all observations and get AI's final response for this "turn".

        # Step 1: Get initial AI response
        manage_conversation_history_and_summarize()
        if interrupt_handler.is_interrupted():
            print_system_console_message("Operation interrupted during context management.")
            continue # To next user input

        ai_client.set_interrupted(interrupt_handler.is_interrupted())
        ai_response_text, stop_reason = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

        if interrupt_handler.is_interrupted() or stop_reason == "interrupted":
            print_system_console_message("AI response generation interrupted.")
            continue
        
        if not ai_response_text:
            print_system_console_message("AI did not provide a response. Please try again.", is_error=True)
            if conversation_history and conversation_history[-1]["role"] == "user" and not conversation_history[-1]["content"].startswith("Observation:"):
                 conversation_history.pop()
            continue

        conversation_history.append({"role": "assistant", "content": ai_response_text})
        
        actions = parse_ai_response_for_actions(ai_response_text)
        logger.debug(f"Parsed actions from AI response: {actions}")

        all_tool_outputs_for_this_turn = []
        ai_had_further_text_after_tools = False

        for action_type, action_content in actions:
            if interrupt_handler.is_interrupted():
                print_system_console_message("Processing of AI actions interrupted.")
                break # Break from processing actions for this AI response

            if action_type == "text":
                print_ai_message(action_content)
                ai_had_further_text_after_tools = True # If text appears after any tool call was processed
            
            elif action_type == "tool":
                tool_name, tool_args = action_content
                print_tool_being_used(tool_name, tool_args) # Notify user
                
                tool_output_str = execute_tool(tool_name, tool_args)

                if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                    print_system_console_message("Command confirmation was interrupted by user.")
                    # Observation about this specific interruption
                    all_tool_outputs_for_this_turn.append(f"Observation for {tool_name}: User interrupted command confirmation.")
                    # We should probably stop processing further actions from this AI response
                    # and let the AI react to the confirmation interruption.
                    break # Stop processing more actions from this AI response
                
                print_tool_output(tool_name, tool_output_str)
                all_tool_outputs_for_this_turn.append(f"Observation for tool '{tool_name}':\n{tool_output_str}")
                
                if "Command interrupted." in tool_output_str and interrupt_handler.is_interrupted():
                    print_system_console_message(f"Tool '{tool_name}' execution was interrupted by user.")
                    # The observation already contains this info.
                    break # Stop processing more actions

        if interrupt_handler.is_interrupted(): # If loop above was broken by interrupt
            # Add collected outputs so far as observations for AI context if any
            if all_tool_outputs_for_this_turn:
                combined_observations = "\n\n".join(all_tool_outputs_for_this_turn)
                conversation_history.append({"role": "user", "content": combined_observations})
            print_system_console_message("Sequence interrupted. Awaiting next user input or AI recovery.")
            continue # To next user input outer loop

        # After processing all actions from the AI's response (text and tools)
        if all_tool_outputs_for_this_turn:
            combined_observations = "\n\n".join(all_tool_outputs_for_this_turn)
            conversation_history.append({"role": "user", "content": combined_observations})

            # Now, get AI's response to these observations
            manage_conversation_history_and_summarize() # Manage context before this follow-up
            if interrupt_handler.is_interrupted():
                print_system_console_message("Interrupted before AI follow-up to tool observations.")
                continue

            ai_client.set_interrupted(interrupt_handler.is_interrupted())
            follow_up_response_text, follow_up_stop_reason = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

            if interrupt_handler.is_interrupted() or follow_up_stop_reason == "interrupted":
                print_system_console_message("AI follow-up response generation interrupted.")
                continue
            
            if follow_up_response_text:
                conversation_history.append({"role": "assistant", "content": follow_up_response_text})
                # This follow-up response could itself contain more text/tool calls.
                # For simplicity now, we'll just print it. A more complex system might re-enter action parsing.
                # The current structure implies this follow-up is the AI's final thought for this "turn".
                actions_from_follow_up = parse_ai_response_for_actions(follow_up_response_text)
                for fu_action_type, fu_action_content in actions_from_follow_up:
                    if fu_action_type == "text":
                        print_ai_message(fu_action_content)
                    elif fu_action_type == "tool":
                        # AI is trying to call a tool immediately after observations.
                        # This makes the loop complex. For now, let's just print a warning
                        # and the user would need to prompt again if they want that tool executed.
                        # Or, we could re-enter the action processing loop.
                        # For now, treat this as AI's textual plan for the *next* turn.
                        print_ai_message(f"(AI is planning to use tool: {fu_action_content[0]} with args: {fu_action_content[1]})")
                        logger.warning(f"AI attempted a tool call in immediate follow-up: {fu_action_content}. This is treated as text for now.")
                
                if follow_up_stop_reason == "max_tokens":
                     print_system_console_message("Warning: AI's follow-up response was cut short due to maximum token limit.", is_error=True)
                     conversation_history.append({"role": "user", "content": "Observation: Your previous follow-up response was truncated."})


            else: # AI failed to provide follow-up
                print_system_console_message("AI did not provide a follow-up response after tool observations.", is_error=True)
                # Remove the combined observations as AI didn't process them.
                if conversation_history and conversation_history[-1]["role"] == "user" and conversation_history[-1]["content"] == combined_observations:
                    conversation_history.pop()
        
        elif stop_reason == "max_tokens" and not actions: # AI response was empty and hit max_tokens (unlikely) or only preamble hit max_tokens
            print_system_console_message("Warning: AI's response was cut short. It may try to continue.", is_error=True)
            conversation_history.append({"role": "user", "content": "Observation: Your previous response was truncated. Please continue."})
            # This will loop back to get AI's next response in the main loop.
            # No, this needs to be handled by re-prompting the AI immediately.
            # This case is tricky. If actions list is empty but stop_reason is max_tokens,
            # it means the AI's *entire initial response* was truncated.
            # The current logic adds the truncated response to history. The next user prompt will continue.
            # Or, we can add an immediate re-prompt here.
            # For now, the outer loop structure will handle this by going to next user input,
            # and AI will see its truncated response in history.

        # If no tool outputs, and AI's initial response was just text (already printed if `actions` had text),
        # then the turn is complete.

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        logger.critical(f"--- An unexpected critical error occurred in main: {e} ---", exc_info=True)
        print(f"\n--- CRITICAL ERROR: {e} ---", file=sys.stderr)
    finally:
        logger.info("Application terminated.")
        current_exception = sys.exc_info()
        if not (current_exception[0] is None or current_exception[0] is SystemExit):
             print("\nApplication terminated due to an error.", file=sys.stderr)
        else:
             print("\nApplication terminated.")