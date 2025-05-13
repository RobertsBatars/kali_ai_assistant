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

# --- Initial Setup (load prompts, AI client, tools, etc.) ---
try:
    with open("system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
    logger.info("System prompt loaded successfully.")
except FileNotFoundError:
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
# --- Helper Functions (print_*, manage_conversation_history, execute_tool) ---
# These functions (print_ai_message, print_user_message_log, print_tool_being_used,
# print_tool_output, print_system_console_message, manage_conversation_history_and_summarize,
# execute_tool) remain the same as in your last provided version.
# For brevity, I'm omitting them here but they should be included.

def print_ai_message(message: str):
    logger.info(f"AI: {message[:1000]}{'...' if len(message) > 1000 else ''}")
    print(f"\n🤖 Assistant:\n{message}")

def print_user_message_log(message: str):
    logger.info(f"User: {message}")

def print_tool_being_used(tool_name: str, tool_args: dict):
    args_str = json.dumps(tool_args)
    if len(args_str) > 100: args_str = args_str[:100] + "..."
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
    global conversation_history
    current_tokens = estimate_messages_token_count(conversation_history)
    logger.debug(f"Current estimated token count: {current_tokens}. Soft limit: {config.CONTEXT_TOKEN_SOFT_LIMIT}")
    if current_tokens > config.CONTEXT_TOKEN_SOFT_LIMIT and len(conversation_history) > config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:
        print_system_console_message(f"Context length ({current_tokens} tokens) nearing limit. Attempting summarization...")
        # ... (rest of summarization logic from previous version)
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
            # ... (token calculation and logging)
            conversation_history = new_history
            print_system_console_message("Conversation history summarized.")
            return True
        else:
            print_system_console_message("Failed to summarize conversation history.", is_error=True)
            # ... (hard limit warning)
            return True
    return False

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
                        return "User declined command execution."
                except (EOFError, KeyboardInterrupt): # Simplified handling
                    interrupt_handler.handle_interrupt(None, None)
                    return "User interrupted command confirmation."
        tool.set_interrupted(interrupt_handler.is_interrupted())
        return tool.execute(arguments)
    return f"Error: Tool '{tool_name}' not found."

def parse_ai_response_for_actions(ai_response: str) -> list:
    """
    Parses the AI's response into a sequence of actions (text or tool calls).
    Returns a list of tuples: e.g., [("text", "Some text"), ("tool", (tool_name, args)), ...].
    """
    actions = []
    if not ai_response:
        return actions
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
                actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>"))
        except json.JSONDecodeError:
            logger.warning(f"JSONDecodeError for tool call content: {tool_json_str}")
            actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>"))
        last_end = match.end()
    remaining_text = ai_response[last_end:].strip()
    if remaining_text:
        actions.append(("text", remaining_text))
    if not actions and ai_response: # No tool calls, whole response is text
        actions.append(("text", ai_response.strip()))
    return actions

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

        # --- Iterative AI Turn Processing ---
        # This loop continues as long as the AI's response leads to tool executions.
        # If the AI responds with only text, this inner loop completes one pass and breaks.
        processing_ai_turn = True
        while processing_ai_turn:
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted before getting AI response.")
                processing_ai_turn = False # Break from this AI turn, go to next user input
                continue

            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted():
                print_system_console_message("Operation interrupted during context management.")
                processing_ai_turn = False
                continue

            ai_client.set_interrupted(interrupt_handler.is_interrupted())
            ai_response_text, stop_reason = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

            if interrupt_handler.is_interrupted() or stop_reason == "interrupted":
                print_system_console_message("AI response generation interrupted.")
                processing_ai_turn = False
                continue
            
            if not ai_response_text:
                print_system_console_message("AI did not provide a response. Please try again.", is_error=True)
                if conversation_history and conversation_history[-1]["role"] == "user" and \
                   not conversation_history[-1]["content"].startswith("Observation:"):
                     conversation_history.pop() # Pop user query if AI failed to respond to it
                processing_ai_turn = False
                continue

            conversation_history.append({"role": "assistant", "content": ai_response_text})
            
            actions = parse_ai_response_for_actions(ai_response_text)
            logger.debug(f"Parsed actions from AI response: {actions}")

            executed_tools_this_pass = False
            all_tool_outputs_for_this_pass = []

            for action_type, action_content in actions:
                if interrupt_handler.is_interrupted():
                    print_system_console_message("Processing of AI actions interrupted by user.")
                    break # Break from processing actions for this AI response

                if action_type == "text":
                    print_ai_message(action_content)
                
                elif action_type == "tool":
                    tool_name, tool_args = action_content
                    print_tool_being_used(tool_name, tool_args)
                    
                    tool_output_str = execute_tool(tool_name, tool_args)

                    if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                        print_system_console_message("Command confirmation was interrupted by user.")
                        all_tool_outputs_for_this_pass.append(f"Observation for {tool_name}: User interrupted command confirmation.")
                        # Stop further actions from *this specific* AI response, let AI react to this interruption.
                        executed_tools_this_pass = True # Mark that an interaction (even if interrupted) happened
                        break 
                    
                    print_tool_output(tool_name, tool_output_str)
                    all_tool_outputs_for_this_pass.append(f"Observation for tool '{tool_name}':\n{tool_output_str}")
                    executed_tools_this_pass = True
                    
                    if "Command interrupted." in tool_output_str and interrupt_handler.is_interrupted():
                        print_system_console_message(f"Tool '{tool_name}' execution was interrupted by user.")
                        # The observation already contains this. Stop further actions.
                        break 

            if interrupt_handler.is_interrupted(): # If action loop was broken by interrupt
                processing_ai_turn = False # Stop this AI turn, go to next user input
                # Add any collected observations before breaking
                if all_tool_outputs_for_this_pass:
                    combined_observations = "\n\n".join(all_tool_outputs_for_this_pass)
                    conversation_history.append({"role": "user", "content": combined_observations})
                continue # To the outer user input loop

            if executed_tools_this_pass:
                # Tools were run (or attempted), so we need to send observations and let AI continue.
                if all_tool_outputs_for_this_pass:
                    combined_observations = "\n\n".join(all_tool_outputs_for_this_pass)
                    conversation_history.append({"role": "user", "content": combined_observations})
                # Loop back in the `while processing_ai_turn:` to get AI's response to these observations.
                # The `processing_ai_turn` flag remains True.
            else:
                # No tools were executed in this pass (AI's response was text-only or invalid tools).
                # AI's current "thought" is complete for this user input.
                processing_ai_turn = False 

            if stop_reason == "max_tokens":
                print_system_console_message("Warning: AI's response was cut short due to maximum token limit.", is_error=True)
                conversation_history.append({
                    "role": "user",
                    "content": "Observation: Your previous response was truncated because it reached the maximum token limit. Please continue your thought process or provide the next single step if you were in the middle of a multi-step plan."
                })
                if not executed_tools_this_pass: # If truncation happened on a text-only response
                    processing_ai_turn = True # Force another pass to let AI continue
                # If tools were executed, the loop will continue anyway due to executed_tools_this_pass.

        # End of `while processing_ai_turn` loop (inner loop)
    # End of `while True` (outer loop for user input)

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
