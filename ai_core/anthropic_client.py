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

# --- Helper Functions (print_*, manage_conversation_history, execute_tool, parse_ai_response_for_actions) ---
# These functions are assumed to be the same as in the previous version you provided.
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
            reduction = old_tokens_summarized_part - new_tokens_summary_part # noqa: F841
            conversation_history = new_history
            new_total_tokens = estimate_messages_token_count(conversation_history) # noqa: F841
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

def execute_tool(tool_name: str, arguments: dict) -> str:
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

def parse_ai_response_for_actions(ai_response: str) -> list:
    actions = []
    if not ai_response: return actions
    pattern = re.compile(r"(.*?)<tool_call>(.*?)</tool_call>", re.DOTALL)
    last_end = 0
    for match in pattern.finditer(ai_response):
        pre_text = match.group(1).strip()
        tool_json_str = match.group(2).strip()
        if pre_text: actions.append(("text", pre_text))
        try:
            tool_data = json.loads(tool_json_str)
            tool_name, arguments = tool_data.get("tool_name"), tool_data.get("arguments")
            if tool_name and isinstance(arguments, dict):
                actions.append(("tool", (tool_name, arguments)))
                logger.info(f"Parsed tool action: {tool_name}, Args: {arguments}")
            else:
                actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>")) # Malformed
        except json.JSONDecodeError:
            actions.append(("text", f"<tool_call>{tool_json_str}</tool_call>")) # Unparsable
        last_end = match.end()
    remaining_text = ai_response[last_end:].strip()
    if remaining_text: actions.append(("text", remaining_text))
    if not actions and ai_response: actions.append(("text", ai_response.strip()))
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
        except KeyboardInterrupt: # Ctrl+C at "You:" prompt
            if interrupt_handler.is_interrupted(): # Second Ctrl+C
                print_system_console_message("Exiting due to repeated Ctrl+C.")
                break
            interrupt_handler.handle_interrupt(None, None) # First Ctrl+C
            print_system_console_message("Input interrupted. Type 'exit' or press Ctrl+C again to quit.")
            continue
        except EOFError: # Ctrl+D
            print_system_console_message("EOF received. Exiting.")
            break
        
        if user_input.lower() in ["exit", "quit"]:
            print_system_console_message(f"Exiting {config.SERVICE_NAME}.")
            break
        if not user_input: continue

        print_user_message_log(user_input)
        conversation_history.append({"role": "user", "content": user_input})

        # --- Iterative AI Turn Processing ---
        # This loop continues as long as the AI's response leads to further actions (tool calls or continued text).
        
        # Start with no prior AI response text for this user turn
        current_ai_response_text_to_process = None 
        is_first_pass_for_user_input = True

        while True: # Inner loop for sequential AI responses and tool calls within a single user "turn"
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted by user flag.")
                break # Break from this AI's multi-step turn, go to next user input

            if is_first_pass_for_user_input or current_ai_response_text_to_process is None:
                # Get initial AI response or response to previous tool's observation
                manage_conversation_history_and_summarize()
                if interrupt_handler.is_interrupted(): break

                ai_client.set_interrupted(interrupt_handler.is_interrupted())
                ai_response_text, stop_reason = ai_client.get_response(SYSTEM_PROMPT, conversation_history)
                is_first_pass_for_user_input = False # Subsequent passes are follow-ups

                if interrupt_handler.is_interrupted() or stop_reason == "interrupted":
                    print_system_console_message("AI response generation interrupted.")
                    break 
                
                if not ai_response_text:
                    print_system_console_message("AI did not provide a response. Please try again.", is_error=True)
                    # If this was a response to user's direct input, pop user input.
                    # If it was a response to an observation, the observation remains.
                    if conversation_history and conversation_history[-1]["role"] == "user" and \
                       not conversation_history[-1]["content"].startswith("Observation:"):
                         conversation_history.pop()
                    break 
                
                conversation_history.append({"role": "assistant", "content": ai_response_text})
                current_ai_response_text_to_process = ai_response_text
            
            # Now, process actions from current_ai_response_text_to_process
            actions = parse_ai_response_for_actions(current_ai_response_text_to_process)
            logger.debug(f"Processing actions: {actions}")

            if not actions: # AI provided empty text or unparsable response
                logger.info("No further actions from AI in this pass. AI turn complete.")
                break # AI's current thought process is done.

            # Assume this is the last part of AI's response unless a tool is called
            next_iteration_needed_for_this_ai_turn = False 
            
            # Process only the first action. If it's text, print. If tool, execute.
            # The loop will then decide if it needs to re-prompt AI based on tool output.
            
            action_type, action_content = actions.pop(0) # Get and remove first action
            remaining_ai_text_to_process_later = None
            if actions: # If there are more actions in the AI's current response block
                # Temporarily convert remaining actions back to a string to be processed in the next iteration
                # This is a simplification. A more robust way would be to queue these actions.
                # For now, we'll focus on the AI reacting to one tool at a time.
                # If AI gives Text1 <Tool1> Text2 <Tool2>, we process Text1, then Tool1.
                # After Tool1's observation, AI should give Text2 <Tool2>.
                
                # If the first action was text, and there's a tool call immediately after,
                # we should process that tool call.
                # The logic here is tricky if AI gives: Text -> Tool -> Text -> Tool.
                # The system prompt guides it to give Text -> Tool(s) -> Wait for Obs -> Text -> Tool(s)
                
                # Let's process all text actions first from the current AI response
                if action_type == "text":
                    print_ai_message(action_content)
                    # See if next action is a tool
                    if actions and actions[0][0] == "tool":
                        action_type, action_content = actions.pop(0)
                    else: # Next action is text or no more actions from this AI block
                        current_ai_response_text_to_process = None # Signal this AI block is done if only text
                        if actions: # More text follows
                           current_ai_response_text_to_process = actions[0][1] # Assuming it's ("text", content)
                           actions.pop(0)
                           if actions: logger.warning("Unhandled subsequent actions after text block.")
                        if not current_ai_response_text_to_process:
                            break # End of AI turn if only text was processed
                        # If there was more text, loop will re-parse and print it.
                        # This is still not perfect for Text -> Tool -> Text -> Tool from one AI response.
                        # The expectation is AI gives Text -> Tool(s), then waits.

            # Refined logic: Process all text, then the first tool call.
            # The `parse_ai_response_for_actions` gives a list. We iterate it.
            # If a tool is run, we break from iterating these actions and send observation.
            
            # Reset for current_ai_response_text_to_process
            current_ai_response_text_to_process = None # Assume this block is done unless a tool runs

            for i, (act_type, act_content) in enumerate(actions): # Use original `actions` list
                if interrupt_handler.is_interrupted(): break

                if act_type == "text":
                    print_ai_message(act_content)
                elif act_type == "tool":
                    tool_name, tool_args = act_content
                    print_tool_being_used(tool_name, tool_args)
                    tool_output_str = execute_tool(tool_name, tool_args)

                    if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                        print_system_console_message("Command confirmation was interrupted by user.")
                        conversation_history.append({"role": "user", "content": f"Observation for {tool_name}: User interrupted command confirmation."})
                        next_iteration_needed_for_this_ai_turn = True
                        break # Stop processing this list of actions, AI needs to react
                    
                    print_tool_output(tool_name, tool_output_str)
                    conversation_history.append({"role": "user", "content": f"Observation for tool '{tool_name}':\n{tool_output_str}"})
                    next_iteration_needed_for_this_ai_turn = True # AI needs to react to this observation

                    # If a command is still running or blocked, AI *must* react to this before anything else.
                    if tool_name == "command_line" and \
                       ("Process is running" in tool_output_str or "Error: Another command" in tool_output_str or "Command interrupted." in tool_output_str):
                        logger.info(f"Command '{tool_args.get('command')}' state requires AI re-evaluation. Stopping further actions from current AI plan.")
                        break # Break from iterating current AI's planned actions

            # After iterating through actions from one AI response block
            if interrupt_handler.is_interrupted(): break # Break from inner while loop

            if not next_iteration_needed_for_this_ai_turn:
                # No tools were run, or tools run didn't require immediate AI follow-up (e.g. finished cleanly without being interactive)
                # and no other condition (like max_tokens) requires re-prompting.
                # So, AI's current "thought" or sequence for this user input is complete.
                logger.debug("No tools run or no immediate follow-up needed from AI. Ending AI turn.")
                break # Break from inner while loop, wait for next user input.
            
            # If next_iteration_needed_for_this_ai_turn is True, the loop continues,
            # and it will call get_response() again with updated history.
            # current_ai_response_text_to_process will be set by the next get_response() call.

            if stop_reason == "max_tokens": # Check original stop_reason for the block of actions
                print_system_console_message("Warning: AI's response may have been cut short. It will try to continue.", is_error=True)
                conversation_history.append({"role": "user", "content": "Observation: Your previous response might have been truncated due to maximum token limit. Please continue if you had more to say or do."})
                # The loop will continue and re-prompt AI.
        # End of inner `while True` loop (sequential AI responses/tool calls)
    # End of outer `while True` (user input loop)

if __name__ == "__main__":
    # ... (same as before) ...
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
