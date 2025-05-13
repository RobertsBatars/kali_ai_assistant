# kali_ai_tool.py
import json
import readline
import sys
import logging
import time
import re 

import config
from ai_core.anthropic_client import AnthropicClient # Ensure this path is correct
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
    sys.exit(1)
# ... (rest of initial setup: AI client, tools, interrupt handler, history)
try:
    ai_client = AnthropicClient()
except ValueError as e:
    print(f"CRITICAL: Error initializing AI Client: {e}", file=sys.stderr)
    sys.exit(1)

available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(),
    "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(),
}
interrupt_handler = InterruptHandler()
conversation_history = []

# --- Helper Functions ---
# print_user_message_log, print_tool_being_used, print_tool_output, 
# print_system_console_message, manage_conversation_history_and_summarize, execute_tool
# parse_ai_response_for_actions (NO LONGER USED, logic integrated into stream handling)
# For brevity, only print_ai_message is shown if unchanged, others assumed present.

def print_ai_message_stream_chunk(text_chunk: str):
    """Prints AI message chunk to console immediately."""
    print(text_chunk, end="", flush=True)

def print_ai_message_complete(full_message: str):
    """Prints a newline after a complete AI message segment."""
    print() # Newline after streamed message
    logger.info(f"AI (complete segment): {full_message[:1000]}{'...' if len(full_message) > 1000 else ''}")

def print_user_message_log(message: str): logger.info(f"User: {message}")
def print_tool_being_used(tool_name: str, tool_args: dict):
    args_str = json.dumps(tool_args)
    if len(args_str) > 100: args_str = args_str[:100] + "..."
    message = f"Attempting to use tool: '{tool_name}' with arguments: {args_str}"
    logger.info(message)
    print(f"\n\n⚙️ System: {message}") # Extra newline for clarity after streamed AI output

def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:1000]}{'...' if len(output) > 1000 else ''}")
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_console_message(message: str, is_error=False):
    log_level = logging.ERROR if is_error else logging.INFO
    logger.log(log_level, f"SystemConsole: {message}")
    print(f"\n⚙️ System:\n{message}")

def manage_conversation_history_and_summarize():
    # (This function remains the same as the previous version, ensure it uses print_system_console_message)
    global conversation_history
    current_tokens = estimate_messages_token_count(conversation_history)
    logger.debug(f"Current estimated token count: {current_tokens}. Soft limit: {config.CONTEXT_TOKEN_SOFT_LIMIT}")
    if current_tokens > config.CONTEXT_TOKEN_SOFT_LIMIT and len(conversation_history) > config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:
        print_system_console_message(f"Context length ({current_tokens} tokens) nearing limit. Attempting summarization...")
        messages_to_keep_suffix = conversation_history[-config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY:]
        messages_to_summarize = conversation_history[:-config.MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY]
        if not messages_to_summarize: return False
        summary_text = ai_client.summarize_conversation(messages_to_summarize, config.SUMMARIZED_HISTORY_TARGET_TOKENS)
        if interrupt_handler.is_interrupted():
            print_system_console_message("Summarization interrupted by user.")
            return True
        if summary_text:
            new_history = [{"role": "system", "content": f"Previous conversation summary: {summary_text}"}]
            new_history.extend(messages_to_keep_suffix)
            conversation_history = new_history
            print_system_console_message("Conversation history summarized.")
            return True
        else:
            print_system_console_message("Failed to summarize conversation history.", is_error=True)
            if current_tokens > config.CONTEXT_TOKEN_HARD_LIMIT:
                 print_system_console_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit.", is_error=True)
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
                    if user_confirmation != "yes": return "User declined command execution."
                except (EOFError, KeyboardInterrupt):
                    interrupt_handler.handle_interrupt(None, None)
                    return "User interrupted command confirmation."
        tool.set_interrupted(interrupt_handler.is_interrupted())
        return tool.execute(arguments)
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
            user_input = input("\n\n👤 You: ").strip() # Extra newline for spacing
        except KeyboardInterrupt:
            if interrupt_handler.is_interrupted(): break
            interrupt_handler.handle_interrupt(None, None)
            print_system_console_message("Input interrupted. Type 'exit' or press Ctrl+C again to quit.")
            continue
        except EOFError: break
        
        if user_input.lower() in ["exit", "quit"]: break
        if not user_input: continue

        print_user_message_log(user_input)
        conversation_history.append({"role": "user", "content": user_input})
        
        # --- Iterative AI Turn Processing ---
        needs_ai_response = True
        while needs_ai_response:
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted by user flag.")
                break 

            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted(): break

            accumulated_ai_preamble = [] # Text from AI before a tool call or end of stream
            tool_to_execute = None # Tuple: (tool_name, tool_args)
            
            print(f"\n🤖 Assistant: ", end="", flush=True) # Start AI response line

            for event_type, data, *extra in ai_client.get_response_stream(SYSTEM_PROMPT, conversation_history):
                if interrupt_handler.is_interrupted(): # Check during stream consumption
                    print_system_console_message("\nStream consumption interrupted.")
                    # Add whatever was accumulated as assistant message
                    if accumulated_ai_preamble:
                        conversation_history.append({"role": "assistant", "content": "".join(accumulated_ai_preamble)})
                    needs_ai_response = False # Stop this AI turn
                    break 

                if event_type == "text_delta":
                    print_ai_message_stream_chunk(data)
                    accumulated_ai_preamble.append(data)
                elif event_type == "tool_call":
                    tool_name, tool_args = data, extra[0]
                    tool_to_execute = (tool_name, tool_args)
                    # Preamble (accumulated_ai_preamble) is handled after loop
                    logger.info(f"Tool call received from stream: {tool_name}")
                    break # Stop processing this stream, execute tool
                elif event_type == "stream_end":
                    # data is full_text, extra[0] is stop_reason
                    final_stop_reason = extra[0]
                    # The text_buffer from client is 'data'. We've already printed & accumulated it.
                    print_ai_message_complete("".join(accumulated_ai_preamble)) # Ensure newline
                    
                    # Add the complete text segment to history
                    # If accumulated_ai_preamble is empty but data (full_text_from_client_buffer) is not, use data.
                    # This handles cases where the stream might end without yielding individual deltas (e.g. very short messages)
                    # but this shouldn't happen with current client logic.
                    final_text_segment = "".join(accumulated_ai_preamble) if accumulated_ai_preamble else data
                    if final_text_segment: # Only add if there was text
                         conversation_history.append({"role": "assistant", "content": final_text_segment})
                    
                    if final_stop_reason == "max_tokens":
                        print_system_console_message("Warning: AI's response was cut short. It may try to continue.", is_error=True)
                        conversation_history.append({"role": "user", "content": "Observation: Your previous response was truncated. Please continue."})
                        needs_ai_response = True # Re-prompt AI
                    else:
                        needs_ai_response = False # AI turn complete if no tool and no truncation
                    break # End of stream processing
                elif event_type == "error":
                    print_ai_message_complete("".join(accumulated_ai_preamble)) # Print what we have
                    print_system_console_message(f"Error from AI stream: {data}", is_error=True)
                    needs_ai_response = False # Stop this AI turn
                    break
                elif event_type == "interrupted": # Interrupted from within client
                    print_ai_message_complete("".join(accumulated_ai_preamble))
                    print_system_console_message("AI stream was interrupted.", is_error=True)
                    needs_ai_response = False
                    break
            
            if interrupt_handler.is_interrupted(): # If the for-loop broke due to interrupt
                needs_ai_response = False # Don't try to get more AI response now

            if tool_to_execute:
                # Add preamble to history (if any) before tool execution observation
                full_preamble = "".join(accumulated_ai_preamble)
                print_ai_message_complete(full_preamble) # Ensure newline after preamble
                if full_preamble:
                    conversation_history.append({"role": "assistant", "content": full_preamble})

                tool_name, tool_args = tool_to_execute
                print_tool_being_used(tool_name, tool_args)
                tool_output_str = execute_tool(tool_name, tool_args)

                if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                    print_system_console_message("Command confirmation was interrupted.")
                    # Don't add this specific observation, let AI retry based on original user query + history.
                    # Or, add a specific observation about *confirmation* interruption.
                    conversation_history.append({"role": "user", "content": f"Observation: I interrupted the confirmation for command: {tool_args.get('command')}"})
                    needs_ai_response = True # AI needs to react to this specific interruption
                    continue # Back to top of inner while loop to get AI's reaction

                print_tool_output(tool_name, tool_output_str)
                observation_content = f"Observation for tool '{tool_name}':\n{tool_output_str}"
                conversation_history.append({"role": "user", "content": observation_content})
                
                if "Command interrupted." in tool_output_str and interrupt_handler.is_interrupted():
                     print_system_console_message(f"Tool '{tool_name}' execution was interrupted by user.")
                
                needs_ai_response = True # AI needs to respond to this tool's observation
            
            # If needs_ai_response is still True, the inner loop continues, re-prompting AI.
            # If False, inner loop breaks, waiting for next user input.

    # End of outer `while True` (user input loop)
    print_system_console_message(f"Exiting {config.SERVICE_NAME}.") # Ensure exit message
if __name__ == "__main__":
    try:
        main()
    except SystemExit: pass
    except Exception as e:
        logger.critical(f"--- Main loop critical error: {e} ---", exc_info=True)
        print(f"\n--- CRITICAL ERROR: {e} ---", file=sys.stderr)
    finally:
        logger.info("Application terminated.")
        # Avoid double printing if already exited cleanly via sys.exit()
        # if not (sys.exc_info()[0] is None or sys.exc_info()[0] is SystemExit):
        #      print("\nApplication terminated due to an error.", file=sys.stderr)
        # else:
        #      print("\nApplication terminated.") # Already handled by main exit path
