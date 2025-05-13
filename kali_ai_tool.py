# kali_ai_tool.py
import json
import readline
import sys
import logging
import time
# import re # No longer needed if parse_ai_response_for_actions is removed

import config
from ai_core.anthropic_client import AnthropicClient # Ensure this path is correct
from tools.base_tool import BaseTool
from tools.command_line_tool import CommandLineTool
from tools.web_search_tool import WebSearchTool
from tools.cve_search_tool import CVESearchTool
from tools.wait_tool import WaitTool # Import the new WaitTool
from utils.interrupt_handler import InterruptHandler
from utils.logger_setup import setup_logging
from utils.token_estimator import estimate_messages_token_count

logger = setup_logging(
    log_file_path=config.LOG_FILE_PATH,
    log_level_file=config.LOG_LEVEL_FILE,
    log_level_console=config.LOG_LEVEL_CONSOLE,
    service_name=config.SERVICE_NAME
)

# --- Initial Setup ---
try:
    with open("system_prompt.txt", "r") as f: SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print("CRITICAL: system_prompt.txt not found.", file=sys.stderr); sys.exit(1)
try:
    ai_client = AnthropicClient()
except ValueError as e:
    print(f"CRITICAL: AI Client Error: {e}", file=sys.stderr); sys.exit(1)

available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(),
    "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(),
    "wait": WaitTool(), # Add the new WaitTool instance
}
interrupt_handler = InterruptHandler()
conversation_history = []

# --- Helper Functions ---
# (print_user_message_log, print_tool_being_used, print_tool_output, 
# print_system_console_message, manage_conversation_history_and_summarize, execute_tool
# are assumed to be the same as your latest working version.
# parse_ai_response_for_actions is NO LONGER USED with the new stream handling)

def print_ai_message_segment(full_message_segment: str):
    """Prints a complete segment of AI's text response."""
    if full_message_segment: # Only print if there's content
        print(f"\n🤖 Assistant:\n{full_message_segment.strip()}")
        logger.info(f"AI Segment: {full_message_segment.strip()[:1000]}{'...' if len(full_message_segment) > 1000 else ''}")

def print_user_message_log(message: str): logger.info(f"User: {message}")
def print_tool_being_used(tool_name: str, tool_args: dict):
    args_str = json.dumps(tool_args)
    if len(args_str) > 100: args_str = args_str[:100] + "..."
    message = f"AI is requesting to use tool: '{tool_name}' with arguments: {args_str}"
    logger.info(message)
    print(f"\n⚙️ System: {message}") # Ensure newlines for clarity

def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:1000]}{'...' if len(output) > 1000 else ''}")
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_console_message(message: str, is_error=False):
    log_level = logging.ERROR if is_error else logging.INFO
    logger.log(log_level, f"SystemConsole: {message}")
    print(f"\n⚙️ System:\n{message}")

def manage_conversation_history_and_summarize():
    # (Assumed to be the same, using print_system_console_message)
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
        else: # Summarization failed
            print_system_console_message("Failed to summarize conversation history.", is_error=True)
            if current_tokens > config.CONTEXT_TOKEN_HARD_LIMIT:
                 print_system_console_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit.", is_error=True)
            return True
    return False

def execute_tool(tool_name: str, arguments: dict) -> str:
    # (Assumed to be the same, handles confirmation and execution)
    if tool_name in available_tools:
        tool = available_tools[tool_name]
        if tool_name == "command_line" and config.REQUIRE_COMMAND_CONFIRMATION:
            command_to_run = arguments.get("command")
            if command_to_run and not arguments.get("stdin_input") and not arguments.get("terminate_interactive"):
                print() # Newline before input prompt for clarity
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

        print() # Ensure a clean line for user input
        try:
            user_input = input("👤 You: ").strip()
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
        
        needs_ai_to_respond = True
        while needs_ai_to_respond:
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted by user flag.")
                break 

            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted(): break

            accumulated_text_this_segment = []
            tool_call_action = None 
            final_stop_reason_for_segment = None
            preamble_for_tool_call = "" # Text specifically before a detected tool call
            
            # No initial "Assistant:" print here; will be done by print_ai_message_segment

            for event_type, data, *extra in ai_client.get_response_stream(SYSTEM_PROMPT, conversation_history):
                if interrupt_handler.is_interrupted():
                    print_system_console_message("\nStream consumption interrupted by user.")
                    if accumulated_text_this_segment:
                        # Add what we got to history before breaking
                        full_segment = "".join(accumulated_text_this_segment)
                        # Don't print here, it might interfere with subsequent input()
                        # print_ai_message_segment(full_segment) # Print it
                        conversation_history.append({"role": "assistant", "content": full_segment})
                    needs_ai_to_respond = False 
                    break 

                if event_type == "text_chunk":
                    accumulated_text_this_segment.append(data)
                elif event_type == "tool_call_detected":
                    preamble_for_tool_call = data 
                    tool_name, tool_args = extra[0], extra[1]
                    tool_call_action = (tool_name, tool_args)
                    logger.info(f"Tool call detected mid-stream: {tool_name}. Preamble: '{preamble_for_tool_call}'")
                    break 
                elif event_type == "stream_complete":
                    final_stop_reason_for_segment = extra[0]
                    full_segment_text = "".join(accumulated_text_this_segment) if accumulated_text_this_segment else data
                    if full_segment_text: 
                         # Print accumulated text before history add
                         print_ai_message_segment(full_segment_text)
                         conversation_history.append({"role": "assistant", "content": full_segment_text})
                    logger.info(f"AI stream segment ended. Reason: {final_stop_reason_for_segment}")
                    break 
                elif event_type == "error":
                    # Print accumulated text before error message
                    if accumulated_text_this_segment: print_ai_message_segment("".join(accumulated_text_this_segment))
                    print_system_console_message(f"Error from AI stream: {data}", is_error=True)
                    if accumulated_text_this_segment:
                        conversation_history.append({"role": "assistant", "content": "".join(accumulated_text_this_segment)})
                    needs_ai_to_respond = False 
                    break
                elif event_type == "interrupted":
                    if accumulated_text_this_segment: print_ai_message_segment("".join(accumulated_text_this_segment))
                    print_system_console_message(f"AI stream was interrupted: {data}", is_error=True)
                    if accumulated_text_this_segment:
                         conversation_history.append({"role": "assistant", "content": "".join(accumulated_text_this_segment)})
                    needs_ai_to_respond = False
                    break
            
            if not needs_ai_to_respond: break 

            if tool_call_action:
                if preamble_for_tool_call:
                    print_ai_message_segment(preamble_for_tool_call)
                    if not any(msg["role"] == "assistant" and msg["content"] == preamble_for_tool_call.strip() for msg in conversation_history[-2:]):
                         if preamble_for_tool_call.strip():
                            conversation_history.append({"role": "assistant", "content": preamble_for_tool_call.strip()})

                tool_name, tool_args = tool_call_action
                print_tool_being_used(tool_name, tool_args)
                tool_output_str = execute_tool(tool_name, tool_args)

                if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                    print_system_console_message("Command confirmation was interrupted.")
                    conversation_history.append({"role": "user", "content": f"Observation: I interrupted the confirmation for your request to run '{tool_args.get('command')}'."})
                else:
                    print_tool_output(tool_name, tool_output_str)
                    observation_content = f"Observation for tool '{tool_name}':\n{tool_output_str}"
                    conversation_history.append({"role": "user", "content": observation_content})
                    if "Command interrupted." in tool_output_str and interrupt_handler.is_interrupted():
                         print_system_console_message(f"Tool '{tool_name}' execution was interrupted.")
                
                needs_ai_to_respond = True 
            
            elif final_stop_reason_for_segment == "max_tokens":
                # Text already printed and added to history by stream_complete handling.
                print_system_console_message("Warning: AI's response was cut short. It may try to continue.", is_error=True)
                conversation_history.append({"role": "user", "content": "Observation: Your previous response was truncated. Please continue."})
                needs_ai_to_respond = True
            
            else: 
                needs_ai_to_respond = False 

    print_system_console_message(f"Exiting {config.SERVICE_NAME}.")

if __name__ == "__main__":
    try: main()
    except SystemExit: pass
    except Exception as e:
        logger.critical(f"--- Main loop critical error: {e} ---", exc_info=True)
        print(f"\n--- CRITICAL ERROR: {e} ---", file=sys.stderr)
    finally:
        logger.info("Application terminated.")
        print("\nApplication terminated.")
