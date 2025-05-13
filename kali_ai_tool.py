# kali_ai_tool.py
import json
import readline
import sys
import logging
import time
# import re # No longer needed

import config
from ai_core.anthropic_client import AnthropicClient
from tools.base_tool import BaseTool
from tools.command_line_tool import CommandLineTool
from tools.web_search_tool import WebSearchTool
from tools.cve_search_tool import CVESearchTool
from tools.wait_tool import WaitTool
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
    logger.info("System prompt loaded successfully.")
except FileNotFoundError:
    print("CRITICAL: system_prompt.txt not found.", file=sys.stderr); sys.exit(1)
try:
    ai_client = AnthropicClient()
    logger.info(f"AnthropicClient initialized with model: {ai_client.model_name}")
except ValueError as e: # Catches API key not configured
    print(f"CRITICAL: AI Client Error: {e}", file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Unexpected error initializing AI Client: {e}", file=sys.stderr)
    logger.critical(f"CRITICAL: Unexpected error initializing AI Client: {e}", exc_info=True)
    sys.exit(1)


available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(),
    "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(),
    "wait": WaitTool(),
}
logger.info(f"Available tools initialized: {list(available_tools.keys())}")

interrupt_handler = InterruptHandler()
logger.info("InterruptHandler initialized.")
conversation_history = []

# --- Helper Functions ---
def print_ai_chunk_and_log(text_chunk: str, full_log_list: list):
    """Prints AI message chunk to console immediately and appends to log list."""
    print(text_chunk, end="", flush=True)
    full_log_list.append(text_chunk)

def print_user_message_log(message: str): logger.info(f"User: {message}")

def print_tool_being_used(tool_name: str, tool_args: dict):
    args_str = json.dumps(tool_args)
    if len(args_str) > 100: args_str = args_str[:100] + "..."
    message = f"AI is requesting to use tool: '{tool_name}' with arguments: {args_str}"
    logger.info(message)
    print(f"\n\n⚙️ System: {message}") # Ensure newlines for clarity after streamed AI output

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
        if not messages_to_summarize: logger.info("Not enough messages to summarize."); return False
        summary_text = ai_client.summarize_conversation(messages_to_summarize, config.SUMMARIZED_HISTORY_TARGET_TOKENS)
        if interrupt_handler.is_interrupted():
            print_system_console_message("Summarization interrupted by user.")
            return True # Attempted
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
            return True # Attempted
    return False

def execute_tool(tool_name: str, arguments: dict) -> str:
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
    logger.info(f"Application main loop started. Model: {ai_client.model_name}, Max Output Tokens: {config.MAX_AI_OUTPUT_TOKENS}")
    
    while True: # Outer loop for user input
        interrupt_handler.reset()
        ai_client.set_interrupted(False)
        for tool in available_tools.values(): tool.set_interrupted(False)

        print() 
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
        while needs_ai_to_respond: # Inner loop for AI's turn (can be multiple steps if tools are used)
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted by user flag.")
                break 

            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted(): break

            # --- Stream AI Response ---
            print(f"\n🤖 Assistant: ", end="", flush=True) # Start AI response line
            
            streamed_text_parts_for_log = [] # For logging the full text segment
            accumulated_preamble_for_history = [] # Text before a tool call
            tool_call_action = None 
            final_stop_reason_for_segment = None
            stream_ended_or_tool_found = False

            for event_type, data, *extra in ai_client.get_response_stream(SYSTEM_PROMPT, conversation_history):
                if interrupt_handler.is_interrupted():
                    if streamed_text_parts_for_log: print() # Newline if text was being printed
                    print_system_console_message("Stream consumption interrupted by user.")
                    needs_ai_to_respond = False 
                    break 

                if event_type == "text_chunk":
                    print_ai_chunk_and_log(data, streamed_text_parts_for_log)
                    accumulated_preamble_for_history.append(data)
                elif event_type == "tool_call_detected":
                    # data is preamble_text from client, extra[0] is tool_name, extra[1] is tool_args
                    # The preamble should have already been printed by text_chunks.
                    # The `data` (preamble_text from client) is what client parsed as before the tool.
                    # `accumulated_preamble_for_history` contains all text streamed so far for this segment.
                    # The true preamble for history is "".join(accumulated_preamble_for_history)
                    
                    tool_name, tool_args = extra[0], extra[1]
                    tool_call_action = (tool_name, tool_args)
                    final_stop_reason_for_segment = "tool_call_yielded" # Mark why stream stopped
                    stream_ended_or_tool_found = True
                    logger.info(f"Tool call detected mid-stream: {tool_name}. Preamble logged.")
                    break 
                elif event_type == "stream_complete":
                    # data is full_text from client buffer, extra[0] is stop_reason
                    final_stop_reason_for_segment = extra[0]
                    # Ensure all text is logged if client buffer had more than chunks yielded
                    if data and not streamed_text_parts_for_log: # e.g. very short message
                        print_ai_chunk_and_log(data, streamed_text_parts_for_log)
                    accumulated_preamble_for_history.append(data[len("".join(accumulated_preamble_for_history)):]) # Append remaining if any
                    stream_ended_or_tool_found = True
                    logger.info(f"AI stream segment ended. Reason: {final_stop_reason_for_segment}")
                    break 
                elif event_type in ["error", "interrupted"]: # Error or interrupt from client
                    if streamed_text_parts_for_log: print() # Newline
                    print_system_console_message(f"Stream error/interrupt from client: {event_type} - {data}", is_error=True)
                    final_stop_reason_for_segment = extra[0] if extra else event_type
                    needs_ai_to_respond = False
                    stream_ended_or_tool_found = True 
                    break
            
            # After stream consumption loop
            if streamed_text_parts_for_log: # If any text was streamed
                print() # Ensure a final newline after AI's text segment
                logger.info(f"AI Full Segment Log: {''.join(streamed_text_parts_for_log)}")


            if not needs_ai_to_respond: break # If stream error/interrupt broke inner event loop

            # Add assistant's message (preamble or full text) to history
            assistant_message_content = "".join(accumulated_preamble_for_history).strip()
            if assistant_message_content: # Only add if there was text
                conversation_history.append({"role": "assistant", "content": assistant_message_content})

            if tool_call_action:
                tool_name, tool_args = tool_call_action
                print_tool_being_used(tool_name, tool_args) # Prints its own newlines
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
                
                needs_ai_to_respond = True # AI needs to respond to this tool's observation
            
            elif final_stop_reason_for_segment == "max_tokens":
                print_system_console_message("Warning: AI's response was cut short. It may try to continue.", is_error=True)
                conversation_history.append({"role": "user", "content": "Observation: Your previous response was truncated. Please continue."})
                needs_ai_to_respond = True
            
            else: # Stream ended naturally without a tool call actioned or max_tokens
                needs_ai_to_respond = False # AI's turn is done

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
