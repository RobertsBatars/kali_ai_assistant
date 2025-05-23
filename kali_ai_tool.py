# kali_ai_tool.py
import json
import readline
import sys
import logging
import time

import config
# Updated imports for AI clients
from ai_core.base_llm_client import BaseLLMClient
from ai_core.anthropic_llm_client import AnthropicLLMClient
from ai_core.lm_studio_llm_client import LMStudioLLMClient

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

# Initialize AI Client based on configuration
ai_client: BaseLLMClient # Type hint
if config.AI_PROVIDER == "anthropic":
    try:
        ai_client = AnthropicLLMClient()
        logger.info(f"AnthropicLLMClient initialized with model: {ai_client.model_name}")
    except ValueError as e: 
        print(f"CRITICAL: Anthropic Client Error: {e}", file=sys.stderr); sys.exit(1)
elif config.AI_PROVIDER == "lm_studio":
    try:
        ai_client = LMStudioLLMClient()
        logger.info(f"LMStudioLLMClient initialized. API Base: {ai_client.api_base}, Model: {ai_client.model_name}")
    except ValueError as e:
        print(f"CRITICAL: LM Studio Client Error: {e}", file=sys.stderr); sys.exit(1)
else:
    print(f"CRITICAL: Unknown AI_PROVIDER '{config.AI_PROVIDER}' in configuration.", file=sys.stderr)
    sys.exit(1)

# ... (rest of initial setup: available_tools, interrupt_handler, conversation_history)
available_tools: dict[str, BaseTool] = {
    "command_line": CommandLineTool(), "web_search": WebSearchTool(),
    "cve_search": CVESearchTool(), "wait": WaitTool(),
}
logger.info(f"Available tools initialized: {list(available_tools.keys())}")
interrupt_handler = InterruptHandler()
logger.info("InterruptHandler initialized.")
conversation_history = []


# --- Helper Functions (print_*, manage_conversation_history, execute_tool) ---
# These should remain the same as your latest working version.
# For brevity, I'm omitting them here but they should be included.
# Ensure they are present in your actual file.

def print_ai_chunk(text_chunk: str):
    print(text_chunk, end="", flush=True)

def print_user_message_log(message: str): logger.info(f"User: {message}")

def print_tool_being_used(tool_name: str, tool_args: dict):
    args_str = json.dumps(tool_args)
    if len(args_str) > 100: args_str = args_str[:100] + "..."
    message = f"AI is requesting to use tool: '{tool_name}' with arguments: {args_str}"
    logger.info(message)
    print(f"\n\n⚙️ System: {message}") 

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
        if interrupt_handler.is_interrupted(): print_system_console_message("Summarization interrupted."); return True
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
    if tool_name in available_tools:
        tool = available_tools[tool_name]
        if tool_name == "command_line" and config.REQUIRE_COMMAND_CONFIRMATION:
            command_to_run = arguments.get("command")
            if command_to_run and not arguments.get("stdin_input") and not arguments.get("terminate_interactive"):
                print() 
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

# --- Main Application Loop (largely same as your last version, uses the polymorphic ai_client) ---
def main():
    global conversation_history
    print_system_console_message(f"{config.SERVICE_NAME} started. AI Provider: {config.AI_PROVIDER.upper()}. Type 'exit' or 'quit' to end.")
    logger.info(f"Application main loop started. AI Provider: {config.AI_PROVIDER.upper()}, Model: {config.ACTIVE_AI_MODEL}, Max Output Tokens: {config.MAX_AI_OUTPUT_TOKENS}")
    
    while True: 
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
        while needs_ai_to_respond:
            if interrupt_handler.is_interrupted():
                print_system_console_message("AI turn processing interrupted by user flag.")
                break 

            manage_conversation_history_and_summarize()
            if interrupt_handler.is_interrupted(): break

            print(f"\n🤖 Assistant: ", end="", flush=True)
            
            accumulated_text_chunks_for_log = []
            current_ai_speech_segment_text = "" # Store full text of current speech segment
            tool_call_action = None 
            final_stop_reason_for_segment = None
            
            for event_type, data, *extra in ai_client.get_response_stream(SYSTEM_PROMPT, conversation_history):
                if interrupt_handler.is_interrupted():
                    if accumulated_text_chunks_for_log: print()
                    print_system_console_message("Stream consumption interrupted by user.")
                    needs_ai_to_respond = False 
                    break 

                if event_type == "text_chunk":
                    print_ai_chunk(data) 
                    accumulated_text_chunks_for_log.append(data)
                    current_ai_speech_segment_text += data
                elif event_type == "first_tool_call_details":
                    # data is preamble_text, extra[0] is tool_name, extra[1] is tool_args
                    # The preamble (data) should be what was streamed *before* the tool call tag.
                    # current_ai_speech_segment_text will hold this preamble.
                    tool_name, tool_args = extra[0], extra[1]
                    tool_call_action = (tool_name, tool_args)
                    final_stop_reason_for_segment = "first_tool_call_yielded"
                    logger.info(f"Tool call received from stream: {tool_name}. Preamble from client: '{data}'")
                    # The preamble text (data) is what the client parsed.
                    # We've already printed chunks that form this preamble.
                    # current_ai_speech_segment_text should now equal 'data'.
                    current_ai_speech_segment_text = data # Trust client's parsed preamble
                    break 
                elif event_type == "stream_complete":
                    final_stop_reason_for_segment = extra[0]
                    # 'data' is full_text from client buffer.
                    # current_ai_speech_segment_text should match this if all chunks were processed.
                    current_ai_speech_segment_text = data if data else "".join(accumulated_text_chunks_for_log)
                    logger.info(f"AI stream segment ended. Reason: {final_stop_reason_for_segment}")
                    break 
                elif event_type in ["error", "interrupted"]:
                    if accumulated_text_chunks_for_log: print()
                    print_system_console_message(f"Stream error/interrupt from client: {event_type} - {data}", is_error=True)
                    final_stop_reason_for_segment = extra[0] if extra else event_type
                    current_ai_speech_segment_text = "".join(accumulated_text_chunks_for_log) # Use what we got
                    needs_ai_to_respond = False 
                    break
            
            if accumulated_text_chunks_for_log or (not tool_call_action and current_ai_speech_segment_text):
                print() # Ensure a final newline after AI's text
                logger.info(f"AI Full Segment Log: {current_ai_speech_segment_text}")

            if not needs_ai_to_respond: break 

            if current_ai_speech_segment_text.strip(): # Add non-empty segment to history
                conversation_history.append({"role": "assistant", "content": current_ai_speech_segment_text.strip()})

            if tool_call_action:
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
