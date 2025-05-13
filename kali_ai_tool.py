# kali_ai_tool.py
import json
import readline
import sys
import logging
import time

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

def print_ai_message(message: str):
    logger.info(f"AI: {message[:1000]}{'...' if len(message) > 1000 else ''}")
    print(f"\n🤖 Assistant:\n{message}")

def print_user_message_log(message: str):
    logger.info(f"User: {message}")

def print_tool_output(tool_name: str, output: str):
    logger.info(f"Tool ({tool_name}) Output: {output[:1000]}{'...' if len(output) > 1000 else ''}")
    print(f"\n🛠️ Tool Output ({tool_name}):\n{output}")

def print_system_console_message(message: str):
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
                 print_system_console_message(f"WARNING: Token count ({current_tokens}) exceeds hard limit. AI may truncate or fail.")
            return True 
    return False

def parse_tool_call(ai_response: str) -> tuple[str | None, dict | None, str | None]:
    """
    Parses the AI's response to find a tool call and any text before it.
    Returns: (tool_name, arguments, preamble_text)
    Preamble_text is the text part of the AI's response before the tool call tag.
    If no tool call tag is found, the entire ai_response is preamble_text.
    """
    tool_name, arguments, preamble_text = None, None, None
    try:
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        start_index = ai_response.find(start_tag)
        end_index = ai_response.find(end_tag)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Valid tool call syntax found
            if start_index > 0:
                preamble_text = ai_response[:start_index].strip()
            
            # The rest of the AI's message might be just the tool call, or text after it.
            # System prompt asks AI to not put text after tool_call if making a call.
            # So we assume the content of tool_call is JSON.
            tool_call_json_str = ai_response[start_index + len(start_tag):end_index].strip()
            logger.debug(f"Raw tool call JSON string: {tool_call_json_str}")
            
            tool_call_data = json.loads(tool_call_json_str)
            parsed_tool_name = tool_call_data.get("tool_name")
            parsed_arguments = tool_call_data.get("arguments")

            if not parsed_tool_name or not isinstance(parsed_arguments, dict):
                logger.warning(f"AI tool call format error: Name or args missing. Data: {tool_call_data}")
                print_system_console_message("Error: AI tried to call a tool with invalid format.")
                # If tool call is bad, treat the whole thing as text if no preamble was extracted yet
                if preamble_text is None: preamble_text = ai_response.strip()
            else:
                tool_name = parsed_tool_name
                arguments = parsed_arguments
                logger.info(f"Parsed tool call: {tool_name}, Args: {arguments}. Preamble: '{preamble_text}'")
        else:
            # No tool call syntax found, entire response is preamble/text
            preamble_text = ai_response.strip()
            
    except json.JSONDecodeError as e:
        logger.warning(f"JSONDecodeError parsing tool call: {e}. String was: '{tool_call_json_str if 'tool_call_json_str' in locals() else 'N/A'}'")
        preamble_text = ai_response.strip() # Treat as text if JSON is bad
    except Exception as e:
        logger.error(f"Error parsing tool call: {e}", exc_info=True)
        preamble_text = ai_response.strip() # Treat as text on other errors

    return tool_name, arguments, preamble_text


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

def main():
    global conversation_history
    print_system_console_message(f"{config.SERVICE_NAME} started. Type 'exit' or 'quit' to end.")
    logger.info(f"Application main loop started. Model: {config.DEFAULT_AI_MODEL}")
    
    while True:
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
        
        if interrupt_handler.is_interrupted():
            print_system_console_message("Operation cancelled by user after input.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print_system_console_message(f"Exiting {config.SERVICE_NAME}.")
            break
        if not user_input: continue

        print_user_message_log(user_input)
        conversation_history.append({"role": "user", "content": user_input})

        manage_conversation_history_and_summarize()
        if interrupt_handler.is_interrupted():
            print_system_console_message("Operation interrupted during context management.")
            if conversation_history and conversation_history[-1]["role"] == "user": conversation_history.pop()
            continue
        
        ai_client.set_interrupted(interrupt_handler.is_interrupted())
        ai_response_text = ai_client.get_response(SYSTEM_PROMPT, conversation_history)

        if interrupt_handler.is_interrupted():
            print_system_console_message("AI response generation interrupted.")
            if conversation_history and conversation_history[-1]["role"] == "user": conversation_history.pop()
            continue
        
        if not ai_response_text:
            print_system_console_message("AI did not provide a response. Please try again.")
            if conversation_history and conversation_history[-1]["role"] == "user": conversation_history.pop()
            continue

        # Add AI's full response to history before parsing.
        # If it's a tool call, the content is the structured call.
        # If it's text, it's text.
        conversation_history.append({"role": "assistant", "content": ai_response_text})
        
        tool_name, tool_args, preamble_text = parse_tool_call(ai_response_text)

        if preamble_text: # This is AI's conversational text.
            print_ai_message(preamble_text)

        if tool_name and tool_args:
            # If preamble_text was None (AI only sent tool call as per strict system prompt),
            # and we haven't printed anything for this AI turn yet.
            if not preamble_text: 
                 print_ai_message(f"Okay, I will use the '{tool_name}' tool.")

            tool_output_str = execute_tool(tool_name, tool_args)
            
            if "User interrupted command confirmation." in tool_output_str and interrupt_handler.is_interrupted():
                print_system_console_message("Command confirmation was interrupted by user.")
                # AI's last message (tool call request) should be kept.
                # The observation of interruption will be sent next.
                # No, we should pop the AI's tool call request because it wasn't fulfilled.
                if conversation_history and conversation_history[-1]["role"] == "assistant":
                    logger.debug("Popping AI's last message (tool call) due to confirmation interruption.")
                    conversation_history.pop()
                # Also pop the user message that led to this, so user can re-issue or AI can try something else.
                # This might be too aggressive. Let's keep the user message.
                # The AI will see the user's original request and then the interruption.
                # For now, just continue to next user prompt.
                continue # Go to next "You:" prompt

            print_tool_output(tool_name, tool_output_str)
            
            observation_content = f"Observation: {tool_output_str}"
            # If the tool execution itself was interrupted (not just confirmation)
            if "Command interrupted." in tool_output_str and interrupt_handler.is_interrupted():
                 print_system_console_message(f"Tool '{tool_name}' execution was interrupted.")
                 # Observation already reflects this from CommandLineTool's output
            
            conversation_history.append({"role": "user", "content": observation_content}) # Tool output is like a user message to AI

            manage_conversation_history_and_summarize()
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
            else: # AI failed to respond after tool use
                print_system_console_message("AI did not provide a follow-up response after tool execution.")
                # Remove the observation as AI didn't process it.
                if conversation_history and conversation_history[-1]["role"] == "user" and "Observation:" in conversation_history[-1]["content"]:
                    conversation_history.pop()
        
        # If no tool_name and tool_args, then preamble_text (if it existed) was the full AI response and already printed.
        # If preamble_text was also None (e.g. AI sent empty response), nothing more to do here.

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
