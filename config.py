# config.py
import os
import logging # For log level constants
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# --- AI Configuration ---
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "claude-3-7-sonnet-latest")
# Context summarization settings
# Max tokens before considering summarization (estimate)
CONTEXT_TOKEN_HARD_LIMIT = int(os.getenv("CONTEXT_TOKEN_HARD_LIMIT", 200000)) # e.g. Claude 3 Opus has 200k
CONTEXT_TOKEN_SOFT_LIMIT = int(os.getenv("CONTEXT_TOKEN_SOFT_LIMIT", 150000)) # Trigger summarization earlier
# Target token count for the summarized history.
SUMMARIZED_HISTORY_TARGET_TOKENS = int(os.getenv("SUMMARIZED_HISTORY_TARGET_TOKENS", 20000))
# Min messages to keep before summarization (e.g., always keep last N exchanges)
MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY = int(os.getenv("MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY", 4)) # Keep last 2 user/assistant turns

# --- Tool Configuration ---
# Default timeout for commands executed by CommandLineTool (in seconds)
DEFAULT_COMMAND_TIMEOUT = int(os.getenv("DEFAULT_COMMAND_TIMEOUT", 300)) # 5 minutes

# --- Logging Configuration ---
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/kali_ai_tool.log")
LOG_LEVEL_FILE_STR = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()
LOG_LEVEL_CONSOLE_STR = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
SERVICE_NAME = "KaliAIAssistant"

# Convert string log levels to logging constants
LOG_LEVEL_FILE = getattr(logging, LOG_LEVEL_FILE_STR, logging.DEBUG)
LOG_LEVEL_CONSOLE = getattr(logging, LOG_LEVEL_CONSOLE_STR, logging.INFO)


# --- Validation ---
if not ANTHROPIC_API_KEY:
    print("CRITICAL: ANTHROPIC_API_KEY not found. Please set it in .env or environment variables.", flush=True)
    # Depending on setup, you might want to exit here if critical keys are missing.
    # For now, it will raise an error in AnthropicClient if used without a key.

# Example .env entries for these new configs:
# CONTEXT_TOKEN_SOFT_LIMIT=150000
# SUMMARIZED_HISTORY_TARGET_TOKENS=20000
# MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY=4
# DEFAULT_COMMAND_TIMEOUT=300
# LOG_FILE_PATH="app_logs/assistant.log"
# LOG_LEVEL_FILE="DEBUG"
# LOG_LEVEL_CONSOLE="INFO"
