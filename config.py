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
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "claude-3-7-sonnet-latest") # Consider claude-3-sonnet for faster/cheaper tests
# Context summarization settings
CONTEXT_TOKEN_HARD_LIMIT = int(os.getenv("CONTEXT_TOKEN_HARD_LIMIT", 200000))
CONTEXT_TOKEN_SOFT_LIMIT = int(os.getenv("CONTEXT_TOKEN_SOFT_LIMIT", 150000))
SUMMARIZED_HISTORY_TARGET_TOKENS = int(os.getenv("SUMMARIZED_HISTORY_TARGET_TOKENS", 20000))
MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY = int(os.getenv("MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY", 6)) # Keep last 3 user/assistant turns

# --- Tool Configuration ---
DEFAULT_COMMAND_TIMEOUT = int(os.getenv("DEFAULT_COMMAND_TIMEOUT", 300)) # 5 minutes
REQUIRE_COMMAND_CONFIRMATION = os.getenv("REQUIRE_COMMAND_CONFIRMATION", "False").lower() == "true"


# --- Logging Configuration ---
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/kali_ai_tool.log")
LOG_LEVEL_FILE_STR = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()
# Changed default console log level to WARNING to reduce verbosity
LOG_LEVEL_CONSOLE_STR = os.getenv("LOG_LEVEL_CONSOLE", "WARNING").upper()
SERVICE_NAME = "KaliAIAssistant"

LOG_LEVEL_FILE = getattr(logging, LOG_LEVEL_FILE_STR, logging.DEBUG)
LOG_LEVEL_CONSOLE = getattr(logging, LOG_LEVEL_CONSOLE_STR, logging.WARNING)


# --- Validation ---
if not ANTHROPIC_API_KEY:
    # Use print for this critical startup message as logger might not be fully set up or respected for console
    print("CRITICAL: ANTHROPIC_API_KEY not found. Please set it in .env or environment variables.", flush=True)
    # Exit if critical key is missing
    # sys.exit(1) # Consider uncommenting if you want to enforce this

# Example .env entries for these new configs:
# REQUIRE_COMMAND_CONFIRMATION=True # or False
# LOG_LEVEL_CONSOLE="INFO" # To override default WARNING for console
