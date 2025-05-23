# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# --- AI Provider Selection ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic").lower() # Default to anthropic

# --- Anthropic API Configuration ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_ANTHROPIC_MODEL = os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# --- LM Studio Configuration ---
LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
LM_STUDIO_MODEL_NAME = os.getenv("LM_STUDIO_MODEL_NAME", "local-model") # This might be a placeholder

# --- General AI Configuration ---
# MAX_AI_OUTPUT_TOKENS is used by both clients to hint generation length
MAX_AI_OUTPUT_TOKENS = int(os.getenv("MAX_AI_OUTPUT_TOKENS", 4096))
CONTEXT_TOKEN_HARD_LIMIT = int(os.getenv("CONTEXT_TOKEN_HARD_LIMIT", 180000))
CONTEXT_TOKEN_SOFT_LIMIT = int(os.getenv("CONTEXT_TOKEN_SOFT_LIMIT", 150000))
SUMMARIZED_HISTORY_TARGET_TOKENS = int(os.getenv("SUMMARIZED_HISTORY_TARGET_TOKENS", 20000))
MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY = int(os.getenv("MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY", 6))

# --- Search API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# --- Tool Configuration ---
DEFAULT_COMMAND_TIMEOUT = int(os.getenv("DEFAULT_COMMAND_TIMEOUT", 300))
REQUIRE_COMMAND_CONFIRMATION = os.getenv("REQUIRE_COMMAND_CONFIRMATION", "True").lower() == "true"

# --- Logging Configuration ---
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/kali_ai_tool.log")
LOG_LEVEL_FILE_STR = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()
LOG_LEVEL_CONSOLE_STR = os.getenv("LOG_LEVEL_CONSOLE", "WARNING").upper()
SERVICE_NAME = "KaliAIAssistant"

LOG_LEVEL_FILE = getattr(logging, LOG_LEVEL_FILE_STR, logging.DEBUG)
LOG_LEVEL_CONSOLE = getattr(logging, LOG_LEVEL_CONSOLE_STR, logging.WARNING)

# --- Validation & Provider Specific Defaults ---
if AI_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    print("CRITICAL: AI_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.", flush=True)
    # Consider exiting: sys.exit(1)

# Determine active model based on provider
ACTIVE_AI_MODEL = ""
if AI_PROVIDER == "anthropic":
    ACTIVE_AI_MODEL = DEFAULT_ANTHROPIC_MODEL
elif AI_PROVIDER == "lm_studio":
    ACTIVE_AI_MODEL = LM_STUDIO_MODEL_NAME # This is more of an identifier for LM Studio
else:
    print(f"Warning: Unknown AI_PROVIDER '{AI_PROVIDER}'. Defaulting to anthropic logic if possible.", flush=True)
    AI_PROVIDER = "anthropic" # Fallback, though might fail if ANTHROPIC_API_KEY is missing
    ACTIVE_AI_MODEL = DEFAULT_ANTHROPIC_MODEL

