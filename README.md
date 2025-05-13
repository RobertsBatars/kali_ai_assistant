# Kali AI Assistant

The Kali AI Assistant is an interactive command-line tool that leverages AI (Anthropic's Claude models) to assist with tasks in a Kali Linux environment, particularly focused on penetration testing, vulnerability analysis, and information gathering. It can execute commands, search the web using multiple APIs, and look up CVE information.

## Features

* **Interactive AI Chat**: Converse with an AI assistant that understands penetration testing context.
* **Command Execution**: Allows the AI to request execution of shell commands (with optional user confirmation). Supports interactive commands and configurable timeouts.
* **Web Search**: Integrated with Google, Tavily, and Brave Search APIs for information gathering.
* **CVE Lookup**: Dedicated tool for searching CVE details (currently uses web search).
* **Modular Tool System**: Easily extendable with new tools.
* **Context Management**: Automatic summarization of long conversations to stay within token limits.
* **Interruption Handling**: Supports Ctrl+C for interrupting ongoing operations.
* **Detailed Logging**: Logs application activity to a file for debugging and review.
* **Streaming AI Responses**: Uses streaming for potentially long AI responses to improve perceived responsiveness.
* **Multiple Tool Calls**: Can process multiple tool calls from a single AI response.

## Project Structure



kali_ai_assistant/
├── kali_ai_tool.py # Main application script
├── system_prompt.txt # Defines the AI's persona and capabilities
├── config.py # Loads and manages configuration
├── .env # (You create this) Stores API keys and sensitive configs
├── .env.example # Example environment file
├── requirements.txt # Python dependencies
├── ai_core/ # AI interaction logic
│ └── anthropic_client.py
├── tools/ # Tool implementations
│ ├── base_tool.py
│ ├── command_line_tool.py
│ ├── web_search_tool.py
│ └── cve_search_tool.py
├── utils/ # Utility modules
│ ├── interrupt_handler.py
│ ├── logger_setup.py
│ └── token_estimator.py
└── logs/ # Directory for log files (created automatically)
## Setup Instructions

### 1. Prerequisites

* Python 3.9 or higher.
* `pip` for installing Python packages.
* Access to a Kali Linux environment (or a Linux system with similar tools if you adapt the use case).
* API keys for:
    * Anthropic (required for AI)
    * Google Custom Search API (optional, for Google search)
    * Tavily API (optional, for Tavily search)
    * Brave Search API (optional, for Brave search)

### 2. Clone the Repository (or Create Files)

If this project is in a repository:
```bash
git clone <repository_url>
cd kali_ai_assistant


Otherwise, ensure all the provided Python files (.py), system_prompt.txt, and requirements.txt are in the correct directory structure as shown above.
3. Create a Python Virtual Environment (Recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate    # On Windows


4. Install Dependencies
pip install -r requirements.txt


5. Configure Environment Variables
Copy the example environment file to a new .env file:
cp .env.example .env


Now, edit the .env file with your actual API keys and any custom configurations:
# .env file

# --- API Keys ---
ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: Google Custom Search API (needs API Key and Programmable Search Engine ID)
GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_CSE_ID="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: Tavily Search API
TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Optional: Brave Search API
BRAVE_SEARCH_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# --- AI Configuration ---
# Model examples: "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
DEFAULT_AI_MODEL="claude-3-sonnet-20240229"
# Max tokens the AI is allowed to generate in a single response.
# Anthropic SDK might have its own limits for non-streaming calls if this is set too high.
# Streaming is now used, so this primarily guides the AI's output length.
MAX_AI_OUTPUT_TOKENS=4096

# Context summarization settings (estimated token counts)
CONTEXT_TOKEN_HARD_LIMIT=180000 # Model's absolute max context (e.g., Claude 3 Opus 200k)
CONTEXT_TOKEN_SOFT_LIMIT=150000 # Trigger summarization well before hard limit
SUMMARIZED_HISTORY_TARGET_TOKENS=20000 # Aim for summary to be around this many tokens
MIN_MESSAGES_TO_KEEP_BEFORE_SUMMARY=6 # Keep last N user/assistant turns before summarizing older parts

# --- Tool Configuration ---
# Default timeout for commands executed by CommandLineTool (in seconds)
DEFAULT_COMMAND_TIMEOUT=300
# Whether to require user confirmation before executing commands suggested by AI
REQUIRE_COMMAND_CONFIRMATION="True" # Set to "False" to disable

# --- Logging Configuration ---
LOG_FILE_PATH="logs/kali_ai_tool.log"
LOG_LEVEL_FILE="DEBUG"    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL_CONSOLE="WARNING" # Typically INFO or WARNING for console


Important:
Replace placeholder values with your actual API keys.
For Google Search, you need to create a Programmable Search Engine and get its ID (GOOGLE_CSE_ID).
Keep your .env file secure and do not commit it to public repositories.
Running the Tool
Once the setup is complete, you can run the assistant from the project's root directory (kali_ai_assistant/):
python kali_ai_tool.py


You can then start interacting with the AI by typing your requests or
