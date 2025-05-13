# Kali AI Assistant

The Kali AI Assistant is an interactive command-line tool that leverages AI (Anthropic's Claude models) to assist with tasks in a Kali Linux environment, particularly focused on penetration testing, vulnerability analysis, and information gathering. It can execute commands, search the web using multiple APIs, introduce timed delays, and look up CVE information.

## Features

* **Interactive AI Chat**: Converse with an AI assistant that understands penetration testing context.
* **Command Execution**: Allows the AI to request execution of shell commands (with optional user confirmation). Supports interactive commands and configurable timeouts.
* **Web Search**: Integrated with Google, Tavily, and Brave Search APIs for information gathering.
* **CVE Lookup**: Dedicated tool for searching CVE details.
* **Wait Tool**: Allows the AI to introduce a timed pause in its execution flow.
* **Modular Tool System**: Easily extendable with new tools.
* **Context Management**: Automatic summarization of long conversations to stay within token limits.
* **Interruption Handling**: Supports Ctrl+C for interrupting ongoing operations.
* **Detailed Logging**: Logs application activity to a file for debugging and review.
* **Streaming AI Responses**: Uses streaming for AI responses to improve perceived responsiveness and handle potentially long outputs.
* **Sequential Tool Processing**: Designed to process one tool call at a time from the AI's response, ensuring better control over execution flow.

## Project Structure

```
kali_ai_assistant/
├── kali_ai_tool.py         # Main application script
├── system_prompt.txt       # Defines the AI's persona and capabilities
├── config.py               # Loads and manages configuration
├── .env                    # (You create this) Stores API keys and sensitive configs
├── .env.example            # Example environment file
├── requirements.txt        # Python dependencies
├── ai_core/                # AI interaction logic
│   └── anthropic_client.py
├── tools/                  # Tool implementations
│   ├── base_tool.py
│   ├── command_line_tool.py
│   ├── web_search_tool.py
│   ├── cve_search_tool.py
│   └── wait_tool.py
├── utils/                  # Utility modules
│   ├── interrupt_handler.py
│   ├── logger_setup.py
│   └── token_estimator.py
└── logs/                   # Directory for log files (created automatically)
```

## Setup Instructions

### 1. Prerequisites

* Python 3.9 or higher.
* `pip` for installing Python packages.
* Access to a Kali Linux environment (or a Linux system with similar tools if you adapt the use case).
* API keys for:
    * Anthropic (required for AI)
    * Google Custom Search API (optional, for Google search)
    * Tavily API (optional, for Tavily search)
    * Brave Search API (optional, for Brave search and CVE search)

### 2. Clone the Repository (or Create Files)

If this project is in a repository:
```bash
git clone <repository_url>
cd kali_ai_assistant
```
Otherwise, ensure all the provided Python files (`.py`), `system_prompt.txt` (matching the content of `system_prompt_txt_multiline_fix`), and `requirements.txt` are in the correct directory structure as shown above.

### 3. Create a Python Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate    # On Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the example environment file to a new `.env` file:
```bash
cp .env.example .env
```
Now, edit the `.env` file with your actual API keys and any custom configurations. Refer to `.env.example` for the available settings. Key variables include:
* `ANTHROPIC_API_KEY`
* `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`
* `TAVILY_API_KEY`
* `BRAVE_SEARCH_API_KEY`
* `DEFAULT_AI_MODEL`
* `MAX_AI_OUTPUT_TOKENS`
* `REQUIRE_COMMAND_CONFIRMATION`
* Logging configurations

**Important:**
* Replace placeholder values with your actual API keys.
* For Google Search, you need to create a Programmable Search Engine and get its ID (`GOOGLE_CSE_ID`).
* Keep your `.env` file secure and do not commit it to public repositories.

## Running the Tool

Once the setup is complete, you can run the assistant from the project's root directory (`kali_ai_assistant/`):
```bash
python kali_ai_tool.py
```
You can then start interacting with the AI by typing your requests or questions.

## Usage Tips

* **Be Specific**: Clearly state your goals or the information you're looking for.
* **Tool Usage**: The AI will inform you when it intends to use a tool. If `REQUIRE_COMMAND_CONFIRMATION` is true (default), you'll be asked to approve command-line executions. The AI should ideally request one tool at a time.
* **Active Commands**: Pay attention to messages indicating a command is still running. The AI is instructed to manage these by either providing input, terminating the command, or waiting (potentially using the `wait` tool).
* **Interruption**: Press `Ctrl+C` to interrupt an ongoing operation (like AI response generation or a tool running). Pressing `Ctrl+C` a second time usually exits the application.
* **Exiting**: Type `exit` or `quit` to close the assistant.
* **Troubleshooting**: Check the `logs/kali_ai_tool.log` file for detailed error messages and activity logs if you encounter issues. The console log level is set to `WARNING` by default to be less verbose; check the file for `INFO` and `DEBUG` messages.

## Extending with New Tools

1.  Create a new Python file in the `tools/` directory (e.g., `my_new_tool.py`).
2.  Define a class that inherits from `tools.base_tool.BaseTool`.
3.  Implement the `__init__` method (calling `super().__init__(name="your_tool_name", description="...")`) and the `execute(self, arguments: dict) -> str` method.
4.  In `kali_ai_tool.py`, import your new tool class.
5.  Add an instance of your new tool to the `available_tools` dictionary:
    ```python
    from tools.my_new_tool import MyNewTool
    # ...
    available_tools: dict[str, BaseTool] = {
        # ... existing tools ...
        "your_tool_name": MyNewTool(),
    }
    ```
6.  Update `system_prompt.txt` (referencing the content of `system_prompt_txt_multiline_fix`) to inform the AI about the new tool, its name, purpose, and expected arguments.

## Disclaimer

This tool is designed to assist with penetration testing and security tasks. Always ensure you have proper authorization before performing any testing on systems or networks you do not own. The user is responsible for all actions taken by or through this tool.
