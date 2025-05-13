# tools/base_tool.py
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    """
    def __init__(self, name, description):
        """
        Initializes the tool.
        Args:
            name (str): The name of the tool (should match what AI uses).
            description (str): A brief description of what the tool does.
        """
        self.name = name
        self.description = description
        self.interrupted = False # Flag for interruption

    def set_interrupted(self, interrupted_status):
        """Sets the interruption status."""
        self.interrupted = interrupted_status

    @abstractmethod
    def execute(self, arguments: dict) -> str:
        """
        Executes the tool with the given arguments.
        Args:
            arguments (dict): A dictionary of arguments for the tool,
                              as specified by the AI.
        Returns:
            str: The output or result of the tool execution.
                 This will be fed back to the AI as an "Observation".
        """
        pass

    def get_tool_info(self) -> dict:
        """
        Returns information about the tool.
        Can be used to dynamically provide tool info to the AI if needed.
        """
        return {
            "name": self.name,
            "description": self.description,
            # Potentially add argument schema here in the future
        }

