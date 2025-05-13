# tools/wait_tool.py
import time
from .base_tool import BaseTool
import logging
import config

logger = logging.getLogger(f"{config.SERVICE_NAME}.WaitTool")

class WaitTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="wait",
            description="Pauses execution for a specified number of seconds. Useful for waiting for background processes or before retrying an operation."
        )

    def execute(self, arguments: dict) -> str:
        """
        Pauses execution.
        Args:
            arguments (dict): Must contain 'duration_seconds' (int or float) - the time to wait.
        Returns:
            str: A message indicating how long the tool waited.
        """
        if self.interrupted:
            return "Wait operation interrupted by user."

        duration = arguments.get("duration_seconds")
        if duration is None:
            return "Error: 'duration_seconds' argument is missing for wait tool."
        
        try:
            duration_val = float(duration)
            if duration_val <= 0:
                return "Error: 'duration_seconds' must be a positive number."
            if duration_val > 300: # Max wait of 5 minutes to prevent accidental long waits
                duration_val = 300
                logger.warning("Wait duration capped at 300 seconds.")
            
            logger.info(f"Waiting for {duration_val} seconds...")
            
            # Check for interruption periodically during the wait
            wait_interval = 0.5 # Check for interrupt every 0.5 seconds
            elapsed_time = 0
            while elapsed_time < duration_val:
                if self.interrupted:
                    logger.info("Wait interrupted during sleep.")
                    return f"Wait operation interrupted by user after approximately {elapsed_time:.1f} seconds."
                time.sleep(min(wait_interval, duration_val - elapsed_time))
                elapsed_time += wait_interval
            
            return f"Successfully waited for {duration_val} seconds."
        except ValueError:
            return "Error: 'duration_seconds' must be a valid number."
        except Exception as e:
            logger.error(f"Error during wait operation: {e}", exc_info=True)
            return f"An unexpected error occurred during wait: {str(e)}"

if __name__ == '__main__':
    # Example usage
    # Setup basic logging for the test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Mock config for service name if needed for logger
    class MockConfig: SERVICE_NAME = "TestApp"
    config = MockConfig()


    wait_tool = WaitTool()
    
    print("Testing wait tool for 3 seconds...")
    result1 = wait_tool.execute({"duration_seconds": 3})
    print(f"Result 1: {result1}")

    print("\nTesting wait tool with invalid duration...")
    result2 = wait_tool.execute({"duration_seconds": "abc"})
    print(f"Result 2: {result2}")

    print("\nTesting wait tool with missing duration...")
    result3 = wait_tool.execute({})
    print(f"Result 3: {result3}")

    print("\nTesting wait tool with excessive duration (should be capped)...")
    result4 = wait_tool.execute({"duration_seconds": 600})
    print(f"Result 4: {result4}") # Expect it to wait for 300s

    # Manual interruption test (conceptual)
    # print("\nTesting wait tool interruption (interrupt manually within 5s)...")
    # wait_tool.set_interrupted(False) # Reset if needed
    # result_interrupt = wait_tool.execute({"duration_seconds": 5})
    # print(f"Result (interrupt): {result_interrupt}")
