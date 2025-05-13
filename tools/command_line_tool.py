# tools/command_line_tool.py
import subprocess
import shlex
import threading
import time
import select
import os
import fcntl # For non-blocking reads on POSIX systems (Linux/macOS)

from .base_tool import BaseTool
import config # Import from the root directory's config.py
import logging

logger = logging.getLogger(f"{config.SERVICE_NAME}.CommandLineTool")

# Store active processes, mapping a unique ID (e.g. command string or a generated UUID) to the process object and its state
# This is a simple in-memory store. For more robust applications, consider a more persistent or managed approach.
# For now, we'll handle one interactive command at a time within the tool's execution context.
# A more advanced system might assign PIDs or job IDs that the AI can refer to.
# Current approach: if a command is interactive, the 'execute' method will manage its lifecycle
# or return a state indicating it's ongoing.

class CommandLineTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="command_line",
            description="Executes a shell command on the Kali Linux system. Can be interactive."
        )
        self.active_process = None
        self.process_output_buffer = []
        self.process_error_buffer = []
        self.process_lock = threading.Lock() # To protect access to process and buffers

    def _read_from_pipe(self, pipe, buffer_list, pipe_name):
        """Helper to read non-blockingly from a pipe and store in buffer."""
        try:
            # Set the pipe to non-blocking mode
            fd = pipe.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            while True:
                try:
                    # Read up to 1KB at a time. Adjust as needed.
                    data = pipe.read(1024)
                    if not data: # EOF
                        # logger.debug(f"EOF reached for {pipe_name}")
                        break 
                    # logger.debug(f"Read from {pipe_name}: {data.strip()}")
                    with self.process_lock:
                        buffer_list.append(data)
                except BlockingIOError: # No data available right now
                    # logger.debug(f"BlockingIOError on {pipe_name}, no data currently.")
                    break # Exit loop if no data, will be called again
                except Exception as e:
                    logger.error(f"Error reading from {pipe_name}: {e}", exc_info=True)
                    break
        except Exception as e:
            # This might happen if the pipe is closed unexpectedly
            logger.warning(f"Exception in _read_from_pipe for {pipe_name}: {e}")


    def _get_buffered_output(self) -> str:
        """Returns and clears the current output and error buffers."""
        output_str = ""
        with self.process_lock:
            if self.process_output_buffer:
                output_str += "STDOUT:\n" + "".join(self.process_output_buffer)
                self.process_output_buffer.clear()
            if self.process_error_buffer:
                output_str += "STDERR:\n" + "".join(self.process_error_buffer)
                self.process_error_buffer.clear()
        return output_str.strip() if output_str else ""


    def execute(self, arguments: dict) -> str:
        """
        Executes a shell command, potentially interactively.
        Args:
            arguments (dict):
                'command' (str): The command to execute.
                'timeout' (int, optional): Timeout in seconds. Uses config.DEFAULT_COMMAND_TIMEOUT if not set.
                'initial_input' (str, optional): Initial input to send to the command's stdin.
                'stdin_input' (str, optional): Input to send to an already running interactive command.
                                                If 'stdin_input' is present, 'command' is ignored (assumes process is active).
                'terminate_interactive' (bool, optional): If true, attempts to terminate the current active interactive command.

        Returns:
            str: Output from the command, or status messages.
                 If a command is interactive and ongoing, it might return partial output and indicate it's still running.
        """
        if self.interrupted:
            if self.active_process:
                logger.info("Interrupt received, attempting to terminate active command.")
                self.active_process.terminate()
                try:
                    self.active_process.wait(timeout=5) # Wait for termination
                except subprocess.TimeoutExpired:
                    self.active_process.kill()
                self.active_process = None
                return "Command execution interrupted by user and process terminated."
            return "Command execution interrupted by user before start."

        timeout = arguments.get("timeout", config.DEFAULT_COMMAND_TIMEOUT)
        command_str = arguments.get("command")
        initial_input = arguments.get("initial_input")
        stdin_input = arguments.get("stdin_input")
        terminate_interactive = arguments.get("terminate_interactive", False)

        with self.process_lock: # Ensure thread-safe access to self.active_process
            # Handle termination request
            if terminate_interactive:
                if self.active_process:
                    logger.info(f"Received request to terminate interactive command (PID: {self.active_process.pid}).")
                    self.active_process.terminate()
                    try:
                        self.active_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.active_process.kill()
                    self.active_process = None
                    self.process_output_buffer.clear()
                    self.process_error_buffer.clear()
                    return "Interactive command terminated by request."
                else:
                    return "No active interactive command to terminate."

            # Handle input to an existing interactive command
            if stdin_input is not None:
                if self.active_process and self.active_process.poll() is None: # Process is still running
                    try:
                        logger.info(f"Sending to STDIN of PID {self.active_process.pid}: {stdin_input}")
                        self.active_process.stdin.write(stdin_input + '\n') # Add newline typically expected by CLIs
                        self.active_process.stdin.flush()
                        
                        # Give the process a moment to react and produce output
                        time.sleep(0.5) # Adjust as needed, or implement more robust non-blocking read after write

                        # Read any immediate output
                        self._read_from_pipe(self.active_process.stdout, self.process_output_buffer, "stdout")
                        self._read_from_pipe(self.active_process.stderr, self.process_error_buffer, "stderr")
                        
                        buffered_output = self._get_buffered_output()
                        
                        if self.active_process.poll() is not None: # Process ended after input
                            exit_code = self.active_process.returncode
                            self.active_process = None # Clear active process
                            return f"{buffered_output}\nProcess finished with exit code: {exit_code}".strip()
                        else:
                            if buffered_output:
                                return f"{buffered_output}\nProcess is still running. Ready for more input or output.".strip()
                            else:
                                return "Input sent. Process is still running. Waiting for more output or ready for input.".strip()

                    except BrokenPipeError:
                        logger.warning(f"BrokenPipeError writing to stdin of PID {self.active_process.pid}. Process likely terminated.")
                        self.active_process.poll() # Update status
                        exit_code = self.active_process.returncode
                        self.active_process = None
                        return f"{self._get_buffered_output()}\nProcess terminated unexpectedly (BrokenPipeError). Exit code: {exit_code}".strip()
                    except Exception as e:
                        logger.error(f"Error writing to stdin or reading output for PID {self.active_process.pid}: {e}", exc_info=True)
                        return f"Error interacting with active process: {e}"
                else:
                    self.active_process = None # Clear if it was set but process is dead
                    return "Error: No active interactive command to send input to, or process has ended."

            # Start a new command
            if not command_str:
                return "Error: 'command' argument is missing for command_line tool."

            if self.active_process and self.active_process.poll() is None:
                return f"Error: Another command (PID: {self.active_process.pid}) is still active. Terminate it or provide stdin_input."

            logger.info(f"Executing command: {command_str} with timeout: {timeout}s")
            self.process_output_buffer.clear()
            self.process_error_buffer.clear()

            try:
                # For interactive processes, use Popen with pipes for stdin, stdout, stderr
                self.active_process = subprocess.Popen(
                    command_str,
                    shell=True, # shell=True is a security risk if command_str is not carefully vetted.
                                # Consider shlex.split(command_str) and shell=False for safer execution
                                # if complex shell features (pipes, redirects in the command_str itself) are not needed.
                                # The system prompt already mandates user confirmation for risky commands.
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1, # Line-buffered
                    universal_newlines=True # Ensures text mode
                )
                logger.info(f"Command '{command_str}' started with PID: {self.active_process.pid}")

                # Send initial input if provided
                if initial_input:
                    logger.info(f"Sending initial input to PID {self.active_process.pid}: {initial_input}")
                    try:
                        self.active_process.stdin.write(initial_input + '\n')
                        self.active_process.stdin.flush()
                    except BrokenPipeError:
                        logger.warning(f"BrokenPipeError on initial input for PID {self.active_process.pid}. Process may have exited early.")
                        # Process might have exited after initial input, check status below
                
                # Non-blocking read of initial output (if any)
                # This is a simplified approach. A more robust solution would use select() or threads for continuous monitoring.
                time.sleep(0.2) # Brief pause to allow process to start and output something
                self._read_from_pipe(self.active_process.stdout, self.process_output_buffer, "stdout")
                self._read_from_pipe(self.active_process.stderr, self.process_error_buffer, "stderr")
                initial_buffered_output = self._get_buffered_output()

                # Check if process ended quickly (e.g., simple non-interactive command or error)
                if self.active_process.poll() is not None:
                    # Process has finished, collect all output
                    stdout, stderr = self.active_process.communicate(timeout=timeout) # communicate will read remaining
                    if stdout: self.process_output_buffer.append(stdout)
                    if stderr: self.process_error_buffer.append(stderr)
                    
                    final_output = self._get_buffered_output()
                    exit_code = self.active_process.returncode
                    self.active_process = None # Clear active process
                    return f"{initial_buffered_output}\n{final_output}\nProcess finished with exit code: {exit_code}".strip()
                else:
                    # Process is still running, likely interactive or long-running
                    if initial_buffered_output:
                         return f"{initial_buffered_output}\nProcess is running (PID: {self.active_process.pid}). Ready for input or will provide more output.".strip()
                    else:
                         return f"Process started (PID: {self.active_process.pid}) and is running. Ready for input or will provide output.".strip()

            except subprocess.TimeoutExpired:
                logger.warning(f"Command '{command_str}' timed out after {timeout} seconds.")
                if self.active_process:
                    self.active_process.kill() # Ensure process is killed
                    self.active_process.communicate() # Clean up pipes
                self.active_process = None
                return f"{self._get_buffered_output()}\nError: Command '{command_str}' timed out after {timeout} seconds."
            except FileNotFoundError:
                logger.error(f"Command not found: {command_str.split()[0] if command_str else 'N/A'}")
                self.active_process = None
                return f"Error: Command not found. Make sure it's in PATH or provide the full path: {command_str.split()[0] if command_str else 'N/A'}"
            except PermissionError:
                logger.error(f"Permission denied for command: {command_str}")
                self.active_process = None
                return f"Error: Permission denied to execute command: {command_str}"
            except Exception as e:
                logger.error(f"Error executing command '{command_str}': {e}", exc_info=True)
                if self.active_process: # Try to clean up if process was started
                    self.active_process.kill()
                    self.active_process.communicate()
                self.active_process = None
                return f"Error executing command '{command_str}': {str(e)}"

if __name__ == '__main__':
    # Setup basic logging for the test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger_cmd = logging.getLogger(f"{config.SERVICE_NAME}.CommandLineToolTest")
    config.DEFAULT_COMMAND_TIMEOUT = 10 # Short timeout for tests

    cmd_tool = CommandLineTool()
    
    test_logger_cmd.info("\n--- Test 1: Simple non-interactive command (ls -l) ---")
    args1 = {"command": "ls -l"}
    result1 = cmd_tool.execute(args1)
    test_logger_cmd.info(f"Result 1:\n{result1}")

    test_logger_cmd.info("\n--- Test 2: Command with error (nonexistentcommand) ---")
    args2 = {"command": "nonexistentcommand"}
    result2 = cmd_tool.execute(args2)
    test_logger_cmd.info(f"Result 2:\n{result2}")

    test_logger_cmd.info("\n--- Test 3: Interactive command (python script) ---")
    # Create a dummy interactive script for testing
    interactive_script_content = """
print("Hello from interactive script!")
name = input("What is your name? ")
print(f"Nice to meet you, {name}!")
color = input("What is your favorite color? ")
print(f"Ah, {color} is a great color!")
print("Script finished.")
"""
    with open("test_interactive.py", "w") as f:
        f.write(interactive_script_content)

    test_logger_cmd.info("Starting interactive script...")
    args3_start = {"command": "python test_interactive.py"}
    result3_start = cmd_tool.execute(args3_start)
    test_logger_cmd.info(f"Initial output:\n{result3_start}")

    if "Process is running" in result3_start:
        test_logger_cmd.info("Sending name 'Alice'...")
        args3_input1 = {"stdin_input": "Alice"}
        result3_input1 = cmd_tool.execute(args3_input1)
        test_logger_cmd.info(f"Output after 'Alice':\n{result3_input1}")

        if "Process is running" in result3_input1 or "Nice to meet you, Alice!" in result3_input1:
            test_logger_cmd.info("Sending color 'Blue'...")
            args3_input2 = {"stdin_input": "Blue"}
            result3_input2 = cmd_tool.execute(args3_input2)
            test_logger_cmd.info(f"Output after 'Blue':\n{result3_input2}")
        
        # Optionally, terminate if still running
        if cmd_tool.active_process and cmd_tool.active_process.poll() is None:
            test_logger_cmd.info("Terminating script...")
            args3_term = {"terminate_interactive": True}
            result3_term = cmd_tool.execute(args3_term)
            test_logger_cmd.info(f"Termination result: {result3_term}")
    else:
        test_logger_cmd.warning("Interactive script did not seem to start correctly or finished too quickly.")

    # Clean up dummy script
    try:
        os.remove("test_interactive.py")
    except OSError:
        pass
    
    test_logger_cmd.info("\n--- Test 4: Timeout test ---")
    # Command that sleeps longer than timeout
    args4 = {"command": "sleep 5", "timeout": 2}
    result4 = cmd_tool.execute(args4)
    test_logger_cmd.info(f"Result 4 (timeout):\n{result4}")

    test_logger_cmd.info("\n--- Test 5: Command with initial input ---")
    # Recreate script that expects immediate input
    script_initial_input = """
data = input()
print(f"Script received: {data}")
"""
    with open("test_initial_input.py", "w") as f:
        f.write(script_initial_input)
    
    args5 = {"command": "python test_initial_input.py", "initial_input": "HelloFromInitial"}
    result5 = cmd_tool.execute(args5)
    test_logger_cmd.info(f"Result 5 (initial input):\n{result5}")

    try:
        os.remove("test_initial_input.py")
    except OSError:
        pass

    test_logger_cmd.info("All tests finished.")
