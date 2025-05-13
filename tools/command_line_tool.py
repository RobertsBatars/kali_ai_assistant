# tools/command_line_tool.py
import subprocess
import shlex # Keep for potential future use if not using shell=True
import threading
import time
import queue # For thread-safe communication from reader threads
import os

from .base_tool import BaseTool
import config
import logging

logger = logging.getLogger(f"{config.SERVICE_NAME}.CommandLineTool")

class CommandLineTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="command_line",
            description="Executes a shell command on the Kali Linux system. Can be interactive."
        )
        self.active_process = None
        self.process_lock = threading.Lock() # Protects self.active_process
        
        # Queues for thread-safe collection of stdout/stderr
        self.stdout_q = queue.Queue()
        self.stderr_q = queue.Queue()
        self.process_threads = []

    def _reader_thread(self, pipe, q, pipe_name):
        """Reads lines from a pipe and puts them into a queue."""
        try:
            for line in iter(pipe.readline, ''):
                if self.interrupted: # Check if main execution was interrupted
                    logger.debug(f"Reader thread for {pipe_name} stopping due to interruption flag.")
                    break
                # logger.debug(f"Pipe {pipe_name} read: {line.strip()}")
                q.put(line)
        except Exception as e:
            # This can happen if the pipe is closed abruptly, e.g., process killed
            logger.warning(f"Exception in reader thread for {pipe_name}: {e}")
        finally:
            pipe.close() # Important to close the pipe from this thread's perspective
            q.put(None) # Signal EOF for this pipe by putting None in the queue
            logger.debug(f"Reader thread for {pipe_name} finished and put None marker.")

    def _get_queued_output(self, clear_queues=True) -> str:
        """Retrieves all current output from stdout and stderr queues."""
        output_parts = []

        # Process stdout
        # logger.debug(f"Checking stdout_q (approx size: {self.stdout_q.qsize()})")
        temp_stdout_lines = []
        while True:
            try:
                line = self.stdout_q.get_nowait()
                if line is None: # EOF marker for this queue
                    if clear_queues: self.stdout_q.put(None) # Put it back if we're not clearing, for next check
                    # logger.debug("None marker found in stdout_q.")
                    break
                temp_stdout_lines.append(line)
            except queue.Empty:
                # logger.debug("stdout_q is empty.")
                break
        if temp_stdout_lines:
            output_parts.append("STDOUT:\n" + "".join(temp_stdout_lines))

        # Process stderr
        # logger.debug(f"Checking stderr_q (approx size: {self.stderr_q.qsize()})")
        temp_stderr_lines = []
        while True:
            try:
                line = self.stderr_q.get_nowait()
                if line is None: # EOF marker
                    if clear_queues: self.stderr_q.put(None)
                    # logger.debug("None marker found in stderr_q.")
                    break
                temp_stderr_lines.append(line)
            except queue.Empty:
                # logger.debug("stderr_q is empty.")
                break
        if temp_stderr_lines:
            output_parts.append("STDERR:\n" + "".join(temp_stderr_lines))
        
        return "".join(output_parts).strip() if output_parts else ""

    def _start_process(self, command_str, initial_input_str, timeout):
        """Starts a new process and its reader threads."""
        # Clear previous queues and threads if any (shouldn't happen if logic is correct)
        self.stdout_q = queue.Queue()
        self.stderr_q = queue.Queue()
        self.process_threads = []

        self.active_process = subprocess.Popen(
            command_str,
            shell=True, # Ensure this is understood; user confirmation is key.
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        logger.info(f"Command '{command_str}' started with PID: {self.active_process.pid}")

        # Start reader threads for stdout and stderr
        t_stdout = threading.Thread(target=self._reader_thread, args=(self.active_process.stdout, self.stdout_q, "stdout"), daemon=True)
        t_stderr = threading.Thread(target=self._reader_thread, args=(self.active_process.stderr, self.stderr_q, "stderr"), daemon=True)
        self.process_threads.extend([t_stdout, t_stderr])
        t_stdout.start()
        t_stderr.start()

        if initial_input_str:
            try:
                logger.info(f"Sending initial input to PID {self.active_process.pid}: {initial_input_str}")
                self.active_process.stdin.write(initial_input_str + '\n')
                self.active_process.stdin.flush()
            except BrokenPipeError:
                logger.warning(f"BrokenPipeError on initial input for PID {self.active_process.pid}. Process may have exited early.")
            except Exception as e:
                logger.error(f"Error sending initial input: {e}", exc_info=True)


    def _terminate_active_process(self, message_prefix="Process"):
        """Safely terminates the active process and joins reader threads."""
        if self.active_process:
            logger.info(f"{message_prefix} (PID: {self.active_process.pid}) is being terminated.")
            if self.active_process.stdin:
                try:
                    self.active_process.stdin.close() # Close stdin to signal no more input
                except Exception as e:
                    logger.warning(f"Error closing stdin for PID {self.active_process.pid}: {e}")
            
            self.active_process.terminate() # Send SIGTERM
            try:
                self.active_process.wait(timeout=5) # Wait for graceful termination
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {self.active_process.pid} did not terminate gracefully, killing.")
                self.active_process.kill() # Send SIGKILL
                try:
                    self.active_process.wait(timeout=2) # Wait for kill
                except subprocess.TimeoutExpired:
                    logger.error(f"Process {self.active_process.pid} failed to be killed.")

            # Wait for reader threads to finish (they should exit once pipes are closed)
            for t in self.process_threads:
                t.join(timeout=2) # Give threads time to exit
                if t.is_alive():
                    logger.warning(f"Reader thread {t.name} did not exit cleanly.")
            
            self.active_process = None
            self.process_threads = []
            logger.info(f"{message_prefix} terminated.")


    def execute(self, arguments: dict) -> str:
        with self.process_lock: # Ensure only one command execution logic path runs at a time
            if self.interrupted: # Check at the beginning of execution
                if self.active_process and self.active_process.poll() is None:
                    self._terminate_active_process("Active process (interrupted)")
                    return "Command execution interrupted by user and active process terminated."
                return "Command execution interrupted by user before start."

            timeout_duration = arguments.get("timeout", config.DEFAULT_COMMAND_TIMEOUT)
            command_str = arguments.get("command")
            initial_input = arguments.get("initial_input")
            stdin_input = arguments.get("stdin_input")
            terminate_interactive = arguments.get("terminate_interactive", False)

            # Handle termination request
            if terminate_interactive:
                if self.active_process:
                    self._terminate_active_process("Interactive command (terminated by request)")
                    # Get any remaining output from queues after termination
                    remaining_output = self._get_queued_output()
                    return f"Interactive command terminated by request.\n{remaining_output}".strip()
                else:
                    return "No active interactive command to terminate."

            # Handle input to an existing interactive command
            if stdin_input is not None:
                if self.active_process and self.active_process.poll() is None:
                    try:
                        logger.info(f"Sending to STDIN of PID {self.active_process.pid}: {stdin_input}")
                        self.active_process.stdin.write(stdin_input + '\n')
                        self.active_process.stdin.flush()
                        
                        time.sleep(0.3) # Give process time to react and reader threads to catch output
                        current_output = self._get_queued_output()
                        
                        if self.active_process.poll() is not None: # Process ended after input
                            self._terminate_active_process("Process (finished after input)") # Cleans up threads
                            exit_code = self.active_process.returncode if hasattr(self.active_process, 'returncode') else 'N/A'
                            return f"{current_output}\nProcess finished with exit code: {exit_code}".strip()
                        return f"{current_output}\nProcess is still running. Ready for more input or output.".strip()

                    except BrokenPipeError:
                        logger.warning(f"BrokenPipeError writing to stdin of PID {self.active_process.pid}. Process likely terminated.")
                        exit_code = self.active_process.poll() # Get exit code
                        self._terminate_active_process("Process (BrokenPipeError)")
                        return f"{self._get_queued_output()}\nProcess terminated unexpectedly (BrokenPipeError). Exit code: {exit_code}".strip()
                    except Exception as e:
                        logger.error(f"Error interacting with active process: {e}", exc_info=True)
                        self._terminate_active_process("Process (error during interaction)")
                        return f"Error interacting with active process: {e}\n{self._get_queued_output()}".strip()
                else:
                    if self.active_process: # Process exists but is not running
                         self._terminate_active_process("Process (found dead before stdin_input)")
                    return "Error: No active interactive command to send input to, or process has ended."

            # Start a new command
            if not command_str:
                return "Error: 'command' argument is missing for command_line tool."

            if self.active_process and self.active_process.poll() is None:
                return f"Error: Another command (PID: {self.active_process.pid}) is still active. Terminate it or provide stdin_input."

            try:
                self._start_process(command_str, initial_input, timeout_duration)
                
                # For non-interactive or fast commands, wait for them to finish or timeout
                # For interactive ones, this loop will also apply but might be interrupted by user/AI for more input
                start_time = time.time()
                while True:
                    if self.interrupted:
                        logger.info("Command execution loop interrupted by flag.")
                        self._terminate_active_process("Process (interrupted during execution loop)")
                        return f"Command interrupted.\n{self._get_queued_output()}".strip()

                    process_status = self.active_process.poll()
                    current_output_chunk = self._get_queued_output(clear_queues=False) # Get output without clearing EOF markers yet

                    if process_status is not None: # Process finished
                        logger.info(f"Process {self.active_process.pid} finished with exit code: {process_status}.")
                        # Wait for reader threads to complete fully by reading until None marker
                        for t in self.process_threads:
                            t.join(timeout=2.0) # Ensure threads have time to push their None marker
                        
                        final_output = self._get_queued_output(clear_queues=True) # Get all remaining output
                        self.active_process = None # Clear active process
                        self.process_threads = []
                        return f"{final_output}\nProcess finished with exit code: {process_status}".strip()

                    if current_output_chunk: # If there's output, return it and indicate it's running
                        # This is for commands that stream output but are not necessarily "interactive" for stdin
                        # Or for interactive commands showing a prompt
                        # The AI needs to decide if it's waiting for input or just seeing output
                        # We might need a small delay to batch up quick outputs
                        # For now, any output means we return and let AI decide next step
                        return f"{current_output_chunk}\nProcess is running (PID: {self.active_process.pid}). Check output for prompts or provide more input if needed.".strip()


                    if time.time() - start_time > timeout_duration:
                        logger.warning(f"Command '{command_str}' timed out after {timeout_duration} seconds.")
                        self._terminate_active_process(f"Process (timed out after {timeout_duration}s)")
                        return f"Error: Command '{command_str}' timed out.\n{self._get_queued_output()}".strip()
                    
                    # If no output and not finished and not timed out, it might be an interactive prompt
                    # or just a long-running silent command.
                    # The AI should be prompted to either provide input if it expects one, or wait, or terminate.
                    # A short sleep to prevent busy-waiting if command is silent but running.
                    # If no output after a brief period, we can assume it's waiting or done.
                    # This logic is tricky. For now, if there's no output for a bit, we return a "still running" message.
                    # The previous "if current_output_chunk" handles cases where there IS output.
                    # If we reach here, there was no new output in this check.
                    
                    # Let's check if both reader threads are done (put None in queue)
                    # This is a more reliable way to see if output is truly finished for now
                    # (assuming the command isn't just idle before producing more output later)
                    # This check is complex because queues might be temporarily empty.
                    # The poll() check is the primary for process termination.

                    time.sleep(0.2) # Polling interval

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
                if self.active_process:
                    self._terminate_active_process("Process (error during execution)")
                return f"Error executing command '{command_str}': {str(e)}\n{self._get_queued_output()}".strip()
        
if __name__ == '__main__':
    # Setup basic logging for the test
    # This setup is minimal; use the main app's logger setup for full features.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger_cmd = logging.getLogger(f"{config.SERVICE_NAME}.CommandLineToolTest")
    config.DEFAULT_COMMAND_TIMEOUT = 10 # Short timeout for tests

    cmd_tool = CommandLineTool()
    
    test_logger_cmd.info("\n--- Test 1: Simple non-interactive command (ls -la) ---")
    args1 = {"command": "ls -la"} # Using -la to ensure some output
    result1 = cmd_tool.execute(args1)
    test_logger_cmd.info(f"Result 1:\n{result1}")
    assert "Process finished with exit code: 0" in result1

    test_logger_cmd.info("\n--- Test 2: Command with error (nonexistentcommand) ---")
    args2 = {"command": "nonexistentcommand"}
    result2 = cmd_tool.execute(args2)
    test_logger_cmd.info(f"Result 2:\n{result2}")
    assert "Error: Command not found" in result2 or "exit code: 127" in result2 # Depending on shell

    test_logger_cmd.info("\n--- Test 3: Interactive command (python script) ---")
    interactive_script_content = """
import time
print("Hello from interactive script!", flush=True)
name = input("What is your name? ")
print(f"Nice to meet you, {name}!", flush=True)
time.sleep(0.1) # ensure output is flushed
color = input("What is your favorite color? ")
print(f"Ah, {color} is a great color!", flush=True)
print("Script finished.", flush=True)
"""
    with open("test_interactive.py", "w") as f:
        f.write(interactive_script_content)

    test_logger_cmd.info("Starting interactive script...")
    # Use python -u for unbuffered output from the script itself
    args3_start = {"command": "python -u test_interactive.py"}
    result3_start = cmd_tool.execute(args3_start)
    test_logger_cmd.info(f"Initial output:\n{result3_start}")
    assert "Hello from interactive script!" in result3_start
    assert "What is your name?" in result3_start # Expect prompt in output
    assert "Process is running" in result3_start


    if "Process is running" in result3_start:
        test_logger_cmd.info("Sending name 'Alice'...")
        args3_input1 = {"stdin_input": "Alice"}
        result3_input1 = cmd_tool.execute(args3_input1)
        test_logger_cmd.info(f"Output after 'Alice':\n{result3_input1}")
        assert "Nice to meet you, Alice!" in result3_input1
        assert "What is your favorite color?" in result3_input1
        assert "Process is running" in result3_input1


        if "Process is running" in result3_input1:
            test_logger_cmd.info("Sending color 'Blue'...")
            args3_input2 = {"stdin_input": "Blue"}
            result3_input2 = cmd_tool.execute(args3_input2)
            test_logger_cmd.info(f"Output after 'Blue':\n{result3_input2}")
            assert "Ah, Blue is a great color!" in result3_input2
            assert "Script finished." in result3_input2
            assert "Process finished with exit code: 0" in result3_input2
        
        if cmd_tool.active_process and cmd_tool.active_process.poll() is None: # Should be finished
            test_logger_cmd.warning("Script should have finished but reported as running. Terminating.")
            args3_term = {"terminate_interactive": True}
            cmd_tool.execute(args3_term)
    else:
        test_logger_cmd.warning("Interactive script did not seem to start/prompt correctly.")

    try:
        os.remove("test_interactive.py")
    except OSError:
        pass
    
    test_logger_cmd.info("\n--- Test 4: Timeout test ---")
    args4 = {"command": "sleep 5", "timeout": 2} # Command sleeps 5s, timeout is 2s
    result4 = cmd_tool.execute(args4)
    test_logger_cmd.info(f"Result 4 (timeout):\n{result4}")
    assert "Error: Command 'sleep 5' timed out" in result4

    test_logger_cmd.info("\n--- Test 5: Command with initial input ---")
    script_initial_input = """
data = input() # Reads one line
print(f"Script received: {data}", flush=True)
"""
    with open("test_initial_input.py", "w") as f:
        f.write(script_initial_input)
    
    args5 = {"command": "python -u test_initial_input.py", "initial_input": "HelloFromInitial"}
    result5 = cmd_tool.execute(args5)
    test_logger_cmd.info(f"Result 5 (initial input):\n{result5}")
    assert "Script received: HelloFromInitial" in result5
    assert "Process finished with exit code: 0" in result5

    try:
        os.remove("test_initial_input.py")
    except OSError:
        pass

    test_logger_cmd.info("All tests finished.")
