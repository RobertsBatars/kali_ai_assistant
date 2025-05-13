# tools/command_line_tool.py
import subprocess
import shlex
import threading
import time
import queue
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
        self.process_lock = threading.Lock()
        self.stdout_q = queue.Queue()
        self.stderr_q = queue.Queue()
        self.process_threads = []

    def _reader_thread(self, pipe, q, pipe_name):
        try:
            for line in iter(pipe.readline, ''):
                if self.interrupted:
                    logger.debug(f"Reader thread for {pipe_name} stopping due to interruption flag.")
                    break
                q.put(line)
        except Exception as e:
            logger.warning(f"Exception in reader thread for {pipe_name}: {e}")
        finally:
            if pipe: # Ensure pipe is not already None (e.g. if process creation failed)
                try:
                    pipe.close()
                except Exception as e:
                    logger.warning(f"Error closing pipe {pipe_name} in reader thread: {e}")
            q.put(None) # Signal EOF
            logger.debug(f"Reader thread for {pipe_name} finished and put None marker.")

    def _get_queued_output(self, clear_eof_markers=True) -> tuple[str, bool, bool]:
        """
        Retrieves current output from queues.
        Returns: (output_string, stdout_eof_reached, stderr_eof_reached)
        """
        output_parts = []
        stdout_eof = False
        stderr_eof = False

        temp_stdout_lines = []
        while True:
            try:
                line = self.stdout_q.get_nowait()
                if line is None:
                    stdout_eof = True
                    if not clear_eof_markers: self.stdout_q.put(None) # Put back if not clearing
                    break
                temp_stdout_lines.append(line)
            except queue.Empty:
                break
        if temp_stdout_lines:
            output_parts.append("STDOUT:\n" + "".join(temp_stdout_lines))

        temp_stderr_lines = []
        while True:
            try:
                line = self.stderr_q.get_nowait()
                if line is None:
                    stderr_eof = True
                    if not clear_eof_markers: self.stderr_q.put(None) # Put back
                    break
                temp_stderr_lines.append(line)
            except queue.Empty:
                break
        if temp_stderr_lines:
            output_parts.append("STDERR:\n" + "".join(temp_stderr_lines))
        
        return "".join(output_parts).strip(), stdout_eof, stderr_eof

    def _start_process(self, command_str, initial_input_str):
        self.stdout_q = queue.Queue()
        self.stderr_q = queue.Queue()
        self.process_threads = []

        # Ensure active_process is None before starting a new one
        if self.active_process and self.active_process.poll() is None:
             logger.error("Attempted to start a new process while another is active. This should not happen.")
             # For safety, try to terminate existing, though this indicates a logic flaw elsewhere
             self._terminate_active_process("Stale process before new start")
        
        self.active_process = subprocess.Popen(
            command_str,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            errors='replace' # Handle potential decoding errors in output
        )
        logger.info(f"Command '{command_str}' started with PID: {self.active_process.pid}")

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
                logger.warning(f"BrokenPipeError on initial input for PID {self.active_process.pid}.")
            except Exception as e:
                logger.error(f"Error sending initial input: {e}", exc_info=True)

    def _terminate_active_process(self, message_prefix="Process"):
        if self.active_process:
            pid = self.active_process.pid # Store PID before process object is cleared
            logger.info(f"{message_prefix} (PID: {pid}) is being terminated.")
            
            # Close stdin first
            if self.active_process.stdin and not self.active_process.stdin.closed:
                try:
                    self.active_process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing stdin for PID {pid}: {e}")
            
            # Terminate and kill if necessary
            if self.active_process.poll() is None: # Check if still running
                self.active_process.terminate()
                try:
                    self.active_process.wait(timeout=3) # Reduced timeout
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {pid} did not terminate gracefully, killing.")
                    self.active_process.kill()
                    try:
                        self.active_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.error(f"Process {pid} failed to be killed.")
            
            # Wait for reader threads
            for t in self.process_threads:
                if t.is_alive():
                    t.join(timeout=1.0) # Reduced timeout
                if t.is_alive():
                    logger.warning(f"Reader thread {t.name} for PID {pid} did not exit cleanly.")
            
            self.active_process = None
            self.process_threads = []
            logger.info(f"{message_prefix} (PID: {pid}) termination sequence complete.")
        else:
            logger.debug(f"{message_prefix}: No active process to terminate.")


    def execute(self, arguments: dict) -> str:
        with self.process_lock:
            if self.interrupted:
                if self.active_process and self.active_process.poll() is None:
                    self._terminate_active_process("Active process (interrupted at execute start)")
                    return f"Command execution interrupted by user and active process terminated.\n{self._get_queued_output(clear_eof_markers=True)[0]}".strip()
                return "Command execution interrupted by user before start."

            timeout_duration = arguments.get("timeout", config.DEFAULT_COMMAND_TIMEOUT)
            command_str = arguments.get("command")
            initial_input = arguments.get("initial_input")
            stdin_input = arguments.get("stdin_input")
            terminate_interactive = arguments.get("terminate_interactive", False)

            if terminate_interactive:
                if self.active_process:
                    self._terminate_active_process("Interactive command (terminated by request)")
                    return f"Interactive command terminated by request.\n{self._get_queued_output(clear_eof_markers=True)[0]}".strip()
                return "No active interactive command to terminate."

            if stdin_input is not None:
                if self.active_process and self.active_process.poll() is None:
                    try:
                        logger.info(f"Sending to STDIN of PID {self.active_process.pid}: {stdin_input}")
                        self.active_process.stdin.write(stdin_input + '\n')
                        self.active_process.stdin.flush()
                        time.sleep(0.3) # Allow reaction
                        output_chunk, _, _ = self._get_queued_output(clear_eof_markers=False)
                        
                        if self.active_process.poll() is not None:
                            exit_code = self.active_process.returncode
                            self._terminate_active_process("Process (finished after stdin_input)")
                            final_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                            return f"{output_chunk}\n{final_output}\nProcess finished with exit code: {exit_code}".strip()
                        return f"{output_chunk}\nProcess is still running. Ready for more input or output.".strip()
                    except BrokenPipeError:
                        logger.warning(f"BrokenPipeError writing to stdin of PID {self.active_process.pid}.")
                        exit_code = self.active_process.poll()
                        self._terminate_active_process("Process (BrokenPipeError on stdin)")
                        final_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        return f"{final_output}\nProcess terminated unexpectedly (BrokenPipeError). Exit code: {exit_code}".strip()
                    except Exception as e:
                        logger.error(f"Error interacting with active process: {e}", exc_info=True)
                        self._terminate_active_process("Process (error on stdin interaction)")
                        final_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        return f"Error interacting with active process: {e}\n{final_output}".strip()
                else: # No active process or it ended
                    if self.active_process : self._terminate_active_process("Process (found dead before stdin_input)")
                    return "Error: No active interactive command to send input to, or process has ended."

            if not command_str: return "Error: 'command' argument is missing."
            if self.active_process and self.active_process.poll() is None:
                return f"Error: Another command (PID: {self.active_process.pid}) is still active."

            try:
                self._start_process(command_str, initial_input)
                start_time = time.time()
                
                # Loop to monitor process and gather output
                accumulated_output_parts = []
                while True:
                    if self.interrupted:
                        logger.info("Command execution loop interrupted by flag.")
                        self._terminate_active_process("Process (interrupted during exec loop)")
                        # Get any output gathered so far plus final remnants
                        current_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        accumulated_output_parts.append(current_output)
                        return f"Command interrupted.\n{''.join(accumulated_output_parts)}".strip()

                    process_status = self.active_process.poll()
                    output_chunk, stdout_eof, stderr_eof = self._get_queued_output(clear_eof_markers=(process_status is not None))

                    if output_chunk:
                        accumulated_output_parts.append(output_chunk)

                    if process_status is not None: # Process finished
                        logger.info(f"Process {self.active_process.pid} finished with exit code: {process_status}.")
                        # Ensure reader threads are joined
                        for t in self.process_threads:
                            if t.is_alive(): t.join(timeout=0.5) # Short join, they should be exiting
                        
                        # One final read from queues after threads are expected to be done
                        final_bits, _, _ = self._get_queued_output(clear_eof_markers=True)
                        if final_bits: accumulated_output_parts.append(final_bits)
                        
                        self.active_process = None
                        self.process_threads = []
                        return f"{''.join(accumulated_output_parts)}\nProcess finished with exit code: {process_status}".strip()

                    # If process is running and we got some output in this chunk, return intermediate
                    if output_chunk: # and process_status is None
                         return f"{''.join(accumulated_output_parts)}\nProcess is running (PID: {self.active_process.pid}). Check output for prompts or provide more input if needed.".strip()
                    
                    # If process is running, no output in this chunk, check timeout
                    if time.time() - start_time > timeout_duration:
                        logger.warning(f"Command '{command_str}' timed out after {timeout_duration} seconds.")
                        self._terminate_active_process(f"Process (timed out after {timeout_duration}s)")
                        current_output, _, _ = self._get_queued_output(clear_eof_markers=True) # Get final output
                        accumulated_output_parts.append(current_output)
                        return f"Error: Command '{command_str}' timed out.\n{''.join(accumulated_output_parts)}".strip()
                    
                    # If process running, no output, not timed out, wait briefly
                    time.sleep(0.1) # Polling interval

            except FileNotFoundError:
                logger.error(f"Command not found: {command_str.split()[0] if command_str else 'N/A'}")
                self.active_process = None # Ensure it's cleared
                return f"Error: Command not found: {command_str.split()[0] if command_str else 'N/A'}"
            except PermissionError:
                logger.error(f"Permission denied for command: {command_str}")
                self.active_process = None
                return f"Error: Permission denied to execute command: {command_str}"
            except Exception as e:
                logger.error(f"Error executing command '{command_str}': {e}", exc_info=True)
                if self.active_process: self._terminate_active_process("Process (error during execution init)")
                # Try to get any output that might have been queued before the error
                error_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                return f"Error executing command '{command_str}': {str(e)}\n{error_output}".strip()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger_cmd = logging.getLogger(f"{config.SERVICE_NAME}.CommandLineToolTest")
    config.DEFAULT_COMMAND_TIMEOUT = 10

    cmd_tool = CommandLineTool()
    
    test_logger_cmd.info("\n--- Test 1: Simple non-interactive command (uname -a) ---")
    args1 = {"command": "uname -a"}
    result1 = cmd_tool.execute(args1)
    test_logger_cmd.info(f"Result 1:\n{result1}")
    assert "Process finished with exit code: 0" in result1
    assert "Linux" in result1 # Check for actual output content

    test_logger_cmd.info("\n--- Test 1.1: Simple non-interactive command (whoami) ---")
    args1_1 = {"command": "whoami"}
    result1_1 = cmd_tool.execute(args1_1)
    test_logger_cmd.info(f"Result 1.1:\n{result1_1}")
    assert "Process finished with exit code: 0" in result1_1
    # Add an assertion for typical whoami output if possible, e.g., non-empty STDOUT before exit code line.
    assert len(result1_1.splitlines()) > 1 # Expecting at least the output and the "Process finished" line

    # ... (rest of the tests from previous version can be re-added here if needed) ...
    test_logger_cmd.info("Basic tests finished.")
