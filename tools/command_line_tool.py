# tools/command_line_tool.py
import subprocess
# import shlex # Not currently used due to shell=True, but good for future
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
            if pipe:
                try: pipe.close()
                except Exception: pass # Ignore errors on close
            q.put(None) # Signal EOF
            logger.debug(f"Reader thread for {pipe_name} finished and put None marker.")

    def _get_queued_output(self, clear_eof_markers=True) -> tuple[str, bool, bool]:
        output_parts = []
        stdout_eof, stderr_eof = False, False
        # ... (rest of _get_queued_output logic from previous version) ...
        temp_stdout_lines = []
        while True:
            try:
                line = self.stdout_q.get_nowait()
                if line is None:
                    stdout_eof = True
                    if not clear_eof_markers: self.stdout_q.put(None)
                    break
                temp_stdout_lines.append(line)
            except queue.Empty: break
        if temp_stdout_lines: output_parts.append("STDOUT:\n" + "".join(temp_stdout_lines))

        temp_stderr_lines = []
        while True:
            try:
                line = self.stderr_q.get_nowait()
                if line is None:
                    stderr_eof = True
                    if not clear_eof_markers: self.stderr_q.put(None)
                    break
                temp_stderr_lines.append(line)
            except queue.Empty: break
        if temp_stderr_lines: output_parts.append("STDERR:\n" + "".join(temp_stderr_lines))
        return "".join(output_parts).strip(), stdout_eof, stderr_eof


    def _start_process(self, command_str, initial_input_str):
        self.stdout_q = queue.Queue()
        self.stderr_q = queue.Queue()
        self.process_threads = []
        
        self.active_process = subprocess.Popen(
            command_str, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, errors='replace'
        )
        logger.info(f"Command '{command_str}' started with PID: {self.active_process.pid}")

        t_stdout = threading.Thread(target=self._reader_thread, args=(self.active_process.stdout, self.stdout_q, "stdout"), daemon=True)
        t_stderr = threading.Thread(target=self._reader_thread, args=(self.active_process.stderr, self.stderr_q, "stderr"), daemon=True)
        self.process_threads.extend([t_stdout, t_stderr])
        t_stdout.start(); t_stderr.start()

        if initial_input_str:
            try:
                logger.info(f"Sending initial input to PID {self.active_process.pid}: {initial_input_str}")
                self.active_process.stdin.write(initial_input_str + '\n')
                self.active_process.stdin.flush()
            except Exception as e: # Catch BrokenPipeError and others
                logger.warning(f"Error sending initial input for PID {self.active_process.pid}: {e}")


    def _terminate_active_process(self, message_prefix="Process"):
        # ... (Same as previous version, ensures graceful termination) ...
        if self.active_process:
            pid = self.active_process.pid 
            logger.info(f"{message_prefix} (PID: {pid}) is being terminated.")
            if self.active_process.stdin and not self.active_process.stdin.closed:
                try: self.active_process.stdin.close()
                except Exception: pass
            if self.active_process.poll() is None:
                self.active_process.terminate()
                try: self.active_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.active_process.kill()
                    try: self.active_process.wait(timeout=2)
                    except subprocess.TimeoutExpired: logger.error(f"Process {pid} failed to be killed.")
            for t in self.process_threads:
                if t.is_alive(): t.join(timeout=1.0)
            self.active_process = None
            self.process_threads = []
            logger.info(f"{message_prefix} (PID: {pid}) termination sequence complete.")


    def execute(self, arguments: dict) -> str:
        with self.process_lock:
            # ... (interrupt handling, terminate_interactive, stdin_input logic mostly same as previous) ...
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
                        time.sleep(0.3) 
                        output_chunk, _, _ = self._get_queued_output(clear_eof_markers=False)
                        
                        if self.active_process.poll() is not None:
                            exit_code = self.active_process.returncode
                            self._terminate_active_process("Process (finished after stdin_input)")
                            final_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                            return f"{output_chunk}\n{final_output}\nProcess finished with exit code: {exit_code}".strip()
                        # Enhanced message for AI
                        return f"{output_chunk}\nProcess is still running (PID: {self.active_process.pid}). You can send more input via 'stdin_input', terminate with 'terminate_interactive: true', or wait.".strip()
                    except Exception as e: # Catch BrokenPipe and others
                        logger.warning(f"Error during stdin_input for PID {self.active_process.pid}: {e}")
                        exit_code = self.active_process.poll() if self.active_process else "N/A"
                        self._terminate_active_process(f"Process (error on stdin_input, PID: {self.active_process.pid if self.active_process else 'Unknown'})")
                        final_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        return f"{final_output}\nProcess interaction error. Exit code: {exit_code}. Error: {e}".strip()
                else: 
                    if self.active_process : self._terminate_active_process("Process (found dead before stdin_input)")
                    return "Error: No active interactive command to send input to, or process has ended."

            if not command_str: return "Error: 'command' argument is missing."
            if self.active_process and self.active_process.poll() is None:
                # Enhanced error message for AI
                return f"Error: Another command (PID: {self.active_process.pid}) is still active. You must manage it (send input via 'stdin_input' or terminate with 'terminate_interactive: true') before starting a new command."

            try:
                self._start_process(command_str, initial_input)
                start_time = time.time()
                accumulated_output_parts = []
                
                while True:
                    if self.interrupted:
                        self._terminate_active_process("Process (interrupted during exec loop)")
                        current_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        accumulated_output_parts.append(current_output)
                        return f"Command interrupted.\n{''.join(accumulated_output_parts)}".strip()

                    process_status = self.active_process.poll()
                    output_chunk, _, _ = self._get_queued_output(clear_eof_markers=(process_status is not None))

                    if output_chunk:
                        accumulated_output_parts.append(output_chunk)

                    if process_status is not None:
                        logger.info(f"Process {self.active_process.pid} finished with exit code: {process_status}.")
                        for t in self.process_threads:
                            if t.is_alive(): t.join(timeout=0.5)
                        final_bits, _, _ = self._get_queued_output(clear_eof_markers=True)
                        if final_bits: accumulated_output_parts.append(final_bits)
                        self.active_process = None; self.process_threads = []
                        return f"{''.join(accumulated_output_parts)}\nProcess finished with exit code: {process_status}".strip()

                    if output_chunk: # Process running, has output
                        # Enhanced message for AI
                        return f"{''.join(accumulated_output_parts)}\nProcess is still running (PID: {self.active_process.pid}). You can send input via 'stdin_input', terminate with 'terminate_interactive: true', or wait by timing yourself out.".strip()
                    
                    if time.time() - start_time > timeout_duration:
                        self._terminate_active_process(f"Process (timed out after {timeout_duration}s)")
                        current_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                        accumulated_output_parts.append(current_output)
                        return f"Error: Command '{command_str}' timed out.\n{''.join(accumulated_output_parts)}".strip()
                    
                    time.sleep(0.1)
            except Exception as e: # Catch FileNotFoundError, PermissionError, etc.
                logger.error(f"Error executing command '{command_str}': {e}", exc_info=True)
                if self.active_process: self._terminate_active_process("Process (error during execution init)")
                error_output, _, _ = self._get_queued_output(clear_eof_markers=True)
                return f"Error executing command '{command_str}': {str(e)}\n{error_output}".strip()

# if __name__ == '__main__':
# ... (Test code can be re-added here. Ensure it tests the new observation messages)
