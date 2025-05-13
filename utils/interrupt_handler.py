# utils/interrupt_handler.py
import signal
import sys

class InterruptHandler:
    def __init__(self):
        self.interrupted = False
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        """
        Handles SIGINT (Ctrl+C). Sets the interrupted flag and exits if pressed again.
        """
        if self.interrupted: # Second Ctrl+C
            print("\nExiting immediately...")
            if self._original_sigint_handler:
                 signal.signal(signal.SIGINT, self._original_sigint_handler) # Restore before exit
            sys.exit(1)
        
        self.interrupted = True
        print("\nInterrupt signal received. Finishing current operation or press Ctrl+C again to exit.")
        # The flag `self.interrupted` should be checked by long-running operations.

    def reset(self):
        """Resets the interrupted state."""
        self.interrupted = False

    def is_interrupted(self) -> bool:
        """Checks if an interrupt has been signalled."""
        return self.interrupted
    
    def __del__(self):
        # Restore original SIGINT handler when object is deleted (e.g. program exit)
        if self._original_sigint_handler:
            try:
                current_handler = signal.getsignal(signal.SIGINT)
                if current_handler == self.handle_interrupt: # only restore if it's still our handler
                    signal.signal(signal.SIGINT, self._original_sigint_handler)
            except Exception as e:
                # This can happen during interpreter shutdown
                # print(f"Note: Could not restore SIGINT handler: {e}")
                pass


# Global instance (optional, can be managed within the main app)
# interrupt_handler_instance = InterruptHandler()
