# utils/logger_setup.py
import logging
import sys
import os # For creating log directory

def setup_logging(log_file_path="logs/kali_ai_tool.log",
                  log_level_file=logging.DEBUG,
                  log_level_console=logging.INFO,
                  log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                  service_name="KaliAIAssistant"):
    """
    Sets up logging for the application.

    Args:
        log_file_path (str): Path to the log file.
        log_level_file (int): Logging level for the file handler.
        log_level_console (int): Logging level for the console handler.
        log_format (str): Format string for log messages.
        service_name (str): The root logger name.
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.DEBUG)  # Set root logger to lowest level to capture all messages

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)
            # Fallback to current directory if log dir creation fails
            log_file_path = os.path.basename(log_file_path)


    # File Handler
    try:
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(log_level_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not set up file logging to {log_file_path}: {e}", file=sys.stderr)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Use sys.stdout for console
    ch.setLevel(log_level_console)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Add a filter to prevent duplicate logs if a handler is added multiple times
    # (though typically we ensure handlers are added only once)
    # logger.addFilter(lambda record: record.levelno >= logger.getEffectiveLevel())


    # Example: Get a logger for a specific module
    # module_logger = logging.getLogger(f"{service_name}.module_name")
    # module_logger.info("This is a test from a module.")

    logger.info(f"Logging setup complete. Console level: {logging.getLevelName(log_level_console)}, File level: {logging.getLevelName(log_level_file)} at {log_file_path}")
    return logger

if __name__ == '__main__':
    # Example of how to use it:
    # In your main script:
    # from utils.logger_setup import setup_logging
    # main_logger = setup_logging(service_name="MyTestApp")
    # main_logger.debug("This is a debug message.")
    # main_logger.info("This is an info message.")
    # main_logger.warning("This is a warning message.")
    # main_logger.error("This is an error message.")
    # main_logger.critical("This is a critical message.")

    # To get a logger in other modules:
    # import logging
    # module_logger = logging.getLogger("MyTestApp.module_name") # Use the same service_name prefix
    # module_logger.info("Hello from module!")
    
    # Test the setup
    test_logger = setup_logging(log_file_path="test_app.log", service_name="TestApp")
    test_logger.info("Test logger initialized.")
    
    sub_logger = logging.getLogger("TestApp.submodule")
    sub_logger.debug("This is a debug message from submodule, should go to file but not console by default.")
    sub_logger.info("This is an info message from submodule.")
