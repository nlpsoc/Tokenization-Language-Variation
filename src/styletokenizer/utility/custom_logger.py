# custom_logger.py
import logging
import sys

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create file handler with line buffering
file_handler = logging.FileHandler('app.log', buffering=1)  # Enable line buffering
file_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define a log_and_flush function
def log_and_flush(message, level=logging.DEBUG):
    """
    Log a message and flush the handlers immediately.

    Args:
        message (str): The message to be logged.
        level (int): The logging level (default is logging.DEBUG).
    """
    logger.log(level, message)
    for handler in logger.handlers:
        handler.flush()