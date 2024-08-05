# custom_logger.py
import logging
import sys

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler to output logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a file handler to output logs to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create formatter and add it to both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def log_and_flush(message, level=logging.DEBUG):
    """
    Log a message and flush the handlers immediately.

    Args:
        message (str): The message to be logged.
        level (int): The logging level (default is logging.DEBUG).
    """
    logger.log(level, message)
    # Flush each handler to ensure the message is written immediately
    for handler in logger.handlers:
        handler.flush()
