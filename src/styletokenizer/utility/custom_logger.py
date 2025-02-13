"""
    setting for custom logger
"""
import logging
import sys

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler to output logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


def log_and_flush(message, level=logging.DEBUG):
    """
    Log a message and flush the console handler immediately.

    Args:
        message (str): The message to be logged.
        level (int): The logging level (default is logging.DEBUG).
    """
    logger.log(level, message)
    # Flush the console handler to ensure the message is written immediately
    console_handler.flush()
