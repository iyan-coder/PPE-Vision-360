"""
logger.py

This module sets up a configurable logging system for machine learning or general Python projects.
It logs both to a file (with a timestamped filename) and to the console using a consistent format.
This logging system is reusable across the entire project.

Usage:
    from logger import logger

    logger.info("Training started")
    logger.error("Something went wrong")
"""

import logging
import os
from datetime import datetime

# --------------------------
# 1. Create logs directory
# --------------------------
# All logs will be saved in the 'logs/' folder.
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# ------------------------------------------------
# 2. Generate a unique timestamped log file name
# ------------------------------------------------
# The log file will be named like '06_02_2025_14_30_21.log'
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# --------------------------
# 3. Create a logger object
# --------------------------
# This logger can be imported and reused across all modules.
logger = logging.getLogger("ml_project_logger")
logger.setLevel(logging.INFO)  # Set logging level to INFO and above
logger.propagate = (
    False  # Avoid duplicate logs if this logger is reused in multiple modules
)

# --------------------------------------------------
# 4. Define a standard format for all log messages
# --------------------------------------------------
# Example log message:
# [2025-06-02 14:32:10] INFO [train.py:42] - Model training started
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)

# --------------------------------------------
# 5. File Handler: Send logs to a log file
# --------------------------------------------
# All logs will be stored persistently in a timestamped file
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ------------------------------------------------
# 6. Console Handler: Print logs to the terminal
# ------------------------------------------------
# Useful for real-time monitoring while running scripts
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# # Optional: Debug
# logger.info(f"Logger initialized. Logs being written to: {LOG_FILE_PATH}")