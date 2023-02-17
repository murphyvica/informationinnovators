"""
Initializes a logger with console and file output logging. The default log
level is INFO.
"""
import logging

from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

file = logging.FileHandler(f'{datetime.now().strftime("%Y-%m-%d")}.log')

console_formatter = logging.Formatter('%(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file.setFormatter(file_formatter)
console.setFormatter(console_formatter)

logger.addHandler(console)
logger.addHandler(file)
