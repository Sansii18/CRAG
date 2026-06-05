# During development it is essential to know which queries triggered web search, how long Qdrant took, whether an API call failed. Python has a built-in logging module but it requires 10+ lines of boilerplate to configure properly. loguru does the same thing in 3 lines with a much better API.

from loguru import logger
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger.add(
    f"{log_dir}/CRAG.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
    rotation="500 MB",
    retention="10 days"
)

logger.add(
    lambda msg: print(msg, end=""),
    format="{time:HH:mm:ss} | {level: <8} | {message}",
    level="INFO"
)