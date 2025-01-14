import logging
from ml_pipeline.config import Config

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_message(message):
    """Log a message."""
    logger.info(message)