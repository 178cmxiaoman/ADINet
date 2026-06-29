import os
import logging
from datetime import datetime


def setup_logger(log_dir="logs", task_type="Train", data_type="", extra_log_dirs=None):
    """
    Configure console and file logging for one experiment run.

    Args:
        log_dir: Primary log directory.
        task_type: Experiment stage, such as "Train" or "Test".
        data_type: Dataset split name, such as "HR-single".
        extra_log_dirs: Optional additional directories for mirrored logs.

    Returns:
        Configured logger.
    """
    log_dirs = [log_dir]
    if extra_log_dirs:
        log_dirs.extend(extra_log_dirs)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if data_type:
        log_filename = f"{task_type}-{data_type}-{current_time}.log"
    else:
        log_filename = f"{task_type}-{current_time}.log"

    handlers = [logging.StreamHandler()]
    for directory in dict.fromkeys(log_dirs):
        os.makedirs(directory, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(directory, log_filename), mode="w", encoding="utf-8"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)
    return logger
