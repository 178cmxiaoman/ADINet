import os
import logging
from datetime import datetime


def setup_logger(log_dir="logs", task_type="Train", data_type="", extra_log_dirs=None):
    """
    设置日志记录器，支持动态生成日志文件名并同时输出到多个文件和控制台。

    参数:
        log_dir (str): 主日志文件存储目录，默认为 "logs"。
        task_type (str): 任务类型，如 "Train" 或 "Test"，默认为 "Train"。
        data_type (str): 数据集类型，如 "HR-single"，默认为空字符串。
        extra_log_dirs (list[str] | None): 额外日志目录列表，例如保存到配置中的 save_path。

    返回:
        logger (logging.Logger): 配置好的日志记录器。
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
