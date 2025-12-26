"""
文件路径: config/logging.py
功能: 日志系统配置
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(debug: bool = True, log_dir: str = "logs") -> None:
    """配置日志系统
    
    Args:
        debug: 是否开启调试模式
        log_dir: 日志目录路径
    """
    # 移除默认处理器
    logger.remove()
    
    # 日志格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
    
    # 控制台输出
    logger.add(
        sys.stderr,
        format=console_format,
        level="DEBUG" if debug else "INFO",
        colorize=True,
        backtrace=True,
        diagnose=debug
    )
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 普通日志文件
    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )
    
    # 错误日志文件
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info("日志系统初始化完成")


def get_logger(name: str = None):
    """获取命名日志器
    
    Args:
        name: 日志器名称，通常使用 __name__
    
    Returns:
        绑定名称的日志器
    """
    if name:
        return logger.bind(name=name)
    return logger

