"""
文件路径: utils/helpers.py
功能: 通用辅助函数
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Generator


def generate_id(prefix: str = "") -> str:
    """生成唯一ID
    
    Args:
        prefix: ID前缀
    
    Returns:
        格式: {prefix}_{uuid} 或 {uuid}
    """
    uid = str(uuid.uuid4())
    return f"{prefix}_{uid}" if prefix else uid


def generate_hash(content: str) -> str:
    """生成内容SHA256哈希
    
    Args:
        content: 要哈希的内容
    
    Returns:
        64位十六进制哈希字符串
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def timestamp_now() -> str:
    """获取当前ISO格式时间戳
    
    Returns:
        ISO格式时间字符串，如 2024-12-26T10:30:00.123456
    """
    return datetime.now().isoformat()


def timestamp_to_datetime(ts: str) -> datetime:
    """ISO时间戳转datetime对象
    
    Args:
        ts: ISO格式时间字符串
    
    Returns:
        datetime对象
    """
    return datetime.fromisoformat(ts)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定长度
    
    Args:
        text: 原始文本
        max_length: 最大长度（包含后缀）
        suffix: 截断后缀
    
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """将列表分成大小为n的块
    
    Args:
        lst: 原始列表
        n: 每块大小
    
    Yields:
        子列表
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """安全获取嵌套字典值
    
    Args:
        data: 字典
        key: 点分隔的键路径，如 "a.b.c"
        default: 默认值
    
    Returns:
        对应值或默认值
    """
    keys = key.split('.')
    result = data
    for k in keys:
        if isinstance(result, dict):
            result = result.get(k)
        else:
            return default
        if result is None:
            return default
    return result

