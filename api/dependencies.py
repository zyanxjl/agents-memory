"""
文件路径: api/dependencies.py
功能: FastAPI 依赖注入模块

提供服务实例的依赖注入，支持单例管理和请求级别的资源。
使用 lru_cache 实现服务单例，避免重复初始化。
"""

from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ==================== 服务单例 ====================

@lru_cache()
def get_memory_service():
    """
    获取记忆服务单例
    
    使用 lru_cache 确保整个应用生命周期内只创建一个实例。
    
    Returns:
        MemoryService: 记忆服务实例
    """
    from services import MemoryService
    logger.info("初始化 MemoryService 单例")
    return MemoryService()


@lru_cache()
def get_rag_service():
    """
    获取RAG服务单例
    
    Returns:
        RAGService: RAG服务实例
    """
    from services import RAGService
    logger.info("初始化 RAGService 单例")
    return RAGService()


@lru_cache()
def get_graph_service():
    """
    获取图谱服务单例
    
    Returns:
        GraphService: 图谱服务实例
    """
    from services import GraphService
    logger.info("初始化 GraphService 单例")
    return GraphService()


@lru_cache()
def get_analytics_service():
    """
    获取分析服务单例
    
    Returns:
        AnalyticsService: 分析服务实例
    """
    from services import AnalyticsService
    logger.info("初始化 AnalyticsService 单例")
    return AnalyticsService()


# ==================== 用户上下文 ====================

def get_current_user_id() -> str:
    """
    获取当前用户ID
    
    简化实现，实际应用中应从认证中间件获取。
    可通过 JWT Token 或 Session 获取真实用户信息。
    
    Returns:
        str: 用户ID
    """
    # TODO: 实际应用中应从请求头或认证中间件获取
    return "default_user"


# ==================== 分页参数 ====================

class PaginationParams:
    """
    分页参数类
    
    用于API路由中的分页查询参数。
    
    Attributes:
        page: 页码（从1开始）
        page_size: 每页数量（最大100）
    """
    
    def __init__(self, page: int = 1, page_size: int = 20):
        """
        初始化分页参数
        
        Args:
            page: 页码，最小为1
            page_size: 每页数量，范围1-100
        """
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.page_size

