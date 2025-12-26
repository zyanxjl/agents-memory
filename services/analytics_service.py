"""
文件路径: services/analytics_service.py
功能: 分析服务层 - 提供系统统计和监控功能

提供:
- 记忆系统统计分析
- 使用趋势报告
- 存储状态监控
- 系统健康检查
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import os
import logging

from pydantic import BaseModel, Field

from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 请求/响应数据模型 ====================

class TimeSeriesPoint(BaseModel):
    """时间序列数据点"""
    timestamp: datetime
    value: float
    label: Optional[str] = None


class MemoryDistribution(BaseModel):
    """记忆类型分布"""
    working: int = 0
    episodic: int = 0
    semantic: int = 0
    perceptual: int = 0

    @property
    def total(self) -> int:
        return self.working + self.episodic + self.semantic + self.perceptual


class StorageStatus(BaseModel):
    """存储状态"""
    qdrant_status: str = "unknown"
    qdrant_vector_count: int = 0
    neo4j_status: str = "unknown"
    neo4j_node_count: int = 0
    sqlite_status: str = "ok"
    sqlite_size_mb: float = 0.0


class DashboardSummary(BaseModel):
    """仪表盘摘要数据"""
    total_memories: int = 0
    today_added: int = 0
    total_documents: int = 0
    total_entities: int = 0
    memory_distribution: MemoryDistribution = Field(default_factory=MemoryDistribution)
    storage_status: StorageStatus = Field(default_factory=StorageStatus)
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)


class TrendReport(BaseModel):
    """趋势报告"""
    period: str  # "day", "week", "month"
    memory_growth: List[TimeSeriesPoint] = Field(default_factory=list)
    avg_importance: List[TimeSeriesPoint] = Field(default_factory=list)
    top_memory_types: Dict[str, int] = Field(default_factory=dict)


class SystemHealth(BaseModel):
    """系统健康状态"""
    overall_status: str = "healthy"  # healthy, degraded, unhealthy
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_check: datetime = Field(default_factory=datetime.now)
    issues: List[str] = Field(default_factory=list)


# ==================== 服务类 ====================

class AnalyticsService:
    """
    分析服务类
    
    提供系统统计、趋势分析和健康监控功能。
    聚合来自其他服务的数据。
    """

    def __init__(
        self,
        memory_service=None,
        rag_service=None,
        graph_service=None
    ):
        """
        初始化分析服务
        
        Args:
            memory_service: 记忆服务实例
            rag_service: RAG服务实例
            graph_service: 图谱服务实例
        """
        self._memory_service = memory_service
        self._rag_service = rag_service
        self._graph_service = graph_service
        
        # 活动日志（内存存储）
        self._activity_log: List[Dict[str, Any]] = []
        self._query_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("✅ AnalyticsService 初始化完成")

    def _get_memory_service(self):
        """延迟获取记忆服务"""
        if self._memory_service is None:
            from services.memory_service import MemoryService
            self._memory_service = MemoryService()
        return self._memory_service

    def _get_rag_service(self):
        """延迟获取RAG服务"""
        if self._rag_service is None:
            from services.rag_service import RAGService
            self._rag_service = RAGService()
        return self._rag_service

    def _get_graph_service(self):
        """延迟获取图谱服务"""
        if self._graph_service is None:
            from services.graph_service import GraphService
            self._graph_service = GraphService()
        return self._graph_service

    # ==================== 仪表盘数据 ====================

    def get_dashboard_summary(self, user_id: Optional[str] = None) -> DashboardSummary:
        """
        获取仪表盘摘要数据
        
        Args:
            user_id: 可选的用户ID过滤
            
        Returns:
            仪表盘摘要
        """
        # 1. 记忆统计
        memory_stats = self._get_memory_service().get_stats()
        
        # 2. RAG统计
        rag_stats = self._get_rag_service().get_stats(user_id=user_id)
        
        # 3. 图谱统计
        graph_stats = self._get_graph_service().get_stats()
        
        # 4. 存储状态
        storage_status = self._get_storage_status()
        
        # 5. 今日新增
        today_added = self._count_today_memories()
        
        # 6. 最近活动
        recent_activity = self._get_recent_activity(limit=10)
        
        return DashboardSummary(
            total_memories=memory_stats.total_count,
            today_added=today_added,
            total_documents=rag_stats.total_documents,
            total_entities=graph_stats.total_entities,
            memory_distribution=MemoryDistribution(
                working=memory_stats.working_count,
                episodic=memory_stats.episodic_count,
                semantic=memory_stats.semantic_count,
                perceptual=memory_stats.perceptual_count
            ),
            storage_status=storage_status,
            recent_activity=recent_activity
        )

    def _get_storage_status(self) -> StorageStatus:
        """获取存储状态"""
        status = StorageStatus()
        settings = get_settings()
        
        # Qdrant 状态
        try:
            from core.storage import QdrantConnectionManager
            qdrant = QdrantConnectionManager.get_instance()
            if qdrant.health_check():
                status.qdrant_status = "connected"
                stats = qdrant.get_collection_stats()
                status.qdrant_vector_count = stats.get("vector_count", 0)
            else:
                status.qdrant_status = "disconnected"
        except Exception as e:
            status.qdrant_status = f"error"
            logger.debug(f"Qdrant状态检查: {e}")
        
        # Neo4j 状态
        try:
            graph_healthy = self._get_graph_service().health_check()
            if graph_healthy:
                status.neo4j_status = "connected"
                graph_stats = self._get_graph_service().get_stats()
                status.neo4j_node_count = graph_stats.total_entities
            else:
                status.neo4j_status = "disconnected"
        except Exception as e:
            status.neo4j_status = "error"
            logger.debug(f"Neo4j状态检查: {e}")
        
        # SQLite 状态
        try:
            db_path = os.path.join(settings.database_settings.sqlite_path, "memory.db")
            if os.path.exists(db_path):
                status.sqlite_status = "ok"
                status.sqlite_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            else:
                status.sqlite_status = "not_initialized"
        except Exception as e:
            status.sqlite_status = "error"
            logger.debug(f"SQLite状态检查: {e}")
        
        return status

    def _count_today_memories(self) -> int:
        """统计今日新增记忆数"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            result = self._get_memory_service().list_memories(page=1, page_size=1000)
            count = 0
            for item in result.get("items", []):
                if item.timestamp >= today_start:
                    count += 1
            return count
        except Exception:
            return 0

    def _get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近活动"""
        return self._activity_log[-limit:][::-1]

    # ==================== 趋势分析 ====================

    def get_trend_report(
        self,
        period: str = "week",
        user_id: Optional[str] = None
    ) -> TrendReport:
        """
        获取趋势报告
        
        Args:
            period: 时间周期 ("day", "week", "month")
            user_id: 可选的用户ID过滤
            
        Returns:
            趋势报告
        """
        # 确定时间范围
        now = datetime.now()
        if period == "day":
            start_time = now - timedelta(days=1)
            interval = timedelta(hours=1)
        elif period == "week":
            start_time = now - timedelta(weeks=1)
            interval = timedelta(days=1)
        else:  # month
            start_time = now - timedelta(days=30)
            interval = timedelta(days=1)
        
        # 获取记忆列表
        result = self._get_memory_service().list_memories(page=1, page_size=10000)
        memories = result.get("items", [])
        
        # 按时间分组统计
        memory_growth = []
        importance_data = []
        type_counts = defaultdict(int)
        
        current_time = start_time
        while current_time <= now:
            next_time = current_time + interval
            
            # 该时间段的记忆
            period_memories = [
                m for m in memories 
                if current_time <= m.timestamp < next_time
            ]
            
            # 记忆增长
            memory_growth.append(TimeSeriesPoint(
                timestamp=current_time,
                value=float(len(period_memories)),
                label=current_time.strftime("%Y-%m-%d %H:%M")
            ))
            
            # 平均重要性
            if period_memories:
                avg_imp = sum(m.importance for m in period_memories) / len(period_memories)
            else:
                avg_imp = 0.0
            importance_data.append(TimeSeriesPoint(
                timestamp=current_time,
                value=avg_imp
            ))
            
            current_time = next_time
        
        # 统计类型分布
        for m in memories:
            if m.timestamp >= start_time:
                type_counts[m.memory_type] += 1
        
        return TrendReport(
            period=period,
            memory_growth=memory_growth,
            avg_importance=importance_data,
            top_memory_types=dict(type_counts)
        )

    # ==================== 系统健康 ====================

    def get_system_health(self) -> SystemHealth:
        """
        获取系统健康状态
        
        Returns:
            系统健康状态
        """
        health = SystemHealth()
        issues = []
        components = {}
        
        # 1. Qdrant 检查
        try:
            from core.storage import QdrantConnectionManager
            qdrant = QdrantConnectionManager.get_instance()
            qdrant_healthy = qdrant.health_check()
            components["qdrant"] = {
                "status": "healthy" if qdrant_healthy else "unhealthy",
                "message": "连接正常" if qdrant_healthy else "连接失败"
            }
            if not qdrant_healthy:
                issues.append("Qdrant向量数据库连接失败")
        except Exception as e:
            components["qdrant"] = {"status": "error", "message": str(e)[:50]}
            issues.append("Qdrant检查异常")
        
        # 2. Neo4j 检查
        try:
            neo4j_healthy = self._get_graph_service().health_check()
            components["neo4j"] = {
                "status": "healthy" if neo4j_healthy else "unhealthy",
                "message": "连接正常" if neo4j_healthy else "连接失败"
            }
            if not neo4j_healthy:
                issues.append("Neo4j图数据库连接失败")
        except Exception as e:
            components["neo4j"] = {"status": "error", "message": str(e)[:50]}
            issues.append("Neo4j检查异常")
        
        # 3. SQLite 检查
        try:
            settings = get_settings()
            db_path = os.path.join(settings.database_settings.sqlite_path, "memory.db")
            sqlite_exists = os.path.exists(db_path)
            components["sqlite"] = {
                "status": "healthy" if sqlite_exists else "not_initialized",
                "message": "数据库正常" if sqlite_exists else "数据库未初始化"
            }
        except Exception as e:
            components["sqlite"] = {"status": "error", "message": str(e)[:50]}
            issues.append("SQLite检查异常")
        
        # 4. 嵌入模型检查
        try:
            from core.embedding import get_text_embedder
            embedder = get_text_embedder()
            test_vec = embedder.encode("health check")
            components["embedding"] = {
                "status": "healthy",
                "message": f"嵌入模型正常，维度: {len(test_vec)}"
            }
        except Exception as e:
            components["embedding"] = {"status": "error", "message": str(e)[:50]}
            issues.append("嵌入模型异常")
        
        # 确定整体状态
        statuses = [c.get("status", "unknown") for c in components.values()]
        if all(s == "healthy" for s in statuses):
            health.overall_status = "healthy"
        elif any(s in ("error", "unhealthy") for s in statuses):
            health.overall_status = "degraded"
        else:
            health.overall_status = "degraded"
        
        health.components = components
        health.issues = issues
        health.last_check = datetime.now()
        
        return health

    # ==================== 活动记录 ====================

    def log_activity(
        self,
        action: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        记录用户活动
        
        Args:
            action: 动作类型
            user_id: 用户ID
            details: 额外详情
        """
        activity = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": user_id,
            "details": details or {}
        }
        
        self._activity_log.append(activity)
        
        # 更新查询计数
        if action in ("search", "ask"):
            date_key = datetime.now().strftime("%Y-%m-%d")
            self._query_counts[date_key] += 1
        
        # 限制日志大小
        if len(self._activity_log) > 10000:
            self._activity_log = self._activity_log[-5000:]
        
        logger.debug(f"活动记录: {action} by {user_id}")

    def get_query_stats(self, days: int = 7) -> Dict[str, int]:
        """
        获取查询统计
        
        Args:
            days: 统计天数
            
        Returns:
            每日查询数量
        """
        result = {}
        now = datetime.now()
        
        for i in range(days):
            date = now - timedelta(days=i)
            date_key = date.strftime("%Y-%m-%d")
            result[date_key] = self._query_counts.get(date_key, 0)
        
        return result

