"""
服务层模块 (services/)

提供业务逻辑封装，连接核心层与API层。
每个服务类封装特定领域的业务逻辑。

服务列表:
- MemoryService: 记忆管理服务（CRUD、搜索、整合、遗忘）
- RAGService: RAG检索增强生成服务（文档处理、检索、问答）
- GraphService: 知识图谱服务（实体查询、可视化）
- AnalyticsService: 分析统计服务（仪表盘、趋势、健康监控）
"""

from .memory_service import (
    MemoryService,
    MemoryCreateRequest,
    MemoryUpdateRequest,
    MemorySearchRequest,
    MemoryResponse,
    MemoryStatsResponse,
    ConsolidateRequest,
    ForgetRequest,
)

from .rag_service import (
    RAGService,
    DocumentUploadRequest,
    DocumentInfo,
    ChunkInfo,
    SearchRequest,
    AdvancedSearchRequest,
    SearchResult,
    AskRequest,
    AskResponse,
    RAGStatsResponse,
)

from .graph_service import (
    GraphService,
    EntityInfo,
    RelationshipInfo,
    EntitySearchRequest,
    PathQueryRequest,
    PathInfo,
    VisualizationData,
    VisualizationNode,
    VisualizationLink,
    GraphStatsResponse,
)

from .analytics_service import (
    AnalyticsService,
    DashboardSummary,
    TrendReport,
    SystemHealth,
    StorageStatus,
    MemoryDistribution,
    TimeSeriesPoint,
)

__all__ = [
    # Memory Service
    "MemoryService",
    "MemoryCreateRequest",
    "MemoryUpdateRequest",
    "MemorySearchRequest",
    "MemoryResponse",
    "MemoryStatsResponse",
    "ConsolidateRequest",
    "ForgetRequest",
    
    # RAG Service
    "RAGService",
    "DocumentUploadRequest",
    "DocumentInfo",
    "ChunkInfo",
    "SearchRequest",
    "AdvancedSearchRequest",
    "SearchResult",
    "AskRequest",
    "AskResponse",
    "RAGStatsResponse",
    
    # Graph Service
    "GraphService",
    "EntityInfo",
    "RelationshipInfo",
    "EntitySearchRequest",
    "PathQueryRequest",
    "PathInfo",
    "VisualizationData",
    "VisualizationNode",
    "VisualizationLink",
    "GraphStatsResponse",
    
    # Analytics Service
    "AnalyticsService",
    "DashboardSummary",
    "TrendReport",
    "SystemHealth",
    "StorageStatus",
    "MemoryDistribution",
    "TimeSeriesPoint",
]
