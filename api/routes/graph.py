"""
文件路径: api/routes/graph.py
功能: 知识图谱API路由

提供实体查询、关系遍历和可视化数据接口。
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
import logging

from api.dependencies import get_graph_service
from api.schemas.common import DataResponse, ListResponse
from api.schemas.graph import (
    EntityInfo, RelationshipInfo, EntitySearch, PathQuery,
    PathInfo, VisualizationData, GraphStats
)
from services import GraphService, EntitySearchRequest, PathQueryRequest

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# ==================== 实体查询 ====================

@router.get("/entities", response_model=ListResponse[EntityInfo], summary="列出实体")
async def list_entities(
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: GraphService = Depends(get_graph_service)
):
    """分页列出知识图谱中的实体"""
    result = service.list_entities(
        entity_type=entity_type,
        page=page,
        page_size=page_size
    )
    return ListResponse(
        success=True,
        data=[EntityInfo(**e.model_dump()) for e in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"]
    )


@router.get("/entities/{entity_id}", response_model=DataResponse[EntityInfo], summary="获取实体")
async def get_entity(
    entity_id: str = Path(..., description="实体ID"),
    service: GraphService = Depends(get_graph_service)
):
    """获取单个实体的详细信息"""
    result = service.get_entity(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail="实体不存在")
    return DataResponse(success=True, data=EntityInfo(**result.model_dump()))


@router.post("/entities/search", response_model=ListResponse[EntityInfo], summary="搜索实体")
async def search_entities(
    request: EntitySearch,
    service: GraphService = Depends(get_graph_service)
):
    """
    搜索实体
    
    支持按名称关键词和类型进行搜索。
    """
    req = EntitySearchRequest(
        query=request.query,
        entity_types=request.entity_types,
        limit=request.limit
    )
    results = service.search_entities(req)
    return ListResponse(
        success=True,
        data=[EntityInfo(**e.model_dump()) for e in results],
        total=len(results)
    )


@router.get("/entities/{entity_id}/related", response_model=ListResponse[EntityInfo], summary="相关实体")
async def get_related_entities(
    entity_id: str = Path(..., description="实体ID"),
    relationship_types: Optional[str] = Query(None, description="关系类型过滤,逗号分隔"),
    max_depth: int = Query(2, ge=1, le=5, description="最大搜索深度"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    service: GraphService = Depends(get_graph_service)
):
    """
    查找与指定实体相关的其他实体
    
    可以指定关系类型和搜索深度。
    """
    # 解析关系类型
    rel_types = relationship_types.split(",") if relationship_types else None
    
    results = service.find_related_entities(
        entity_id=entity_id,
        relationship_types=rel_types,
        max_depth=max_depth,
        limit=limit
    )
    return ListResponse(
        success=True,
        data=[EntityInfo(**e.model_dump()) for e in results],
        total=len(results)
    )


# ==================== 路径查询 ====================

@router.post("/path", response_model=DataResponse[PathInfo], summary="查找路径")
async def find_path(
    request: PathQuery,
    service: GraphService = Depends(get_graph_service)
):
    """
    查找两个实体之间的路径
    
    使用广度优先搜索查找最短路径。
    """
    req = PathQueryRequest(
        from_entity_id=request.from_entity_id,
        to_entity_id=request.to_entity_id,
        max_depth=request.max_depth
    )
    result = service.find_path(req)
    
    if result is None:
        raise HTTPException(status_code=404, detail="未找到路径")
    
    return DataResponse(
        success=True,
        data=PathInfo(
            entities=[EntityInfo(**e.model_dump()) for e in result.entities],
            relationships=[RelationshipInfo(**r.model_dump()) for r in result.relationships],
            path_length=result.path_length
        )
    )


# ==================== 可视化 ====================

@router.get("/visualization", response_model=DataResponse[VisualizationData], summary="可视化数据")
async def get_visualization_data(
    center_entity_id: Optional[str] = Query(None, description="中心实体ID（可选）"),
    depth: int = Query(2, ge=1, le=5, description="展开深度"),
    limit: int = Query(100, ge=1, le=500, description="节点数量限制"),
    service: GraphService = Depends(get_graph_service)
):
    """
    获取图谱可视化数据
    
    返回适用于 ECharts 等图表库的节点和边数据。
    如果指定中心实体，则从该实体开始展开；否则返回全局视图。
    """
    result = service.get_visualization_data(
        center_entity_id=center_entity_id,
        depth=depth,
        limit=limit
    )
    return DataResponse(success=True, data=VisualizationData(**result.model_dump()))


# ==================== 统计 ====================

@router.get("/stats", response_model=DataResponse[GraphStats], summary="获取统计")
async def get_stats(
    service: GraphService = Depends(get_graph_service)
):
    """获取知识图谱统计信息"""
    stats = service.get_stats()
    return DataResponse(success=True, data=GraphStats(**stats.model_dump()))


@router.get("/health", summary="健康检查")
async def health_check(
    service: GraphService = Depends(get_graph_service)
):
    """检查图数据库（Neo4j）连接状态"""
    is_healthy = service.health_check()
    return DataResponse(
        success=True,
        data={
            "connected": is_healthy,
            "status": "healthy" if is_healthy else "disconnected"
        }
    )

