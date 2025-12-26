"""
文件路径: api/routes/memory.py
功能: 记忆管理API路由

提供记忆的CRUD、搜索和管理接口。
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
import logging

from api.dependencies import get_memory_service, get_current_user_id
from api.schemas.common import DataResponse, ListResponse
from api.schemas.memory import (
    MemoryCreate, MemoryUpdate, MemorySearch,
    MemoryResponse, MemoryStats, ConsolidateRequest, ForgetRequest
)
from services import (
    MemoryService, 
    MemoryCreateRequest, 
    MemoryUpdateRequest, 
    MemorySearchRequest
)
from services import ConsolidateRequest as ServiceConsolidateRequest
from services import ForgetRequest as ServiceForgetRequest

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# ==================== CRUD 操作 ====================

@router.post("", response_model=DataResponse[MemoryResponse], summary="添加记忆")
async def create_memory(
    request: MemoryCreate,
    user_id: str = Depends(get_current_user_id),
    service: MemoryService = Depends(get_memory_service)
):
    """
    添加新记忆
    
    - **content**: 记忆内容（必填，1-10000字符）
    - **memory_type**: 记忆类型，可选 working/episodic/semantic/auto
    - **importance**: 重要性分数 0.0-1.0
    - **metadata**: 额外元数据字典
    """
    try:
        # 构建服务层请求
        req = MemoryCreateRequest(
            content=request.content,
            memory_type=request.memory_type,
            user_id=user_id,
            importance=request.importance,
            metadata=request.metadata
        )
        # 调用服务
        result = service.add_memory(req)
        
        # 返回响应
        return DataResponse(
            success=True,
            message="记忆添加成功",
            data=MemoryResponse(**result.model_dump())
        )
    except Exception as e:
        logger.error(f"添加记忆失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{memory_id}", response_model=DataResponse[MemoryResponse], summary="获取记忆")
async def get_memory(
    memory_id: str = Path(..., description="记忆ID"),
    service: MemoryService = Depends(get_memory_service)
):
    """根据ID获取单个记忆详情"""
    result = service.get_memory(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return DataResponse(
        success=True,
        data=MemoryResponse(**result.model_dump())
    )


@router.put("/{memory_id}", response_model=DataResponse[bool], summary="更新记忆")
async def update_memory(
    request: MemoryUpdate,
    memory_id: str = Path(..., description="记忆ID"),
    service: MemoryService = Depends(get_memory_service)
):
    """更新记忆的内容或属性"""
    req = MemoryUpdateRequest(
        content=request.content,
        importance=request.importance,
        metadata=request.metadata
    )
    success = service.update_memory(memory_id, req)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在或更新失败")
    return DataResponse(success=True, message="更新成功", data=True)


@router.delete("/{memory_id}", response_model=DataResponse[bool], summary="删除记忆")
async def delete_memory(
    memory_id: str = Path(..., description="记忆ID"),
    service: MemoryService = Depends(get_memory_service)
):
    """删除指定ID的记忆"""
    success = service.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return DataResponse(success=True, message="删除成功", data=True)


# ==================== 搜索与列表 ====================

@router.post("/search", response_model=ListResponse[MemoryResponse], summary="搜索记忆")
async def search_memories(
    request: MemorySearch,
    user_id: str = Depends(get_current_user_id),
    service: MemoryService = Depends(get_memory_service)
):
    """
    搜索记忆
    
    支持跨多种记忆类型的语义搜索，返回按相关性排序的结果。
    """
    req = MemorySearchRequest(
        query=request.query,
        memory_types=request.memory_types,
        user_id=user_id,
        limit=request.limit,
        min_importance=request.min_importance
    )
    results = service.search_memories(req)
    return ListResponse(
        success=True,
        data=[MemoryResponse(**r.model_dump()) for r in results],
        total=len(results),
        page=1,
        page_size=request.limit
    )


@router.get("", response_model=ListResponse[MemoryResponse], summary="列出记忆")
async def list_memories(
    memory_type: Optional[str] = Query(None, description="记忆类型过滤"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    sort_by: str = Query("timestamp", description="排序字段: timestamp/importance"),
    sort_order: str = Query("desc", description="排序方向: asc/desc"),
    service: MemoryService = Depends(get_memory_service)
):
    """
    分页列出记忆
    
    支持按类型过滤和自定义排序。
    """
    result = service.list_memories(
        memory_type=memory_type,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )
    return ListResponse(
        success=True,
        data=[MemoryResponse(**item.model_dump()) for item in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        total_pages=result["total_pages"]
    )


# ==================== 管理操作 ====================

@router.get("/stats/overview", response_model=DataResponse[MemoryStats], summary="获取统计")
async def get_stats(
    service: MemoryService = Depends(get_memory_service)
):
    """获取记忆系统的统计信息"""
    stats = service.get_stats()
    return DataResponse(
        success=True,
        data=MemoryStats(**stats.model_dump())
    )


@router.post("/consolidate", summary="整合记忆")
async def consolidate_memories(
    request: ConsolidateRequest,
    service: MemoryService = Depends(get_memory_service)
):
    """
    执行记忆整合
    
    将短期记忆（如工作记忆）中重要的内容整合到长期记忆（如情景记忆）。
    """
    req = ServiceConsolidateRequest(
        source_type=request.source_type,
        target_type=request.target_type,
        importance_threshold=request.importance_threshold
    )
    result = service.consolidate(req)
    return DataResponse(success=True, message="整合完成", data=result)


@router.post("/forget", summary="遗忘记忆")
async def forget_memories(
    request: ForgetRequest,
    service: MemoryService = Depends(get_memory_service)
):
    """
    执行记忆遗忘
    
    根据策略删除低重要性或过期的记忆。
    """
    req = ServiceForgetRequest(
        strategy=request.strategy,
        threshold=request.threshold,
        max_age_days=request.max_age_days
    )
    result = service.forget(req)
    return DataResponse(success=True, message="遗忘完成", data=result)


@router.post("/export", summary="导出记忆")
async def export_memories(
    memory_type: Optional[str] = Query(None, description="类型过滤"),
    service: MemoryService = Depends(get_memory_service)
):
    """导出记忆数据为JSON格式"""
    data = service.export_memories(memory_type=memory_type)
    return DataResponse(success=True, data=data)


@router.post("/import", summary="导入记忆")
async def import_memories(
    data: dict,
    service: MemoryService = Depends(get_memory_service)
):
    """从JSON格式导入记忆数据"""
    result = service.import_memories(data)
    return DataResponse(success=True, message="导入完成", data=result)

