# 阶段3：API层实现 - 详细任务规划

## 概述

**目标**：使用 FastAPI 实现 RESTful API 接口，将服务层暴露为 Web API。

**预计时间**：2天

**输出目录**：`api/`

**依赖**：Phase 2 服务层实现完成

---

## 目录结构

```
api/
├── __init__.py              # API模块初始化
├── main.py                  # FastAPI应用入口
├── dependencies.py          # 依赖注入
├── routes/                  # 路由模块
│   ├── __init__.py
│   ├── memory.py            # 记忆API路由
│   ├── rag.py               # RAG API路由
│   ├── graph.py             # 图谱API路由
│   └── analytics.py         # 分析统计API路由
├── schemas/                 # Pydantic请求/响应模型
│   ├── __init__.py
│   ├── common.py            # 通用模型
│   ├── memory.py            # 记忆相关模型
│   ├── rag.py               # RAG相关模型
│   └── graph.py             # 图谱相关模型
└── middleware/              # 中间件
    ├── __init__.py
    ├── cors.py              # CORS配置
    └── logging.py           # 请求日志
```

---

## Task 3.1：FastAPI应用搭建

### 3.1.1 功能描述

创建 FastAPI 主应用，配置中间件、路由和依赖注入。

### 3.1.2 主应用实现

```python
# api/main.py

"""
FastAPI 应用入口

功能:
- 创建 FastAPI 应用实例
- 配置 CORS 中间件
- 注册路由
- 配置异常处理
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config.settings import get_settings

# 路由导入
from api.routes import memory, rag, graph, analytics

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("🚀 Agent Memory System API 启动中...")
    yield
    # 关闭时
    logger.info("👋 Agent Memory System API 关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    settings = get_settings()
    
    app = FastAPI(
        title="Agent Memory System API",
        description="智能体记忆系统 - 可视化管理平台 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(memory.router, prefix="/api/v1/memory", tags=["记忆管理"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG知识库"])
    app.include_router(graph.router, prefix="/api/v1/graph", tags=["知识图谱"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["分析统计"])
    
    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "error": str(exc)}
        )
    
    # 健康检查端点
    @app.get("/health", tags=["系统"])
    async def health_check():
        """健康检查"""
        return {"status": "ok", "service": "Agent Memory System"}
    
    # 根路径
    @app.get("/", tags=["系统"])
    async def root():
        """API根路径"""
        return {
            "message": "Agent Memory System API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 3.1.3 依赖注入

```python
# api/dependencies.py

"""
依赖注入模块

提供服务实例的依赖注入，支持请求级别的资源管理。
"""

from functools import lru_cache
from typing import Generator

from services import MemoryService, RAGService, GraphService, AnalyticsService
from core.memory import MemoryConfig


# ==================== 服务单例 ====================

@lru_cache()
def get_memory_service() -> MemoryService:
    """获取记忆服务单例"""
    return MemoryService()


@lru_cache()
def get_rag_service() -> RAGService:
    """获取RAG服务单例"""
    return RAGService()


@lru_cache()
def get_graph_service() -> GraphService:
    """获取图谱服务单例"""
    return GraphService()


@lru_cache()
def get_analytics_service() -> AnalyticsService:
    """获取分析服务单例"""
    return AnalyticsService()


# ==================== 用户上下文 ====================

def get_current_user_id() -> str:
    """
    获取当前用户ID
    
    简化实现，实际应从认证中间件获取。
    """
    return "default_user"


# ==================== 分页参数 ====================

class PaginationParams:
    """分页参数"""
    def __init__(self, page: int = 1, page_size: int = 20):
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))
```

### 3.1.4 验证方法

```python
# 启动服务器后访问 http://localhost:8000/docs 查看 API 文档
# 或访问 http://localhost:8000/health 检查健康状态
```

---

## Task 3.2：通用Schema定义

### 3.2.1 通用响应模型

```python
# api/schemas/common.py

"""
通用Pydantic模型

定义API层通用的请求/响应模型。
"""

from typing import TypeVar, Generic, List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseBase(BaseModel):
    """基础响应模型"""
    success: bool = Field(True, description="请求是否成功")
    message: str = Field("", description="响应消息")


class DataResponse(ResponseBase, Generic[T]):
    """带数据的响应模型"""
    data: Optional[T] = Field(None, description="响应数据")


class ListResponse(ResponseBase, Generic[T]):
    """列表响应模型（带分页）"""
    data: List[T] = Field(default_factory=list, description="数据列表")
    total: int = Field(0, description="总数")
    page: int = Field(1, description="当前页")
    page_size: int = Field(20, description="每页数量")
    total_pages: int = Field(0, description="总页数")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(False)
    message: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误")
    error_code: Optional[str] = Field(None, description="错误代码")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="状态: ok/degraded/unhealthy")
    components: Dict[str, Any] = Field(default_factory=dict, description="组件状态")
    timestamp: datetime = Field(default_factory=datetime.now)


class StatsResponse(BaseModel):
    """统计信息响应"""
    total_count: int = Field(0)
    details: Dict[str, Any] = Field(default_factory=dict)
```

---

## Task 3.3：记忆API路由

### 3.3.1 记忆Schema

```python
# api/schemas/memory.py

"""
记忆相关的Pydantic模型
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryCreate(BaseModel):
    """创建记忆请求"""
    content: str = Field(..., min_length=1, max_length=10000, description="记忆内容")
    memory_type: str = Field("auto", description="类型: working/episodic/semantic/auto")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="重要性")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Python是一种高级编程语言",
                "memory_type": "semantic",
                "importance": 0.8,
                "metadata": {"source": "learning"}
            }
        }


class MemoryUpdate(BaseModel):
    """更新记忆请求"""
    content: Optional[str] = Field(None, max_length=10000)
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class MemorySearch(BaseModel):
    """搜索记忆请求"""
    query: str = Field(..., min_length=1, description="搜索查询")
    memory_types: List[str] = Field(
        default=["working", "episodic", "semantic"],
        description="记忆类型"
    )
    limit: int = Field(10, ge=1, le=100, description="返回数量")
    min_importance: float = Field(0.0, ge=0.0, le=1.0, description="最低重要性")


class MemoryResponse(BaseModel):
    """记忆响应"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = None


class MemoryStats(BaseModel):
    """记忆统计"""
    total_count: int = 0
    working_count: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    perceptual_count: int = 0
    avg_importance: float = 0.0


class ConsolidateRequest(BaseModel):
    """整合请求"""
    source_type: str = Field("working", description="源类型")
    target_type: str = Field("episodic", description="目标类型")
    importance_threshold: float = Field(0.7, ge=0.0, le=1.0)


class ForgetRequest(BaseModel):
    """遗忘请求"""
    strategy: str = Field("importance_based", description="策略")
    threshold: float = Field(0.1, ge=0.0, le=1.0)
    max_age_days: int = Field(30, ge=1)
```

### 3.3.2 记忆路由

```python
# api/routes/memory.py

"""
记忆管理API路由

提供记忆的CRUD、搜索和管理接口。
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from api.dependencies import get_memory_service, get_current_user_id
from api.schemas.common import DataResponse, ListResponse, ErrorResponse
from api.schemas.memory import (
    MemoryCreate, MemoryUpdate, MemorySearch,
    MemoryResponse, MemoryStats, ConsolidateRequest, ForgetRequest
)
from services import (
    MemoryService, MemoryCreateRequest, MemoryUpdateRequest, 
    MemorySearchRequest
)

router = APIRouter()


# ==================== CRUD ====================

@router.post("", response_model=DataResponse[MemoryResponse], summary="添加记忆")
async def create_memory(
    request: MemoryCreate,
    user_id: str = Depends(get_current_user_id),
    service: MemoryService = Depends(get_memory_service)
):
    """
    添加新记忆
    
    - **content**: 记忆内容（必填）
    - **memory_type**: 记忆类型，可选 working/episodic/semantic/auto
    - **importance**: 重要性分数 0.0-1.0
    - **metadata**: 额外元数据
    """
    try:
        req = MemoryCreateRequest(
            content=request.content,
            memory_type=request.memory_type,
            user_id=user_id,
            importance=request.importance,
            metadata=request.metadata
        )
        result = service.add_memory(req)
        return DataResponse(
            success=True,
            message="记忆添加成功",
            data=MemoryResponse(**result.model_dump())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{memory_id}", response_model=DataResponse[MemoryResponse], summary="获取记忆")
async def get_memory(
    memory_id: str = Path(..., description="记忆ID"),
    service: MemoryService = Depends(get_memory_service)
):
    """根据ID获取单个记忆"""
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
    """更新记忆内容或属性"""
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
    """删除指定记忆"""
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
    
    支持跨多种记忆类型的语义搜索。
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


@router.get("/list", response_model=ListResponse[MemoryResponse], summary="列出记忆")
async def list_memories(
    memory_type: Optional[str] = Query(None, description="记忆类型过滤"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    sort_by: str = Query("timestamp", description="排序字段"),
    sort_order: str = Query("desc", description="排序方向"),
    service: MemoryService = Depends(get_memory_service)
):
    """分页列出记忆"""
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

@router.get("/stats", response_model=DataResponse[MemoryStats], summary="获取统计")
async def get_stats(
    service: MemoryService = Depends(get_memory_service)
):
    """获取记忆统计信息"""
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
    """将短期记忆整合到长期记忆"""
    from services import ConsolidateRequest as ServiceReq
    req = ServiceReq(
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
    """执行记忆遗忘策略"""
    from services import ForgetRequest as ServiceReq
    req = ServiceReq(
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
    """导出记忆数据"""
    data = service.export_memories(memory_type=memory_type)
    return DataResponse(success=True, data=data)


@router.post("/import", summary="导入记忆")
async def import_memories(
    data: dict,
    service: MemoryService = Depends(get_memory_service)
):
    """导入记忆数据"""
    result = service.import_memories(data)
    return DataResponse(success=True, message="导入完成", data=result)
```

---

## Task 3.4：RAG API路由

### 3.4.1 RAG Schema

```python
# api/schemas/rag.py

"""
RAG相关的Pydantic模型
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUpload(BaseModel):
    """文档上传请求（用于JSON方式）"""
    filename: str = Field(..., description="文件名")
    content_base64: str = Field(..., description="Base64编码的文件内容")
    chunk_size: int = Field(800, ge=100, le=4000, description="分块大小")
    chunk_overlap: int = Field(100, ge=0, le=500, description="重叠大小")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    filename: str
    user_id: str
    upload_time: datetime
    chunk_count: int
    total_chars: int
    status: str


class ChunkInfo(BaseModel):
    """分块信息"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """检索请求"""
    query: str = Field(..., min_length=1, description="查询内容")
    limit: int = Field(5, ge=1, le=50, description="返回数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_advanced: bool = Field(False, description="是否使用高级检索")


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskQuery(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, description="问题")
    context_limit: int = Field(5, ge=1, le=20, description="上下文数量")
    include_sources: bool = Field(True, description="是否返回来源")


class AskResult(BaseModel):
    """问答结果"""
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0


class RAGStats(BaseModel):
    """RAG统计"""
    total_documents: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    collections: List[str] = Field(default_factory=list)
```

### 3.4.2 RAG路由

```python
# api/routes/rag.py

"""
RAG知识库API路由

提供文档管理、知识检索和问答接口。
"""

from typing import Optional, List
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, Path, UploadFile, File
from fastapi.responses import JSONResponse

from api.dependencies import get_rag_service, get_current_user_id
from api.schemas.common import DataResponse, ListResponse
from api.schemas.rag import (
    DocumentUpload, DocumentInfo, ChunkInfo,
    SearchQuery, SearchResult, AskQuery, AskResult, RAGStats
)
from services import RAGService, DocumentUploadRequest, SearchRequest, AdvancedSearchRequest, AskRequest

router = APIRouter()


# ==================== 文档管理 ====================

@router.post("/documents", response_model=DataResponse[DocumentInfo], summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(800, ge=100, le=4000),
    chunk_overlap: int = Query(100, ge=0, le=500),
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """
    上传文档到RAG知识库
    
    支持多种格式：PDF、Word、Markdown、TXT等。
    文档会被自动解析、分块并向量化。
    """
    try:
        content = await file.read()
        req = DocumentUploadRequest(
            filename=file.filename,
            user_id=user_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        result = service.upload_document(content, req)
        return DataResponse(
            success=True,
            message="文档上传成功",
            data=DocumentInfo(**result.model_dump())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/json", response_model=DataResponse[DocumentInfo], summary="上传文档(JSON)")
async def upload_document_json(
    request: DocumentUpload,
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """通过JSON方式上传文档（Base64编码）"""
    try:
        content = base64.b64decode(request.content_base64)
        req = DocumentUploadRequest(
            filename=request.filename,
            user_id=user_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata
        )
        result = service.upload_document(content, req)
        return DataResponse(
            success=True,
            message="文档上传成功",
            data=DocumentInfo(**result.model_dump())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/documents", response_model=ListResponse[DocumentInfo], summary="列出文档")
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """列出用户的所有文档"""
    result = service.list_documents(user_id=user_id, page=page, page_size=page_size)
    return ListResponse(
        success=True,
        data=[DocumentInfo(**d.model_dump()) for d in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"]
    )


@router.get("/documents/{doc_id}", response_model=DataResponse[DocumentInfo], summary="获取文档")
async def get_document(
    doc_id: str = Path(..., description="文档ID"),
    service: RAGService = Depends(get_rag_service)
):
    """获取文档详情"""
    result = service.get_document(doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DataResponse(success=True, data=DocumentInfo(**result.model_dump()))


@router.delete("/documents/{doc_id}", summary="删除文档")
async def delete_document(
    doc_id: str = Path(..., description="文档ID"),
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """删除文档及其所有分块"""
    success = service.delete_document(doc_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在或无权限")
    return DataResponse(success=True, message="删除成功")


@router.get("/documents/{doc_id}/chunks", response_model=ListResponse[ChunkInfo], summary="获取分块")
async def get_document_chunks(
    doc_id: str = Path(..., description="文档ID"),
    service: RAGService = Depends(get_rag_service)
):
    """获取文档的所有分块"""
    chunks = service.get_document_chunks(doc_id)
    return ListResponse(
        success=True,
        data=[ChunkInfo(**c.model_dump()) for c in chunks],
        total=len(chunks)
    )


# ==================== 检索 ====================

@router.post("/search", response_model=ListResponse[SearchResult], summary="知识检索")
async def search(
    request: SearchQuery,
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """
    检索知识库
    
    - **query**: 查询内容
    - **limit**: 返回数量
    - **use_advanced**: 是否使用高级检索（MQE/重排序）
    """
    if request.use_advanced:
        req = AdvancedSearchRequest(
            query=request.query,
            user_id=user_id,
            limit=request.limit,
            score_threshold=request.score_threshold,
            use_mqe=True,
            use_rerank=True
        )
        results = service.advanced_search(req)
    else:
        req = SearchRequest(
            query=request.query,
            user_id=user_id,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        results = service.search(req)
    
    return ListResponse(
        success=True,
        data=[SearchResult(**r.model_dump()) for r in results],
        total=len(results)
    )


# ==================== 问答 ====================

@router.post("/ask", response_model=DataResponse[AskResult], summary="知识问答")
async def ask(
    request: AskQuery,
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """
    基于知识库的问答
    
    系统会检索相关内容，并生成答案。
    """
    req = AskRequest(
        question=request.question,
        user_id=user_id,
        context_limit=request.context_limit,
        include_sources=request.include_sources
    )
    result = service.ask(req)
    return DataResponse(
        success=True,
        data=AskResult(
            answer=result.answer,
            sources=[SearchResult(**s.model_dump()) for s in result.sources],
            retrieval_time_ms=result.retrieval_time_ms,
            generation_time_ms=result.generation_time_ms
        )
    )


# ==================== 统计 ====================

@router.get("/stats", response_model=DataResponse[RAGStats], summary="获取统计")
async def get_stats(
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """获取RAG统计信息"""
    stats = service.get_stats(user_id=user_id)
    return DataResponse(success=True, data=RAGStats(**stats.model_dump()))
```

---

## Task 3.5：图谱API路由

### 3.5.1 图谱Schema

```python
# api/schemas/graph.py

"""
知识图谱相关的Pydantic模型
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EntityInfo(BaseModel):
    """实体信息"""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    related_count: int = 0


class RelationshipInfo(BaseModel):
    """关系信息"""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntitySearch(BaseModel):
    """实体搜索请求"""
    query: str = Field(..., min_length=1)
    entity_types: Optional[List[str]] = None
    limit: int = Field(20, ge=1, le=100)


class PathQuery(BaseModel):
    """路径查询请求"""
    from_entity_id: str
    to_entity_id: str
    max_depth: int = Field(4, ge=1, le=10)


class PathInfo(BaseModel):
    """路径信息"""
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    path_length: int = 0


class VisualizationNode(BaseModel):
    """可视化节点"""
    id: str
    name: str
    category: str
    value: float = 1.0


class VisualizationLink(BaseModel):
    """可视化边"""
    source: str
    target: str
    relationship: str
    value: float = 1.0


class VisualizationData(BaseModel):
    """可视化数据"""
    nodes: List[VisualizationNode] = Field(default_factory=list)
    links: List[VisualizationLink] = Field(default_factory=list)
    categories: List[Dict[str, str]] = Field(default_factory=list)


class GraphStats(BaseModel):
    """图谱统计"""
    total_entities: int = 0
    total_relationships: int = 0
    entity_types: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)
    is_connected: bool = False
```

### 3.5.2 图谱路由

```python
# api/routes/graph.py

"""
知识图谱API路由

提供实体查询、关系遍历和可视化数据接口。
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path

from api.dependencies import get_graph_service
from api.schemas.common import DataResponse, ListResponse
from api.schemas.graph import (
    EntityInfo, RelationshipInfo, EntitySearch, PathQuery,
    PathInfo, VisualizationData, GraphStats
)
from services import GraphService, EntitySearchRequest, PathQueryRequest

router = APIRouter()


# ==================== 实体查询 ====================

@router.get("/entities", response_model=ListResponse[EntityInfo], summary="列出实体")
async def list_entities(
    entity_type: Optional[str] = Query(None, description="类型过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: GraphService = Depends(get_graph_service)
):
    """分页列出实体"""
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
    """获取实体详情"""
    result = service.get_entity(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail="实体不存在")
    return DataResponse(success=True, data=EntityInfo(**result.model_dump()))


@router.post("/entities/search", response_model=ListResponse[EntityInfo], summary="搜索实体")
async def search_entities(
    request: EntitySearch,
    service: GraphService = Depends(get_graph_service)
):
    """搜索实体"""
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
    relationship_types: Optional[str] = Query(None, description="关系类型,逗号分隔"),
    max_depth: int = Query(2, ge=1, le=5),
    limit: int = Query(20, ge=1, le=100),
    service: GraphService = Depends(get_graph_service)
):
    """查找相关实体"""
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
    """查找两个实体之间的路径"""
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
    center_entity_id: Optional[str] = Query(None, description="中心实体ID"),
    depth: int = Query(2, ge=1, le=5, description="展开深度"),
    limit: int = Query(100, ge=1, le=500, description="节点限制"),
    service: GraphService = Depends(get_graph_service)
):
    """
    获取图谱可视化数据
    
    返回适用于 ECharts 等图表库的节点和边数据。
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
    """获取图谱统计信息"""
    stats = service.get_stats()
    return DataResponse(success=True, data=GraphStats(**stats.model_dump()))


@router.get("/health", summary="健康检查")
async def health_check(
    service: GraphService = Depends(get_graph_service)
):
    """检查图数据库连接状态"""
    is_healthy = service.health_check()
    return DataResponse(
        success=True,
        data={
            "connected": is_healthy,
            "status": "healthy" if is_healthy else "disconnected"
        }
    )
```

---

## Task 3.6：分析统计API路由

### 3.6.1 分析路由

```python
# api/routes/analytics.py

"""
分析统计API路由

提供仪表盘、趋势分析和系统健康检查接口。
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_analytics_service, get_current_user_id
from api.schemas.common import DataResponse, HealthResponse
from services import AnalyticsService

router = APIRouter()


@router.get("/dashboard", summary="仪表盘数据")
async def get_dashboard(
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取仪表盘摘要数据
    
    包含：总记忆数、今日新增、文档数、实体数、存储状态等。
    """
    summary = service.get_dashboard_summary(user_id=user_id)
    return DataResponse(success=True, data=summary.model_dump())


@router.get("/trends", summary="趋势报告")
async def get_trends(
    period: str = Query("week", description="周期: day/week/month"),
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取趋势报告
    
    包含：记忆增长曲线、平均重要性变化、类型分布等。
    """
    if period not in ("day", "week", "month"):
        period = "week"
    report = service.get_trend_report(period=period, user_id=user_id)
    return DataResponse(success=True, data=report.model_dump())


@router.get("/health", response_model=HealthResponse, summary="系统健康")
async def get_system_health(
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取系统健康状态
    
    检查各组件（Qdrant、Neo4j、SQLite、嵌入模型）的连接状态。
    """
    health = service.get_system_health()
    return HealthResponse(
        status=health.overall_status,
        components=health.components,
        timestamp=health.last_check
    )


@router.get("/query-stats", summary="查询统计")
async def get_query_stats(
    days: int = Query(7, ge=1, le=30, description="统计天数"),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """获取每日查询统计"""
    stats = service.get_query_stats(days=days)
    return DataResponse(success=True, data=stats)


@router.post("/log-activity", summary="记录活动")
async def log_activity(
    action: str,
    details: Optional[dict] = None,
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """记录用户活动"""
    service.log_activity(action=action, user_id=user_id, details=details)
    return DataResponse(success=True, message="活动已记录")
```

---

## Task 3.7：路由模块初始化

### 3.7.1 路由 `__init__.py`

```python
# api/routes/__init__.py

"""
API路由模块

导出所有路由供主应用注册。
"""

from . import memory
from . import rag
from . import graph
from . import analytics

__all__ = ["memory", "rag", "graph", "analytics"]
```

### 3.7.2 Schema `__init__.py`

```python
# api/schemas/__init__.py

"""
Pydantic模型模块

导出所有请求/响应模型。
"""

from .common import (
    ResponseBase,
    DataResponse,
    ListResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse
)

from .memory import (
    MemoryCreate,
    MemoryUpdate,
    MemorySearch,
    MemoryResponse,
    MemoryStats,
    ConsolidateRequest,
    ForgetRequest
)

from .rag import (
    DocumentUpload,
    DocumentInfo,
    ChunkInfo,
    SearchQuery,
    SearchResult,
    AskQuery,
    AskResult,
    RAGStats
)

from .graph import (
    EntityInfo,
    RelationshipInfo,
    EntitySearch,
    PathQuery,
    PathInfo,
    VisualizationData,
    VisualizationNode,
    VisualizationLink,
    GraphStats
)

__all__ = [
    # Common
    "ResponseBase", "DataResponse", "ListResponse", "ErrorResponse", 
    "HealthResponse", "StatsResponse",
    # Memory
    "MemoryCreate", "MemoryUpdate", "MemorySearch", "MemoryResponse",
    "MemoryStats", "ConsolidateRequest", "ForgetRequest",
    # RAG
    "DocumentUpload", "DocumentInfo", "ChunkInfo", "SearchQuery",
    "SearchResult", "AskQuery", "AskResult", "RAGStats",
    # Graph
    "EntityInfo", "RelationshipInfo", "EntitySearch", "PathQuery",
    "PathInfo", "VisualizationData", "VisualizationNode", "VisualizationLink",
    "GraphStats"
]
```

---

## Task 3.8：阶段验证

### 3.8.1 验证脚本

```python
# scripts/verify_phase3.py

"""
阶段3验证脚本 - API层实现验证

验证项目:
1. API模块导入
2. FastAPI应用创建
3. 路由注册
4. API端点访问（需启动服务器）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def verify_imports():
    """验证API模块导入"""
    print("1. 验证API模块导入...")
    try:
        from api.main import app, create_app
        from api.dependencies import get_memory_service, get_rag_service
        from api.schemas import DataResponse, MemoryCreate, SearchQuery
        from api.routes import memory, rag, graph, analytics
        print("  [OK] 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_app_creation():
    """验证FastAPI应用创建"""
    print("2. 验证FastAPI应用创建...")
    try:
        from api.main import create_app
        app = create_app()
        assert app is not None
        assert app.title == "Agent Memory System API"
        print(f"  [OK] 应用创建成功: {app.title}")
        return True
    except Exception as e:
        print(f"  [FAIL] 应用创建失败: {e}")
        return False


def verify_routes():
    """验证路由注册"""
    print("3. 验证路由注册...")
    try:
        from api.main import app
        routes = [r.path for r in app.routes]
        
        # 检查关键路由
        required_routes = [
            "/api/v1/memory",
            "/api/v1/rag",
            "/api/v1/graph",
            "/api/v1/analytics",
            "/health"
        ]
        
        found = 0
        for req in required_routes:
            matching = [r for r in routes if req in r]
            if matching:
                found += 1
                print(f"  - 找到路由: {req}")
        
        print(f"  [OK] 路由注册成功: {found}/{len(required_routes)}")
        return found >= 3  # 至少3个
    except Exception as e:
        print(f"  [FAIL] 路由验证失败: {e}")
        return False


def verify_schemas():
    """验证Schema模型"""
    print("4. 验证Schema模型...")
    try:
        from api.schemas import (
            DataResponse, ListResponse, MemoryCreate, 
            MemoryResponse, SearchQuery, AskQuery
        )
        
        # 测试创建模型实例
        resp = DataResponse(success=True, message="test")
        assert resp.success == True
        
        mem = MemoryCreate(content="测试内容")
        assert mem.content == "测试内容"
        assert mem.memory_type == "auto"
        
        print("  [OK] Schema模型验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] Schema验证失败: {e}")
        return False


def verify_dependencies():
    """验证依赖注入"""
    print("5. 验证依赖注入...")
    try:
        from api.dependencies import (
            get_memory_service, get_rag_service,
            get_graph_service, get_analytics_service
        )
        
        # 测试获取服务
        mem_service = get_memory_service()
        assert mem_service is not None
        
        rag_service = get_rag_service()
        assert rag_service is not None
        
        print("  [OK] 依赖注入验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] 依赖注入验证失败: {e}")
        return False


def main():
    """运行所有验证"""
    print("=" * 60)
    print("Agent Memory System - 阶段3验证")
    print("API层实现验证")
    print("=" * 60)
    print()
    
    results = {
        "API模块导入": verify_imports(),
        "FastAPI应用": verify_app_creation(),
        "路由注册": verify_routes(),
        "Schema模型": verify_schemas(),
        "依赖注入": verify_dependencies(),
    }
    
    print()
    print("=" * 60)
    print("验证结果:")
    print("-" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print("-" * 60)
    print(f"  通过: {passed}/{total}")
    print("=" * 60)
    
    if passed == total:
        print("阶段3验证通过! API层实现完成。")
        print()
        print("启动API服务器:")
        print("  python -m uvicorn api.main:app --reload --port 8000")
        print()
        print("访问API文档:")
        print("  http://localhost:8000/docs")
        print()
        print("下一步: 可以开始阶段4 - 前端实现")
        return 0
    else:
        print("阶段3验证失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## API端点汇总

### 记忆管理 `/api/v1/memory`

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/` | 添加记忆 |
| GET | `/{memory_id}` | 获取记忆 |
| PUT | `/{memory_id}` | 更新记忆 |
| DELETE | `/{memory_id}` | 删除记忆 |
| POST | `/search` | 搜索记忆 |
| GET | `/list` | 列出记忆 |
| GET | `/stats` | 获取统计 |
| POST | `/consolidate` | 整合记忆 |
| POST | `/forget` | 遗忘记忆 |
| POST | `/export` | 导出记忆 |
| POST | `/import` | 导入记忆 |

### RAG知识库 `/api/v1/rag`

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/documents` | 上传文档 |
| GET | `/documents` | 列出文档 |
| GET | `/documents/{doc_id}` | 获取文档 |
| DELETE | `/documents/{doc_id}` | 删除文档 |
| GET | `/documents/{doc_id}/chunks` | 获取分块 |
| POST | `/search` | 知识检索 |
| POST | `/ask` | 知识问答 |
| GET | `/stats` | 获取统计 |

### 知识图谱 `/api/v1/graph`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/entities` | 列出实体 |
| GET | `/entities/{entity_id}` | 获取实体 |
| POST | `/entities/search` | 搜索实体 |
| GET | `/entities/{entity_id}/related` | 相关实体 |
| POST | `/path` | 查找路径 |
| GET | `/visualization` | 可视化数据 |
| GET | `/stats` | 获取统计 |
| GET | `/health` | 健康检查 |

### 分析统计 `/api/v1/analytics`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/dashboard` | 仪表盘数据 |
| GET | `/trends` | 趋势报告 |
| GET | `/health` | 系统健康 |
| GET | `/query-stats` | 查询统计 |
| POST | `/log-activity` | 记录活动 |

---

## 验证清单

| 任务 | 验证项 | 验证方法 |
|------|--------|----------|
| Task 3.1 | FastAPI应用创建 | `python -c "from api.main import app; print(app.title)"` |
| Task 3.2 | Schema模型 | 验证脚本 |
| Task 3.3 | 记忆API | 启动服务后访问 `/docs` |
| Task 3.4 | RAG API | 启动服务后访问 `/docs` |
| Task 3.5 | 图谱API | 启动服务后访问 `/docs` |
| Task 3.6 | 分析API | 启动服务后访问 `/docs` |
| Task 3.7 | 模块初始化 | 验证脚本 |
| Task 3.8 | 完整验证 | `python scripts/verify_phase3.py` |

---

## 注意事项

1. **异常处理**：所有路由都应有适当的异常处理
2. **参数验证**：使用Pydantic进行严格的参数验证
3. **日志记录**：关键操作应记录日志
4. **文档注释**：每个端点应有清晰的文档字符串
5. **CORS配置**：生产环境应限制允许的源

---

## 下一步

完成阶段3后，可以进入阶段4：前端基础实现，使用Jinja2模板和静态资源创建Web界面。

