"""
文件路径: api/routes/rag.py
功能: RAG知识库API路由

提供文档管理、知识检索和问答接口。
"""

from typing import Optional
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, Path, UploadFile, File
import logging

from api.dependencies import get_rag_service, get_current_user_id
from api.schemas.common import DataResponse, ListResponse
from api.schemas.rag import (
    DocumentUpload, DocumentInfo, ChunkInfo,
    SearchQuery, SearchResult, AskQuery, AskResult, RAGStats
)
from services import (
    RAGService, 
    DocumentUploadRequest, 
    SearchRequest, 
    AdvancedSearchRequest, 
    AskRequest
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


# ==================== 文档管理 ====================

@router.post("/documents", response_model=DataResponse[DocumentInfo], summary="上传文档")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(800, ge=100, le=4000, description="分块大小"),
    chunk_overlap: int = Query(100, ge=0, le=500, description="分块重叠"),
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """
    上传文档到RAG知识库
    
    支持多种格式：PDF、Word、Markdown、TXT等。
    文档会被自动解析、分块并向量化存储。
    """
    try:
        # 读取文件内容
        content = await file.read()
        
        # 构建请求
        req = DocumentUploadRequest(
            filename=file.filename or "unknown",
            user_id=user_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 调用服务
        result = service.upload_document(content, req)
        
        return DataResponse(
            success=True,
            message="文档上传成功",
            data=DocumentInfo(**result.model_dump())
        )
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/json", response_model=DataResponse[DocumentInfo], summary="上传文档(JSON)")
async def upload_document_json(
    request: DocumentUpload,
    user_id: str = Depends(get_current_user_id),
    service: RAGService = Depends(get_rag_service)
):
    """
    通过JSON方式上传文档
    
    文件内容需Base64编码，适用于API调用。
    """
    try:
        # 解码Base64内容
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
        logger.error(f"文档上传失败: {e}")
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
    """获取文档详细信息"""
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
        # 高级检索
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
        # 基础检索
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
    
    系统会检索相关内容，并使用LLM生成答案。
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
    """获取RAG知识库统计信息"""
    stats = service.get_stats(user_id=user_id)
    return DataResponse(success=True, data=RAGStats(**stats.model_dump()))

