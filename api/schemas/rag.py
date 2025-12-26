"""
文件路径: api/schemas/rag.py
功能: RAG相关的Pydantic模型

定义RAG知识库API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUpload(BaseModel):
    """
    文档上传请求（JSON方式）
    
    用于通过Base64编码上传文档。
    """
    filename: str = Field(..., description="文件名")
    content_base64: str = Field(..., description="Base64编码的文件内容")
    chunk_size: int = Field(800, ge=100, le=4000, description="分块大小")
    chunk_overlap: int = Field(100, ge=0, le=500, description="分块重叠")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class DocumentInfo(BaseModel):
    """文档信息响应"""
    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    user_id: str = Field(..., description="用户ID")
    upload_time: datetime = Field(..., description="上传时间")
    chunk_count: int = Field(..., description="分块数量")
    total_chars: int = Field(..., description="总字符数")
    status: str = Field(..., description="状态: processing/ready/error")


class ChunkInfo(BaseModel):
    """分块信息响应"""
    chunk_id: str = Field(..., description="分块ID")
    doc_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="分块内容")
    chunk_index: int = Field(..., description="分块索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分块元数据")


class SearchQuery(BaseModel):
    """
    检索请求
    
    用于在知识库中进行语义检索。
    """
    query: str = Field(..., min_length=1, description="查询内容")
    limit: int = Field(5, ge=1, le=50, description="返回数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    use_advanced: bool = Field(False, description="是否使用高级检索(MQE/重排序)")


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str = Field(..., description="分块ID")
    content: str = Field(..., description="内容")
    score: float = Field(..., description="相似度分数")
    doc_id: Optional[str] = Field(None, description="文档ID")
    filename: Optional[str] = Field(None, description="文件名")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AskQuery(BaseModel):
    """
    问答请求
    
    基于知识库的问答。
    """
    question: str = Field(..., min_length=1, description="问题")
    context_limit: int = Field(5, ge=1, le=20, description="上下文数量")
    include_sources: bool = Field(True, description="是否返回来源")


class AskResult(BaseModel):
    """问答结果"""
    answer: str = Field(..., description="答案")
    sources: List[SearchResult] = Field(default_factory=list, description="来源")
    retrieval_time_ms: float = Field(0, description="检索耗时(毫秒)")
    generation_time_ms: float = Field(0, description="生成耗时(毫秒)")


class RAGStats(BaseModel):
    """RAG统计信息"""
    total_documents: int = Field(0, description="文档总数")
    total_chunks: int = Field(0, description="分块总数")
    total_chars: int = Field(0, description="总字符数")
    collections: List[str] = Field(default_factory=list, description="集合列表")

