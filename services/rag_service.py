"""
文件路径: services/rag_service.py
功能: RAG服务层 - 封装检索增强生成的业务逻辑

提供:
- 文档上传与处理
- 向量检索（基础/高级）
- 问答生成
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import tempfile
import os
import uuid
import logging

from pydantic import BaseModel, Field

from core.rag import load_and_chunk_texts
from core.rag.pipeline import index_chunks, search_vectors, search_vectors_expanded, embed_query
from core.embedding import get_text_embedder, get_dimension
from core.storage import QdrantConnectionManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 请求/响应数据模型 ====================

class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    filename: str = Field(..., description="文件名")
    user_id: str = Field("default", description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    chunk_size: int = Field(800, ge=100, le=4000, description="分块大小")
    chunk_overlap: int = Field(100, ge=0, le=500, description="分块重叠")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    filename: str
    user_id: str
    upload_time: datetime
    chunk_count: int
    total_chars: int
    status: str  # processing/ready/error


class ChunkInfo(BaseModel):
    """分块信息"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., min_length=1, description="查询内容")
    user_id: str = Field("default", description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    limit: int = Field(5, ge=1, le=50, description="返回数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")


class AdvancedSearchRequest(SearchRequest):
    """高级检索请求"""
    use_mqe: bool = Field(False, description="是否使用多查询扩展")
    use_hyde: bool = Field(False, description="是否使用假设文档嵌入")
    use_rerank: bool = Field(True, description="是否使用重排序")


class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, description="问题")
    user_id: str = Field("default", description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    context_limit: int = Field(5, ge=1, le=20, description="上下文数量")
    include_sources: bool = Field(True, description="是否返回来源")


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0


class RAGStatsResponse(BaseModel):
    """RAG统计响应"""
    total_documents: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    collections: List[str] = Field(default_factory=list)


# ==================== 服务类 ====================

class RAGService:
    """
    RAG服务类
    
    封装文档处理、知识检索和问答生成功能。
    """

    def __init__(self):
        """初始化RAG服务"""
        self.settings = get_settings()
        self.embedder = get_text_embedder()
        self.vector_dim = get_dimension()
        
        # 文档元数据存储（内存）
        self._documents: Dict[str, DocumentInfo] = {}
        self._chunks: Dict[str, List[ChunkInfo]] = {}  # doc_id -> chunks
        
        logger.info(f"✅ RAGService 初始化完成，嵌入维度: {self.vector_dim}")

    def _get_vector_store(self, collection_name: str):
        """获取向量存储实例"""
        return QdrantConnectionManager.get_instance(
            collection_name=f"rag_{collection_name}",
            vector_size=self.vector_dim
        )

    # ==================== 文档管理 ====================

    def upload_document(
        self,
        file_content: bytes,
        request: DocumentUploadRequest
    ) -> DocumentInfo:
        """
        上传并处理文档
        
        Args:
            file_content: 文件二进制内容
            request: 上传请求
            
        Returns:
            文档信息
        """
        doc_id = str(uuid.uuid4())
        collection = request.collection_name or request.user_id
        
        # 写入临时文件
        suffix = Path(request.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            # 1. 加载并分块
            chunks = load_and_chunk_texts(
                paths=[tmp_path],
                chunk_size=request.chunk_size,
                overlap=request.chunk_overlap
            )
            
            if not chunks:
                raise ValueError("文档解析失败或内容为空")
            
            # 2. 为每个分块添加文档级元数据
            chunk_infos = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk["id"] = chunk_id
                chunk["metadata"] = {
                    **chunk.get("metadata", {}),
                    "doc_id": doc_id,
                    "filename": request.filename,
                    "user_id": request.user_id,
                    "chunk_index": i,
                    **request.metadata
                }
                
                chunk_infos.append(ChunkInfo(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk["content"],
                    chunk_index=i,
                    metadata=chunk["metadata"]
                ))
            
            # 3. 索引到向量数据库
            store = self._get_vector_store(collection)
            index_chunks(
                store=store,
                chunks=chunks,
                rag_namespace=collection
            )
            
            # 4. 保存文档信息
            total_chars = sum(len(c["content"]) for c in chunks)
            doc_info = DocumentInfo(
                doc_id=doc_id,
                filename=request.filename,
                user_id=request.user_id,
                upload_time=datetime.now(),
                chunk_count=len(chunks),
                total_chars=total_chars,
                status="ready"
            )
            
            self._documents[doc_id] = doc_info
            self._chunks[doc_id] = chunk_infos
            
            logger.info(f"文档上传成功: {request.filename}, 分块数={len(chunks)}")
            return doc_info
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        """获取文档信息"""
        return self._documents.get(doc_id)

    def list_documents(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """列出用户的文档"""
        # 过滤用户文档
        user_docs = [d for d in self._documents.values() if d.user_id == user_id]
        user_docs.sort(key=lambda d: d.upload_time, reverse=True)
        
        # 分页
        total = len(user_docs)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": user_docs[start:end]
        }

    def delete_document(self, doc_id: str, user_id: str) -> bool:
        """删除文档及其分块"""
        doc = self._documents.get(doc_id)
        if not doc or doc.user_id != user_id:
            return False
        
        # 删除向量（通过删除所有该文档的分块ID）
        chunks = self._chunks.get(doc_id, [])
        chunk_ids = [c.chunk_id for c in chunks]
        
        if chunk_ids:
            try:
                store = self._get_vector_store(user_id)
                store.delete_memories(chunk_ids)
            except Exception as e:
                logger.warning(f"删除向量失败: {e}")
        
        # 删除元数据
        del self._documents[doc_id]
        if doc_id in self._chunks:
            del self._chunks[doc_id]
        
        logger.info(f"文档删除成功: {doc_id}")
        return True

    def get_document_chunks(self, doc_id: str) -> List[ChunkInfo]:
        """获取文档的所有分块"""
        return self._chunks.get(doc_id, [])

    # ==================== 检索操作 ====================

    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        基础向量检索
        
        Args:
            request: 检索请求
            
        Returns:
            检索结果列表
        """
        collection = request.collection_name or request.user_id
        
        try:
            store = self._get_vector_store(collection)
            hits = search_vectors(
                store=store,
                query=request.query,
                top_k=request.limit,
                rag_namespace=collection,
                score_threshold=request.score_threshold
            )
            
            # 转换结果
            results = []
            for hit in hits:
                meta = hit.get("metadata", {})
                results.append(SearchResult(
                    chunk_id=hit.get("id", meta.get("memory_id", "")),
                    content=meta.get("content", ""),
                    score=hit.get("score", 0.0),
                    doc_id=meta.get("doc_id"),
                    filename=meta.get("filename"),
                    metadata=meta
                ))
            
            logger.debug(f"检索完成: query='{request.query[:30]}...', 结果数={len(results)}")
            return results
            
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

    def advanced_search(self, request: AdvancedSearchRequest) -> List[SearchResult]:
        """
        高级检索（支持MQE、HyDE）
        
        Args:
            request: 高级检索请求
            
        Returns:
            检索结果列表
        """
        collection = request.collection_name or request.user_id
        
        try:
            store = self._get_vector_store(collection)
            hits = search_vectors_expanded(
                store=store,
                query=request.query,
                top_k=request.limit,
                rag_namespace=collection,
                score_threshold=request.score_threshold,
                enable_mqe=request.use_mqe,
                enable_hyde=request.use_hyde
            )
            
            # 转换结果
            results = []
            for hit in hits:
                meta = hit.get("metadata", {})
                results.append(SearchResult(
                    chunk_id=hit.get("id", meta.get("memory_id", "")),
                    content=meta.get("content", ""),
                    score=hit.get("score", 0.0),
                    doc_id=meta.get("doc_id"),
                    filename=meta.get("filename"),
                    metadata=meta
                ))
            
            # 重排序（简单关键词增强）
            if request.use_rerank and results:
                results = self._rerank(request.query, results)
            
            return results[:request.limit]
            
        except Exception as e:
            logger.warning(f"高级检索失败: {e}")
            return self.search(SearchRequest(
                query=request.query,
                user_id=request.user_id,
                collection_name=request.collection_name,
                limit=request.limit
            ))

    def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """简单重排序：基于关键词匹配增强分数"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            content_lower = result.content.lower()
            content_words = set(content_lower.split())
            
            # 计算关键词重叠
            overlap = len(query_words & content_words)
            boost = overlap / max(len(query_words), 1) * 0.2
            
            # 调整分数
            result.score = result.score * 0.8 + boost
        
        # 重新排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ==================== 问答生成 ====================

    def ask(self, request: AskRequest) -> AskResponse:
        """
        执行问答
        
        Args:
            request: 问答请求
            
        Returns:
            问答响应
        """
        import time
        start_time = time.time()
        
        # 1. 检索相关内容
        retrieval_start = time.time()
        search_req = AdvancedSearchRequest(
            query=request.question,
            user_id=request.user_id,
            collection_name=request.collection_name,
            limit=request.context_limit,
            use_mqe=True,
            use_rerank=True
        )
        sources = self.advanced_search(search_req)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # 2. 构建上下文
        context_parts = []
        for i, src in enumerate(sources):
            context_parts.append(f"[{i+1}] {src.content}")
        context = "\n\n".join(context_parts)
        
        # 3. 生成答案
        gen_start = time.time()
        answer = self._generate_answer(request.question, context)
        gen_time = (time.time() - gen_start) * 1000
        
        return AskResponse(
            answer=answer,
            sources=sources if request.include_sources else [],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=gen_time
        )

    def _generate_answer(self, question: str, context: str) -> str:
        """
        使用LLM生成答案
        
        Args:
            question: 问题
            context: 检索到的上下文
            
        Returns:
            生成的答案
        """
        if not context.strip():
            return "抱歉，没有找到相关信息来回答这个问题。"
        
        # 尝试调用 DashScope LLM
        try:
            from dashscope import Generation
            
            prompt = f"""基于以下参考信息回答问题。如果信息不足，请说明。

参考信息:
{context}

问题: {question}

请提供准确、简洁的答案:"""
            
            response = Generation.call(
                model=self.settings.llm_settings.model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            if response and hasattr(response, 'output'):
                text = response.output.get("text", "")
                if text:
                    return text
                    
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
        
        # 回退：返回上下文摘要
        return f"根据检索到的信息:\n\n{context[:800]}..."

    # ==================== 统计操作 ====================

    def get_stats(self, user_id: Optional[str] = None) -> RAGStatsResponse:
        """获取RAG统计信息"""
        if user_id:
            docs = [d for d in self._documents.values() if d.user_id == user_id]
        else:
            docs = list(self._documents.values())
        
        total_chunks = sum(d.chunk_count for d in docs)
        total_chars = sum(d.total_chars for d in docs)
        collections = list(set(d.user_id for d in docs))
        
        return RAGStatsResponse(
            total_documents=len(docs),
            total_chunks=total_chunks,
            total_chars=total_chars,
            collections=collections
        )

