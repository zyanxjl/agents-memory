# 阶段2：服务层实现 - 详细任务规划

## 概述

**目标**：实现业务服务层，封装核心模块功能，为API层提供统一的业务接口。

**预计时间**：2天

**输出目录**：`services/`

**依赖**：Phase 1 核心模块重构完成

---

## 目录结构

```
services/
├── __init__.py              # 服务层统一导出
├── memory_service.py        # 记忆服务（CRUD、搜索、管理）
├── rag_service.py           # RAG服务（文档处理、检索、问答）
├── graph_service.py         # 图谱服务（查询、可视化）
└── analytics_service.py     # 分析服务（统计、报告）
```

---

## Task 2.1：MemoryService 实现

### 2.1.1 功能描述

MemoryService 是记忆系统的业务服务层，封装 MemoryManager 和各类记忆类型的操作，提供：
- 记忆的增删改查（CRUD）
- 多类型记忆的统一搜索
- 记忆整合与遗忘策略
- 记忆导入导出

### 2.1.2 类设计

```python
# services/memory_service.py

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from core.memory import MemoryManager, MemoryItem, MemoryConfig
from core.memory import WorkingMemory, EpisodicMemory, SemanticMemory, PerceptualMemory
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class MemoryCreateRequest(BaseModel):
    """创建记忆请求"""
    content: str = Field(..., min_length=1, max_length=10000, description="记忆内容")
    memory_type: str = Field("auto", description="记忆类型: working/episodic/semantic/perceptual/auto")
    user_id: str = Field(..., description="用户ID")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="重要性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "用户询问了Python函数的使用方法",
                "memory_type": "episodic",
                "user_id": "user_001",
                "importance": 0.7,
                "metadata": {"session_id": "sess_abc", "source": "chat"}
            }
        }


class MemoryUpdateRequest(BaseModel):
    """更新记忆请求"""
    content: Optional[str] = Field(None, max_length=10000, description="新内容")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="新重要性")
    metadata: Optional[Dict[str, Any]] = Field(None, description="要更新的元数据")


class MemorySearchRequest(BaseModel):
    """搜索记忆请求"""
    query: str = Field(..., min_length=1, description="搜索查询")
    memory_types: List[str] = Field(default_factory=lambda: ["working", "episodic", "semantic"], 
                                     description="要搜索的记忆类型")
    user_id: Optional[str] = Field(None, description="用户ID过滤")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")
    importance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="最低重要性阈值")
    time_range_start: Optional[datetime] = Field(None, description="时间范围起始")
    time_range_end: Optional[datetime] = Field(None, description="时间范围结束")


class MemoryResponse(BaseModel):
    """记忆响应"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None  # 搜索时的相关性分数
    
    @classmethod
    def from_memory_item(cls, item: MemoryItem, relevance_score: float = None) -> "MemoryResponse":
        """从MemoryItem转换"""
        return cls(
            id=item.id,
            content=item.content,
            memory_type=item.memory_type,
            user_id=item.user_id,
            timestamp=item.timestamp,
            importance=item.importance,
            metadata=item.metadata,
            relevance_score=relevance_score
        )


class MemoryStatsResponse(BaseModel):
    """记忆统计响应"""
    total_count: int = Field(..., description="总记忆数")
    working_count: int = Field(0, description="工作记忆数")
    episodic_count: int = Field(0, description="情景记忆数")
    semantic_count: int = Field(0, description="语义记忆数")
    perceptual_count: int = Field(0, description="感知记忆数")
    avg_importance: float = Field(0.0, description="平均重要性")
    storage_stats: Dict[str, Any] = Field(default_factory=dict, description="存储统计")


class ConsolidateRequest(BaseModel):
    """记忆整合请求"""
    user_id: str = Field(..., description="用户ID")
    source_type: str = Field("working", description="源记忆类型")
    target_type: str = Field("episodic", description="目标记忆类型")
    importance_threshold: float = Field(0.3, ge=0.0, le=1.0, description="整合阈值")


class ForgetRequest(BaseModel):
    """记忆遗忘请求"""
    user_id: Optional[str] = Field(None, description="用户ID")
    memory_type: Optional[str] = Field(None, description="记忆类型")
    strategy: str = Field("importance_based", description="遗忘策略: importance_based/time_based/capacity_based")
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="遗忘阈值")
    max_age_days: int = Field(30, ge=1, description="最大保留天数（time_based策略）")


# ==================== 服务类 ====================

class MemoryService:
    """
    记忆服务类
    封装记忆系统的所有业务逻辑，提供统一的服务接口。
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化记忆服务。
        Args:
            config: 可选的记忆配置，如果不提供则使用默认配置。
        """
        self.config = config or MemoryConfig()
        self.manager = MemoryManager(self.config)
        logger.info("✅ MemoryService 初始化完成")
    
    # ==================== CRUD 操作 ====================
    
    def add_memory(self, request: MemoryCreateRequest) -> MemoryResponse:
        """
        添加新记忆。
        Args:
            request: 创建记忆请求。
        Returns:
            创建的记忆响应。
        """
        # 构建MemoryItem
        memory_item = MemoryItem(
            id="",  # 由底层生成
            content=request.content,
            memory_type=request.memory_type if request.memory_type != "auto" else "working",
            user_id=request.user_id,
            timestamp=datetime.now(),
            importance=request.importance,
            metadata=request.metadata
        )
        
        # 自动分类（如果需要）
        if request.memory_type == "auto":
            memory_item.memory_type = self._auto_classify(request.content)
        
        # 通过Manager添加
        memory_id = self.manager.add_memory(memory_item)
        memory_item.id = memory_id
        
        logger.info(f"添加记忆: ID={memory_id[:8]}..., 类型={memory_item.memory_type}, 用户={request.user_id}")
        return MemoryResponse.from_memory_item(memory_item)
    
    def get_memory(self, memory_id: str) -> Optional[MemoryResponse]:
        """
        获取单个记忆。
        Args:
            memory_id: 记忆ID。
        Returns:
            记忆响应，如果不存在返回None。
        """
        item = self.manager.get_memory(memory_id)
        if item:
            return MemoryResponse.from_memory_item(item)
        return None
    
    def update_memory(self, memory_id: str, request: MemoryUpdateRequest) -> bool:
        """
        更新记忆。
        Args:
            memory_id: 记忆ID。
            request: 更新请求。
        Returns:
            是否更新成功。
        """
        success = self.manager.update_memory(
            memory_id=memory_id,
            content=request.content,
            importance=request.importance,
            metadata=request.metadata
        )
        if success:
            logger.info(f"更新记忆: ID={memory_id[:8]}...")
        return success
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆。
        Args:
            memory_id: 记忆ID。
        Returns:
            是否删除成功。
        """
        success = self.manager.remove_memory(memory_id)
        if success:
            logger.info(f"删除记忆: ID={memory_id[:8]}...")
        return success
    
    # ==================== 搜索操作 ====================
    
    def search_memories(self, request: MemorySearchRequest) -> List[MemoryResponse]:
        """
        搜索记忆（多类型并行检索）。
        Args:
            request: 搜索请求。
        Returns:
            匹配的记忆列表，按相关性排序。
        """
        all_results: List[tuple] = []  # (score, MemoryItem)
        
        # 构建时间范围
        time_range = None
        if request.time_range_start or request.time_range_end:
            time_range = (
                request.time_range_start or datetime.min,
                request.time_range_end or datetime.max
            )
        
        # 并行搜索各类型记忆
        for mem_type in request.memory_types:
            try:
                results = self.manager.search(
                    query=request.query,
                    memory_type=mem_type,
                    user_id=request.user_id,
                    limit=request.limit * 2,  # 获取更多候选进行重排
                    importance_threshold=request.importance_threshold,
                    time_range=time_range
                )
                for item in results:
                    score = item.metadata.get("relevance_score", 0.5)
                    all_results.append((score, item))
            except Exception as e:
                logger.warning(f"搜索 {mem_type} 记忆失败: {e}")
        
        # 按相关性排序并去重
        all_results.sort(key=lambda x: x[0], reverse=True)
        seen_ids = set()
        final_results = []
        
        for score, item in all_results:
            if item.id not in seen_ids and len(final_results) < request.limit:
                seen_ids.add(item.id)
                final_results.append(MemoryResponse.from_memory_item(item, relevance_score=score))
        
        logger.debug(f"搜索完成: query='{request.query[:20]}...', 结果数={len(final_results)}")
        return final_results
    
    def list_memories(
        self,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "timestamp",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        分页列出记忆。
        Args:
            memory_type: 可选的记忆类型过滤。
            user_id: 可选的用户ID过滤。
            page: 页码（从1开始）。
            page_size: 每页数量。
            sort_by: 排序字段 (timestamp/importance)。
            sort_order: 排序方向 (asc/desc)。
        Returns:
            包含分页信息和记忆列表的字典。
        """
        # 获取所有记忆
        all_memories = self.manager.get_all_memories(
            memory_type=memory_type,
            user_id=user_id
        )
        
        # 排序
        reverse = sort_order == "desc"
        if sort_by == "importance":
            all_memories.sort(key=lambda m: m.importance, reverse=reverse)
        else:  # 默认按时间排序
            all_memories.sort(key=lambda m: m.timestamp, reverse=reverse)
        
        # 分页
        total = len(all_memories)
        start = (page - 1) * page_size
        end = start + page_size
        page_memories = all_memories[start:end]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "items": [MemoryResponse.from_memory_item(m) for m in page_memories]
        }
    
    # ==================== 管理操作 ====================
    
    def get_stats(self, user_id: Optional[str] = None) -> MemoryStatsResponse:
        """
        获取记忆统计信息。
        Args:
            user_id: 可选的用户ID过滤。
        Returns:
            记忆统计响应。
        """
        stats = self.manager.get_stats(user_id=user_id)
        
        return MemoryStatsResponse(
            total_count=stats.get("total_count", 0),
            working_count=stats.get("working_count", 0),
            episodic_count=stats.get("episodic_count", 0),
            semantic_count=stats.get("semantic_count", 0),
            perceptual_count=stats.get("perceptual_count", 0),
            avg_importance=stats.get("avg_importance", 0.0),
            storage_stats=stats.get("storage_stats", {})
        )
    
    def consolidate(self, request: ConsolidateRequest) -> Dict[str, Any]:
        """
        执行记忆整合（将短期记忆转化为长期记忆）。
        Args:
            request: 整合请求。
        Returns:
            整合结果统计。
        """
        result = self.manager.consolidate(
            user_id=request.user_id,
            source_type=request.source_type,
            target_type=request.target_type,
            importance_threshold=request.importance_threshold
        )
        
        logger.info(f"记忆整合完成: {result.get('consolidated_count', 0)} 条记忆从 {request.source_type} 转移到 {request.target_type}")
        return result
    
    def forget(self, request: ForgetRequest) -> Dict[str, Any]:
        """
        执行记忆遗忘。
        Args:
            request: 遗忘请求。
        Returns:
            遗忘结果统计。
        """
        forgotten_count = self.manager.forget(
            user_id=request.user_id,
            memory_type=request.memory_type,
            strategy=request.strategy,
            threshold=request.threshold,
            max_age_days=request.max_age_days
        )
        
        logger.info(f"记忆遗忘完成: {forgotten_count} 条记忆被删除 (策略: {request.strategy})")
        return {
            "forgotten_count": forgotten_count,
            "strategy": request.strategy,
            "threshold": request.threshold
        }
    
    def export_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        导出记忆。
        Args:
            user_id: 可选的用户ID过滤。
            memory_type: 可选的记忆类型过滤。
            format: 导出格式 (json/csv)。
        Returns:
            导出数据。
        """
        memories = self.manager.get_all_memories(
            memory_type=memory_type,
            user_id=user_id
        )
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_count": len(memories),
            "filters": {
                "user_id": user_id,
                "memory_type": memory_type
            },
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type,
                    "user_id": m.user_id,
                    "timestamp": m.timestamp.isoformat(),
                    "importance": m.importance,
                    "metadata": m.metadata
                }
                for m in memories
            ]
        }
        
        logger.info(f"导出记忆: {len(memories)} 条")
        return export_data
    
    def import_memories(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        导入记忆。
        Args:
            data: 导入数据（export_memories的输出格式）。
            user_id: 导入到的用户ID。
        Returns:
            导入结果统计。
        """
        imported_count = 0
        failed_count = 0
        
        memories = data.get("memories", [])
        for mem_data in memories:
            try:
                request = MemoryCreateRequest(
                    content=mem_data["content"],
                    memory_type=mem_data.get("memory_type", "episodic"),
                    user_id=user_id,  # 使用新的用户ID
                    importance=mem_data.get("importance", 0.5),
                    metadata=mem_data.get("metadata", {})
                )
                self.add_memory(request)
                imported_count += 1
            except Exception as e:
                logger.warning(f"导入记忆失败: {e}")
                failed_count += 1
        
        logger.info(f"导入记忆完成: 成功={imported_count}, 失败={failed_count}")
        return {
            "imported_count": imported_count,
            "failed_count": failed_count,
            "total": len(memories)
        }
    
    # ==================== 辅助方法 ====================
    
    def _auto_classify(self, content: str) -> str:
        """
        自动分类记忆类型。
        基于内容特征判断应该存储为哪种记忆类型。
        Args:
            content: 记忆内容。
        Returns:
            记忆类型字符串。
        """
        content_lower = content.lower()
        
        # 简单规则分类
        # 1. 事件性内容 -> 情景记忆
        event_keywords = ["发生", "完成", "开始", "结束", "做了", "happened", "did", "completed"]
        if any(kw in content_lower for kw in event_keywords):
            return "episodic"
        
        # 2. 概念性内容 -> 语义记忆
        concept_keywords = ["是什么", "定义", "概念", "原理", "what is", "definition", "means"]
        if any(kw in content_lower for kw in concept_keywords):
            return "semantic"
        
        # 3. 短小内容 -> 工作记忆
        if len(content) < 100:
            return "working"
        
        # 4. 默认 -> 情景记忆
        return "episodic"
```

### 2.1.3 验证方法

```python
# scripts/verify_memory_service.py

import sys
sys.path.insert(0, ".")

from datetime import datetime

def verify_memory_service():
    """验证MemoryService"""
    print("验证MemoryService...")
    
    from services.memory_service import (
        MemoryService, 
        MemoryCreateRequest, 
        MemoryUpdateRequest,
        MemorySearchRequest,
        ConsolidateRequest,
        ForgetRequest
    )
    
    # 1. 初始化服务
    service = MemoryService()
    print("  [OK] 服务初始化")
    
    # 2. 添加记忆
    create_req = MemoryCreateRequest(
        content="Python是一种高级编程语言，广泛用于数据科学和人工智能领域。",
        memory_type="semantic",
        user_id="test_user",
        importance=0.8,
        metadata={"source": "test"}
    )
    response = service.add_memory(create_req)
    assert response.id, "记忆ID不应为空"
    assert response.content == create_req.content
    memory_id = response.id
    print(f"  [OK] 添加记忆: {memory_id[:8]}...")
    
    # 3. 获取记忆
    retrieved = service.get_memory(memory_id)
    assert retrieved is not None, "应能获取到记忆"
    assert retrieved.content == create_req.content
    print("  [OK] 获取记忆")
    
    # 4. 更新记忆
    update_req = MemoryUpdateRequest(importance=0.9)
    success = service.update_memory(memory_id, update_req)
    assert success, "更新应成功"
    updated = service.get_memory(memory_id)
    assert updated.importance == 0.9
    print("  [OK] 更新记忆")
    
    # 5. 搜索记忆
    search_req = MemorySearchRequest(
        query="Python编程语言",
        memory_types=["semantic", "episodic"],
        user_id="test_user",
        limit=5
    )
    results = service.search_memories(search_req)
    assert len(results) > 0, "应有搜索结果"
    print(f"  [OK] 搜索记忆: 找到 {len(results)} 条")
    
    # 6. 获取统计
    stats = service.get_stats(user_id="test_user")
    assert stats.total_count > 0
    print(f"  [OK] 获取统计: 总数={stats.total_count}")
    
    # 7. 列出记忆
    list_result = service.list_memories(user_id="test_user", page=1, page_size=10)
    assert "items" in list_result
    assert list_result["total"] > 0
    print(f"  [OK] 列出记忆: 总数={list_result['total']}")
    
    # 8. 导出记忆
    export_data = service.export_memories(user_id="test_user")
    assert "memories" in export_data
    assert len(export_data["memories"]) > 0
    print(f"  [OK] 导出记忆: {len(export_data['memories'])} 条")
    
    # 9. 删除记忆
    success = service.delete_memory(memory_id)
    assert success, "删除应成功"
    deleted = service.get_memory(memory_id)
    assert deleted is None, "删除后应获取不到"
    print("  [OK] 删除记忆")
    
    print("MemoryService验证通过!")
    return True

if __name__ == "__main__":
    verify_memory_service()
```

---

## Task 2.2：RAGService 实现

### 2.2.1 功能描述

RAGService 是 RAG（检索增强生成）系统的业务服务层，封装文档处理、向量检索和问答生成功能：
- 文档上传与处理（解析、分块、向量化）
- 知识检索（基础检索、高级检索）
- 问答生成（结合LLM）

### 2.2.2 类设计

```python
# services/rag_service.py

from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
from pathlib import Path
import tempfile
import os
import logging

from pydantic import BaseModel, Field

from core.rag import RAGPipeline, DocumentProcessor, TextChunker
from core.embedding import get_text_embedder, get_dimension
from core.storage import QdrantConnectionManager
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    filename: str = Field(..., description="文件名")
    user_id: str = Field(..., description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称，默认使用用户ID")
    chunk_size: int = Field(800, ge=100, le=4000, description="分块大小（字符数）")
    chunk_overlap: int = Field(100, ge=0, le=500, description="分块重叠大小")
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
    metadata: Dict[str, Any]


class ChunkInfo(BaseModel):
    """分块信息"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., min_length=1, description="查询内容")
    user_id: str = Field(..., description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    limit: int = Field(5, ge=1, le=50, description="返回结果数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="元数据过滤条件")


class AdvancedSearchRequest(SearchRequest):
    """高级检索请求"""
    use_mge: bool = Field(False, description="是否使用多查询扩展(MQE)")
    use_hyde: bool = Field(False, description="是否使用假设文档嵌入(HyDE)")
    use_rerank: bool = Field(True, description="是否使用重排序")
    expand_queries: int = Field(3, ge=1, le=5, description="MQE扩展查询数量")


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
    user_id: str = Field(..., description="用户ID")
    collection_name: Optional[str] = Field(None, description="集合名称")
    context_limit: int = Field(5, ge=1, le=20, description="上下文数量限制")
    use_advanced_retrieval: bool = Field(True, description="是否使用高级检索")
    include_sources: bool = Field(True, description="是否返回引用来源")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="生成温度")


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0


class RAGStatsResponse(BaseModel):
    """RAG统计响应"""
    total_documents: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    collections: List[str] = Field(default_factory=list)
    vector_store_stats: Dict[str, Any] = Field(default_factory=dict)


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
        
        # 文档处理器
        self.doc_processor = DocumentProcessor()
        self.chunker = TextChunker()
        
        # 文档元数据存储（内存中，生产环境应使用数据库）
        self._documents: Dict[str, DocumentInfo] = {}
        self._chunks: Dict[str, List[ChunkInfo]] = {}  # doc_id -> chunks
        
        logger.info("✅ RAGService 初始化完成")
    
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
        上传并处理文档。
        Args:
            file_content: 文件二进制内容。
            request: 上传请求。
        Returns:
            文档信息。
        """
        import uuid
        doc_id = str(uuid.uuid4())
        collection_name = request.collection_name or request.user_id
        
        # 创建临时文件
        suffix = Path(request.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # 1. 解析文档
            text_content = self.doc_processor.process_file(tmp_path)
            if not text_content:
                raise ValueError("文档解析失败或内容为空")
            
            # 2. 分块
            chunks = self.chunker.chunk_text(
                text_content,
                chunk_size=request.chunk_size,
                overlap=request.chunk_overlap
            )
            
            # 3. 向量化并存储
            vector_store = self._get_vector_store(collection_name)
            chunk_infos = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # 生成向量
                embedding = self.embedder.encode(chunk["text"])
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                
                # 构建元数据
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": request.filename,
                    "user_id": request.user_id,
                    "chunk_index": i,
                    "content": chunk["text"][:500],  # 存储内容摘要
                    **request.metadata
                }
                
                # 存储向量
                vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[chunk_metadata],
                    ids=[chunk_id]
                )
                
                # 记录分块信息
                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk["text"],
                    chunk_index=i,
                    start_char=chunk.get("start", 0),
                    end_char=chunk.get("end", len(chunk["text"])),
                    metadata=chunk_metadata
                )
                chunk_infos.append(chunk_info)
            
            # 4. 记录文档信息
            doc_info = DocumentInfo(
                doc_id=doc_id,
                filename=request.filename,
                user_id=request.user_id,
                upload_time=datetime.now(),
                chunk_count=len(chunks),
                total_chars=len(text_content),
                status="ready",
                metadata=request.metadata
            )
            
            self._documents[doc_id] = doc_info
            self._chunks[doc_id] = chunk_infos
            
            logger.info(f"文档上传成功: {request.filename}, 分块数={len(chunks)}")
            return doc_info
            
        finally:
            # 清理临时文件
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
        user_docs = [
            doc for doc in self._documents.values()
            if doc.user_id == user_id
        ]
        user_docs.sort(key=lambda d: d.upload_time, reverse=True)
        
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
        """删除文档及其所有分块"""
        doc = self._documents.get(doc_id)
        if not doc or doc.user_id != user_id:
            return False
        
        # 删除向量
        collection_name = user_id
        vector_store = self._get_vector_store(collection_name)
        chunks = self._chunks.get(doc_id, [])
        chunk_ids = [c.chunk_id for c in chunks]
        
        if chunk_ids:
            vector_store.delete_memories(chunk_ids)
        
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
        基础向量检索。
        Args:
            request: 检索请求。
        Returns:
            检索结果列表。
        """
        collection_name = request.collection_name or request.user_id
        vector_store = self._get_vector_store(collection_name)
        
        # 生成查询向量
        query_embedding = self.embedder.encode(request.query)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        
        # 构建过滤条件
        where_filter = {"user_id": request.user_id}
        if request.filter_metadata:
            where_filter.update(request.filter_metadata)
        
        # 执行检索
        hits = vector_store.search_similar(
            query_vector=query_embedding,
            limit=request.limit,
            where=where_filter,
            score_threshold=request.score_threshold
        )
        
        # 转换结果
        results = []
        for hit in hits:
            meta = hit.get("metadata", {})
            # 获取完整内容
            chunk_id = meta.get("chunk_id")
            full_content = meta.get("content", "")
            
            # 尝试从内存获取完整内容
            doc_id = meta.get("doc_id")
            if doc_id and doc_id in self._chunks:
                for chunk in self._chunks[doc_id]:
                    if chunk.chunk_id == chunk_id:
                        full_content = chunk.content
                        break
            
            results.append(SearchResult(
                chunk_id=chunk_id or hit.get("id", ""),
                content=full_content,
                score=hit.get("score", 0.0),
                doc_id=doc_id,
                filename=meta.get("filename"),
                metadata=meta
            ))
        
        logger.debug(f"检索完成: query='{request.query[:30]}...', 结果数={len(results)}")
        return results
    
    def advanced_search(self, request: AdvancedSearchRequest) -> List[SearchResult]:
        """
        高级检索（支持MQE、HyDE、重排序）。
        Args:
            request: 高级检索请求。
        Returns:
            检索结果列表。
        """
        all_results: List[SearchResult] = []
        queries = [request.query]
        
        # 1. 多查询扩展 (MQE)
        if request.use_mge:
            expanded = self._expand_query(request.query, request.expand_queries)
            queries.extend(expanded)
            logger.debug(f"MQE扩展查询: {queries}")
        
        # 2. 假设文档嵌入 (HyDE)
        if request.use_hyde:
            hypothetical = self._generate_hypothetical_answer(request.query)
            if hypothetical:
                queries.append(hypothetical)
                logger.debug(f"HyDE假设答案: {hypothetical[:100]}...")
        
        # 3. 对每个查询执行检索
        seen_ids = set()
        for q in queries:
            base_request = SearchRequest(
                query=q,
                user_id=request.user_id,
                collection_name=request.collection_name,
                limit=request.limit * 2,
                score_threshold=request.score_threshold,
                filter_metadata=request.filter_metadata
            )
            results = self.search(base_request)
            
            for r in results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    all_results.append(r)
        
        # 4. 重排序
        if request.use_rerank and len(all_results) > 1:
            all_results = self._rerank(request.query, all_results)
        
        return all_results[:request.limit]
    
    def _expand_query(self, query: str, count: int = 3) -> List[str]:
        """
        查询扩展（简单实现）。
        生产环境应使用LLM生成更多查询变体。
        """
        # 简单的同义词扩展
        expansions = []
        
        # 添加问句形式
        if not query.endswith("?"):
            expansions.append(f"什么是{query}?")
            expansions.append(f"如何理解{query}?")
        
        # 添加更具体的查询
        expansions.append(f"{query}的详细说明")
        
        return expansions[:count]
    
    def _generate_hypothetical_answer(self, query: str) -> Optional[str]:
        """
        生成假设答案（HyDE）。
        生产环境应使用LLM生成。
        """
        # 简单模板
        return f"关于'{query}'的答案是：这是一个关于{query}的详细解释，包含了相关的概念、原理和应用。"
    
    def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        结果重排序。
        使用简单的关键词匹配增强，生产环境应使用Cross-Encoder。
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            content_lower = result.content.lower()
            content_words = set(content_lower.split())
            
            # 计算关键词重叠度
            overlap = len(query_words & content_words)
            keyword_boost = overlap / max(len(query_words), 1) * 0.2
            
            # 调整分数
            result.score = result.score * 0.8 + keyword_boost
        
        # 重新排序
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    # ==================== 问答生成 ====================
    
    def ask(self, request: AskRequest) -> AskResponse:
        """
        执行问答。
        Args:
            request: 问答请求。
        Returns:
            问答响应。
        """
        import time
        start_time = time.time()
        
        # 1. 检索相关内容
        retrieval_start = time.time()
        if request.use_advanced_retrieval:
            search_request = AdvancedSearchRequest(
                query=request.question,
                user_id=request.user_id,
                collection_name=request.collection_name,
                limit=request.context_limit,
                use_mge=True,
                use_rerank=True
            )
            sources = self.advanced_search(search_request)
        else:
            search_request = SearchRequest(
                query=request.question,
                user_id=request.user_id,
                collection_name=request.collection_name,
                limit=request.context_limit
            )
            sources = self.search(search_request)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # 2. 构建上下文
        context_parts = []
        for i, source in enumerate(sources):
            context_parts.append(f"[{i+1}] {source.content}")
        context = "\n\n".join(context_parts)
        
        # 3. 生成答案
        generation_start = time.time()
        answer = self._generate_answer(request.question, context, request.temperature)
        generation_time = (time.time() - generation_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return AskResponse(
            answer=answer,
            sources=sources if request.include_sources else [],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time
        )
    
    def _generate_answer(self, question: str, context: str, temperature: float) -> str:
        """
        使用LLM生成答案。
        Args:
            question: 问题。
            context: 检索到的上下文。
            temperature: 生成温度。
        Returns:
            生成的答案。
        """
        if not context.strip():
            return "抱歉，我没有找到相关信息来回答这个问题。"
        
        # 尝试调用LLM
        try:
            from dashscope import Generation
            
            prompt = f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请说明。

参考信息:
{context}

问题: {question}

请提供准确、简洁的答案:"""
            
            response = Generation.call(
                model=self.settings.llm_settings.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=1000
            )
            
            if response and response.output:
                return response.output.get("text", "生成失败")
            
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
        
        # 回退：返回上下文摘要
        return f"根据检索到的信息:\n\n{context[:500]}..."
    
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
```

### 2.2.3 验证方法

```python
# scripts/verify_rag_service.py

import sys
sys.path.insert(0, ".")

def verify_rag_service():
    """验证RAGService"""
    print("验证RAGService...")
    
    from services.rag_service import (
        RAGService,
        DocumentUploadRequest,
        SearchRequest,
        AdvancedSearchRequest,
        AskRequest
    )
    
    # 1. 初始化服务
    service = RAGService()
    print("  [OK] 服务初始化")
    
    # 2. 上传测试文档
    test_content = b"""# Python Programming Guide

Python is a high-level programming language known for its simplicity and readability.

## Variables and Data Types

Python supports various data types including integers, floats, strings, and lists.

## Functions

Functions in Python are defined using the `def` keyword.

```python
def greet(name):
    return f"Hello, {name}!"
```

## Classes

Python supports object-oriented programming with classes.
"""
    
    upload_request = DocumentUploadRequest(
        filename="python_guide.md",
        user_id="test_user",
        chunk_size=200,
        chunk_overlap=50,
        metadata={"category": "programming"}
    )
    
    doc_info = service.upload_document(test_content, upload_request)
    assert doc_info.doc_id, "文档ID不应为空"
    assert doc_info.chunk_count > 0, "应有分块"
    print(f"  [OK] 上传文档: {doc_info.filename}, 分块数={doc_info.chunk_count}")
    
    # 3. 获取文档
    retrieved_doc = service.get_document(doc_info.doc_id)
    assert retrieved_doc is not None
    print("  [OK] 获取文档")
    
    # 4. 列出文档
    doc_list = service.list_documents(user_id="test_user")
    assert doc_list["total"] > 0
    print(f"  [OK] 列出文档: 总数={doc_list['total']}")
    
    # 5. 获取分块
    chunks = service.get_document_chunks(doc_info.doc_id)
    assert len(chunks) > 0
    print(f"  [OK] 获取分块: {len(chunks)} 个")
    
    # 6. 基础检索
    search_request = SearchRequest(
        query="Python functions",
        user_id="test_user",
        limit=3
    )
    results = service.search(search_request)
    assert len(results) > 0, "应有检索结果"
    print(f"  [OK] 基础检索: 结果数={len(results)}")
    
    # 7. 高级检索
    adv_search_request = AdvancedSearchRequest(
        query="How to define functions in Python?",
        user_id="test_user",
        limit=3,
        use_mge=True,
        use_rerank=True
    )
    adv_results = service.advanced_search(adv_search_request)
    assert len(adv_results) > 0
    print(f"  [OK] 高级检索: 结果数={len(adv_results)}")
    
    # 8. 问答（不依赖LLM的基础测试）
    ask_request = AskRequest(
        question="What is Python?",
        user_id="test_user",
        context_limit=3,
        use_advanced_retrieval=False
    )
    answer = service.ask(ask_request)
    assert answer.answer, "应有答案"
    print(f"  [OK] 问答: 耗时={answer.total_time_ms:.1f}ms")
    
    # 9. 获取统计
    stats = service.get_stats(user_id="test_user")
    assert stats.total_documents > 0
    print(f"  [OK] 获取统计: 文档数={stats.total_documents}, 分块数={stats.total_chunks}")
    
    # 10. 删除文档
    success = service.delete_document(doc_info.doc_id, "test_user")
    assert success
    print("  [OK] 删除文档")
    
    print("RAGService验证通过!")
    return True

if __name__ == "__main__":
    verify_rag_service()
```

---

## Task 2.3：GraphService 实现

### 2.3.1 功能描述

GraphService 是知识图谱的业务服务层，封装 Neo4j 图数据库的操作，提供：
- 实体和关系的查询
- 图谱遍历和路径查找
- 可视化数据生成（用于前端展示）
- 图谱统计分析

### 2.3.2 类设计

```python
# services/graph_service.py

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from core.storage import Neo4jGraphStore
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class EntityInfo(BaseModel):
    """实体信息"""
    id: str = Field(..., description="实体ID")
    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    related_memory_count: int = Field(0, description="关联记忆数量")


class RelationshipInfo(BaseModel):
    """关系信息"""
    from_entity_id: str
    to_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(1.0, description="关系权重")


class EntitySearchRequest(BaseModel):
    """实体搜索请求"""
    query: str = Field(..., min_length=1, description="搜索关键词")
    entity_types: Optional[List[str]] = Field(None, description="实体类型过滤")
    limit: int = Field(20, ge=1, le=100, description="返回数量限制")


class PathQueryRequest(BaseModel):
    """路径查询请求"""
    from_entity_id: str = Field(..., description="起始实体ID")
    to_entity_id: str = Field(..., description="目标实体ID")
    max_depth: int = Field(4, ge=1, le=10, description="最大路径深度")
    relationship_types: Optional[List[str]] = Field(None, description="关系类型过滤")


class PathInfo(BaseModel):
    """路径信息"""
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    path_length: int = 0


class VisualizationNode(BaseModel):
    """可视化节点"""
    id: str
    name: str
    category: str  # 实体类型，用于颜色分类
    value: float = 1.0  # 节点大小
    properties: Dict[str, Any] = Field(default_factory=dict)


class VisualizationLink(BaseModel):
    """可视化边"""
    source: str
    target: str
    relationship: str
    value: float = 1.0  # 边的粗细


class VisualizationData(BaseModel):
    """可视化数据（用于ECharts等图表库）"""
    nodes: List[VisualizationNode] = Field(default_factory=list)
    links: List[VisualizationLink] = Field(default_factory=list)
    categories: List[Dict[str, str]] = Field(default_factory=list)  # 节点类别


class GraphStatsResponse(BaseModel):
    """图谱统计响应"""
    total_entities: int = 0
    total_relationships: int = 0
    entity_types: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)
    avg_connections_per_entity: float = 0.0


# ==================== 服务类 ====================

class GraphService:
    """
    知识图谱服务类
    封装图数据库操作，提供图谱查询和可视化功能。
    """
    
    def __init__(self):
        """初始化图谱服务"""
        self.settings = get_settings()
        self.graph_store = Neo4jGraphStore()
        logger.info("✅ GraphService 初始化完成")
    
    # ==================== 实体操作 ====================
    
    def get_entity(self, entity_id: str) -> Optional[EntityInfo]:
        """
        获取单个实体详情。
        Args:
            entity_id: 实体ID。
        Returns:
            实体信息，不存在返回None。
        """
        entity_data = self.graph_store.get_entity(entity_id)
        if not entity_data:
            return None
        
        # 获取关联记忆数量
        relationships = self.graph_store.get_entity_relationships(entity_id)
        memory_count = sum(1 for r in relationships if r.get("relationship_type") == "HAS_MEMORY")
        
        return EntityInfo(
            id=entity_data.get("id", entity_id),
            name=entity_data.get("name", "Unknown"),
            entity_type=entity_data.get("type", "UNKNOWN"),
            properties=entity_data,
            related_memory_count=memory_count
        )
    
    def search_entities(self, request: EntitySearchRequest) -> List[EntityInfo]:
        """
        搜索实体。
        Args:
            request: 搜索请求。
        Returns:
            匹配的实体列表。
        """
        # 简单实现：获取所有实体并过滤
        # 生产环境应使用Neo4j的全文索引
        all_entities = self._get_all_entities(limit=1000)
        
        query_lower = request.query.lower()
        results = []
        
        for entity in all_entities:
            # 名称匹配
            if query_lower in entity.name.lower():
                # 类型过滤
                if request.entity_types is None or entity.entity_type in request.entity_types:
                    results.append(entity)
                    if len(results) >= request.limit:
                        break
        
        return results
    
    def list_entities(
        self,
        entity_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        分页列出实体。
        Args:
            entity_type: 可选的类型过滤。
            page: 页码。
            page_size: 每页数量。
        Returns:
            分页结果。
        """
        all_entities = self._get_all_entities(limit=10000)
        
        if entity_type:
            all_entities = [e for e in all_entities if e.entity_type == entity_type]
        
        total = len(all_entities)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": all_entities[start:end]
        }
    
    def _get_all_entities(self, limit: int = 1000) -> List[EntityInfo]:
        """获取所有实体（内部方法）"""
        # 使用Neo4j查询获取所有实体
        stats = self.graph_store.get_stats()
        entities = []
        
        # 通过遍历实体类型获取
        for entity_type, count in stats.get("node_types", {}).items():
            # 这里需要Neo4j的批量查询支持
            # 简化实现：返回空列表，实际应实现批量查询
            pass
        
        return entities
    
    # ==================== 关系操作 ====================
    
    def get_entity_relationships(self, entity_id: str) -> List[RelationshipInfo]:
        """
        获取实体的所有关系。
        Args:
            entity_id: 实体ID。
        Returns:
            关系列表。
        """
        relationships_data = self.graph_store.get_entity_relationships(entity_id)
        
        results = []
        for rel in relationships_data:
            results.append(RelationshipInfo(
                from_entity_id=rel.get("from_entity", {}).get("id", ""),
                to_entity_id=rel.get("to_entity", {}).get("id", ""),
                relationship_type=rel.get("relationship_type", "RELATED"),
                properties=rel.get("relationship", {}),
                weight=rel.get("relationship", {}).get("weight", 1.0)
            ))
        
        return results
    
    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[EntityInfo]:
        """
        查找相关实体。
        Args:
            entity_id: 起始实体ID。
            relationship_types: 关系类型过滤。
            max_depth: 最大深度。
            limit: 返回数量限制。
        Returns:
            相关实体列表。
        """
        related_data = self.graph_store.find_related_entities(
            entity_id=entity_id,
            relationship_types=relationship_types,
            max_depth=max_depth,
            limit=limit
        )
        
        results = []
        for data in related_data:
            results.append(EntityInfo(
                id=data.get("id", ""),
                name=data.get("name", "Unknown"),
                entity_type=data.get("type", "UNKNOWN"),
                properties=data
            ))
        
        return results
    
    # ==================== 路径查询 ====================
    
    def find_path(self, request: PathQueryRequest) -> Optional[PathInfo]:
        """
        查找两个实体之间的路径。
        Args:
            request: 路径查询请求。
        Returns:
            路径信息，未找到返回None。
        """
        # 使用BFS查找最短路径
        visited: Set[str] = set()
        queue = [(request.from_entity_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == request.to_entity_id:
                # 找到路径，构建PathInfo
                entities = []
                relationships = []
                
                for i, entity_id in enumerate(path + [current_id]):
                    entity = self.get_entity(entity_id)
                    if entity:
                        entities.append(entity)
                    
                    if i > 0:
                        # 获取关系
                        prev_id = path[i-1] if i > 0 and i <= len(path) else path[-1]
                        rels = self.get_entity_relationships(prev_id)
                        for rel in rels:
                            if rel.to_entity_id == entity_id or rel.from_entity_id == entity_id:
                                relationships.append(rel)
                                break
                
                return PathInfo(
                    entities=entities,
                    relationships=relationships,
                    path_length=len(entities) - 1
                )
            
            if current_id in visited or len(path) >= request.max_depth:
                continue
            
            visited.add(current_id)
            
            # 获取邻居节点
            related = self.graph_store.find_related_entities(
                entity_id=current_id,
                relationship_types=request.relationship_types,
                max_depth=1,
                limit=50
            )
            
            for neighbor in related:
                neighbor_id = neighbor.get("id")
                if neighbor_id and neighbor_id not in visited:
                    queue.append((neighbor_id, path + [current_id]))
        
        return None  # 未找到路径
    
    # ==================== 可视化数据 ====================
    
    def get_visualization_data(
        self,
        center_entity_id: Optional[str] = None,
        depth: int = 2,
        limit: int = 100
    ) -> VisualizationData:
        """
        获取可视化数据（用于前端图谱展示）。
        Args:
            center_entity_id: 中心实体ID，为空则获取全局视图。
            depth: 展开深度。
            limit: 节点数量限制。
        Returns:
            可视化数据。
        """
        nodes: Dict[str, VisualizationNode] = {}
        links: List[VisualizationLink] = []
        entity_types: Set[str] = set()
        
        if center_entity_id:
            # 从中心实体展开
            self._expand_from_entity(
                entity_id=center_entity_id,
                depth=depth,
                nodes=nodes,
                links=links,
                entity_types=entity_types,
                limit=limit
            )
        else:
            # 获取全局视图（取重要的实体）
            stats = self.graph_store.get_stats()
            # 简化实现：返回空数据
            pass
        
        # 构建类别列表
        categories = [{"name": et} for et in sorted(entity_types)]
        
        return VisualizationData(
            nodes=list(nodes.values()),
            links=links,
            categories=categories
        )
    
    def _expand_from_entity(
        self,
        entity_id: str,
        depth: int,
        nodes: Dict[str, VisualizationNode],
        links: List[VisualizationLink],
        entity_types: Set[str],
        limit: int,
        current_depth: int = 0
    ):
        """递归展开实体"""
        if current_depth > depth or len(nodes) >= limit:
            return
        
        if entity_id in nodes:
            return
        
        # 获取实体信息
        entity = self.get_entity(entity_id)
        if not entity:
            return
        
        # 添加节点
        entity_types.add(entity.entity_type)
        nodes[entity_id] = VisualizationNode(
            id=entity_id,
            name=entity.name,
            category=entity.entity_type,
            value=1.0 + entity.related_memory_count * 0.1,
            properties=entity.properties
        )
        
        if current_depth < depth and len(nodes) < limit:
            # 获取关系并展开
            relationships = self.get_entity_relationships(entity_id)
            
            for rel in relationships:
                # 添加边
                links.append(VisualizationLink(
                    source=rel.from_entity_id,
                    target=rel.to_entity_id,
                    relationship=rel.relationship_type,
                    value=rel.weight
                ))
                
                # 递归展开相邻节点
                neighbor_id = rel.to_entity_id if rel.from_entity_id == entity_id else rel.from_entity_id
                self._expand_from_entity(
                    entity_id=neighbor_id,
                    depth=depth,
                    nodes=nodes,
                    links=links,
                    entity_types=entity_types,
                    limit=limit,
                    current_depth=current_depth + 1
                )
    
    # ==================== 统计操作 ====================
    
    def get_stats(self) -> GraphStatsResponse:
        """获取图谱统计信息"""
        stats = self.graph_store.get_stats()
        
        total_entities = stats.get("total_nodes", 0)
        total_relationships = stats.get("total_relationships", 0)
        
        avg_connections = 0.0
        if total_entities > 0:
            avg_connections = (total_relationships * 2) / total_entities
        
        return GraphStatsResponse(
            total_entities=total_entities,
            total_relationships=total_relationships,
            entity_types=stats.get("node_types", {}),
            relationship_types=stats.get("relationship_types", {}),
            avg_connections_per_entity=avg_connections
        )
    
    def health_check(self) -> bool:
        """检查图数据库健康状态"""
        return self.graph_store.health_check()
```

### 2.3.3 验证方法

```python
# scripts/verify_graph_service.py

import sys
sys.path.insert(0, ".")

def verify_graph_service():
    """验证GraphService"""
    print("验证GraphService...")
    
    from services.graph_service import (
        GraphService,
        EntitySearchRequest,
        PathQueryRequest
    )
    
    # 1. 初始化服务
    service = GraphService()
    print("  [OK] 服务初始化")
    
    # 2. 健康检查
    is_healthy = service.health_check()
    print(f"  [{'OK' if is_healthy else 'WARN'}] 健康检查: {'连接正常' if is_healthy else 'Neo4j未连接'}")
    
    # 3. 获取统计
    stats = service.get_stats()
    print(f"  [OK] 获取统计: 实体数={stats.total_entities}, 关系数={stats.total_relationships}")
    
    # 4. 搜索实体（即使没有数据也应正常运行）
    search_request = EntitySearchRequest(
        query="Python",
        limit=10
    )
    results = service.search_entities(search_request)
    print(f"  [OK] 搜索实体: 结果数={len(results)}")
    
    # 5. 列出实体
    list_result = service.list_entities(page=1, page_size=10)
    assert "total" in list_result
    assert "items" in list_result
    print(f"  [OK] 列出实体: 总数={list_result['total']}")
    
    # 6. 获取可视化数据
    viz_data = service.get_visualization_data(depth=1, limit=50)
    assert hasattr(viz_data, "nodes")
    assert hasattr(viz_data, "links")
    print(f"  [OK] 可视化数据: 节点数={len(viz_data.nodes)}, 边数={len(viz_data.links)}")
    
    print("GraphService验证通过!")
    return True

if __name__ == "__main__":
    verify_graph_service()
```

---

## Task 2.4：AnalyticsService 实现

### 2.4.1 功能描述

AnalyticsService 是分析统计的业务服务层，提供：
- 记忆系统的统计分析
- 使用趋势报告
- 存储状态监控
- 性能指标收集

### 2.4.2 类设计

```python
# services/analytics_service.py

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from pydantic import BaseModel, Field

from services.memory_service import MemoryService
from services.rag_service import RAGService
from services.graph_service import GraphService
from config.settings import get_settings

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class TimeSeriesPoint(BaseModel):
    """时间序列数据点"""
    timestamp: datetime
    value: float
    label: Optional[str] = None


class MemoryTypeDistribution(BaseModel):
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
    """仪表盘摘要"""
    total_memories: int = 0
    today_added: int = 0
    total_documents: int = 0
    total_entities: int = 0
    memory_distribution: MemoryTypeDistribution = Field(default_factory=MemoryTypeDistribution)
    storage_status: StorageStatus = Field(default_factory=StorageStatus)
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)


class TrendReport(BaseModel):
    """趋势报告"""
    period: str  # "day", "week", "month"
    memory_growth: List[TimeSeriesPoint] = Field(default_factory=list)
    query_count: List[TimeSeriesPoint] = Field(default_factory=list)
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
    """
    
    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        rag_service: Optional[RAGService] = None,
        graph_service: Optional[GraphService] = None
    ):
        """
        初始化分析服务。
        Args:
            memory_service: 可选的记忆服务实例。
            rag_service: 可选的RAG服务实例。
            graph_service: 可选的图谱服务实例。
        """
        self.memory_service = memory_service or MemoryService()
        self.rag_service = rag_service or RAGService()
        self.graph_service = graph_service or GraphService()
        
        # 活动日志（内存中，生产环境应使用数据库）
        self._activity_log: List[Dict[str, Any]] = []
        self._query_counts: Dict[str, int] = defaultdict(int)
        
        logger.info("✅ AnalyticsService 初始化完成")
    
    # ==================== 仪表盘数据 ====================
    
    def get_dashboard_summary(self, user_id: Optional[str] = None) -> DashboardSummary:
        """
        获取仪表盘摘要数据。
        Args:
            user_id: 可选的用户ID过滤。
        Returns:
            仪表盘摘要。
        """
        # 1. 记忆统计
        memory_stats = self.memory_service.get_stats(user_id=user_id)
        
        # 2. RAG统计
        rag_stats = self.rag_service.get_stats(user_id=user_id)
        
        # 3. 图谱统计
        graph_stats = self.graph_service.get_stats()
        
        # 4. 存储状态
        storage_status = self._get_storage_status()
        
        # 5. 今日新增（简化实现）
        today_added = self._count_today_memories(user_id)
        
        # 6. 最近活动
        recent_activity = self._get_recent_activity(limit=10)
        
        return DashboardSummary(
            total_memories=memory_stats.total_count,
            today_added=today_added,
            total_documents=rag_stats.total_documents,
            total_entities=graph_stats.total_entities,
            memory_distribution=MemoryTypeDistribution(
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
        
        # Qdrant状态
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
            status.qdrant_status = f"error: {str(e)[:50]}"
        
        # Neo4j状态
        try:
            if self.graph_service.health_check():
                status.neo4j_status = "connected"
                graph_stats = self.graph_service.get_stats()
                status.neo4j_node_count = graph_stats.total_entities
            else:
                status.neo4j_status = "disconnected"
        except Exception as e:
            status.neo4j_status = f"error: {str(e)[:50]}"
        
        # SQLite状态
        try:
            import os
            settings = get_settings()
            db_path = os.path.join(settings.database_settings.sqlite_path, "memory.db")
            if os.path.exists(db_path):
                status.sqlite_status = "ok"
                status.sqlite_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            else:
                status.sqlite_status = "not_initialized"
        except Exception as e:
            status.sqlite_status = f"error: {str(e)[:50]}"
        
        return status
    
    def _count_today_memories(self, user_id: Optional[str] = None) -> int:
        """统计今日新增记忆数"""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            from services.memory_service import MemorySearchRequest
            # 简化实现：通过列表过滤
            result = self.memory_service.list_memories(
                user_id=user_id,
                page=1,
                page_size=1000
            )
            
            today_count = 0
            for item in result.get("items", []):
                if item.timestamp >= today_start:
                    today_count += 1
            
            return today_count
        except Exception:
            return 0
    
    def _get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近活动"""
        # 返回最近的活动日志
        return self._activity_log[-limit:][::-1]
    
    # ==================== 趋势分析 ====================
    
    def get_trend_report(
        self,
        period: str = "week",
        user_id: Optional[str] = None
    ) -> TrendReport:
        """
        获取趋势报告。
        Args:
            period: 时间周期 ("day", "week", "month")。
            user_id: 可选的用户ID过滤。
        Returns:
            趋势报告。
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
        result = self.memory_service.list_memories(
            user_id=user_id,
            page=1,
            page_size=10000
        )
        memories = result.get("items", [])
        
        # 按时间分组统计
        memory_growth = []
        importance_by_time = defaultdict(list)
        type_counts = defaultdict(int)
        
        current_time = start_time
        while current_time <= now:
            next_time = current_time + interval
            
            # 统计该时间段的记忆数
            count = sum(1 for m in memories if current_time <= m.timestamp < next_time)
            memory_growth.append(TimeSeriesPoint(
                timestamp=current_time,
                value=count,
                label=current_time.strftime("%Y-%m-%d %H:%M")
            ))
            
            # 统计平均重要性
            period_memories = [m for m in memories if current_time <= m.timestamp < next_time]
            if period_memories:
                avg_imp = sum(m.importance for m in period_memories) / len(period_memories)
            else:
                avg_imp = 0.0
            importance_by_time[current_time.isoformat()].append(avg_imp)
            
            current_time = next_time
        
        # 统计类型分布
        for m in memories:
            if m.timestamp >= start_time:
                type_counts[m.memory_type] += 1
        
        return TrendReport(
            period=period,
            memory_growth=memory_growth,
            query_count=[],  # 需要实现查询计数
            avg_importance=[
                TimeSeriesPoint(timestamp=datetime.fromisoformat(k), value=sum(v)/len(v) if v else 0)
                for k, v in importance_by_time.items()
            ],
            top_memory_types=dict(type_counts)
        )
    
    # ==================== 系统健康 ====================
    
    def get_system_health(self) -> SystemHealth:
        """
        获取系统健康状态。
        Returns:
            系统健康状态。
        """
        health = SystemHealth()
        issues = []
        
        # 检查各组件
        components = {}
        
        # 1. Qdrant
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
            components["qdrant"] = {"status": "error", "message": str(e)}
            issues.append(f"Qdrant检查异常: {str(e)[:50]}")
        
        # 2. Neo4j
        try:
            neo4j_healthy = self.graph_service.health_check()
            components["neo4j"] = {
                "status": "healthy" if neo4j_healthy else "unhealthy",
                "message": "连接正常" if neo4j_healthy else "连接失败"
            }
            if not neo4j_healthy:
                issues.append("Neo4j图数据库连接失败")
        except Exception as e:
            components["neo4j"] = {"status": "error", "message": str(e)}
            issues.append(f"Neo4j检查异常: {str(e)[:50]}")
        
        # 3. SQLite
        try:
            import os
            settings = get_settings()
            db_path = os.path.join(settings.database_settings.sqlite_path, "memory.db")
            sqlite_exists = os.path.exists(db_path)
            components["sqlite"] = {
                "status": "healthy" if sqlite_exists else "not_initialized",
                "message": "数据库正常" if sqlite_exists else "数据库未初始化"
            }
        except Exception as e:
            components["sqlite"] = {"status": "error", "message": str(e)}
            issues.append(f"SQLite检查异常: {str(e)[:50]}")
        
        # 4. 嵌入模型
        try:
            from core.embedding import get_text_embedder
            embedder = get_text_embedder()
            test_vec = embedder.encode("health check")
            components["embedding"] = {
                "status": "healthy",
                "message": f"嵌入模型正常，维度: {len(test_vec)}"
            }
        except Exception as e:
            components["embedding"] = {"status": "error", "message": str(e)}
            issues.append(f"嵌入模型异常: {str(e)[:50]}")
        
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
        记录用户活动。
        Args:
            action: 动作类型（如 "add_memory", "search", "ask"）。
            user_id: 用户ID。
            details: 额外详情。
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
        获取查询统计。
        Args:
            days: 统计天数。
        Returns:
            每日查询数量字典。
        """
        result = {}
        now = datetime.now()
        
        for i in range(days):
            date = now - timedelta(days=i)
            date_key = date.strftime("%Y-%m-%d")
            result[date_key] = self._query_counts.get(date_key, 0)
        
        return result
```

### 2.4.3 验证方法

```python
# scripts/verify_analytics_service.py

import sys
sys.path.insert(0, ".")

def verify_analytics_service():
    """验证AnalyticsService"""
    print("验证AnalyticsService...")
    
    from services.analytics_service import AnalyticsService
    
    # 1. 初始化服务
    service = AnalyticsService()
    print("  [OK] 服务初始化")
    
    # 2. 获取仪表盘摘要
    summary = service.get_dashboard_summary()
    assert hasattr(summary, "total_memories")
    assert hasattr(summary, "memory_distribution")
    assert hasattr(summary, "storage_status")
    print(f"  [OK] 仪表盘摘要: 总记忆数={summary.total_memories}")
    
    # 3. 获取趋势报告
    trend = service.get_trend_report(period="week")
    assert trend.period == "week"
    assert hasattr(trend, "memory_growth")
    print(f"  [OK] 趋势报告: 周期={trend.period}, 数据点数={len(trend.memory_growth)}")
    
    # 4. 获取系统健康状态
    health = service.get_system_health()
    assert health.overall_status in ("healthy", "degraded", "unhealthy")
    assert hasattr(health, "components")
    print(f"  [OK] 系统健康: 状态={health.overall_status}, 组件数={len(health.components)}")
    
    # 5. 记录活动
    service.log_activity(
        action="test_action",
        user_id="test_user",
        details={"test": True}
    )
    recent = service._get_recent_activity(limit=1)
    assert len(recent) > 0
    print("  [OK] 活动记录")
    
    # 6. 获取查询统计
    query_stats = service.get_query_stats(days=7)
    assert isinstance(query_stats, dict)
    print(f"  [OK] 查询统计: {len(query_stats)} 天")
    
    print("AnalyticsService验证通过!")
    return True

if __name__ == "__main__":
    verify_analytics_service()
```

---

## Task 2.5：服务层集成与初始化

### 2.5.1 服务层导出模块

```python
# services/__init__.py

"""
服务层模块
提供业务逻辑封装，连接核心层与API层。
"""

from .memory_service import (
    MemoryService,
    MemoryCreateRequest,
    MemoryUpdateRequest,
    MemorySearchRequest,
    MemoryResponse,
    MemoryStatsResponse,
    ConsolidateRequest,
    ForgetRequest
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
    RAGStatsResponse
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
    GraphStatsResponse
)

from .analytics_service import (
    AnalyticsService,
    DashboardSummary,
    TrendReport,
    SystemHealth,
    StorageStatus,
    MemoryTypeDistribution
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
    "MemoryTypeDistribution"
]
```

### 2.5.2 阶段2总验证脚本

```python
# scripts/verify_phase2.py

"""
阶段2验证脚本 - 服务层实现验证
"""

import sys
sys.path.insert(0, ".")

def verify_imports():
    """验证服务层导入"""
    print("1. 验证服务层导入...")
    try:
        from services import (
            MemoryService,
            RAGService,
            GraphService,
            AnalyticsService
        )
        print("  [OK] 所有服务类导入成功")
        return True
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")
        return False

def verify_memory_service():
    """验证MemoryService"""
    print("2. 验证MemoryService...")
    try:
        from services.memory_service import MemoryService, MemoryCreateRequest
        
        service = MemoryService()
        
        # 测试添加记忆
        request = MemoryCreateRequest(
            content="测试记忆内容",
            memory_type="working",
            user_id="test_user",
            importance=0.5
        )
        response = service.add_memory(request)
        assert response.id, "应返回记忆ID"
        
        # 测试获取记忆
        retrieved = service.get_memory(response.id)
        assert retrieved is not None
        
        # 测试删除记忆
        deleted = service.delete_memory(response.id)
        assert deleted
        
        print("  [OK] MemoryService验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] MemoryService验证失败: {e}")
        return False

def verify_rag_service():
    """验证RAGService"""
    print("3. 验证RAGService...")
    try:
        from services.rag_service import RAGService, SearchRequest
        
        service = RAGService()
        
        # 测试基础检索（即使没有文档也应正常运行）
        request = SearchRequest(
            query="test query",
            user_id="test_user",
            limit=5
        )
        results = service.search(request)
        assert isinstance(results, list)
        
        # 测试统计
        stats = service.get_stats()
        assert hasattr(stats, "total_documents")
        
        print("  [OK] RAGService验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] RAGService验证失败: {e}")
        return False

def verify_graph_service():
    """验证GraphService"""
    print("4. 验证GraphService...")
    try:
        from services.graph_service import GraphService
        
        service = GraphService()
        
        # 测试统计（即使Neo4j未连接也应正常运行）
        stats = service.get_stats()
        assert hasattr(stats, "total_entities")
        
        # 测试可视化数据
        viz_data = service.get_visualization_data(depth=1, limit=10)
        assert hasattr(viz_data, "nodes")
        assert hasattr(viz_data, "links")
        
        print("  [OK] GraphService验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] GraphService验证失败: {e}")
        return False

def verify_analytics_service():
    """验证AnalyticsService"""
    print("5. 验证AnalyticsService...")
    try:
        from services.analytics_service import AnalyticsService
        
        service = AnalyticsService()
        
        # 测试仪表盘
        summary = service.get_dashboard_summary()
        assert hasattr(summary, "total_memories")
        
        # 测试健康检查
        health = service.get_system_health()
        assert health.overall_status in ("healthy", "degraded", "unhealthy")
        
        # 测试趋势报告
        trend = service.get_trend_report(period="day")
        assert trend.period == "day"
        
        print("  [OK] AnalyticsService验证通过")
        return True
    except Exception as e:
        print(f"  [FAIL] AnalyticsService验证失败: {e}")
        return False

def main():
    """运行所有验证"""
    print("=" * 60)
    print("Agent Memory System - 阶段2验证")
    print("服务层实现验证")
    print("=" * 60)
    print()
    
    results = {
        "服务层导入": verify_imports(),
        "MemoryService": verify_memory_service(),
        "RAGService": verify_rag_service(),
        "GraphService": verify_graph_service(),
        "AnalyticsService": verify_analytics_service()
    }
    
    print()
    print("=" * 60)
    print("验证结果:")
    print("-" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"  通过: {passed}/{total}")
    print("=" * 60)
    
    if passed == total:
        print("阶段2验证通过! 服务层实现完成。")
        print()
        print("下一步: 可以开始阶段3 - API层实现")
        return 0
    else:
        print("阶段2验证失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 验证清单

| 任务 | 验证项 | 验证方法 |
|------|--------|----------|
| Task 2.1 | MemoryService CRUD | `python scripts/verify_memory_service.py` |
| Task 2.1 | MemoryService 搜索 | 验证脚本中的搜索测试 |
| Task 2.1 | MemoryService 统计 | 验证脚本中的统计测试 |
| Task 2.2 | RAGService 文档上传 | `python scripts/verify_rag_service.py` |
| Task 2.2 | RAGService 检索 | 验证脚本中的检索测试 |
| Task 2.2 | RAGService 问答 | 验证脚本中的问答测试 |
| Task 2.3 | GraphService 实体查询 | `python scripts/verify_graph_service.py` |
| Task 2.3 | GraphService 可视化 | 验证脚本中的可视化数据测试 |
| Task 2.4 | AnalyticsService 仪表盘 | `python scripts/verify_analytics_service.py` |
| Task 2.4 | AnalyticsService 健康检查 | 验证脚本中的健康检查测试 |
| Task 2.5 | 服务层集成 | `python scripts/verify_phase2.py` |

---

## 注意事项

1. **依赖注入**：服务类支持依赖注入，便于测试和替换实现
2. **错误处理**：所有服务方法应有适当的异常处理和日志记录
3. **数据验证**：使用Pydantic模型进行请求/响应数据验证
4. **性能考虑**：对于大量数据操作，应考虑分页和批量处理
5. **线程安全**：服务实例应是线程安全的，支持并发访问

---

## 下一步

完成阶段2后，可以进入阶段3：API层实现，使用FastAPI将服务层暴露为RESTful API。

