"""
文件路径: core/memory/semantic.py
功能: 语义记忆实现

结合向量检索和知识图谱的混合语义记忆:
- 使用嵌入模型进行文本嵌入
- 向量相似度检索进行快速初筛
- 知识图谱进行实体关系推理
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

from .base import BaseMemory, MemoryItem, MemoryConfig
from core.embedding import get_text_embedder, get_dimension


class Entity:
    """实体类"""
    
    def __init__(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "MISC",
        description: str = "",
        properties: Dict[str, Any] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1


class SemanticMemory(BaseMemory):
    """语义记忆实现
    
    特点:
    - 使用嵌入模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 嵌入模型
        self.embedding_model = None
        self._init_embedding_model()
        
        # 存储后端
        self.vector_store = None
        self.graph_store = None
        self._init_storage()
        
        # 本地缓存
        self.entities: Dict[str, Entity] = {}
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, np.ndarray] = {}
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            self.embedding_model = get_text_embedder()
            logger.info("嵌入模型就绪")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def _init_storage(self):
        """初始化存储后端"""
        try:
            from core.storage import QdrantConnectionManager
            from config import settings
            db_settings = settings.database
            
            self.vector_store = QdrantConnectionManager.get_instance(
                url=db_settings.qdrant_url,
                api_key=db_settings.qdrant_api_key,
                collection_name=db_settings.qdrant_collection,
                vector_size=get_dimension(384)
            )
            logger.info("Qdrant向量存储初始化完成")
        except Exception as e:
            logger.warning(f"Qdrant初始化失败: {e}")
            self.vector_store = None

        try:
            from core.storage import Neo4jGraphStore
            self.graph_store = Neo4jGraphStore()
            logger.info("Neo4j图存储初始化完成")
        except Exception as e:
            logger.warning(f"Neo4j初始化失败: {e}")
            self.graph_store = None
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        try:
            # 生成嵌入
            embedding = self.embedding_model.encode(memory_item.content)
            self.memory_embeddings[memory_item.id] = embedding
            
            # 存储到Qdrant
            if self.vector_store:
                metadata = {
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "content": memory_item.content,
                    "memory_type": "semantic",
                    "timestamp": int(memory_item.timestamp.timestamp()),
                    "importance": memory_item.importance,
                }
                
                vec_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                self.vector_store.add_vectors(
                    vectors=[vec_list],
                    metadata=[metadata],
                    ids=[memory_item.id]
                )
            
            # 存储到本地缓存
            self.semantic_memories.append(memory_item)
            
            logger.info(f"添加语义记忆: {memory_item.id}")
            return memory_item.id
        
        except Exception as e:
            logger.error(f"添加语义记忆失败: {e}")
            raise
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆"""
        try:
            user_id = kwargs.get("user_id")
            
            # 向量检索
            results = []
            if self.vector_store and self.embedding_model:
                query_embedding = self.embedding_model.encode(query)
                query_vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
                
                where_filter = {"memory_type": "semantic"}
                if user_id:
                    where_filter["user_id"] = user_id

                hits = self.vector_store.search_similar(
                    query_vector=query_vec,
                    limit=limit * 2,
                    where=where_filter
                )

                now_ts = int(datetime.now().timestamp())
                for hit in hits:
                    meta = hit.get("metadata", {})
                    memory_id = meta.get("memory_id")
                    
                    memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
                    if memory and memory.metadata.get("forgotten", False):
                        continue
                    
                    vec_score = float(hit.get("score", 0.0))
                    timestamp = meta.get("timestamp", now_ts)
                    age_days = max(0.0, (now_ts - timestamp) / 86400.0)
                    recency_score = 1.0 / (1.0 + age_days)
                    importance = meta.get("importance", 0.5)
                    
                    base_relevance = vec_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (importance * 0.4)
                    combined = base_relevance * importance_weight
                    
                    memory_item = MemoryItem(
                        id=memory_id,
                        content=meta.get("content", ""),
                        memory_type="semantic",
                        user_id=meta.get("user_id", ""),
                        timestamp=datetime.fromtimestamp(timestamp),
                        importance=importance,
                        metadata={
                            "combined_score": combined,
                            "vector_score": vec_score,
                        }
                    )
                    results.append((combined, memory_item))
            
            # 排序并返回
            results.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in results[:limit]]
                
        except Exception as e:
            logger.error(f"检索语义记忆失败: {e}")
            return []
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """更新语义记忆"""
        memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
        if not memory:
            return False
        
        try:
            if content is not None:
                embedding = self.embedding_model.encode(content)
                self.memory_embeddings[memory_id] = embedding
                memory.content = content
                
            if importance is not None:
                memory.importance = importance
            
            if metadata is not None:
                memory.metadata.update(metadata)
                
            return True
            
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            return False
    
    def remove(self, memory_id: str) -> bool:
        """删除语义记忆"""
        memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
        if not memory:
            return False
        
        try:
            if self.vector_store:
                self.vector_store.delete_vectors([memory_id])
            
            self.semantic_memories.remove(memory)
            if memory_id in self.memory_embeddings:
                del self.memory_embeddings[memory_id]
                
            return True
            
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False
    
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        return any(m.id == memory_id for m in self.semantic_memories)
    
    def clear(self):
        """清空所有语义记忆"""
        try:
            if self.vector_store:
                self.vector_store.clear_collection()
            if self.graph_store:
                self.graph_store.clear_all()
        except Exception as e:
            logger.warning(f"清空存储失败: {e}")
        
        self.semantic_memories.clear()
        self.memory_embeddings.clear()
        self.entities.clear()

    def get_all(self) -> List[MemoryItem]:
        """获取所有语义记忆"""
        return self.semantic_memories.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取语义记忆统计信息"""
        active_memories = [m for m in self.semantic_memories if not m.metadata.get("forgotten", False)]
        
        graph_stats = {}
        if self.graph_store:
            try:
                graph_stats = self.graph_store.get_stats() or {}
            except Exception:
                pass

        return {
            "count": len(active_memories),
            "forgotten_count": len(self.semantic_memories) - len(active_memories),
            "total_count": len(self.semantic_memories),
            "entities_count": len(self.entities),
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_edges": graph_stats.get("total_relationships", 0),
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "memory_type": "semantic"
        }

