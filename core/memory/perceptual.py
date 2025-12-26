"""
文件路径: core/memory/perceptual.py
功能: 感知记忆实现（多模态）

按照第8章架构设计的感知记忆:
- 多模态数据存储（文本、图像、音频等）
- 结构化元数据 + 向量索引
- 同模态检索
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import os
import random
import logging

logger = logging.getLogger(__name__)

from .base import BaseMemory, MemoryItem, MemoryConfig
from core.embedding import get_text_embedder, get_dimension


class Perception:
    """感知数据实体"""
    
    def __init__(
        self,
        perception_id: str,
        data: Any,
        modality: str,
        encoding: Optional[List[float]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.perception_id = perception_id
        self.data = data
        self.modality = modality
        self.encoding = encoding or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """计算数据哈希"""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()


class PerceptualMemory(BaseMemory):
    """感知记忆实现
    
    特点:
    - 支持多模态数据（文本、图像、音频等）
    - 跨模态相似性搜索
    - 感知数据的语义理解
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 感知数据存储
        self.perceptions: Dict[str, Perception] = {}
        self.perceptual_memories: List[MemoryItem] = []
        
        # 模态索引
        self.modality_index: Dict[str, List[str]] = {}
        
        # 支持的模态
        self.supported_modalities = set(self.config.perceptual_memory_modalities)
        
        # 初始化嵌入模型
        try:
            self.text_embedder = get_text_embedder()
            self.vector_dim = get_dimension(384)
        except Exception as e:
            logger.warning(f"嵌入模型初始化失败: {e}")
            self.text_embedder = None
            self.vector_dim = 384
        
        # 初始化向量存储
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            from core.storage import QdrantConnectionManager
            from config import settings
            db_settings = settings.database
            
            self.vector_store = QdrantConnectionManager.get_instance(
                url=db_settings.qdrant_url,
                api_key=db_settings.qdrant_api_key,
                collection_name=f"{db_settings.qdrant_collection}_perceptual",
                vector_size=self.vector_dim
            )
        except Exception as e:
            logger.warning(f"Qdrant初始化失败: {e}")
            self.vector_store = None
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加感知记忆"""
        modality = memory_item.metadata.get("modality", "text")
        raw_data = memory_item.metadata.get("raw_data", memory_item.content)
        
        if modality not in self.supported_modalities:
            raise ValueError(f"不支持的模态类型: {modality}")

        # 编码感知数据
        encoding = self._encode_data(raw_data, modality)
        
        perception = Perception(
            perception_id=f"perception_{memory_item.id}",
            data=raw_data,
            modality=modality,
            encoding=encoding,
            metadata={"source": "memory_system"}
        )

        # 缓存与索引
        self.perceptions[perception.perception_id] = perception
        if modality not in self.modality_index:
            self.modality_index[modality] = []
        self.modality_index[modality].append(perception.perception_id)

        # 存储记忆项
        memory_item.metadata["perception_id"] = perception.perception_id
        memory_item.metadata["modality"] = modality
        self.perceptual_memories.append(memory_item)

        # Qdrant向量存储
        if self.vector_store:
            try:
                self.vector_store.add_vectors(
                    vectors=[encoding],
                    metadata=[{
                        "memory_id": memory_item.id,
                        "user_id": memory_item.user_id,
                        "memory_type": "perceptual",
                        "modality": modality,
                        "importance": memory_item.importance,
                        "content": memory_item.content,
                    }],
                    ids=[memory_item.id]
                )
            except Exception as e:
                logger.warning(f"向量存储失败: {e}")

        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆"""
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality = kwargs.get("query_modality", target_modality or "text")

        results = []
        
        # 向量检索
        if self.vector_store and self.text_embedder:
            try:
                qvec = self._encode_data(query, query_modality)
                where = {"memory_type": "perceptual"}
                if user_id:
                    where["user_id"] = user_id
                if target_modality:
                    where["modality"] = target_modality
                
                hits = self.vector_store.search_similar(
                    query_vector=qvec,
                    limit=max(limit * 3, 15),
                    where=where
                )
                
                now_ts = int(datetime.now().timestamp())
                for hit in hits:
                    meta = hit.get("metadata", {})
                    mem_id = meta.get("memory_id")
                    if not mem_id:
                        continue
                    
                    memory = next((m for m in self.perceptual_memories if m.id == mem_id), None)
                    if not memory:
                        continue
                    
                    vec_score = float(hit.get("score", 0.0))
                    age_days = max(0.0, (now_ts - int(memory.timestamp.timestamp())) / 86400.0)
                    recency_score = 1.0 / (1.0 + age_days)
                    
                    base_relevance = vec_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (memory.importance * 0.4)
                    combined = base_relevance * importance_weight

                    results.append((combined, memory))
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 回退：关键词匹配
        if not results:
            for m in self.perceptual_memories:
                if target_modality and m.metadata.get("modality") != target_modality:
                    continue
                if query.lower() in (m.content or "").lower():
                    results.append((0.5, m))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """更新感知记忆"""
        for memory in self.perceptual_memories:
            if memory.id == memory_id:
                if content is not None:
                    memory.content = content
                if importance is not None:
                    memory.importance = importance
                if metadata is not None:
                    memory.metadata.update(metadata)
                return True
        return False
    
    def remove(self, memory_id: str) -> bool:
        """删除感知记忆"""
        for i, memory in enumerate(self.perceptual_memories):
            if memory.id == memory_id:
                removed_memory = self.perceptual_memories.pop(i)
                perception_id = removed_memory.metadata.get("perception_id")
                if perception_id and perception_id in self.perceptions:
                    perception = self.perceptions.pop(perception_id)
                    modality = perception.modality
                    if modality in self.modality_index:
                        if perception_id in self.modality_index[modality]:
                            self.modality_index[modality].remove(perception_id)
                
                if self.vector_store:
                    try:
                        self.vector_store.delete_vectors([memory_id])
                    except Exception:
                        pass
                
                return True
        return False
    
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        return any(memory.id == memory_id for memory in self.perceptual_memories)
    
    def clear(self):
        """清空所有感知记忆"""
        self.perceptual_memories.clear()
        self.perceptions.clear()
        self.modality_index.clear()

    def get_all(self) -> List[MemoryItem]:
        """获取所有感知记忆"""
        return self.perceptual_memories.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取感知记忆统计信息"""
        modality_counts = {modality: len(ids) for modality, ids in self.modality_index.items()}
        
        return {
            "count": len(self.perceptual_memories),
            "forgotten_count": 0,
            "total_count": len(self.perceptual_memories),
            "perceptions_count": len(self.perceptions),
            "modality_counts": modality_counts,
            "supported_modalities": list(self.supported_modalities),
            "avg_importance": sum(m.importance for m in self.perceptual_memories) / len(self.perceptual_memories) if self.perceptual_memories else 0.0,
            "memory_type": "perceptual"
        }
    
    def _encode_data(self, data: Any, modality: str) -> List[float]:
        """编码数据为向量"""
        if modality == "text" and self.text_embedder:
            try:
                emb = self.text_embedder.encode(str(data))
                if hasattr(emb, "tolist"):
                    return emb.tolist()
                return list(emb)
            except Exception:
                pass
        
        # 回退到哈希向量
        return self._hash_to_vector(str(data), self.vector_dim)
    
    def _hash_to_vector(self, data_str: str, dim: int) -> List[float]:
        """将字符串哈希为固定维度的向量"""
        seed = int(hashlib.sha256(data_str.encode("utf-8", errors="ignore")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.random() for _ in range(dim)]

