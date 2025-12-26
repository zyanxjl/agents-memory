"""
文件路径: core/memory/episodic.py
功能: 情景记忆实现

按照第8章架构设计的情景记忆，提供:
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- SQLite + Qdrant双存储
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

from .base import BaseMemory, MemoryItem, MemoryConfig
from core.embedding import get_text_embedder, get_dimension


class Episode:
    """情景记忆中的单个情景"""
    
    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class EpisodicMemory(BaseMemory):
    """情景记忆实现
    
    特点:
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    """
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 本地缓存
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}
        
        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 初始化存储后端
        self._init_storage()

    def _init_storage(self):
        """初始化存储后端"""
        db_dir = self.config.storage_path if hasattr(self.config, 'storage_path') else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        
        # 尝试初始化SQLite文档存储
        try:
            from storage.document_store import SQLiteDocumentStore
            db_path = os.path.join(db_dir, "memory.db")
            self.doc_store = SQLiteDocumentStore(db_path=db_path)
        except ImportError:
            logger.warning("SQLiteDocumentStore不可用，使用内存存储")
            self.doc_store = None

        # 统一嵌入模型
        try:
            self.embedder = get_text_embedder()
        except Exception as e:
            logger.warning(f"嵌入模型初始化失败: {e}")
            self.embedder = None

        # 向量存储
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
        except Exception as e:
            logger.warning(f"Qdrant初始化失败: {e}")
            self.vector_store = None
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        
        # 创建情景
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # SQLite权威存储
        if self.doc_store:
            try:
                ts_int = int(memory_item.timestamp.timestamp())
                self.doc_store.add_memory(
                    memory_id=memory_item.id,
                    user_id=memory_item.user_id,
                    content=memory_item.content,
                    memory_type="episodic",
                    timestamp=ts_int,
                    importance=memory_item.importance,
                    properties={
                        "session_id": session_id,
                        "context": context,
                        "outcome": outcome,
                    }
                )
            except Exception as e:
                logger.warning(f"SQLite存储失败: {e}")

        # Qdrant向量索引
        if self.vector_store and self.embedder:
            try:
                embedding = self.embedder.encode(memory_item.content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[{
                        "memory_id": memory_item.id,
                        "user_id": memory_item.user_id,
                        "memory_type": "episodic",
                        "importance": memory_item.importance,
                        "session_id": session_id,
                        "content": memory_item.content
                    }],
                    ids=[memory_item.id]
                )
            except Exception as e:
                logger.warning(f"Qdrant存储失败: {e}")

        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索情景记忆"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")

        results: List[Tuple[float, MemoryItem]] = []
        
        # 向量检索
        if self.vector_store and self.embedder:
            try:
                query_vec = self.embedder.encode(query)
                if hasattr(query_vec, "tolist"):
                    query_vec = query_vec.tolist()
                where = {"memory_type": "episodic"}
                if user_id:
                    where["user_id"] = user_id
                hits = self.vector_store.search_similar(
                    query_vector=query_vec,
                    limit=max(limit * 3, 15),
                    where=where
                )
                
                now_ts = int(datetime.now().timestamp())
                for hit in hits:
                    meta = hit.get("metadata", {})
                    mem_id = meta.get("memory_id")
                    if not mem_id:
                        continue
                    if session_id and meta.get("session_id") != session_id:
                        continue
                    
                    # 从本地缓存获取
                    episode = next((e for e in self.episodes if e.episode_id == mem_id), None)
                    if not episode:
                        continue
                    if episode.context.get("forgotten", False):
                        continue
                    
                    vec_score = float(hit.get("score", 0.0))
                    age_days = max(0.0, (now_ts - int(episode.timestamp.timestamp())) / 86400.0)
                    recency_score = 1.0 / (1.0 + age_days)
                    
                    base_relevance = vec_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (episode.importance * 0.4)
                    combined = base_relevance * importance_weight

                    item = MemoryItem(
                        id=episode.episode_id,
                        content=episode.content,
                        memory_type="episodic",
                        user_id=episode.user_id,
                        timestamp=episode.timestamp,
                        importance=episode.importance,
                        metadata={
                            "session_id": episode.session_id,
                            "context": episode.context,
                            "outcome": episode.outcome,
                            "relevance_score": combined
                        }
                    )
                    results.append((combined, item))
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 回退：关键词匹配
        if not results:
            query_lower = query.lower()
            now_ts = int(datetime.now().timestamp())
            for ep in self.episodes:
                if ep.context.get("forgotten", False):
                    continue
                if user_id and ep.user_id != user_id:
                    continue
                if session_id and ep.session_id != session_id:
                    continue
                if query_lower in ep.content.lower():
                    age_days = max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0)
                    recency_score = 1.0 / (1.0 + age_days)
                    base_relevance = 0.5 * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight
                    
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined
                        }
                    )
                    results.append((combined, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """更新情景记忆"""
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        if self.doc_store:
            try:
                self.doc_store.update_memory(
                    memory_id=memory_id,
                    content=content,
                    importance=importance,
                    properties=metadata
                )
            except Exception:
                pass

        return updated
    
    def remove(self, memory_id: str) -> bool:
        """删除情景记忆"""
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    if memory_id in self.sessions[session_id]:
                        self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break

        if self.doc_store:
            try:
                self.doc_store.delete_memory(memory_id)
            except Exception:
                pass
        
        if self.vector_store:
            try:
                self.vector_store.delete_vectors([memory_id])
            except Exception:
                pass
        
        return removed
    
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在"""
        return any(episode.episode_id == memory_id for episode in self.episodes)
    
    def clear(self):
        """清空所有情景记忆"""
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

    def get_all(self) -> List[MemoryItem]:
        """获取所有情景记忆"""
        return [
            MemoryItem(
                id=ep.episode_id,
                content=ep.content,
                memory_type="episodic",
                user_id=ep.user_id,
                timestamp=ep.timestamp,
                importance=ep.importance,
                metadata={
                    "session_id": ep.session_id,
                    "context": ep.context,
                    "outcome": ep.outcome
                }
            )
            for ep in self.episodes
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取情景记忆统计信息"""
        active_episodes = [e for e in self.episodes if not e.context.get("forgotten", False)]
        
        return {
            "count": len(active_episodes),
            "forgotten_count": len(self.episodes) - len(active_episodes),
            "total_count": len(self.episodes),
            "sessions_count": len(self.sessions),
            "avg_importance": sum(e.importance for e in active_episodes) / len(active_episodes) if active_episodes else 0.0,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic"
        }
    
    def _calculate_time_span(self) -> float:
        """计算记忆时间跨度（天）"""
        if not self.episodes:
            return 0.0
        timestamps = [e.timestamp for e in self.episodes]
        min_time = min(timestamps)
        max_time = max(timestamps)
        return (max_time - min_time).days

