"""
文件路径: core/storage/qdrant.py
功能: Qdrant向量数据库存储实现

主要特性:
- 使用新的配置系统替代环境变量
- 继承 VectorStore 抽象基类
- 单例连接管理器避免重复连接
"""

import logging
import uuid
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings
from .base import VectorStore

# 尝试导入Qdrant客户端
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, 
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Qdrant连接管理器 - 单例模式防止重复连接
    
    使用方法:
        store = QdrantConnectionManager.get_instance(collection_name="my_vectors")
        store = QdrantConnectionManager.get_default_instance()  # 使用配置
    """
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        **kwargs
    ) -> 'QdrantVectorStore':
        """获取或创建Qdrant实例"""
        key = (url or "local", collection_name)
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"创建新的Qdrant连接: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        **kwargs
                    )
        
        return cls._instances[key]
    
    @classmethod
    def get_default_instance(cls) -> 'QdrantVectorStore':
        """使用配置创建默认实例"""
        db_settings = settings.database
        embed_settings = settings.embedding
        
        return cls.get_instance(
            url=db_settings.qdrant_url,
            api_key=db_settings.qdrant_api_key,
            collection_name=db_settings.qdrant_collection,
            vector_size=embed_settings.embed_dimension
        )


class QdrantVectorStore(VectorStore):
    """Qdrant向量数据库存储实现
    
    支持本地和云服务两种部署模式。
    
    Args:
        url: Qdrant服务URL
        api_key: API密钥（云服务需要）
        collection_name: 集合名称
        vector_size: 向量维度
        distance: 距离度量（cosine/dot/euclidean）
        timeout: 超时时间
    """
    
    def __init__(
        self, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client未安装。请运行: pip install qdrant-client>=1.6.0")
        
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        # 距离度量映射
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        
        # 初始化客户端
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Qdrant客户端和集合"""
        try:
            if self.url and self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
                logger.info(f"成功连接到Qdrant云服务: {self.url}")
            elif self.url:
                self.client = QdrantClient(url=self.url, timeout=self.timeout)
                logger.info(f"成功连接到Qdrant服务: {self.url}")
            else:
                self.client = QdrantClient(host="localhost", port=6333, timeout=self.timeout)
                logger.info("成功连接到本地Qdrant服务")
            
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Qdrant连接失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
                )
                logger.info(f"创建Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"使用现有Qdrant集合: {self.collection_name}")
            
            self._ensure_payload_indexes()
                
        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise
    
    def _ensure_payload_indexes(self):
        """创建payload索引以优化过滤查询"""
        index_fields = [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass  # 索引已存在
    
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """添加向量到Qdrant"""
        try:
            if not vectors:
                return False
            
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            points = []
            for vector, meta, point_id in zip(vectors, metadata, ids):
                if len(vector) != self.vector_size:
                    continue
                
                meta_copy = meta.copy()
                meta_copy["timestamp"] = int(datetime.now().timestamp())
                
                # 确保ID格式正确（UUID格式）
                safe_id = point_id if isinstance(point_id, str) else str(uuid.uuid4())
                try:
                    uuid.UUID(safe_id)
                except ValueError:
                    safe_id = str(uuid.uuid4())
                
                points.append(PointStruct(id=safe_id, vector=vector, payload=meta_copy))
            
            if not points:
                return False
            
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            logger.info(f"成功添加 {len(points)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            if len(query_vector) != self.vector_size:
                logger.error(f"查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
                return []
            
            # 构建过滤条件
            query_filter = None
            if where:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in where.items()
                    if isinstance(v, (str, int, float, bool))
                ]
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # 执行搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            
            return [
                {"id": hit.id, "score": hit.score, "metadata": hit.payload or {}}
                for hit in search_result
            ]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            if not ids:
                return True
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True
            )
            logger.info(f"成功删除 {len(ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def delete_by_filter(self, where: Dict[str, Any]) -> bool:
        """按条件删除向量"""
        try:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in where.items()
            ]
            query_filter = Filter(should=conditions)
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True
            )
            return True
        except Exception as e:
            logger.error(f"按条件删除失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"成功清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "store_type": "qdrant",
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "vector_size": self.vector_size,
            }
        except Exception:
            return {"store_type": "qdrant", "name": self.collection_name}
    
    # 兼容性别名方法
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计（兼容性别名）"""
        return self.get_stats()
    
    def delete_memories(self, ids: List[str]) -> bool:
        """删除记忆（兼容性别名）"""
        return self.delete_vectors(ids)

