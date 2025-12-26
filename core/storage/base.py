"""
文件路径: core/storage/base.py
功能: 存储后端抽象基类

定义三种存储类型的接口:
- VectorStore: 向量存储抽象基类
- GraphStore: 图存储抽象基类
- DocumentStore: 文档存储抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class VectorStore(ABC):
    """向量存储抽象基类
    
    定义向量数据库的基本操作接口，支持向量的增删查。
    """
    
    @abstractmethod
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """添加向量
        
        Args:
            vectors: 向量列表
            metadata: 每个向量的元数据列表
            ids: 可选的ID列表
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量
        
        Args:
            query_vector: 查询向量
            limit: 返回数量限制
            score_threshold: 最低分数阈值
            where: 过滤条件
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量
        
        Args:
            ids: 要删除的向量ID列表
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class GraphStore(ABC):
    """图存储抽象基类
    
    定义知识图谱存储的基本操作接口，支持实体和关系的管理。
    """
    
    @abstractmethod
    def add_entity(
        self, 
        entity_id: str, 
        name: str, 
        entity_type: str, 
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加实体节点
        
        Args:
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 额外属性
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加关系
        
        Args:
            from_entity_id: 起始实体ID
            to_entity_id: 目标实体ID
            relationship_type: 关系类型
            properties: 关系属性
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """查找相关实体
        
        Args:
            entity_id: 起始实体ID
            relationship_types: 关系类型过滤
            max_depth: 最大搜索深度
            limit: 返回数量限制
            
        Returns:
            相关实体列表
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class DocumentStore(ABC):
    """文档存储抽象基类
    
    定义结构化文档存储的基本操作接口。
    """
    
    @abstractmethod
    def save_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """保存文档"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索文档"""
        pass

