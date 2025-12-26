"""
文件路径: core/memory/base.py
功能: 记忆系统基础类和配置

按照第8章架构设计的基础组件:
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- BaseMemory: 记忆基类（抽象）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import uuid


class MemoryItem(BaseModel):
    """记忆项数据结构
    
    表示一条记忆的完整信息，包括内容、类型、重要性等。
    """
    id: str                          # 记忆唯一标识
    content: str                     # 记忆内容
    memory_type: str                 # 记忆类型（working/episodic/semantic/perceptual）
    user_id: str                     # 用户ID
    timestamp: datetime              # 创建时间
    importance: float = 0.5          # 重要性分数 [0, 1]
    metadata: Dict[str, Any] = {}    # 额外元数据

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """记忆系统配置
    
    统一管理所有记忆类型的配置参数。
    """
    
    # 存储路径
    storage_path: str = "./memory_data"
    
    # 统计显示用的基础配置
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """记忆基类（抽象）

    定义所有记忆类型的通用接口和行为。
    所有具体记忆类型（WorkingMemory, EpisodicMemory等）必须继承此类。
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        """初始化记忆基类
        
        Args:
            config: 记忆配置对象
            storage_backend: 可选的存储后端
        """
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项

        Args:
            memory_item: 记忆项对象

        Returns:
            记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索相关记忆

        Args:
            query: 查询内容
            limit: 返回数量限制
            **kwargs: 其他检索参数

        Returns:
            相关记忆列表
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在

        Args:
            memory_id: 记忆ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        """生成记忆ID"""
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """计算记忆重要性

        Args:
            content: 记忆内容
            base_importance: 基础重要性

        Returns:
            计算后的重要性分数
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()

