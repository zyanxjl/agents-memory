"""
文件路径: core/memory/manager.py
功能: 记忆管理器 - 记忆核心层的统一管理接口

负责:
- 记忆生命周期管理
- 记忆优先级和重要性评估
- 记忆遗忘和清理机制
- 多类型记忆的协调管理
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from .base import MemoryItem, MemoryConfig
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .perceptual import PerceptualMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口
    
    提供对所有记忆类型的统一访问接口，管理记忆的生命周期。
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False
    ):
        """初始化记忆管理器
        
        Args:
            config: 记忆配置
            user_id: 用户ID
            enable_working: 是否启用工作记忆
            enable_episodic: 是否启用情景记忆
            enable_semantic: 是否启用语义记忆
            enable_perceptual: 是否启用感知记忆
        """
        self.config = config or MemoryConfig()
        self.user_id = user_id
        
        # 初始化各类型记忆
        self.memory_types = {}
        
        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config)
        
        if enable_episodic:
            self.memory_types['episodic'] = EpisodicMemory(self.config)
            
        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config)
            
        if enable_perceptual:
            self.memory_types['perceptual'] = PerceptualMemory(self.config)
        
        logger.info(f"MemoryManager初始化完成，启用记忆类型: {list(self.memory_types.keys())}")
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True
    ) -> str:
        """添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数 (0-1)
            metadata: 元数据
            auto_classify: 是否自动分类到合适的记忆类型
            
        Returns:
            记忆ID
        """
        # 自动分类
        if auto_classify:
            memory_type = self._classify_memory_type(content, metadata)
        
        # 计算重要性
        if importance is None:
            importance = self._calculate_importance(content, metadata)
        
        # 创建记忆项
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # 添加到对应的记忆类型
        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logger.debug(f"添加记忆到 {memory_type}: {memory_id}")
            return memory_id
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
    
    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple] = None
    ) -> List[MemoryItem]:
        """检索记忆
        
        Args:
            query: 查询内容
            memory_types: 要检索的记忆类型列表
            limit: 返回数量限制
            min_importance: 最小重要性阈值
            time_range: 时间范围
            
        Returns:
            检索到的记忆列表
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())
        
        all_results = []
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    type_results = memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    logger.warning(f"检索 {memory_type} 记忆时出错: {e}")
                    continue

        # 按重要性排序
        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新记忆"""
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)
        
        logger.warning(f"未找到记忆: {memory_id}")
        return False
    
    def remove_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)
        
        logger.warning(f"未找到记忆: {memory_id}")
        return False
    
    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        """记忆遗忘机制"""
        total_forgotten = 0
        
        for memory_type, memory_instance in self.memory_types.items():
            if hasattr(memory_instance, 'forget'):
                forgotten = memory_instance.forget(strategy, threshold, max_age_days)
                total_forgotten += forgotten

        logger.info(f"记忆遗忘完成: {total_forgotten} 条记忆")
        return total_forgotten

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> int:
        """记忆整合 - 将重要的短期记忆转换为长期记忆"""
        if from_type not in self.memory_types or to_type not in self.memory_types:
            logger.warning(f"记忆类型不存在: {from_type} -> {to_type}")
            return 0

        source_memory = self.memory_types[from_type]
        target_memory = self.memory_types[to_type]

        all_memories = source_memory.get_all()
        candidates = [m for m in all_memories if m.importance >= importance_threshold]

        consolidated_count = 0
        for memory in candidates:
            if source_memory.remove(memory.id):
                memory.memory_type = to_type
                memory.importance = min(1.0, memory.importance * 1.1)
                target_memory.add(memory)
                consolidated_count += 1

        logger.info(f"记忆整合完成: {consolidated_count} 条记忆从 {from_type} 转移到 {to_type}")
        return consolidated_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor
            }
        }

        for memory_type, memory_instance in self.memory_types.items():
            type_stats = memory_instance.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            stats["total_memories"] += type_stats.get("count", 0)

        return stats

    def clear_all_memories(self):
        """清空所有记忆"""
        for memory_type, memory_instance in self.memory_types.items():
            memory_instance.clear()
        logger.info("所有记忆已清空")

    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """自动分类记忆类型"""
        if metadata and metadata.get("type"):
            return metadata["type"]
        
        if self._is_episodic_content(content):
            return "episodic"
        elif self._is_semantic_content(content):
            return "semantic"
        else:
            return "working"
    
    def _is_episodic_content(self, content: str) -> bool:
        """判断是否为情景记忆内容"""
        episodic_keywords = ["昨天", "今天", "明天", "上次", "记得", "发生", "经历"]
        return any(keyword in content for keyword in episodic_keywords)
    
    def _is_semantic_content(self, content: str) -> bool:
        """判断是否为语义记忆内容"""
        semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "方法"]
        return any(keyword in content for keyword in semantic_keywords)
    
    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        """计算记忆重要性"""
        importance = 0.5
        
        if len(content) > 100:
            importance += 0.1
        
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2
        
        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2
        
        return max(0.0, min(1.0, importance))
    
    def __str__(self) -> str:
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"

