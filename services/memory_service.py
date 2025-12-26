"""
文件路径: services/memory_service.py
功能: 记忆服务层 - 封装记忆系统的业务逻辑

提供:
- 记忆的增删改查 (CRUD)
- 多类型记忆的统一搜索
- 记忆整合与遗忘策略
- 记忆导入导出
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from core.memory import MemoryManager, MemoryItem, MemoryConfig

logger = logging.getLogger(__name__)


# ==================== 请求/响应数据模型 ====================

class MemoryCreateRequest(BaseModel):
    """创建记忆的请求模型"""
    content: str = Field(..., min_length=1, max_length=10000, description="记忆内容")
    memory_type: str = Field("auto", description="记忆类型: working/episodic/semantic/perceptual/auto")
    user_id: str = Field("default", description="用户ID")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="重要性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class MemoryUpdateRequest(BaseModel):
    """更新记忆的请求模型"""
    content: Optional[str] = Field(None, max_length=10000, description="新内容")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="新重要性")
    metadata: Optional[Dict[str, Any]] = Field(None, description="要更新的元数据")


class MemorySearchRequest(BaseModel):
    """搜索记忆的请求模型"""
    query: str = Field(..., min_length=1, description="搜索查询")
    memory_types: List[str] = Field(
        default_factory=lambda: ["working", "episodic", "semantic"],
        description="要搜索的记忆类型"
    )
    user_id: Optional[str] = Field(None, description="用户ID过滤")
    limit: int = Field(10, ge=1, le=100, description="返回结果数量")
    min_importance: float = Field(0.0, ge=0.0, le=1.0, description="最低重要性阈值")


class MemoryResponse(BaseModel):
    """记忆响应模型"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None  # 搜索时的相关性分数

    @classmethod
    def from_memory_item(cls, item: MemoryItem, score: float = None) -> "MemoryResponse":
        """从 MemoryItem 对象转换为响应模型"""
        return cls(
            id=item.id,
            content=item.content,
            memory_type=item.memory_type,
            user_id=item.user_id,
            timestamp=item.timestamp,
            importance=item.importance,
            metadata=item.metadata,
            relevance_score=score
        )


class MemoryStatsResponse(BaseModel):
    """记忆统计响应模型"""
    total_count: int = Field(0, description="总记忆数")
    working_count: int = Field(0, description="工作记忆数")
    episodic_count: int = Field(0, description="情景记忆数")
    semantic_count: int = Field(0, description="语义记忆数")
    perceptual_count: int = Field(0, description="感知记忆数")
    avg_importance: float = Field(0.0, description="平均重要性")


class ConsolidateRequest(BaseModel):
    """记忆整合请求模型"""
    source_type: str = Field("working", description="源记忆类型")
    target_type: str = Field("episodic", description="目标记忆类型")
    importance_threshold: float = Field(0.7, ge=0.0, le=1.0, description="整合阈值")


class ForgetRequest(BaseModel):
    """记忆遗忘请求模型"""
    strategy: str = Field("importance_based", description="遗忘策略")
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="遗忘阈值")
    max_age_days: int = Field(30, ge=1, description="最大保留天数")


# ==================== 服务类 ====================

class MemoryService:
    """
    记忆服务类
    
    封装记忆系统的所有业务逻辑，提供统一的服务接口。
    作为 API 层与核心层之间的桥梁。
    """

    def __init__(self, config: Optional[MemoryConfig] = None, user_id: str = "default"):
        """
        初始化记忆服务
        
        Args:
            config: 可选的记忆配置
            user_id: 默认用户ID
        """
        self.config = config or MemoryConfig()
        self.default_user_id = user_id
        
        # 创建记忆管理器实例
        self.manager = MemoryManager(
            config=self.config,
            user_id=user_id,
            enable_working=True,
            enable_episodic=True,
            enable_semantic=True,
            enable_perceptual=False  # 感知记忆默认关闭
        )
        
        logger.info(f"✅ MemoryService 初始化完成，用户: {user_id}")

    # ==================== CRUD 操作 ====================

    def add_memory(self, request: MemoryCreateRequest) -> MemoryResponse:
        """
        添加新记忆
        
        Args:
            request: 创建记忆请求
            
        Returns:
            创建的记忆响应
        """
        # 确定是否自动分类
        auto_classify = (request.memory_type == "auto")
        memory_type = "working" if auto_classify else request.memory_type
        
        # 调用管理器添加记忆
        memory_id = self.manager.add_memory(
            content=request.content,
            memory_type=memory_type,
            importance=request.importance,
            metadata=request.metadata,
            auto_classify=auto_classify
        )
        
        # 构建响应
        response = MemoryResponse(
            id=memory_id,
            content=request.content,
            memory_type=memory_type if not auto_classify else self._get_memory_type(memory_id),
            user_id=request.user_id or self.default_user_id,
            timestamp=datetime.now(),
            importance=request.importance,
            metadata=request.metadata
        )
        
        logger.info(f"添加记忆: ID={memory_id[:8]}..., 类型={response.memory_type}")
        return response

    def get_memory(self, memory_id: str) -> Optional[MemoryResponse]:
        """
        获取单个记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            记忆响应，不存在返回 None
        """
        # 遍历所有记忆类型查找
        for mem_type, mem_instance in self.manager.memory_types.items():
            if mem_instance.has_memory(memory_id):
                # 从记忆实例获取记忆项
                all_memories = mem_instance.get_all()
                for item in all_memories:
                    if item.id == memory_id:
                        return MemoryResponse.from_memory_item(item)
        return None

    def update_memory(self, memory_id: str, request: MemoryUpdateRequest) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            request: 更新请求
            
        Returns:
            是否更新成功
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
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        success = self.manager.remove_memory(memory_id)
        if success:
            logger.info(f"删除记忆: ID={memory_id[:8]}...")
        return success

    # ==================== 搜索操作 ====================

    def search_memories(self, request: MemorySearchRequest) -> List[MemoryResponse]:
        """
        搜索记忆（多类型并行检索）
        
        Args:
            request: 搜索请求
            
        Returns:
            匹配的记忆列表，按相关性排序
        """
        # 调用管理器检索
        results = self.manager.retrieve_memories(
            query=request.query,
            memory_types=request.memory_types,
            limit=request.limit,
            min_importance=request.min_importance
        )
        
        # 转换为响应模型
        responses = []
        for i, item in enumerate(results):
            # 使用排名作为简单的相关性分数
            score = 1.0 - (i / max(len(results), 1)) * 0.5
            responses.append(MemoryResponse.from_memory_item(item, score=score))
        
        logger.debug(f"搜索完成: query='{request.query[:20]}...', 结果数={len(responses)}")
        return responses

    def list_memories(
        self,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "timestamp",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        分页列出记忆
        
        Args:
            memory_type: 可选的记忆类型过滤
            page: 页码（从1开始）
            page_size: 每页数量
            sort_by: 排序字段
            sort_order: 排序方向
            
        Returns:
            包含分页信息和记忆列表的字典
        """
        # 收集所有记忆
        all_memories = []
        
        if memory_type and memory_type in self.manager.memory_types:
            # 只获取指定类型
            all_memories = self.manager.memory_types[memory_type].get_all()
        else:
            # 获取所有类型
            for mem_instance in self.manager.memory_types.values():
                all_memories.extend(mem_instance.get_all())
        
        # 排序
        reverse = (sort_order == "desc")
        if sort_by == "importance":
            all_memories.sort(key=lambda m: m.importance, reverse=reverse)
        else:
            all_memories.sort(key=lambda m: m.timestamp, reverse=reverse)
        
        # 分页
        total = len(all_memories)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_memories[start:end]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0,
            "items": [MemoryResponse.from_memory_item(m) for m in page_items]
        }

    # ==================== 管理操作 ====================

    def get_stats(self) -> MemoryStatsResponse:
        """
        获取记忆统计信息
        
        Returns:
            记忆统计响应
        """
        stats = self.manager.get_memory_stats()
        by_type = stats.get("memories_by_type", {})
        
        # 计算平均重要性
        total_importance = 0.0
        total_count = stats.get("total_memories", 0)
        
        for mem_type, type_stats in by_type.items():
            avg_imp = type_stats.get("avg_importance", 0.0)
            count = type_stats.get("count", 0)
            total_importance += avg_imp * count
        
        avg_importance = total_importance / total_count if total_count > 0 else 0.0
        
        return MemoryStatsResponse(
            total_count=total_count,
            working_count=by_type.get("working", {}).get("count", 0),
            episodic_count=by_type.get("episodic", {}).get("count", 0),
            semantic_count=by_type.get("semantic", {}).get("count", 0),
            perceptual_count=by_type.get("perceptual", {}).get("count", 0),
            avg_importance=avg_importance
        )

    def consolidate(self, request: ConsolidateRequest) -> Dict[str, Any]:
        """
        执行记忆整合（将短期记忆转化为长期记忆）
        
        Args:
            request: 整合请求
            
        Returns:
            整合结果统计
        """
        count = self.manager.consolidate_memories(
            from_type=request.source_type,
            to_type=request.target_type,
            importance_threshold=request.importance_threshold
        )
        
        logger.info(f"记忆整合完成: {count} 条从 {request.source_type} -> {request.target_type}")
        return {
            "consolidated_count": count,
            "source_type": request.source_type,
            "target_type": request.target_type
        }

    def forget(self, request: ForgetRequest) -> Dict[str, Any]:
        """
        执行记忆遗忘
        
        Args:
            request: 遗忘请求
            
        Returns:
            遗忘结果统计
        """
        count = self.manager.forget_memories(
            strategy=request.strategy,
            threshold=request.threshold,
            max_age_days=request.max_age_days
        )
        
        logger.info(f"记忆遗忘完成: {count} 条被删除 (策略: {request.strategy})")
        return {
            "forgotten_count": count,
            "strategy": request.strategy,
            "threshold": request.threshold
        }

    def export_memories(self, memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        导出记忆
        
        Args:
            memory_type: 可选的记忆类型过滤
            
        Returns:
            导出数据
        """
        memories = []
        
        if memory_type and memory_type in self.manager.memory_types:
            memories = self.manager.memory_types[memory_type].get_all()
        else:
            for mem_instance in self.manager.memory_types.values():
                memories.extend(mem_instance.get_all())
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_count": len(memories),
            "memory_type_filter": memory_type,
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

    def import_memories(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入记忆
        
        Args:
            data: 导入数据（export_memories 的输出格式）
            
        Returns:
            导入结果统计
        """
        imported = 0
        failed = 0
        
        for mem_data in data.get("memories", []):
            try:
                request = MemoryCreateRequest(
                    content=mem_data["content"],
                    memory_type=mem_data.get("memory_type", "episodic"),
                    user_id=mem_data.get("user_id", self.default_user_id),
                    importance=mem_data.get("importance", 0.5),
                    metadata=mem_data.get("metadata", {})
                )
                self.add_memory(request)
                imported += 1
            except Exception as e:
                logger.warning(f"导入记忆失败: {e}")
                failed += 1
        
        logger.info(f"导入记忆完成: 成功={imported}, 失败={failed}")
        return {
            "imported_count": imported,
            "failed_count": failed,
            "total": len(data.get("memories", []))
        }

    def clear_all(self) -> None:
        """清空所有记忆"""
        self.manager.clear_all_memories()
        logger.info("所有记忆已清空")

    # ==================== 辅助方法 ====================

    def _get_memory_type(self, memory_id: str) -> str:
        """根据记忆ID获取其类型"""
        for mem_type, mem_instance in self.manager.memory_types.items():
            if mem_instance.has_memory(memory_id):
                return mem_type
        return "unknown"

