"""
文件路径: api/schemas/memory.py
功能: 记忆相关的Pydantic模型

定义记忆管理API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryCreate(BaseModel):
    """
    创建记忆请求
    
    用于添加新记忆到系统中。
    """
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=10000, 
        description="记忆内容"
    )
    memory_type: str = Field(
        "auto", 
        description="记忆类型: working/episodic/semantic/auto"
    )
    importance: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="重要性分数"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="额外元数据"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Python是一种高级编程语言",
                "memory_type": "semantic",
                "importance": 0.8,
                "metadata": {"source": "learning"}
            }
        }


class MemoryUpdate(BaseModel):
    """
    更新记忆请求
    
    用于更新已存在记忆的内容或属性。
    """
    content: Optional[str] = Field(None, max_length=10000, description="新内容")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="新重要性")
    metadata: Optional[Dict[str, Any]] = Field(None, description="要更新的元数据")


class MemorySearch(BaseModel):
    """
    搜索记忆请求
    
    用于在记忆系统中进行语义搜索。
    """
    query: str = Field(..., min_length=1, description="搜索查询")
    memory_types: List[str] = Field(
        default=["working", "episodic", "semantic"],
        description="要搜索的记忆类型"
    )
    limit: int = Field(10, ge=1, le=100, description="返回数量")
    min_importance: float = Field(0.0, ge=0.0, le=1.0, description="最低重要性")


class MemoryResponse(BaseModel):
    """
    记忆响应模型
    
    返回记忆的完整信息。
    """
    id: str = Field(..., description="记忆ID")
    content: str = Field(..., description="记忆内容")
    memory_type: str = Field(..., description="记忆类型")
    user_id: str = Field(..., description="用户ID")
    timestamp: datetime = Field(..., description="创建时间")
    importance: float = Field(..., description="重要性")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    relevance_score: Optional[float] = Field(None, description="相关性分数")


class MemoryStats(BaseModel):
    """记忆统计信息"""
    total_count: int = Field(0, description="总记忆数")
    working_count: int = Field(0, description="工作记忆数")
    episodic_count: int = Field(0, description="情景记忆数")
    semantic_count: int = Field(0, description="语义记忆数")
    perceptual_count: int = Field(0, description="感知记忆数")
    avg_importance: float = Field(0.0, description="平均重要性")


class ConsolidateRequest(BaseModel):
    """
    记忆整合请求
    
    将短期记忆整合到长期记忆。
    """
    source_type: str = Field("working", description="源记忆类型")
    target_type: str = Field("episodic", description="目标记忆类型")
    importance_threshold: float = Field(0.7, ge=0.0, le=1.0, description="整合阈值")


class ForgetRequest(BaseModel):
    """
    记忆遗忘请求
    
    执行记忆遗忘策略。
    """
    strategy: str = Field("importance_based", description="遗忘策略")
    threshold: float = Field(0.1, ge=0.0, le=1.0, description="遗忘阈值")
    max_age_days: int = Field(30, ge=1, description="最大保留天数")

