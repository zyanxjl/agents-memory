"""
文件路径: api/schemas/common.py
功能: 通用Pydantic响应模型

定义API层通用的请求/响应模型，确保接口返回格式统一。
"""

from typing import TypeVar, Generic, List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

# 泛型类型变量，用于通用响应
T = TypeVar("T")


class ResponseBase(BaseModel):
    """基础响应模型 - 所有响应的基类"""
    success: bool = Field(True, description="请求是否成功")
    message: str = Field("", description="响应消息")


class DataResponse(ResponseBase, Generic[T]):
    """
    带数据的响应模型
    
    用于返回单个对象的API响应。
    
    示例:
        DataResponse(success=True, data={"id": "123", "name": "test"})
    """
    data: Optional[T] = Field(None, description="响应数据")


class ListResponse(ResponseBase, Generic[T]):
    """
    列表响应模型（带分页）
    
    用于返回列表数据的API响应，支持分页信息。
    """
    data: List[T] = Field(default_factory=list, description="数据列表")
    total: int = Field(0, description="总数")
    page: int = Field(1, description="当前页")
    page_size: int = Field(20, description="每页数量")
    total_pages: int = Field(0, description="总页数")


class ErrorResponse(BaseModel):
    """
    错误响应模型
    
    用于API错误时的统一响应格式。
    """
    success: bool = Field(False)
    message: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误")
    error_code: Optional[str] = Field(None, description="错误代码")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="状态: ok/degraded/unhealthy")
    components: Dict[str, Any] = Field(default_factory=dict, description="组件状态")
    timestamp: datetime = Field(default_factory=datetime.now)


class StatsResponse(BaseModel):
    """统计信息响应"""
    total_count: int = Field(0)
    details: Dict[str, Any] = Field(default_factory=dict)

