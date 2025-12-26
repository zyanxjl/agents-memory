"""
文件路径: api/schemas/graph.py
功能: 知识图谱相关的Pydantic模型

定义知识图谱API的请求和响应模型。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EntityInfo(BaseModel):
    """实体信息"""
    id: str = Field(..., description="实体ID")
    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    related_count: int = Field(0, description="关联实体数量")


class RelationshipInfo(BaseModel):
    """关系信息"""
    from_id: str = Field(..., description="源实体ID")
    to_id: str = Field(..., description="目标实体ID")
    relationship_type: str = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")


class EntitySearch(BaseModel):
    """
    实体搜索请求
    
    用于在图谱中搜索实体。
    """
    query: str = Field(..., min_length=1, description="搜索关键词")
    entity_types: Optional[List[str]] = Field(None, description="实体类型过滤")
    limit: int = Field(20, ge=1, le=100, description="返回数量")


class PathQuery(BaseModel):
    """
    路径查询请求
    
    查找两个实体之间的路径。
    """
    from_entity_id: str = Field(..., description="起始实体ID")
    to_entity_id: str = Field(..., description="目标实体ID")
    max_depth: int = Field(4, ge=1, le=10, description="最大搜索深度")


class PathInfo(BaseModel):
    """路径信息"""
    entities: List[EntityInfo] = Field(default_factory=list, description="路径上的实体")
    relationships: List[RelationshipInfo] = Field(default_factory=list, description="路径上的关系")
    path_length: int = Field(0, description="路径长度")


class VisualizationNode(BaseModel):
    """可视化节点（用于前端图谱展示）"""
    id: str = Field(..., description="节点ID")
    name: str = Field(..., description="节点名称")
    category: str = Field(..., description="节点类别/类型")
    value: float = Field(1.0, description="节点大小/权重")


class VisualizationLink(BaseModel):
    """可视化边"""
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    relationship: str = Field(..., description="关系类型")
    value: float = Field(1.0, description="边权重")


class VisualizationData(BaseModel):
    """
    可视化数据
    
    适用于 ECharts 等图表库的节点和边数据。
    """
    nodes: List[VisualizationNode] = Field(default_factory=list, description="节点列表")
    links: List[VisualizationLink] = Field(default_factory=list, description="边列表")
    categories: List[Dict[str, str]] = Field(default_factory=list, description="类别列表")


class GraphStats(BaseModel):
    """图谱统计信息"""
    total_entities: int = Field(0, description="实体总数")
    total_relationships: int = Field(0, description="关系总数")
    entity_types: Dict[str, int] = Field(default_factory=dict, description="各类型实体数量")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="各类型关系数量")
    is_connected: bool = Field(False, description="图数据库是否连接")

