"""
文件路径: services/graph_service.py
功能: 知识图谱服务层 - 封装图数据库的业务逻辑

提供:
- 实体和关系的查询
- 图谱遍历和路径查找
- 可视化数据生成
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== 请求/响应数据模型 ====================

class EntityInfo(BaseModel):
    """实体信息"""
    id: str = Field(..., description="实体ID")
    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    related_count: int = Field(0, description="关联实体数量")


class RelationshipInfo(BaseModel):
    """关系信息"""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntitySearchRequest(BaseModel):
    """实体搜索请求"""
    query: str = Field(..., min_length=1, description="搜索关键词")
    entity_types: Optional[List[str]] = Field(None, description="实体类型过滤")
    limit: int = Field(20, ge=1, le=100, description="返回数量")


class PathQueryRequest(BaseModel):
    """路径查询请求"""
    from_entity_id: str = Field(..., description="起始实体ID")
    to_entity_id: str = Field(..., description="目标实体ID")
    max_depth: int = Field(4, ge=1, le=10, description="最大深度")


class PathInfo(BaseModel):
    """路径信息"""
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    path_length: int = 0


class VisualizationNode(BaseModel):
    """可视化节点（用于前端图谱展示）"""
    id: str
    name: str
    category: str  # 实体类型
    value: float = 1.0  # 节点大小


class VisualizationLink(BaseModel):
    """可视化边"""
    source: str
    target: str
    relationship: str
    value: float = 1.0


class VisualizationData(BaseModel):
    """可视化数据（用于ECharts等图表库）"""
    nodes: List[VisualizationNode] = Field(default_factory=list)
    links: List[VisualizationLink] = Field(default_factory=list)
    categories: List[Dict[str, str]] = Field(default_factory=list)


class GraphStatsResponse(BaseModel):
    """图谱统计响应"""
    total_entities: int = 0
    total_relationships: int = 0
    entity_types: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)
    is_connected: bool = False


# ==================== 服务类 ====================

class GraphService:
    """
    知识图谱服务类
    
    封装图数据库操作，提供图谱查询和可视化功能。
    """

    def __init__(self):
        """初始化图谱服务"""
        self._graph_store = None
        self._connected = False
        self._init_graph_store()
        logger.info(f"✅ GraphService 初始化完成，连接状态: {self._connected}")

    def _init_graph_store(self):
        """初始化图存储"""
        try:
            from core.storage import Neo4jGraphStore
            self._graph_store = Neo4jGraphStore()
            self._connected = True
        except ImportError:
            logger.warning("Neo4j驱动未安装，图谱功能将不可用")
            self._connected = False
        except Exception as e:
            logger.warning(f"Neo4j连接失败: {e}")
            self._connected = False

    # ==================== 实体操作 ====================

    def get_entity(self, entity_id: str) -> Optional[EntityInfo]:
        """
        获取单个实体详情
        
        Args:
            entity_id: 实体ID
            
        Returns:
            实体信息，不存在返回None
        """
        if not self._connected:
            return None
        
        try:
            # 使用按名称搜索（因为Neo4j store没有直接的get_entity方法）
            entities = self._graph_store.search_entities_by_name(entity_id, limit=1)
            
            if entities:
                entity = entities[0]
                # 获取关联数量
                related = self._graph_store.find_related_entities(entity_id, max_depth=1, limit=100)
                
                return EntityInfo(
                    id=entity.get("id", entity_id),
                    name=entity.get("name", "Unknown"),
                    entity_type=entity.get("type", "UNKNOWN"),
                    properties=entity,
                    related_count=len(related)
                )
            return None
            
        except Exception as e:
            logger.error(f"获取实体失败: {e}")
            return None

    def search_entities(self, request: EntitySearchRequest) -> List[EntityInfo]:
        """
        搜索实体
        
        Args:
            request: 搜索请求
            
        Returns:
            匹配的实体列表
        """
        if not self._connected:
            return []
        
        try:
            entities = self._graph_store.search_entities_by_name(
                request.query, 
                limit=request.limit
            )
            
            results = []
            for entity in entities:
                # 类型过滤
                entity_type = entity.get("type", "UNKNOWN")
                if request.entity_types and entity_type not in request.entity_types:
                    continue
                
                results.append(EntityInfo(
                    id=entity.get("id", ""),
                    name=entity.get("name", "Unknown"),
                    entity_type=entity_type,
                    properties=entity
                ))
            
            return results[:request.limit]
            
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            return []

    def list_entities(
        self,
        entity_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        分页列出实体
        
        Args:
            entity_type: 可选的类型过滤
            page: 页码
            page_size: 每页数量
            
        Returns:
            分页结果
        """
        if not self._connected:
            return {"total": 0, "page": page, "page_size": page_size, "items": []}
        
        try:
            # 使用空查询获取所有实体
            all_entities = self._graph_store.search_entities_by_name("", limit=10000)
            
            # 类型过滤
            if entity_type:
                all_entities = [e for e in all_entities if e.get("type") == entity_type]
            
            # 转换
            entity_infos = [
                EntityInfo(
                    id=e.get("id", ""),
                    name=e.get("name", "Unknown"),
                    entity_type=e.get("type", "UNKNOWN"),
                    properties=e
                )
                for e in all_entities
            ]
            
            # 分页
            total = len(entity_infos)
            start = (page - 1) * page_size
            end = start + page_size
            
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": entity_infos[start:end]
            }
            
        except Exception as e:
            logger.error(f"列出实体失败: {e}")
            return {"total": 0, "page": page, "page_size": page_size, "items": []}

    # ==================== 关系操作 ====================

    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[EntityInfo]:
        """
        查找相关实体
        
        Args:
            entity_id: 起始实体ID
            relationship_types: 关系类型过滤
            max_depth: 最大深度
            limit: 返回数量
            
        Returns:
            相关实体列表
        """
        if not self._connected:
            return []
        
        try:
            related = self._graph_store.find_related_entities(
                entity_id=entity_id,
                relationship_types=relationship_types,
                max_depth=max_depth,
                limit=limit
            )
            
            return [
                EntityInfo(
                    id=r.get("id", ""),
                    name=r.get("name", "Unknown"),
                    entity_type=r.get("type", "UNKNOWN"),
                    properties=r
                )
                for r in related
            ]
            
        except Exception as e:
            logger.error(f"查找相关实体失败: {e}")
            return []

    def find_path(self, request: PathQueryRequest) -> Optional[PathInfo]:
        """
        查找两个实体之间的路径
        
        Args:
            request: 路径查询请求
            
        Returns:
            路径信息，未找到返回None
        """
        if not self._connected:
            return None
        
        try:
            # 使用BFS查找路径
            visited: Set[str] = set()
            queue = [(request.from_entity_id, [])]
            
            while queue:
                current_id, path = queue.pop(0)
                
                if current_id == request.to_entity_id:
                    # 构建路径信息
                    entities = []
                    for eid in path + [current_id]:
                        entity = self.get_entity(eid)
                        if entity:
                            entities.append(entity)
                    
                    return PathInfo(
                        entities=entities,
                        relationships=[],  # 简化实现
                        path_length=len(entities) - 1
                    )
                
                if current_id in visited or len(path) >= request.max_depth:
                    continue
                
                visited.add(current_id)
                
                # 获取邻居
                related = self._graph_store.find_related_entities(
                    entity_id=current_id,
                    max_depth=1,
                    limit=50
                )
                
                for neighbor in related:
                    neighbor_id = neighbor.get("id")
                    if neighbor_id and neighbor_id not in visited:
                        queue.append((neighbor_id, path + [current_id]))
            
            return None
            
        except Exception as e:
            logger.error(f"查找路径失败: {e}")
            return None

    # ==================== 可视化数据 ====================

    def get_visualization_data(
        self,
        center_entity_id: Optional[str] = None,
        depth: int = 2,
        limit: int = 100
    ) -> VisualizationData:
        """
        获取可视化数据（用于前端图谱展示）
        
        Args:
            center_entity_id: 中心实体ID
            depth: 展开深度
            limit: 节点数量限制
            
        Returns:
            可视化数据
        """
        nodes: Dict[str, VisualizationNode] = {}
        links: List[VisualizationLink] = []
        entity_types: Set[str] = set()
        
        if not self._connected:
            return VisualizationData(categories=[{"name": "未连接"}])
        
        try:
            if center_entity_id:
                # 从中心实体展开
                self._expand_for_visualization(
                    entity_id=center_entity_id,
                    depth=depth,
                    nodes=nodes,
                    links=links,
                    entity_types=entity_types,
                    limit=limit,
                    current_depth=0
                )
            else:
                # 获取部分实体作为全局视图
                entities = self._graph_store.search_entities_by_name("", limit=limit)
                for entity in entities:
                    eid = entity.get("id", "")
                    etype = entity.get("type", "UNKNOWN")
                    entity_types.add(etype)
                    nodes[eid] = VisualizationNode(
                        id=eid,
                        name=entity.get("name", "Unknown"),
                        category=etype,
                        value=1.0
                    )
            
            # 构建类别列表
            categories = [{"name": et} for et in sorted(entity_types)]
            
            return VisualizationData(
                nodes=list(nodes.values()),
                links=links,
                categories=categories
            )
            
        except Exception as e:
            logger.error(f"获取可视化数据失败: {e}")
            return VisualizationData()

    def _expand_for_visualization(
        self,
        entity_id: str,
        depth: int,
        nodes: Dict[str, VisualizationNode],
        links: List[VisualizationLink],
        entity_types: Set[str],
        limit: int,
        current_depth: int
    ):
        """递归展开实体用于可视化"""
        if current_depth > depth or len(nodes) >= limit or entity_id in nodes:
            return
        
        # 获取实体信息
        entity = self.get_entity(entity_id)
        if not entity:
            return
        
        # 添加节点
        entity_types.add(entity.entity_type)
        nodes[entity_id] = VisualizationNode(
            id=entity_id,
            name=entity.name,
            category=entity.entity_type,
            value=1.0 + entity.related_count * 0.1
        )
        
        if current_depth < depth and len(nodes) < limit:
            # 获取相关实体
            related = self.find_related_entities(entity_id, max_depth=1, limit=20)
            
            for rel_entity in related:
                # 添加边
                links.append(VisualizationLink(
                    source=entity_id,
                    target=rel_entity.id,
                    relationship="RELATED"
                ))
                
                # 递归展开
                self._expand_for_visualization(
                    entity_id=rel_entity.id,
                    depth=depth,
                    nodes=nodes,
                    links=links,
                    entity_types=entity_types,
                    limit=limit,
                    current_depth=current_depth + 1
                )

    # ==================== 统计操作 ====================

    def get_stats(self) -> GraphStatsResponse:
        """获取图谱统计信息"""
        if not self._connected:
            return GraphStatsResponse(is_connected=False)
        
        try:
            stats = self._graph_store.get_statistics()
            
            return GraphStatsResponse(
                total_entities=stats.get("node_count", 0),
                total_relationships=stats.get("relationship_count", 0),
                entity_types=stats.get("node_types", {}),
                relationship_types=stats.get("relationship_types", {}),
                is_connected=True
            )
            
        except Exception as e:
            logger.error(f"获取图谱统计失败: {e}")
            return GraphStatsResponse(is_connected=self._connected)

    def health_check(self) -> bool:
        """检查图数据库健康状态"""
        if not self._connected or not self._graph_store:
            return False
        
        try:
            # 尝试执行简单查询
            self._graph_store.search_entities_by_name("health_check", limit=1)
            return True
        except Exception:
            return False

