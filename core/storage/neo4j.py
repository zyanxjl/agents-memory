"""
文件路径: core/storage/neo4j.py
功能: Neo4j图数据库存储实现

主要特性:
- 使用新的配置系统
- 继承 GraphStore 抽象基类
- 支持实体、关系的CRUD操作
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings
from .base import GraphStore

# 尝试导入Neo4j驱动
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j图数据库存储实现
    
    Args:
        uri: Neo4j服务URI
        username: 用户名
        password: 密码
        database: 数据库名称
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None, 
        password: Optional[str] = None,
        database: str = "neo4j",
        **kwargs
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j未安装。请运行: pip install neo4j>=5.0.0")
        
        # 使用配置或参数
        db_settings = settings.database
        self.uri = uri or db_settings.neo4j_uri
        self.username = username or db_settings.neo4j_username
        self.password = password or db_settings.neo4j_password
        self.database = database
        
        self.driver = None
        self._initialize_driver(**kwargs)
        self._create_indexes()
    
    def _initialize_driver(self, **config):
        """初始化Neo4j驱动"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                **config
            )
            self.driver.verify_connectivity()
            logger.info(f"成功连接到Neo4j: {self.uri}")
        except AuthError as e:
            logger.error(f"Neo4j认证失败: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j服务不可用: {e}")
            raise
    
    def _create_indexes(self):
        """创建必要的索引"""
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for query in indexes:
                try:
                    session.run(query)
                except Exception:
                    pass
    
    def add_entity(
        self, 
        entity_id: str, 
        name: str, 
        entity_type: str, 
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加实体节点"""
        try:
            props = properties or {}
            props.update({
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": datetime.now().isoformat()
            })
            
            query = """
            MERGE (e:Entity {id: $entity_id})
            SET e += $properties
            RETURN e
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, properties=props)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False
    
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加实体间关系"""
        try:
            props = properties or {}
            props["created_at"] = datetime.now().isoformat()
            
            query = f"""
            MATCH (from:Entity {{id: $from_id}})
            MATCH (to:Entity {{id: $to_id}})
            MERGE (from)-[r:{relationship_type}]->(to)
            SET r += $properties
            RETURN r
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, from_id=from_entity_id, to_id=to_entity_id, properties=props)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return False
    
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """查找相关实体"""
        try:
            rel_filter = ""
            if relationship_types:
                rel_filter = ":" + "|".join(relationship_types)
            
            query = f"""
            MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
            WHERE start.id <> related.id
            RETURN DISTINCT related, length(path) as distance
            ORDER BY distance
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                return [
                    {**dict(record["related"]), "distance": record["distance"]}
                    for record in result
                ]
                
        except Exception as e:
            logger.error(f"查找相关实体失败: {e}")
            return []
    
    def search_entities_by_name(self, name_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按名称模式搜索实体"""
        try:
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $pattern
            RETURN e
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, pattern=name_pattern, limit=limit)
                return [dict(record["e"]) for record in result]
                
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            return []
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """获取实体的所有关系"""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
            RETURN type(r) as relationship_type, properties(r) as relationship, other
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                return [
                    {
                        "relationship_type": record["relationship_type"],
                        "relationship": record["relationship"],
                        "other_entity": dict(record["other"])
                    }
                    for record in result
                ]
                
        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return []
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体及其关系"""
        try:
            query = "MATCH (e:Entity {id: $entity_id}) DETACH DELETE e"
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                return result.consume().counters.nodes_deleted > 0
        except Exception as e:
            logger.error(f"删除实体失败: {e}")
            return False
    
    def clear_all(self) -> bool:
        """清空所有数据"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                return True
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as health")
                return result.single()["health"] == 1
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            with self.driver.session(database=self.database) as session:
                nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                return {
                    "store_type": "neo4j",
                    "total_nodes": nodes,
                    "total_relationships": rels
                }
        except Exception:
            return {"store_type": "neo4j"}
    
    def __del__(self):
        """析构时关闭驱动"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass

