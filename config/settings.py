"""
文件路径: config/settings.py
功能: 应用配置管理
"""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用基础配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    app_name: str = Field(default="AgentMemorySystem")
    app_env: Literal["development", "production", "testing"] = Field(default="development")
    debug: bool = Field(default=True)
    secret_key: str = Field(default="change-me-in-production")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection: str = Field(default="hello_agents_vectors")
    
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    
    # SQLite
    sqlite_path: str = Field(default="data/memory.db")


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    embed_model_type: Literal["dashscope", "local", "tfidf"] = Field(default="dashscope")
    embed_model_name: str = Field(default="text-embedding-v3")
    embed_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    embed_dimension: int = Field(default=1024)
    embed_batch_size: int = Field(default=25)


class LLMSettings(BaseSettings):
    """大语言模型配置（参考DashScope兼容模式）"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    llm_model_type: Literal["dashscope", "openai"] = Field(default="dashscope")
    llm_model_name: str = Field(default="qwen-plus")
    llm_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    llm_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.7)


class Settings(BaseSettings):
    """统一配置入口"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 便捷访问
settings = get_settings()

