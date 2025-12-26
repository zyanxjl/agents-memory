"""嵌入服务模块导出

提供统一的文本嵌入接口与多种实现:
- EmbeddingModel: 抽象基类
- LocalTransformerEmbedding: 本地Transformer嵌入
- DashScopeEmbedding: 阿里云DashScope嵌入
- TFIDFEmbedding: TF-IDF兜底嵌入
- get_text_embedder(): 全局单例Provider
"""

from .base import EmbeddingModel
from .local import LocalTransformerEmbedding
from .dashscope import DashScopeEmbedding
from .tfidf import TFIDFEmbedding
from .provider import (
    create_embedding_model,
    create_embedding_model_with_fallback,
    get_text_embedder,
    get_dimension,
    refresh_embedder
)

__all__ = [
    # 基类
    "EmbeddingModel",
    # 实现类
    "LocalTransformerEmbedding",
    "DashScopeEmbedding", 
    "TFIDFEmbedding",
    # Provider函数
    "create_embedding_model",
    "create_embedding_model_with_fallback",
    "get_text_embedder",
    "get_dimension",
    "refresh_embedder",
]
