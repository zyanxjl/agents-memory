"""
文件路径: core/embedding/provider.py
功能: 嵌入模型工厂和单例Provider

提供统一的嵌入模型管理接口:
- create_embedding_model(): 创建嵌入模型实例
- create_embedding_model_with_fallback(): 带回退的创建
- get_text_embedder(): 全局共享的嵌入实例（单例）
- get_dimension(): 获取统一向量维度
- refresh_embedder(): 强制重建嵌入实例
"""

import threading
from typing import Optional
from config import settings
from .base import EmbeddingModel
from .local import LocalTransformerEmbedding
from .dashscope import DashScopeEmbedding
from .tfidf import TFIDFEmbedding


def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """创建嵌入模型实例
    
    Args:
        model_type: "dashscope" | "local" | "tfidf"
        **kwargs: 传递给具体模型的参数
        
    Returns:
        EmbeddingModel实例
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "dashscope":
        return DashScopeEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def create_embedding_model_with_fallback(preferred_type: str = "dashscope", **kwargs) -> EmbeddingModel:
    """带回退的创建：dashscope -> local -> tfidf
    
    Args:
        preferred_type: 首选模型类型
        **kwargs: 传递给具体模型的参数
        
    Returns:
        EmbeddingModel实例
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    
    fallback = ["dashscope", "local", "tfidf"]
    
    # 将首选放最前
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)
    
    last_error = None
    for t in fallback:
        try:
            # TF-IDF 需要特殊处理参数
            if t == "tfidf":
                tfidf_kwargs = {"max_features": kwargs.get("max_features", 384)}
                model = create_embedding_model(t, **tfidf_kwargs)
            else:
                model = create_embedding_model(t, **kwargs)
            
            # 验证模型可用
            _ = model.encode("test")
            logger.info(f"嵌入模型初始化成功: {t}, 维度: {model.dimension}")
            return model
            
        except Exception as e:
            last_error = e
            logger.debug(f"嵌入模型 {t} 初始化失败: {e}")
            continue
    
    # 如果所有模型都失败，尝试强制使用 TF-IDF
    try:
        logger.warning("所有首选嵌入模型不可用，强制使用 TF-IDF 兜底")
        model = TFIDFEmbedding(max_features=384)
        _ = model.encode("test")
        return model
    except Exception as e:
        logger.error(f"TF-IDF 兜底也失败: {e}")
    
    raise RuntimeError(f"所有嵌入模型都不可用: {last_error}")


# ==================
# Provider（单例）
# ==================

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None


def _build_embedder() -> EmbeddingModel:
    """根据配置构建嵌入模型"""
    embed_settings = settings.embedding
    
    preferred = embed_settings.embed_model_type
    model_name = embed_settings.embed_model_name
    api_key = embed_settings.embed_api_key
    
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    if api_key:
        kwargs["api_key"] = api_key
    
    return create_embedding_model_with_fallback(preferred_type=preferred, **kwargs)


def get_text_embedder() -> EmbeddingModel:
    """获取全局共享的文本嵌入实例（线程安全单例）
    
    Returns:
        EmbeddingModel实例
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """获取统一向量维度
    
    Args:
        default: 失败时的默认值
        
    Returns:
        向量维度
    """
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """强制重建嵌入实例（可用于动态切换配置）
    
    Returns:
        新的EmbeddingModel实例
    """
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder

