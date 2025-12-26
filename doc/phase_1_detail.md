# 阶段1：核心层重构 - 详细任务规划

## 阶段概述

| 属性 | 说明 |
|------|------|
| **阶段名称** | 核心层重构 |
| **预计时间** | 2天 |
| **核心目标** | 将现有记忆系统重构为清晰的模块结构，保持原有功能不变 |
| **前置条件** | 阶段0完成，目录结构和配置系统就绪 |

### 重构原则
1. **保持功能不变** - 所有现有功能必须正常工作
2. **优化导入结构** - 使用新的配置系统替代环境变量直接读取
3. **统一接口** - 添加抽象基类，便于后续扩展
4. **向后兼容** - 保留原有的公开API

---

## Task 1.1：嵌入模块重构

### 任务描述
将 `embedding.py` 拆分为独立的嵌入服务模块，使用新的配置系统。

### 源文件分析

**原文件：`embedding.py`**
- `EmbeddingModel` - 嵌入模型基类
- `LocalTransformerEmbedding` - 本地Transformer嵌入
- `TFIDFEmbedding` - TF-IDF兜底嵌入
- `DashScopeEmbedding` - 阿里云百炼嵌入
- `create_embedding_model()` - 工厂函数
- `create_embedding_model_with_fallback()` - 带回退的工厂
- `get_text_embedder()` / `get_dimension()` / `refresh_embedder()` - Provider单例

### 目标结构

```
core/embedding/
├── __init__.py           # 模块导出
├── base.py               # EmbeddingModel 基类
├── local.py              # LocalTransformerEmbedding
├── dashscope.py          # DashScopeEmbedding
├── tfidf.py              # TFIDFEmbedding
└── provider.py           # 单例Provider和工厂函数
```

### 具体实现

#### 1.1.1 core/embedding/base.py

```python
"""
文件路径: core/embedding/base.py
功能: 嵌入模型抽象基类
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbeddingModel(ABC):
    """嵌入模型基类（最小接口）
    
    所有嵌入模型实现必须继承此类并实现 encode() 方法和 dimension 属性。
    """

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            单个向量或向量列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入向量的维度"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimension={self.dimension})"
```

#### 1.1.2 core/embedding/local.py

```python
"""
文件路径: core/embedding/local.py
功能: 本地Transformer嵌入模型
"""

from typing import List, Union
import numpy as np
from .base import EmbeddingModel


class LocalTransformerEmbedding(EmbeddingModel):
    """本地Transformer嵌入（优先 sentence-transformers，缺失回退 transformers+torch）
    
    Args:
        model_name: 模型名称，默认 "sentence-transformers/all-MiniLM-L6-v2"
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None  # "st" 或 "hf"
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        self._load_backend()

    def _load_backend(self):
        """加载嵌入后端"""
        # 优先 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            test_vec = self._st_model.encode("test_text")
            self._dimension = len(test_vec)
            self._backend = "st"
            return
        except Exception:
            self._st_model = None

        # 回退 transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            with torch.no_grad():
                inputs = self._hf_tokenizer("test_text", return_tensors="pt", padding=True, truncation=True)
                outputs = self._hf_model(**inputs)
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                self._dimension = int(test_embedding.shape[1])
            self._backend = "hf"
            return
        except Exception:
            self._hf_tokenizer = None
            self._hf_model = None

        raise ImportError("未找到可用的本地嵌入后端，请安装 sentence-transformers 或 transformers+torch")

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """编码文本为向量"""
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        if self._backend == "st":
            vecs = self._st_model.encode(inputs)
            if hasattr(vecs, "tolist"):
                vecs = [v for v in vecs]
        else:
            import torch
            tokenized = self._hf_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs = [v for v in embeddings]

        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)
```

#### 1.1.3 core/embedding/dashscope.py

```python
"""
文件路径: core/embedding/dashscope.py
功能: 阿里云DashScope嵌入模型（支持REST和SDK两种模式）
"""

import os
from typing import List, Union, Optional
import numpy as np
from .base import EmbeddingModel


class DashScopeEmbedding(EmbeddingModel):
    """阿里云 DashScope（通义千问）Embedding
    
    支持两种模式：
    - REST模式：提供 base_url 时使用 OpenAI 兼容的 REST 接口
    - SDK模式：使用官方 dashscope SDK
    
    Args:
        model_name: 模型名称，默认 "text-embedding-v3"
        api_key: API密钥
        base_url: REST API基础URL（可选）
    """

    def __init__(
        self, 
        model_name: str = "text-embedding-v3", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = None
        
        # 仅在非REST情况下初始化SDK
        if not self.base_url:
            self._init_client()
        
        # 探测维度
        test = self.encode("health_check")
        self._dimension = len(test)

    def _init_client(self):
        """初始化DashScope SDK"""
        try:
            if self.api_key:
                os.environ["DASHSCOPE_API_KEY"] = self.api_key
            import dashscope  # noqa: F401
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """编码文本为向量"""
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        # REST 模式
        if self.base_url:
            vecs = self._encode_rest(inputs)
        else:
            vecs = self._encode_sdk(inputs)

        if single:
            return vecs[0]
        return vecs
    
    def _encode_rest(self, inputs: List[str]) -> List[np.ndarray]:
        """使用REST API编码"""
        import httpx
        
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": inputs}
        
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Embedding REST 调用失败: {resp.status_code} {resp.text}")
        
        data = resp.json()
        items = data.get("data") or []
        return [np.array(item.get("embedding")) for item in items]
    
    def _encode_sdk(self, inputs: List[str]) -> List[np.ndarray]:
        """使用SDK编码"""
        from dashscope import TextEmbedding
        
        rsp = TextEmbedding.call(model=self.model_name, input=inputs)
        embeddings_obj = None
        
        if isinstance(rsp, dict):
            embeddings_obj = (rsp.get("output") or {}).get("embeddings")
        else:
            embeddings_obj = getattr(getattr(rsp, "output", None), "embeddings", None)
        
        if not embeddings_obj:
            raise RuntimeError("DashScope 返回为空或格式不匹配")
        
        return [np.array(item.get("embedding") or item.get("vector")) for item in embeddings_obj]

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)
```

#### 1.1.4 core/embedding/tfidf.py

```python
"""
文件路径: core/embedding/tfidf.py
功能: TF-IDF嵌入模型（轻量级兜底方案）
"""

from typing import List, Union
import numpy as np
from .base import EmbeddingModel


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF 简易兜底嵌入
    
    在无深度学习模型时保证系统可用。
    注意：必须先调用 fit() 训练后才能使用 encode()。
    
    Args:
        max_features: 最大特征数量，即向量维度
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        """初始化TF-IDF向量化器"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")

    def fit(self, texts: List[str]):
        """训练TF-IDF模型
        
        Args:
            texts: 训练语料库
        """
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """编码文本为向量"""
        if not self._is_fitted:
            raise ValueError("TF-IDF模型未训练，请先调用 fit() 方法")
        
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = tfidf_matrix.toarray()
        
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
```

#### 1.1.5 core/embedding/provider.py

```python
"""
文件路径: core/embedding/provider.py
功能: 嵌入模型工厂和单例Provider
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
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    
    fallback = ["dashscope", "local", "tfidf"]
    
    # 将首选放最前
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)
    
    for t in fallback:
        try:
            return create_embedding_model(t, **kwargs)
        except Exception:
            continue
    
    raise RuntimeError("所有嵌入模型都不可用，请安装依赖或检查配置")


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
```

#### 1.1.6 core/embedding/__init__.py

```python
"""嵌入服务模块导出"""

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
```

### 验证方法

**验证脚本：scripts/verify_embedding.py**

```python
"""嵌入模块验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    print("=" * 50)
    print("嵌入模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
    try:
        from core.embedding import (
            EmbeddingModel,
            LocalTransformerEmbedding,
            DashScopeEmbedding,
            TFIDFEmbedding,
            get_text_embedder,
            get_dimension
        )
        print("[OK] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 2. 测试TF-IDF嵌入（不需要外部依赖）
    try:
        tfidf = TFIDFEmbedding(max_features=100)
        tfidf.fit(["hello world", "test document", "sample text"])
        vec = tfidf.encode("hello")
        assert len(vec) == tfidf.dimension, "TF-IDF维度不匹配"
        print(f"[OK] TFIDFEmbedding: dimension={tfidf.dimension}")
    except Exception as e:
        print(f"[FAIL] TFIDFEmbedding: {e}")
        all_passed = False
    
    # 3. 测试Provider（可能失败，取决于配置）
    try:
        embedder = get_text_embedder()
        dim = get_dimension()
        print(f"[OK] Provider: embedder={type(embedder).__name__}, dimension={dim}")
    except Exception as e:
        print(f"[WARN] Provider未配置或不可用: {e}")
    
    # 4. 测试基类接口
    try:
        assert hasattr(EmbeddingModel, 'encode'), "缺少encode方法"
        assert hasattr(EmbeddingModel, 'dimension'), "缺少dimension属性"
        print("[OK] 基类接口完整")
    except Exception as e:
        print(f"[FAIL] 基类接口: {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("嵌入模块验证通过！")
    else:
        print("部分验证失败")
    
    return all_passed

if __name__ == "__main__":
    verify()
```

### 预期结果
```
==================================================
嵌入模块验证
==================================================
[OK] 模块导入成功
[OK] TFIDFEmbedding: dimension=100
[OK] Provider: embedder=DashScopeEmbedding, dimension=1024
[OK] 基类接口完整
==================================================
嵌入模块验证通过！
```

---

## Task 1.2：存储模块重构

### 任务描述
将 `storage/` 下的存储实现迁移到 `core/storage/`，统一接口并使用新配置系统。

### 源文件分析

| 原文件 | 描述 |
|--------|------|
| `storage/qdrant_store.py` | Qdrant向量存储（~540行） |
| `storage/neo4j_store.py` | Neo4j图存储（~456行） |
| `storage/document_store.py` | SQLite文档存储 |

### 目标结构

```
core/storage/
├── __init__.py           # 模块导出
├── base.py               # 存储抽象基类
├── qdrant.py             # Qdrant向量存储
├── neo4j.py              # Neo4j图存储
└── sqlite.py             # SQLite文档存储
```

### 具体实现

#### 1.2.1 core/storage/base.py

```python
"""
文件路径: core/storage/base.py
功能: 存储后端抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """添加向量"""
        pass
    
    @abstractmethod
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class GraphStore(ABC):
    """图存储抽象基类"""
    
    @abstractmethod
    def add_entity(
        self, 
        entity_id: str, 
        name: str, 
        entity_type: str, 
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加实体节点"""
        pass
    
    @abstractmethod
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """添加关系"""
        pass
    
    @abstractmethod
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """查找相关实体"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass


class DocumentStore(ABC):
    """文档存储抽象基类"""
    
    @abstractmethod
    def save_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """保存文档"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索文档"""
        pass
```

#### 1.2.2 core/storage/qdrant.py

```python
"""
文件路径: core/storage/qdrant.py
功能: Qdrant向量数据库存储实现

主要改动：
- 使用新的配置系统替代环境变量
- 继承 VectorStore 抽象基类
- 保持原有功能不变
"""

import logging
import uuid
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings
from .base import VectorStore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, 
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Qdrant连接管理器 - 单例模式防止重复连接"""
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        **kwargs
    ) -> 'QdrantVectorStore':
        """获取或创建Qdrant实例"""
        key = (url or "local", collection_name)
        
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"创建新的Qdrant连接: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        **kwargs
                    )
        
        return cls._instances[key]
    
    @classmethod
    def get_default_instance(cls) -> 'QdrantVectorStore':
        """使用配置创建默认实例"""
        db_settings = settings.database
        embed_settings = settings.embedding
        
        return cls.get_instance(
            url=db_settings.qdrant_url,
            api_key=db_settings.qdrant_api_key,
            collection_name=db_settings.qdrant_collection,
            vector_size=embed_settings.embed_dimension
        )


class QdrantVectorStore(VectorStore):
    """Qdrant向量数据库存储实现
    
    支持本地和云服务两种部署模式。
    """
    
    def __init__(
        self, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client未安装。请运行: pip install qdrant-client>=1.6.0")
        
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        # 距离度量映射
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        
        # 初始化客户端
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化Qdrant客户端和集合"""
        try:
            if self.url and self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
                logger.info(f"成功连接到Qdrant云服务: {self.url}")
            elif self.url:
                self.client = QdrantClient(url=self.url, timeout=self.timeout)
                logger.info(f"成功连接到Qdrant服务: {self.url}")
            else:
                self.client = QdrantClient(host="localhost", port=6333, timeout=self.timeout)
                logger.info("成功连接到本地Qdrant服务")
            
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Qdrant连接失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保集合存在"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
                )
                logger.info(f"创建Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"使用现有Qdrant集合: {self.collection_name}")
            
            self._ensure_payload_indexes()
                
        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise
    
    def _ensure_payload_indexes(self):
        """创建payload索引"""
        index_fields = [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass  # 索引已存在
    
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """添加向量到Qdrant"""
        try:
            if not vectors:
                return False
            
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            points = []
            for vector, meta, point_id in zip(vectors, metadata, ids):
                if len(vector) != self.vector_size:
                    continue
                
                meta_copy = meta.copy()
                meta_copy["timestamp"] = int(datetime.now().timestamp())
                
                # 确保ID格式正确
                safe_id = point_id if isinstance(point_id, str) else str(uuid.uuid4())
                try:
                    uuid.UUID(safe_id)
                except ValueError:
                    safe_id = str(uuid.uuid4())
                
                points.append(PointStruct(id=safe_id, vector=vector, payload=meta_copy))
            
            if not points:
                return False
            
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            logger.info(f"成功添加 {len(points)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            if len(query_vector) != self.vector_size:
                logger.error(f"查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
                return []
            
            query_filter = None
            if where:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in where.items()
                    if isinstance(v, (str, int, float, bool))
                ]
                if conditions:
                    query_filter = Filter(must=conditions)
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            
            return [
                {"id": hit.id, "score": hit.score, "metadata": hit.payload or {}}
                for hit in search_result
            ]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            if not ids:
                return True
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True
            )
            logger.info(f"成功删除 {len(ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def delete_by_filter(self, where: Dict[str, Any]) -> bool:
        """按条件删除向量"""
        try:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in where.items()
            ]
            query_filter = Filter(should=conditions)
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True
            )
            return True
        except Exception as e:
            logger.error(f"按条件删除失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"成功清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "store_type": "qdrant",
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "vector_size": self.vector_size,
            }
        except Exception:
            return {"store_type": "qdrant", "name": self.collection_name}
```

#### 1.2.3 core/storage/neo4j.py

```python
"""
文件路径: core/storage/neo4j.py
功能: Neo4j图数据库存储实现

主要改动：
- 使用新的配置系统
- 继承 GraphStore 抽象基类
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings
from .base import GraphStore

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j图数据库存储实现"""
    
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
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
```

#### 1.2.4 core/storage/__init__.py

```python
"""存储后端模块导出"""

from .base import VectorStore, GraphStore, DocumentStore
from .qdrant import QdrantVectorStore, QdrantConnectionManager
from .neo4j import Neo4jGraphStore

__all__ = [
    # 抽象基类
    "VectorStore",
    "GraphStore", 
    "DocumentStore",
    # Qdrant
    "QdrantVectorStore",
    "QdrantConnectionManager",
    # Neo4j
    "Neo4jGraphStore",
]
```

### 验证方法

```python
# scripts/verify_storage.py
"""存储模块验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    print("=" * 50)
    print("存储模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
    try:
        from core.storage import (
            VectorStore,
            GraphStore,
            QdrantVectorStore,
            Neo4jGraphStore
        )
        print("[OK] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 2. 测试抽象类
    try:
        assert hasattr(VectorStore, 'add_vectors')
        assert hasattr(VectorStore, 'search_similar')
        assert hasattr(GraphStore, 'add_entity')
        assert hasattr(GraphStore, 'find_related_entities')
        print("[OK] 抽象类接口完整")
    except AssertionError as e:
        print(f"[FAIL] 抽象类接口: {e}")
        all_passed = False
    
    # 3. 测试Qdrant类存在
    try:
        assert QdrantVectorStore.__bases__[0].__name__ == 'VectorStore'
        print("[OK] QdrantVectorStore 继承正确")
    except Exception as e:
        print(f"[FAIL] Qdrant继承: {e}")
        all_passed = False
    
    # 4. 测试Neo4j类存在
    try:
        assert Neo4jGraphStore.__bases__[0].__name__ == 'GraphStore'
        print("[OK] Neo4jGraphStore 继承正确")
    except Exception as e:
        print(f"[FAIL] Neo4j继承: {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("存储模块验证通过！")
    return all_passed

if __name__ == "__main__":
    verify()
```

---

## Task 1.3：记忆模块迁移

### 任务描述
将记忆系统代码迁移到 `core/memory/`，保持原有功能。

### 文件迁移映射

| 原文件 | 目标文件 | 操作 |
|--------|----------|------|
| `base.py` | `core/memory/base.py` | 迁移，更新导入 |
| `manager.py` | `core/memory/manager.py` | 迁移，更新导入 |
| `types/working.py` | `core/memory/working.py` | 迁移，更新导入 |
| `types/episodic.py` | `core/memory/episodic.py` | 迁移，更新导入 |
| `types/semantic.py` | `core/memory/semantic.py` | 迁移，更新导入 |
| `types/perceptual.py` | `core/memory/perceptual.py` | 迁移，更新导入 |

### 主要改动

1. **更新导入路径**
   - `from .base import` → 保持不变（同目录）
   - `from ..base import` → `from .base import`
   - `from ..embedding import` → `from core.embedding import`
   - `from ..storage.qdrant_store import` → `from core.storage import`

2. **使用配置系统**
   - 将环境变量读取改为使用 `config.settings`

### core/memory/__init__.py

```python
"""记忆系统核心模块导出"""

from .base import MemoryItem, MemoryConfig, BaseMemory
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .perceptual import PerceptualMemory
from .manager import MemoryManager

__all__ = [
    # 基础类
    "MemoryItem",
    "MemoryConfig",
    "BaseMemory",
    # 记忆类型
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "PerceptualMemory",
    # 管理器
    "MemoryManager",
]
```

### 验证方法

```python
# scripts/verify_memory.py
"""记忆模块验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    print("=" * 50)
    print("记忆模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
    try:
        from core.memory import (
            MemoryItem,
            MemoryConfig,
            BaseMemory,
            WorkingMemory,
            EpisodicMemory,
            SemanticMemory,
            PerceptualMemory,
            MemoryManager
        )
        print("[OK] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 2. 测试MemoryItem创建
    try:
        from datetime import datetime
        item = MemoryItem(
            id="test-1",
            content="测试内容",
            memory_type="working",
            user_id="user-1",
            timestamp=datetime.now(),
            importance=0.5
        )
        assert item.id == "test-1"
        print(f"[OK] MemoryItem创建成功: {item.id}")
    except Exception as e:
        print(f"[FAIL] MemoryItem: {e}")
        all_passed = False
    
    # 3. 测试WorkingMemory
    try:
        config = MemoryConfig()
        wm = WorkingMemory(config)
        
        from datetime import datetime
        item = MemoryItem(
            id="wm-test-1",
            content="工作记忆测试",
            memory_type="working",
            user_id="user-1",
            timestamp=datetime.now(),
            importance=0.7
        )
        
        mem_id = wm.add(item)
        assert wm.has_memory(mem_id)
        
        results = wm.retrieve("测试", limit=5)
        stats = wm.get_stats()
        
        print(f"[OK] WorkingMemory: count={stats['count']}")
    except Exception as e:
        print(f"[FAIL] WorkingMemory: {e}")
        all_passed = False
    
    # 4. 测试MemoryManager
    try:
        manager = MemoryManager(enable_perceptual=False)
        mem_id = manager.add_memory("管理器测试内容", memory_type="working")
        stats = manager.get_memory_stats()
        
        print(f"[OK] MemoryManager: types={stats['enabled_types']}")
    except Exception as e:
        print(f"[FAIL] MemoryManager: {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("记忆模块验证通过！")
    return all_passed

if __name__ == "__main__":
    verify()
```

---

## Task 1.4：RAG模块迁移

### 任务描述
将RAG系统迁移到 `core/rag/`，保持原有功能。

### 文件迁移

| 原文件 | 目标文件 |
|--------|----------|
| `rag/pipeline.py` | `core/rag/pipeline.py` |
| `rag/document.py` | `core/rag/document.py` |

### 主要改动

1. **更新导入路径**
   - `from ..embedding import` → `from core.embedding import`
   - `from ..storage.qdrant_store import` → `from core.storage import`

2. **添加LLM调用工具类**（可选）

### core/rag/__init__.py

```python
"""RAG检索增强生成模块导出"""

from .pipeline import (
    load_and_chunk_texts,
    index_chunks,
    embed_query,
    search_vectors,
    search_vectors_expanded,
    create_rag_pipeline,
    rank,
    merge_snippets,
    merge_snippets_grouped,
)

__all__ = [
    "load_and_chunk_texts",
    "index_chunks",
    "embed_query",
    "search_vectors",
    "search_vectors_expanded",
    "create_rag_pipeline",
    "rank",
    "merge_snippets",
    "merge_snippets_grouped",
]
```

### 验证方法

```python
# scripts/verify_rag.py
"""RAG模块验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    print("=" * 50)
    print("RAG模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
    try:
        from core.rag import (
            load_and_chunk_texts,
            index_chunks,
            embed_query,
            search_vectors,
            create_rag_pipeline
        )
        print("[OK] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 2. 测试文本分块函数
    try:
        import tempfile
        import os
        
        # 创建测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("# 测试标题\n\n这是一段测试文本。\n\n## 子标题\n\n更多内容。")
            test_file = f.name
        
        chunks = load_and_chunk_texts([test_file], chunk_size=100, chunk_overlap=20)
        os.unlink(test_file)
        
        assert len(chunks) > 0, "分块结果为空"
        print(f"[OK] load_and_chunk_texts: {len(chunks)} chunks")
    except Exception as e:
        print(f"[FAIL] load_and_chunk_texts: {e}")
        all_passed = False
    
    # 3. 测试embed_query函数存在
    try:
        assert callable(embed_query)
        print("[OK] embed_query 函数存在")
    except Exception as e:
        print(f"[FAIL] embed_query: {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("RAG模块验证通过！")
    return all_passed

if __name__ == "__main__":
    verify()
```

---

## 阶段1 总验证清单

| 序号 | 任务 | 验证命令 | 预期结果 |
|------|------|----------|----------|
| 1 | 嵌入模块 | `python scripts/verify_embedding.py` | 验证通过 |
| 2 | 存储模块 | `python scripts/verify_storage.py` | 验证通过 |
| 3 | 记忆模块 | `python scripts/verify_memory.py` | 验证通过 |
| 4 | RAG模块 | `python scripts/verify_rag.py` | 验证通过 |
| 5 | 完整导入 | `python -c "from core.embedding import get_text_embedder; from core.memory import MemoryManager; from core.rag import create_rag_pipeline; print('OK')"` | 输出 OK |

---

## 产出物清单

| 文件 | 描述 |
|------|------|
| `core/embedding/base.py` | 嵌入模型基类 |
| `core/embedding/local.py` | 本地Transformer嵌入 |
| `core/embedding/dashscope.py` | DashScope嵌入 |
| `core/embedding/tfidf.py` | TF-IDF嵌入 |
| `core/embedding/provider.py` | 嵌入Provider单例 |
| `core/storage/base.py` | 存储抽象基类 |
| `core/storage/qdrant.py` | Qdrant向量存储 |
| `core/storage/neo4j.py` | Neo4j图存储 |
| `core/memory/base.py` | 记忆基类 |
| `core/memory/manager.py` | 记忆管理器 |
| `core/memory/working.py` | 工作记忆 |
| `core/memory/episodic.py` | 情景记忆 |
| `core/memory/semantic.py` | 语义记忆 |
| `core/memory/perceptual.py` | 感知记忆 |
| `core/rag/pipeline.py` | RAG管道 |

---

## 向后兼容说明

为保持向后兼容，建议在原位置创建兼容模块：

```python
# embedding.py (保留，重定向到新模块)
"""向后兼容：重定向到 core.embedding"""
from core.embedding import *
```

```python
# manager.py (保留，重定向到新模块)
"""向后兼容：重定向到 core.memory"""
from core.memory import MemoryManager, MemoryConfig, MemoryItem
```

---

## 下一阶段预告

**阶段2：服务层实现** 将完成：
- `MemoryService` - 记忆CRUD服务
- `RAGService` - RAG检索服务
- `GraphService` - 知识图谱服务
- `AnalyticsService` - 统计分析服务

