"""
文件路径: core/embedding/dashscope.py
功能: 阿里云DashScope嵌入模型（支持REST和SDK两种模式）

支持两种模式：
- REST模式：提供 base_url 时使用 OpenAI 兼容的 REST 接口
- SDK模式：使用官方 dashscope SDK
"""

import os
from typing import List, Union, Optional
import numpy as np
from .base import EmbeddingModel


class DashScopeEmbedding(EmbeddingModel):
    """阿里云 DashScope（通义千问）Embedding
    
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

