"""
文件路径: core/embedding/local.py
功能: 本地Transformer嵌入模型

优先使用 sentence-transformers，缺失时回退到 transformers+torch。
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

