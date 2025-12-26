"""
文件路径: core/embedding/tfidf.py
功能: TF-IDF嵌入模型（轻量级兜底方案）

在无深度学习模型时保证系统可用。
自动使用默认语料初始化，无需手动调用 fit()。
"""

from typing import List, Union
import numpy as np
from .base import EmbeddingModel


# 默认初始化语料（覆盖常见词汇）
_DEFAULT_CORPUS = [
    "记忆 memory 存储 storage 检索 retrieval 查询 query 搜索 search",
    "Python 编程 programming 代码 code 函数 function 类 class 方法 method",
    "人工智能 AI 机器学习 machine learning 深度学习 deep learning 神经网络",
    "数据 data 信息 information 知识 knowledge 文档 document 文本 text",
    "用户 user 系统 system 服务 service 接口 API 请求 request 响应 response",
    "向量 vector 嵌入 embedding 相似度 similarity 距离 distance 维度 dimension",
    "图谱 graph 实体 entity 关系 relationship 节点 node 边 edge",
    "时间 time 日期 date 事件 event 历史 history 记录 record",
    "重要 important 关键 key 核心 core 主要 main 基础 basic",
    "问题 question 答案 answer 解决 solve 处理 process 分析 analyze",
]


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF 简易兜底嵌入
    
    自动初始化，无需手动调用 fit()。
    
    Args:
        max_features: 最大特征数量，即向量维度
        auto_fit: 是否自动使用默认语料初始化
    """

    def __init__(self, max_features: int = 384, auto_fit: bool = True):
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()
        
        # 自动初始化
        if auto_fit:
            self.fit(_DEFAULT_CORPUS)

    def _init_vectorizer(self):
        """初始化TF-IDF向量化器"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features, 
                stop_words=None,  # 保留中文等非英语词汇
                token_pattern=r"(?u)\b\w+\b"  # 匹配单个字符的词
            )
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")

    def fit(self, texts: List[str]):
        """训练TF-IDF模型
        
        Args:
            texts: 训练语料库
        """
        self._vectorizer.fit(texts)
        self._is_fitted = True
        actual_features = len(self._vectorizer.get_feature_names_out())
        self._dimension = actual_features

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量或向量列表
        """
        # 如果未训练，使用默认语料自动初始化
        if not self._is_fitted:
            self.fit(_DEFAULT_CORPUS)
        
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = tfidf_matrix.toarray()
        
        # 确保维度一致（填充或截断）
        result = []
        for emb in embeddings:
            if len(emb) < self.max_features:
                emb = np.pad(emb, (0, self.max_features - len(emb)), mode='constant')
            elif len(emb) > self.max_features:
                emb = emb[:self.max_features]
            result.append(emb)
        
        if single:
            return result[0]
        return result

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self.max_features  # 使用设定的最大特征数

