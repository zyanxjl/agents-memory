"""
文件路径: core/embedding/base.py
功能: 嵌入模型抽象基类

定义所有嵌入模型必须实现的最小接口:
- encode(): 将文本编码为向量
- dimension: 返回嵌入向量的维度
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

