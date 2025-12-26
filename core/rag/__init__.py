"""RAG检索增强生成模块导出

提供完整的RAG功能:
- 文档加载和分块
- 向量索引构建
- 语义检索
- 查询扩展
- 结果排序和合并
"""

from .document import load_and_chunk_texts
from .pipeline import (
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
    # 文档处理
    "load_and_chunk_texts",
    # 索引和检索
    "index_chunks",
    "embed_query",
    "search_vectors",
    "search_vectors_expanded",
    "create_rag_pipeline",
    # 排序和合并
    "rank",
    "merge_snippets",
    "merge_snippets_grouped",
]
