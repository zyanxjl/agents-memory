"""
文件路径: core/rag/pipeline.py
功能: RAG检索增强生成管道

提供:
- 向量索引构建
- 语义检索
- 查询扩展（MQE/HyDE）
- 重排序
- 结果合并
"""

from typing import List, Dict, Optional, Any
import os

from core.embedding import get_text_embedder, get_dimension
from core.storage import QdrantVectorStore, QdrantConnectionManager
from .document import load_and_chunk_texts


def _preprocess_markdown_for_embedding(text: str) -> str:
    """预处理Markdown文本用于嵌入"""
    import re
    
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def _create_default_vector_store(dimension: int = None) -> QdrantVectorStore:
    """创建默认向量存储"""
    if dimension is None:
        dimension = get_dimension(384)
    
    from config import settings
    db_settings = settings.database
    
    return QdrantConnectionManager.get_instance(
        url=db_settings.qdrant_url,
        api_key=db_settings.qdrant_api_key,
        collection_name="hello_agents_rag_vectors",
        vector_size=dimension,
        distance="cosine"
    )


def index_chunks(
    store=None, 
    chunks: List[Dict] = None, 
    cache_db: Optional[str] = None, 
    batch_size: int = 64,
    rag_namespace: str = "default"
) -> None:
    """索引分块到向量数据库
    
    Args:
        store: 向量存储实例
        chunks: 分块列表
        cache_db: 缓存数据库路径（未使用）
        batch_size: 批处理大小
        rag_namespace: RAG命名空间
    """
    if not chunks:
        print("[RAG] 无分块需要索引")
        return
    
    embedder = get_text_embedder()
    dimension = get_dimension(384)
    
    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] 创建默认Qdrant存储，维度={dimension}")
    
    # 预处理文本
    processed_texts = [_preprocess_markdown_for_embedding(c["content"]) for c in chunks]
    
    print(f"[RAG] 开始嵌入: 总数={len(processed_texts)} 批大小={batch_size}")
    
    # 批量编码
    vecs: List[List[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i:i+batch_size]
        try:
            part_vecs = embedder.encode(part)
            
            if not isinstance(part_vecs, list):
                if hasattr(part_vecs, "tolist"):
                    part_vecs = [part_vecs.tolist()]
                else:
                    part_vecs = [list(part_vecs)]
            else:
                if part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                    if hasattr(part_vecs[0], "tolist"):
                        part_vecs = [v.tolist() for v in part_vecs]
                    else:
                        part_vecs = [list(v) for v in part_vecs]
            
            for v in part_vecs:
                if hasattr(v, "tolist"):
                    v = v.tolist()
                v_norm = [float(x) for x in v]
                if len(v_norm) != dimension:
                    if len(v_norm) < dimension:
                        v_norm.extend([0.0] * (dimension - len(v_norm)))
                    else:
                        v_norm = v_norm[:dimension]
                vecs.append(v_norm)
                
        except Exception as e:
            print(f"[WARNING] 批次 {i} 编码失败: {e}")
            for _ in range(len(part)):
                vecs.append([0.0] * dimension)
        
        print(f"[RAG] 嵌入进度: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")
    
    # 准备元数据
    metas: List[Dict] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],
            "data_source": "rag_pipeline",
            "rag_namespace": rag_namespace,
            "is_rag_data": True,
        }
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])
    
    print(f"[RAG] 开始upsert: n={len(vecs)}")
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Upsert完成: {len(vecs)} 向量已索引")
    else:
        print("[RAG] Upsert失败")
        raise RuntimeError("向量索引失败")


def embed_query(query: str) -> List[float]:
    """嵌入查询"""
    embedder = get_text_embedder()
    dimension = get_dimension(384)
    
    try:
        vec = embedder.encode(query)
        
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        
        if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]
        
        result = [float(x) for x in vec]
        
        if len(result) != dimension:
            if len(result) < dimension:
                result.extend([0.0] * (dimension - len(result)))
            else:
                result = result[:dimension]
        
        return result
    except Exception as e:
        print(f"[WARNING] 查询嵌入失败: {e}")
        return [0.0] * dimension


def search_vectors(
    store=None, 
    query: str = "", 
    top_k: int = 8, 
    rag_namespace: Optional[str] = None, 
    only_rag_data: bool = True, 
    score_threshold: Optional[float] = None
) -> List[Dict]:
    """搜索RAG向量"""
    if not query:
        return []
    
    if store is None:
        store = _create_default_vector_store()
    
    qv = embed_query(query)
    
    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    
    try:
        return store.search_similar(
            query_vector=qv, 
            limit=top_k, 
            score_threshold=score_threshold, 
            where=where
        )
    except Exception as e:
        print(f"[WARNING] RAG搜索失败: {e}")
        return []


def search_vectors_expanded(
    store=None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    only_rag_data: bool = True,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """带查询扩展的搜索"""
    if not query:
        return []
    
    if store is None:
        store = _create_default_vector_store()
    
    expansions: List[str] = [query]
    
    # MQE扩展
    if enable_mqe and mqe_expansions > 0:
        try:
            expansions.extend(_prompt_mqe(query, mqe_expansions))
        except Exception:
            pass
    
    # HyDE扩展
    if enable_hyde:
        try:
            hyde_text = _prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)
        except Exception:
            pass

    # 去重
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h
    
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


def _prompt_mqe(query: str, n: int) -> List[str]:
    """MQE多查询扩展（需要LLM支持）"""
    return [query]


def _prompt_hyde(query: str) -> Optional[str]:
    """HyDE假设文档嵌入（需要LLM支持）"""
    return None


def rank(
    vector_hits: List[Dict], 
    graph_signals: Optional[Dict[str, float]] = None, 
    w_vector: float = 0.7, 
    w_graph: float = 0.3
) -> List[Dict]:
    """排序结果"""
    items: List[Dict] = []
    graph_signals = graph_signals or {}
    
    for h in vector_hits:
        mid = h.get("metadata", {}).get("memory_id", h.get("id"))
        g = float(graph_signals.get(mid, 0.0))
        v = float(h.get("score", 0.0))
        score = w_vector * v + w_graph * g
        items.append({
            "memory_id": mid,
            "score": score,
            "vector_score": v,
            "graph_score": g,
            "content": h.get("metadata", {}).get("content", ""),
            "metadata": h.get("metadata", {}),
        })
    
    items.sort(key=lambda x: x["score"], reverse=True)
    return items


def merge_snippets(ranked_items: List[Dict], max_chars: int = 1200) -> str:
    """合并片段"""
    out: List[str] = []
    total = 0
    
    for it in ranked_items:
        text = it.get("content", "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            out.append(text[:remain])
            total += remain
            break
        out.append(text)
        total += len(text)
    
    return "\n\n".join(out)


def merge_snippets_grouped(ranked_items: List[Dict], max_chars: int = 1200, include_citations: bool = True) -> str:
    """按文档分组合并片段"""
    by_doc: Dict[str, List[Dict]] = {}
    doc_score: Dict[str, float] = {}
    
    for it in ranked_items:
        meta = it.get("metadata", {})
        did = meta.get("doc_id") or meta.get("source_path") or "unknown"
        by_doc.setdefault(did, []).append(it)
        doc_score[did] = doc_score.get(did, 0.0) + float(it.get("score", 0.0))
    
    ordered_docs = sorted(by_doc.keys(), key=lambda d: doc_score.get(d, 0.0), reverse=True)
    
    for d in ordered_docs:
        by_doc[d].sort(key=lambda x: (x.get("metadata", {}).get("start", 0)))
    
    out: List[str] = []
    citations: List[Dict] = []
    total = 0
    cite_index = 1
    
    for did in ordered_docs:
        parts = by_doc[did]
        for it in parts:
            text = (it.get("content", "") or "").strip()
            if not text:
                continue
            
            suffix = ""
            if include_citations:
                suffix = f" [{cite_index}]"
            need = len(text) + (len(suffix) if suffix else 0)
            
            if total + need > max_chars:
                break
            
            out.append(text + suffix)
            total += need
            
            if include_citations:
                m = it.get("metadata", {})
                citations.append({
                    "index": cite_index,
                    "source_path": m.get("source_path"),
                    "doc_id": m.get("doc_id"),
                    "start": m.get("start"),
                    "end": m.get("end"),
                    "heading_path": m.get("heading_path"),
                })
                cite_index += 1
        
        if total >= max_chars:
            break
    
    merged = "\n\n".join(out)
    
    if include_citations and citations:
        lines: List[str] = [merged, "", "References:"]
        for c in citations:
            loc = ""
            if c.get("start") is not None and c.get("end") is not None:
                loc = f" ({c['start']}-{c['end']})"
            hp = f" – {c['heading_path']}" if c.get("heading_path") else ""
            sp = c.get("source_path") or c.get("doc_id") or "source"
            lines.append(f"[{c['index']}] {sp}{loc}{hp}")
        return "\n".join(lines)
    
    return merged


def create_rag_pipeline(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "hello_agents_rag_vectors",
    rag_namespace: str = "default"
) -> Dict[str, Any]:
    """创建完整的RAG管道
    
    Args:
        qdrant_url: Qdrant URL
        qdrant_api_key: Qdrant API Key
        collection_name: 集合名称
        rag_namespace: 命名空间
        
    Returns:
        包含store、namespace和辅助函数的字典
    """
    dimension = get_dimension(384)
    
    store = QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=dimension,
        distance="cosine"
    )
    
    def add_documents(file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100):
        """添加文档到RAG管道"""
        chunks = load_and_chunk_texts(
            paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag"
        )
        index_chunks(
            store=store,
            chunks=chunks,
            rag_namespace=rag_namespace
        )
        return len(chunks)
    
    def search(query: str, top_k: int = 8, score_threshold: Optional[float] = None):
        """搜索RAG知识库"""
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold
        )
    
    def search_advanced(
        query: str, 
        top_k: int = 8, 
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None
    ):
        """高级搜索（带查询扩展）"""
        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold
        )
    
    def get_stats():
        """获取管道统计"""
        return store.get_stats()
    
    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats
    }

