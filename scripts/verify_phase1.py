"""阶段1总验证脚本

验证阶段1所有模块的重构是否完成。
"""
import sys
sys.path.insert(0, '.')


def main():
    print("=" * 60)
    print("Agent Memory System - 阶段1验证")
    print("核心层重构验证")
    print("=" * 60)
    print()
    
    all_passed = True
    results = []
    
    # 1. 验证嵌入模块
    print("1. 验证嵌入模块...")
    try:
        from core.embedding import (
            EmbeddingModel,
            LocalTransformerEmbedding,
            DashScopeEmbedding,
            TFIDFEmbedding,
            get_text_embedder,
            get_dimension,
            create_embedding_model,
            create_embedding_model_with_fallback,
            refresh_embedder
        )
        results.append(("嵌入模块导入", True))
        
        # 测试TF-IDF
        tfidf = TFIDFEmbedding(max_features=50)
        tfidf.fit(["test", "document"])
        vec = tfidf.encode("test")
        assert len(vec) == tfidf.dimension
        results.append(("TF-IDF嵌入", True))
    except Exception as e:
        results.append(("嵌入模块", False))
        print(f"   [FAIL] {e}")
        all_passed = False
    
    # 2. 验证存储模块
    print("2. 验证存储模块...")
    try:
        from core.storage import (
            VectorStore,
            GraphStore,
            DocumentStore,
            QdrantVectorStore,
            QdrantConnectionManager,
            Neo4jGraphStore
        )
        results.append(("存储模块导入", True))
        
        # 验证抽象类
        assert hasattr(VectorStore, 'add_vectors')
        assert hasattr(VectorStore, 'search_similar')
        assert hasattr(GraphStore, 'add_entity')
        results.append(("存储抽象类", True))
    except Exception as e:
        results.append(("存储模块", False))
        print(f"   [FAIL] {e}")
        all_passed = False
    
    # 3. 验证记忆模块
    print("3. 验证记忆模块...")
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
        results.append(("记忆模块导入", True))
        
        # 测试WorkingMemory
        from datetime import datetime
        config = MemoryConfig()
        wm = WorkingMemory(config)
        item = MemoryItem(
            id="test-verify",
            content="验证测试",
            memory_type="working",
            user_id="test",
            timestamp=datetime.now(),
            importance=0.5
        )
        wm.add(item)
        assert wm.has_memory("test-verify")
        results.append(("WorkingMemory", True))
    except Exception as e:
        results.append(("记忆模块", False))
        print(f"   [FAIL] {e}")
        all_passed = False
    
    # 4. 验证RAG模块
    print("4. 验证RAG模块...")
    try:
        from core.rag import (
            load_and_chunk_texts,
            index_chunks,
            embed_query,
            search_vectors,
            search_vectors_expanded,
            create_rag_pipeline,
            rank,
            merge_snippets,
            merge_snippets_grouped
        )
        results.append(("RAG模块导入", True))
        
        # 测试文本分块
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("测试内容")
            test_file = f.name
        chunks = load_and_chunk_texts([test_file])
        os.unlink(test_file)
        assert len(chunks) > 0
        results.append(("文本分块", True))
    except Exception as e:
        results.append(("RAG模块", False))
        print(f"   [FAIL] {e}")
        all_passed = False
    
    # 5. 验证完整导入
    print("5. 验证完整导入...")
    try:
        from core.embedding import get_text_embedder
        from core.memory import MemoryManager
        from core.rag import create_rag_pipeline
        results.append(("完整导入", True))
    except Exception as e:
        results.append(("完整导入", False))
        print(f"   [FAIL] {e}")
        all_passed = False
    
    # 输出结果
    print()
    print("=" * 60)
    print("验证结果:")
    print("-" * 60)
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
    print("-" * 60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"  通过: {passed_count}/{total_count}")
    print("=" * 60)
    
    if all_passed:
        print("阶段1验证通过! 核心层重构完成。")
        print()
        print("下一步: 可以开始阶段2 - 服务层实现")
    else:
        print("阶段1验证失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

