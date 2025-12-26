"""RAG模块验证脚本

验证 core/rag 模块的功能完整性。
"""
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
    
    # 4. 测试create_rag_pipeline函数存在
    try:
        assert callable(create_rag_pipeline)
        print("[OK] create_rag_pipeline 函数存在")
    except Exception as e:
        print(f"[FAIL] create_rag_pipeline: {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("RAG模块验证通过!")
    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

