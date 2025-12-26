"""嵌入模块验证脚本

验证 core/embedding 模块的功能完整性。
"""
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
        print("嵌入模块验证通过!")
    else:
        print("部分验证失败")
    
    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

