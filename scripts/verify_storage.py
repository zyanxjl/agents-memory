"""存储模块验证脚本

验证 core/storage 模块的功能完整性。
"""
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
        print("存储模块验证通过!")
    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

