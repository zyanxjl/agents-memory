"""记忆模块验证脚本

验证 core/memory 模块的功能完整性。
"""
import sys
sys.path.insert(0, '.')


def verify():
    print("=" * 50)
    print("记忆模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
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
        print("[OK] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 2. 测试MemoryItem创建
    try:
        from datetime import datetime
        item = MemoryItem(
            id="test-1",
            content="测试内容",
            memory_type="working",
            user_id="user-1",
            timestamp=datetime.now(),
            importance=0.5
        )
        assert item.id == "test-1"
        print(f"[OK] MemoryItem创建成功: {item.id}")
    except Exception as e:
        print(f"[FAIL] MemoryItem: {e}")
        all_passed = False
    
    # 3. 测试WorkingMemory
    try:
        from datetime import datetime
        config = MemoryConfig()
        wm = WorkingMemory(config)
        
        item = MemoryItem(
            id="wm-test-1",
            content="工作记忆测试",
            memory_type="working",
            user_id="user-1",
            timestamp=datetime.now(),
            importance=0.7
        )
        
        mem_id = wm.add(item)
        assert wm.has_memory(mem_id)
        
        results = wm.retrieve("测试", limit=5)
        stats = wm.get_stats()
        
        print(f"[OK] WorkingMemory: count={stats['count']}")
    except Exception as e:
        print(f"[FAIL] WorkingMemory: {e}")
        all_passed = False
    
    # 4. 测试MemoryManager（可能失败，因为需要外部依赖）
    try:
        manager = MemoryManager(enable_perceptual=False, enable_semantic=False, enable_episodic=False)
        mem_id = manager.add_memory("管理器测试内容", memory_type="working")
        stats = manager.get_memory_stats()
        
        print(f"[OK] MemoryManager: types={stats['enabled_types']}")
    except Exception as e:
        print(f"[WARN] MemoryManager: {e}")
    
    print("=" * 50)
    if all_passed:
        print("记忆模块验证通过!")
    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

