"""
阶段2验证脚本 - 服务层实现验证

验证项目:
1. 服务层模块导入
2. MemoryService 功能验证
3. RAGService 功能验证
4. GraphService 功能验证
5. AnalyticsService 功能验证
"""

import sys
import os

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置控制台编码（Windows）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def verify_imports():
    """验证服务层导入"""
    print("1. 验证服务层导入...")
    try:
        from services import (
            MemoryService,
            RAGService,
            GraphService,
            AnalyticsService
        )
        print("  [OK] 所有服务类导入成功")
        return True
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")
        return False


def verify_memory_service():
    """验证 MemoryService"""
    print("2. 验证 MemoryService...")
    try:
        from services.memory_service import (
            MemoryService, 
            MemoryCreateRequest,
            MemoryUpdateRequest,
            MemorySearchRequest
        )
        
        # 初始化服务
        service = MemoryService()
        print("  - 服务初始化成功")
        
        # 测试添加记忆
        request = MemoryCreateRequest(
            content="这是一个测试记忆，用于验证MemoryService功能。",
            memory_type="working",
            user_id="test_user",
            importance=0.7
        )
        response = service.add_memory(request)
        assert response.id, "记忆ID不应为空"
        memory_id = response.id
        print(f"  - 添加记忆成功: {memory_id[:8]}...")
        
        # 测试获取记忆
        retrieved = service.get_memory(memory_id)
        assert retrieved is not None, "应能获取到记忆"
        print("  - 获取记忆成功")
        
        # 测试更新记忆
        update_req = MemoryUpdateRequest(importance=0.9)
        updated = service.update_memory(memory_id, update_req)
        assert updated, "更新应成功"
        print("  - 更新记忆成功")
        
        # 测试搜索记忆
        search_req = MemorySearchRequest(
            query="测试记忆",
            memory_types=["working"],
            limit=5
        )
        results = service.search_memories(search_req)
        print(f"  - 搜索记忆成功: 找到 {len(results)} 条")
        
        # 测试统计
        stats = service.get_stats()
        assert stats.total_count >= 0
        print(f"  - 获取统计成功: 总数={stats.total_count}")
        
        # 测试删除记忆
        deleted = service.delete_memory(memory_id)
        assert deleted, "删除应成功"
        print("  - 删除记忆成功")
        
        print("  [OK] MemoryService 验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] MemoryService 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_rag_service():
    """验证 RAGService"""
    print("3. 验证 RAGService...")
    try:
        from services.rag_service import RAGService, SearchRequest
        
        # 初始化服务
        service = RAGService()
        print("  - 服务初始化成功")
        
        # 测试基础检索（即使没有文档也应正常运行）
        request = SearchRequest(
            query="test query",
            user_id="test_user",
            limit=5
        )
        results = service.search(request)
        assert isinstance(results, list)
        print(f"  - 基础检索成功: 结果数={len(results)}")
        
        # 测试统计
        stats = service.get_stats()
        assert hasattr(stats, "total_documents")
        print(f"  - 获取统计成功: 文档数={stats.total_documents}")
        
        print("  [OK] RAGService 验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] RAGService 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_graph_service():
    """验证 GraphService"""
    print("4. 验证 GraphService...")
    try:
        from services.graph_service import GraphService, EntitySearchRequest
        
        # 初始化服务
        service = GraphService()
        print("  - 服务初始化成功")
        
        # 测试健康检查
        is_healthy = service.health_check()
        status = "连接正常" if is_healthy else "Neo4j未连接（可忽略）"
        print(f"  - 健康检查: {status}")
        
        # 测试统计
        stats = service.get_stats()
        assert hasattr(stats, "total_entities")
        print(f"  - 获取统计成功: 实体数={stats.total_entities}")
        
        # 测试搜索（即使没有数据也应正常）
        request = EntitySearchRequest(query="test", limit=10)
        results = service.search_entities(request)
        assert isinstance(results, list)
        print(f"  - 搜索实体成功: 结果数={len(results)}")
        
        # 测试可视化数据
        viz_data = service.get_visualization_data(depth=1, limit=10)
        assert hasattr(viz_data, "nodes")
        assert hasattr(viz_data, "links")
        print(f"  - 可视化数据成功: 节点数={len(viz_data.nodes)}")
        
        print("  [OK] GraphService 验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] GraphService 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_analytics_service():
    """验证 AnalyticsService"""
    print("5. 验证 AnalyticsService...")
    try:
        from services.analytics_service import AnalyticsService
        
        # 初始化服务
        service = AnalyticsService()
        print("  - 服务初始化成功")
        
        # 测试仪表盘摘要
        summary = service.get_dashboard_summary()
        assert hasattr(summary, "total_memories")
        assert hasattr(summary, "memory_distribution")
        assert hasattr(summary, "storage_status")
        print(f"  - 仪表盘摘要成功: 总记忆数={summary.total_memories}")
        
        # 测试趋势报告
        trend = service.get_trend_report(period="day")
        assert trend.period == "day"
        assert hasattr(trend, "memory_growth")
        print(f"  - 趋势报告成功: 周期={trend.period}")
        
        # 测试系统健康
        health = service.get_system_health()
        assert health.overall_status in ("healthy", "degraded", "unhealthy")
        assert hasattr(health, "components")
        print(f"  - 系统健康成功: 状态={health.overall_status}")
        
        # 测试活动记录
        service.log_activity(action="test", user_id="test_user", details={"test": True})
        print("  - 活动记录成功")
        
        # 测试查询统计
        query_stats = service.get_query_stats(days=7)
        assert isinstance(query_stats, dict)
        print(f"  - 查询统计成功: {len(query_stats)} 天")
        
        print("  [OK] AnalyticsService 验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] AnalyticsService 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证"""
    print("=" * 60)
    print("Agent Memory System - 阶段2验证")
    print("服务层实现验证")
    print("=" * 60)
    print()
    
    results = {
        "服务层导入": verify_imports(),
        "MemoryService": verify_memory_service(),
        "RAGService": verify_rag_service(),
        "GraphService": verify_graph_service(),
        "AnalyticsService": verify_analytics_service(),
    }
    
    print()
    print("=" * 60)
    print("验证结果:")
    print("-" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"  通过: {passed}/{total}")
    print("=" * 60)
    
    if passed == total:
        print("阶段2验证通过! 服务层实现完成。")
        print()
        print("下一步: 可以开始阶段3 - API层实现")
        return 0
    else:
        print("阶段2验证失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

