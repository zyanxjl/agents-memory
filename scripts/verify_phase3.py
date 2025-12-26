"""
阶段3验证脚本 - API层实现验证

验证项目:
1. API模块导入
2. FastAPI应用创建
3. 路由注册
4. Schema模型
5. 依赖注入
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
    """验证API模块导入"""
    print("1. 验证API模块导入...")
    try:
        from api.main import app, create_app
        from api.dependencies import get_memory_service, get_rag_service
        from api.schemas import DataResponse, MemoryCreate, SearchQuery
        from api.routes import memory, rag, graph, analytics
        print("  [OK] 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_app_creation():
    """验证FastAPI应用创建"""
    print("2. 验证FastAPI应用创建...")
    try:
        from api.main import create_app
        app = create_app()
        assert app is not None, "应用不应为None"
        assert app.title == "Agent Memory System API", f"标题不匹配: {app.title}"
        print(f"  - 应用标题: {app.title}")
        print(f"  - 应用版本: {app.version}")
        print("  [OK] 应用创建成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_routes():
    """验证路由注册"""
    print("3. 验证路由注册...")
    try:
        from api.main import app
        
        # 获取所有路由路径
        routes = [r.path for r in app.routes]
        
        # 检查关键路由前缀
        required_prefixes = [
            "/api/v1/memory",
            "/api/v1/rag",
            "/api/v1/graph",
            "/api/v1/analytics",
            "/health"
        ]
        
        found_count = 0
        for prefix in required_prefixes:
            matching = [r for r in routes if prefix in r]
            if matching:
                found_count += 1
                print(f"  - 找到路由前缀: {prefix} ({len(matching)} 个端点)")
        
        # 统计总路由数
        api_routes = [r for r in routes if r.startswith("/api")]
        print(f"  - API路由总数: {len(api_routes)}")
        
        success = found_count >= 4
        if success:
            print("  [OK] 路由注册成功")
        else:
            print(f"  [WARN] 只找到 {found_count}/{len(required_prefixes)} 个路由前缀")
        return success
        
    except Exception as e:
        print(f"  [FAIL] 路由验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_schemas():
    """验证Schema模型"""
    print("4. 验证Schema模型...")
    try:
        from api.schemas import (
            DataResponse, ListResponse, ErrorResponse,
            MemoryCreate, MemoryResponse, MemorySearch,
            DocumentInfo, SearchQuery, AskQuery,
            EntityInfo, VisualizationData
        )
        
        # 测试创建模型实例
        resp = DataResponse(success=True, message="测试")
        assert resp.success == True
        assert resp.message == "测试"
        print("  - DataResponse: OK")
        
        mem = MemoryCreate(content="测试记忆内容")
        assert mem.content == "测试记忆内容"
        assert mem.memory_type == "auto"
        assert mem.importance == 0.5
        print("  - MemoryCreate: OK")
        
        search = SearchQuery(query="测试查询")
        assert search.query == "测试查询"
        assert search.limit == 5
        print("  - SearchQuery: OK")
        
        print("  [OK] Schema模型验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Schema验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dependencies():
    """验证依赖注入"""
    print("5. 验证依赖注入...")
    try:
        from api.dependencies import (
            get_memory_service, get_rag_service,
            get_graph_service, get_analytics_service,
            get_current_user_id, PaginationParams
        )
        
        # 测试获取服务（使用 lru_cache，会初始化服务）
        mem_service = get_memory_service()
        assert mem_service is not None
        print("  - MemoryService: OK")
        
        rag_service = get_rag_service()
        assert rag_service is not None
        print("  - RAGService: OK")
        
        graph_service = get_graph_service()
        assert graph_service is not None
        print("  - GraphService: OK")
        
        analytics_service = get_analytics_service()
        assert analytics_service is not None
        print("  - AnalyticsService: OK")
        
        # 测试用户ID获取
        user_id = get_current_user_id()
        assert user_id == "default_user"
        print("  - get_current_user_id: OK")
        
        # 测试分页参数
        pagination = PaginationParams(page=2, page_size=50)
        assert pagination.page == 2
        assert pagination.page_size == 50
        assert pagination.offset == 50
        print("  - PaginationParams: OK")
        
        print("  [OK] 依赖注入验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 依赖注入验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_test_client():
    """使用测试客户端验证API端点"""
    print("6. 验证API端点（TestClient）...")
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # 测试根路径
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("  - GET /: OK")
        
        # 测试健康检查
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print("  - GET /health: OK")
        
        # 测试记忆统计
        response = client.get("/api/v1/memory/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        print("  - GET /api/v1/memory/stats/overview: OK")
        
        # 测试RAG统计
        response = client.get("/api/v1/rag/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        print("  - GET /api/v1/rag/stats: OK")
        
        # 测试图谱统计
        response = client.get("/api/v1/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        print("  - GET /api/v1/graph/stats: OK")
        
        # 测试分析仪表盘
        response = client.get("/api/v1/analytics/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        print("  - GET /api/v1/analytics/dashboard: OK")
        
        print("  [OK] API端点验证通过")
        return True
        
    except Exception as e:
        print(f"  [FAIL] API端点验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证"""
    print("=" * 60)
    print("Agent Memory System - 阶段3验证")
    print("API层实现验证")
    print("=" * 60)
    print()
    
    results = {
        "API模块导入": verify_imports(),
        "FastAPI应用": verify_app_creation(),
        "路由注册": verify_routes(),
        "Schema模型": verify_schemas(),
        "依赖注入": verify_dependencies(),
        "API端点测试": verify_test_client(),
    }
    
    print()
    print("=" * 60)
    print("验证结果:")
    print("-" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print("-" * 60)
    print(f"  通过: {passed}/{total}")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ 阶段3验证通过! API层实现完成。")
        print()
        print("启动API服务器:")
        print("  python -m uvicorn api.main:app --reload --port 8000")
        print()
        print("或直接运行:")
        print("  python api/main.py")
        print()
        print("访问API文档:")
        print("  http://localhost:8000/docs")
        print()
        print("下一步: 可以开始阶段4 - 前端实现")
        return 0
    else:
        print("\n❌ 阶段3验证失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

