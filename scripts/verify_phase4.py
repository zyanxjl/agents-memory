"""
文件路径: scripts/verify_phase4.py
功能: 验证 Phase 4 - 前端界面实现

验证内容:
1. 静态文件存在性
2. 模板文件存在性
3. 页面路由功能
4. FastAPI应用集成测试
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_static_files() -> tuple[bool, str]:
    """验证静态文件"""
    static_dir = project_root / "web" / "static"
    required_files = [
        "css/custom.css",
        "js/api.js",
        "js/utils.js",
    ]
    
    missing = []
    for f in required_files:
        if not (static_dir / f).exists():
            missing.append(f)
    
    if missing:
        return False, f"缺少静态文件: {missing}"
    
    return True, f"静态文件验证通过 ({len(required_files)} 个文件)"


def verify_template_files() -> tuple[bool, str]:
    """验证模板文件"""
    template_dir = project_root / "web" / "templates"
    required_files = [
        "base.html",
        "components/sidebar.html",
        "components/header.html",
        "pages/dashboard.html",
        "pages/memory/list.html",
        "pages/memory/search.html",
        "pages/rag/documents.html",
        "pages/rag/search.html",
        "pages/rag/chat.html",
        "pages/graph/explorer.html",
        "pages/settings.html",
    ]
    
    missing = []
    for f in required_files:
        if not (template_dir / f).exists():
            missing.append(f)
    
    if missing:
        return False, f"缺少模板文件: {missing}"
    
    return True, f"模板文件验证通过 ({len(required_files)} 个文件)"


def verify_pages_route() -> tuple[bool, str]:
    """验证页面路由模块"""
    try:
        from api.routes import pages
        
        # 检查路由器
        if not hasattr(pages, 'router'):
            return False, "pages模块缺少router"
        
        # 检查路由数量
        routes = [r for r in pages.router.routes if hasattr(r, 'path')]
        route_paths = [r.path for r in routes]
        
        expected_routes = ['/', '/memory', '/memory/search', '/rag', '/rag/search', 
                          '/rag/chat', '/graph', '/settings']
        
        missing_routes = [r for r in expected_routes if r not in route_paths]
        
        if missing_routes:
            return False, f"缺少路由: {missing_routes}"
        
        return True, f"页面路由验证通过 ({len(routes)} 个路由)"
    except Exception as e:
        return False, f"页面路由验证失败: {e}"


def verify_app_integration() -> tuple[bool, str]:
    """验证FastAPI应用集成"""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # 测试页面访问
        pages_to_test = [
            ('/', 'dashboard'),
            ('/memory', 'memory'),
            ('/memory/search', 'search'),
            ('/rag', 'rag'),
            ('/settings', 'settings'),
        ]
        
        results = []
        for path, name in pages_to_test:
            try:
                response = client.get(path)
                if response.status_code == 200:
                    results.append(f"  [OK] {path}")
                else:
                    results.append(f"  [FAIL] {path}: {response.status_code}")
            except Exception as e:
                results.append(f"  [FAIL] {path}: {e}")
        
        # 测试静态文件
        static_response = client.get('/static/css/custom.css')
        if static_response.status_code == 200:
            results.append("  [OK] Static files")
        else:
            results.append(f"  [FAIL] Static files: {static_response.status_code}")
        
        # 测试 API 端点仍然可用
        api_response = client.get('/health')
        if api_response.status_code == 200:
            results.append("  [OK] API /health")
        else:
            results.append(f"  [FAIL] API /health: {api_response.status_code}")
        
        success = all('[OK]' in r for r in results)
        return success, "App integration test:\n" + "\n".join(results)
        
    except Exception as e:
        return False, f"应用集成测试失败: {e}"


def main():
    """主验证函数"""
    print("=" * 60)
    print("Phase 4 - Frontend Implementation Verification")
    print("=" * 60)
    
    tests = [
        ("Static Files", verify_static_files),
        ("Template Files", verify_template_files),
        ("Page Routes", verify_pages_route),
        ("App Integration", verify_app_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[TEST] {name}...")
        try:
            success, message = test_func()
            results.append((name, success, message))
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {message}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"[FAIL] Error: {e}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, message in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n*** Phase 4 Verification PASSED! ***")
        print("\nStart server:")
        print("  cd E:\\agent_learn\\agents_memory")
        print("  python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        print("\nVisit:")
        print("  http://localhost:8000")
        print("  http://localhost:8000/docs (API Docs)")
        return 0
    else:
        print("\n*** Some tests FAILED. Please check errors above. ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())

