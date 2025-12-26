"""阶段0 总验证脚本"""
import sys
from pathlib import Path

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

def verify_directories():
    """验证目录结构"""
    print("\n" + "=" * 50)
    print("1. 目录结构验证")
    print("=" * 50)
    
    required_dirs = [
        "api", "api/routes", "api/schemas", "api/middleware",
        "core", "core/memory", "core/rag", "core/embedding", "core/storage",
        "services", "web", "web/static", "web/templates",
        "config", "utils", "tests", "scripts", "logs", "data"
    ]
    
    all_passed = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {dir_path}/")
        all_passed = all_passed and exists
    
    return all_passed

def verify_files():
    """验证必要文件"""
    print("\n" + "=" * 50)
    print("2. 必要文件验证")
    print("=" * 50)
    
    required_files = [
        "requirements.txt",
        "pyproject.toml",
        "config/__init__.py",
        "config/settings.py",
        "config/logging.py",
        "utils/__init__.py",
        "utils/helpers.py",
    ]
    
    all_passed = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {file_path}")
        all_passed = all_passed and exists
    
    return all_passed

def verify_imports():
    """验证模块导入"""
    print("\n" + "=" * 50)
    print("3. 模块导入验证")
    print("=" * 50)
    
    all_passed = True
    
    # 测试 config 导入
    try:
        from config import settings, get_settings, setup_logging, get_logger
        print("[OK] from config import settings, get_settings, setup_logging, get_logger")
    except ImportError as e:
        print(f"[FAIL] config 导入失败: {e}")
        all_passed = False
    
    # 测试 utils 导入
    try:
        from utils import generate_id, generate_hash, timestamp_now, chunks, safe_get
        print("[OK] from utils import generate_id, generate_hash, timestamp_now, chunks, safe_get")
    except ImportError as e:
        print(f"[FAIL] utils 导入失败: {e}")
        all_passed = False
    
    return all_passed

def verify_config():
    """验证配置系统"""
    print("\n" + "=" * 50)
    print("4. 配置系统验证")
    print("=" * 50)
    
    from config import settings
    
    checks = [
        ("app.app_name", settings.app.app_name),
        ("app.app_env", settings.app.app_env),
        ("database.qdrant_url", settings.database.qdrant_url),
        ("database.neo4j_uri", settings.database.neo4j_uri),
        ("embedding.embed_model_type", settings.embedding.embed_model_type),
        ("llm.llm_api_base", settings.llm.llm_api_base),
    ]
    
    all_passed = True
    for name, value in checks:
        passed = bool(value)
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}: {value}")
        all_passed = all_passed and passed
    
    return all_passed

def verify_utils():
    """验证工具函数"""
    print("\n" + "=" * 50)
    print("5. 工具函数验证")
    print("=" * 50)
    
    from utils import generate_id, generate_hash, timestamp_now, truncate_text, chunks, safe_get
    
    all_passed = True
    
    # generate_id
    id1 = generate_id("test")
    passed = id1.startswith("test_") and len(id1) > 5
    print(f"{'[OK]' if passed else '[FAIL]'} generate_id: {id1[:30]}...")
    all_passed = all_passed and passed
    
    # generate_hash
    h = generate_hash("hello")
    passed = len(h) == 64
    print(f"{'[OK]' if passed else '[FAIL]'} generate_hash: {h[:16]}...")
    all_passed = all_passed and passed
    
    # timestamp_now
    ts = timestamp_now()
    passed = "T" in ts
    print(f"{'[OK]' if passed else '[FAIL]'} timestamp_now: {ts}")
    all_passed = all_passed and passed
    
    # truncate_text
    t = truncate_text("hello world", 8)
    passed = len(t) == 8
    print(f"{'[OK]' if passed else '[FAIL]'} truncate_text: '{t}'")
    all_passed = all_passed and passed
    
    # chunks
    c = list(chunks([1,2,3,4,5], 2))
    passed = len(c) == 3
    print(f"{'[OK]' if passed else '[FAIL]'} chunks: {c}")
    all_passed = all_passed and passed
    
    # safe_get
    v = safe_get({"a": {"b": 1}}, "a.b")
    passed = v == 1
    print(f"{'[OK]' if passed else '[FAIL]'} safe_get: a.b = {v}")
    all_passed = all_passed and passed
    
    return all_passed

def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("       Phase 0 - Complete Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Directory Structure", verify_directories()))
    results.append(("Required Files", verify_files()))
    results.append(("Module Imports", verify_imports()))
    results.append(("Config System", verify_config()))
    results.append(("Utility Functions", verify_utils()))
    
    print("\n" + "=" * 60)
    print("       Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {name}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    if all_passed:
        print("Phase 0 - All verifications passed!")
    else:
        print("Some verifications failed. Please check the output above.")
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
