"""配置系统验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    from config import settings
    
    print("=" * 50)
    print("配置系统验证")
    print("=" * 50)
    
    # 验证各配置项
    checks = [
        ("应用名称", settings.app.app_name, bool),
        ("运行环境", settings.app.app_env, lambda x: x in ["development", "production", "testing"]),
        ("Qdrant URL", settings.database.qdrant_url, bool),
        ("Neo4j URI", settings.database.neo4j_uri, bool),
        ("嵌入模型类型", settings.embedding.embed_model_type, lambda x: x in ["dashscope", "local", "tfidf"]),
        ("LLM API Base", settings.llm.llm_api_base, lambda x: x.startswith("http")),
    ]
    
    all_passed = True
    for name, value, validator in checks:
        try:
            passed = validator(value) if callable(validator) else bool(value)
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {value}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"✗ {name}: 验证失败 - {e}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("所有配置验证通过！")
    else:
        print("部分配置验证失败！")
    
    return all_passed

if __name__ == "__main__":
    verify()

